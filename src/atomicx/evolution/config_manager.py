"""Live configuration management with regime-specific overrides.

Enables autonomous evolution by making all hardcoded parameters configurable
and trackable. Configuration changes are persisted to the database immediately
and can be rolled back if performance degrades.

Features:
- Regime-specific overrides (e.g., different thresholds in TRENDING_BULLISH vs RANGING)
- Fallback chain: regime-specific → global → default
- Performance tracking per config change
- Version history for rollback
- Async-safe with local cache
"""

from typing import Any
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from loguru import logger
from sqlalchemy import select, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atomicx.data.storage.models import LiveConfig
from atomicx.data.storage.database import get_session


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION VALUES
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ═══ Fusion Engine ═══
    "fusion.bet_threshold": 0.72,
    "fusion.stay_out_zone_low": 0.40,
    "fusion.stay_out_zone_high": 0.60,
    "fusion.min_agent_count": 5,
    "fusion.min_causal_score": 0.60,
    "fusion.dual_confirm_weight_pattern": 0.50,
    "fusion.dual_confirm_weight_causal": 0.50,

    # ═══ Prediction Verification ═══
    "prediction.verification_delay_seconds": 900,  # 15 minutes
    "prediction.correctness_threshold": 0.001,  # 0.1% price movement
    "prediction.max_pending_age_seconds": 1800,  # 30 minutes

    # ═══ Agent Behavior ═══
    "agent.min_confidence_to_signal": 0.40,
    "agent.max_confidence_to_signal": 0.95,
    "agent.auto_prune_after": 200,  # Predictions before removal
    "agent.min_edge_threshold": 0.02,  # 2% edge required
    "agent.weight_adjustment_rate": 0.05,
    "agent.weight_min": 0.1,
    "agent.weight_max": 2.0,
    "agent.disable_threshold": 0.30,  # Win rate below this → disable
    "agent.mutate_threshold": 0.40,  # Win rate below this → mutate

    # ═══ Causal Discovery ═══
    "causal.discovery_interval_minutes": 60,
    "causal.min_data_points": 100,
    "causal.confidence_threshold": 0.70,
    "causal.max_lag": 5,
    "causal.alpha": 0.05,  # Significance level

    # ═══ CausalRL ═══
    "causalrl.learning_rate": 0.001,
    "causalrl.discount_factor": 0.95,
    "causalrl.exploration_rate": 0.10,
    "causalrl.weight_decay": 0.95,
    "causalrl.update_interval_cycles": 10,

    # ═══ Strategy Genome ═══
    "genome.mutation_rate": 0.15,
    "genome.parameter_variance": 0.10,
    "genome.max_generation": 50,
    "genome.prune_after_generations": 10,
    "genome.elite_survival_rate": 0.20,

    # ═══ Pattern Library ═══
    "pattern.min_occurrences": 10,
    "pattern.similarity_threshold": 0.85,
    "pattern.max_age_days": 90,

    # ═══ Memory Tiers ═══
    "memory.tier1_retention_hours": 24,
    "memory.tier2_retention_days": 30,
    "memory.tier3_retention_days": 180,
    "memory.tier4_permanent": True,

    # ═══ News Sentiment ═══
    "news.sentiment_weight": 0.15,
    "news.fetch_interval_minutes": 5,
    "news.max_age_hours": 4,
    "news.min_relevance_score": 0.50,

    # ═══ Regime Detection ═══
    "regime.transition_threshold": 0.75,
    "regime.min_stability_cycles": 3,
    "regime.volatility_window": 20,

    # ═══ Evolution System ═══
    "evolution.diagnosis_interval_cycles": 50,
    "evolution.weight_evolution_interval_cycles": 100,
    "evolution.report_interval_cycles": 100,
    "evolution.min_proposal_confidence": 0.70,
    "evolution.shadow_test_cycles": 20,
    "evolution.deployment_monitor_cycles": 30,
    "evolution.auto_promote_delta": 0.05,  # 5% improvement
    "evolution.auto_rollback_delta": -0.10,  # 10% degradation
    "evolution.max_code_changes_per_100_cycles": 1,
    "evolution.code_change_min_confidence": 0.85,

    # ═══ Dashboard ═══
    "dashboard.refresh_interval_ms": 2000,
    "dashboard.max_history_points": 200,
    "dashboard.alert_threshold_winrate": 0.45,

    # ═══ Swarm Simulation ═══
    "swarm.fast_agent_count": 100,    # 80% of predictions (lightweight)
    "swarm.medium_agent_count": 500,  # 15% of predictions (moderate)
    "swarm.deep_agent_count": 1000,   # 5% of predictions (deep analysis)
    "swarm.simulation_steps": 100,    # Number of simulation steps
}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """Live configuration management with regime-specific overrides.

    Provides a centralized system for reading and updating configuration values
    that can be changed autonomously by the evolution system.

    Usage:
        config = ConfigManager()
        await config.initialize()

        # Get value with regime fallback
        threshold = config.get("fusion.bet_threshold", regime="TRENDING_BULLISH")

        # Update value
        await config.set(
            key="fusion.bet_threshold",
            value=0.75,
            regime="TRENDING_BULLISH",
            reason="A/B test showed 5% improvement in bullish trends"
        )
    """

    def __init__(self):
        self._cache: dict[tuple[str, str | None], Any] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Load all configuration from database and populate defaults if empty."""
        if self._initialized:
            return

        logger.info("[CONFIG] Initializing ConfigManager...")

        async with get_session() as session:
            # Load existing config
            result = await session.execute(select(LiveConfig))
            rows = result.scalars().all()

            loaded_count = 0
            for row in rows:
                cache_key = (row.config_key, row.regime)
                self._cache[cache_key] = row.config_value
                loaded_count += 1

            logger.info(f"[CONFIG] Loaded {loaded_count} config entries from database")

            # Populate missing defaults
            if loaded_count == 0:
                logger.info("[CONFIG] No existing config found, populating defaults...")
                await self._populate_defaults(session)
                await session.commit()

        self._initialized = True
        logger.success("[CONFIG] ConfigManager initialized")

    async def _populate_defaults(self, session) -> None:
        """Insert default configuration values into database."""
        for key, value in DEFAULT_CONFIG.items():
            component = key.split(".")[0]  # Extract component from key

            stmt = pg_insert(LiveConfig).values(
                config_key=key,
                regime="global",  # Global default (NULL not allowed in composite PK)
                config_value=value if isinstance(value, dict) else {"value": value},
                component=component,
                default_value=value if isinstance(value, dict) else {"value": value},
                updated_by="system",
                update_reason="Initial default configuration",
                version=1
            ).on_conflict_do_nothing(
                index_elements=["config_key", "regime"]
            )

            await session.execute(stmt)

        logger.info(f"[CONFIG] Populated {len(DEFAULT_CONFIG)} default values")

    def get(self, key: str, regime: str | None = None, default: Any = None) -> Any:
        """Get configuration value with regime fallback.

        Fallback chain:
        1. Regime-specific value (if regime provided)
        2. Global value (regime="global")
        3. DEFAULT_CONFIG
        4. default parameter

        Args:
            key: Configuration key (e.g., "fusion.bet_threshold")
            regime: Optional regime for regime-specific override (None = global)
            default: Fallback if not found anywhere

        Returns:
            Configuration value
        """
        # Normalize regime: None -> "global"
        regime = regime or "global"

        # Try regime-specific first
        if regime and regime != "global":
            cache_key = (key, regime)
            if cache_key in self._cache:
                value = self._cache[cache_key]
                return value["value"] if isinstance(value, dict) and "value" in value else value

        # Try global
        cache_key = (key, "global")
        if cache_key in self._cache:
            value = self._cache[cache_key]
            return value["value"] if isinstance(value, dict) and "value" in value else value

        # Try DEFAULT_CONFIG
        if key in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[key]

        # Return default parameter
        logger.warning(f"[CONFIG] Key not found: {key}, returning default: {default}")
        return default

    async def set(
        self,
        key: str,
        value: Any,
        regime: str | None = None,
        reason: str = "Manual update",
        updated_by: str = "evolution_engine",
        performance_delta: float | None = None
    ) -> None:
        """Update configuration value and persist to database.

        Args:
            key: Configuration key
            value: New value
            regime: Optional regime for regime-specific override (None = global)
            reason: Explanation for the change
            updated_by: Who/what made the change
            performance_delta: Performance improvement/degradation (e.g., 0.05 = 5% improvement)
        """
        async with self._lock:
            # Normalize regime: None -> "global"
            regime = regime or "global"
            component = key.split(".")[0]

            # Get current value for logging
            old_value = self.get(key, regime=regime)

            # Prepare value for JSONB storage
            if not isinstance(value, dict):
                json_value = {"value": value}
            else:
                json_value = value

            async with get_session() as session:
                # Check if config exists
                stmt = select(LiveConfig).where(
                    LiveConfig.config_key == key,
                    LiveConfig.regime == regime
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing
                    existing.config_value = json_value
                    existing.updated_at = datetime.now(timezone.utc)
                    existing.updated_by = updated_by
                    existing.update_reason = reason
                    existing.performance_delta = Decimal(str(performance_delta)) if performance_delta else None
                    existing.version += 1

                    logger.info(
                        f"[CONFIG] Updated {key} (regime={regime}): "
                        f"{old_value} → {value} | "
                        f"Reason: {reason} | "
                        f"Delta: {performance_delta:+.2%}" if performance_delta else ""
                    )
                else:
                    # Insert new
                    default_value = DEFAULT_CONFIG.get(key, {})
                    if not isinstance(default_value, dict):
                        default_value = {"value": default_value}

                    stmt = insert(LiveConfig).values(
                        config_key=key,
                        regime=regime,
                        config_value=json_value,
                        component=component,
                        default_value=default_value,
                        updated_by=updated_by,
                        update_reason=reason,
                        performance_delta=Decimal(str(performance_delta)) if performance_delta else None,
                        version=1
                    )
                    await session.execute(stmt)

                    logger.info(
                        f"[CONFIG] Created {key} (regime={regime}): {value} | "
                        f"Reason: {reason}"
                    )

                await session.commit()

            # Update cache
            cache_key = (key, regime)
            self._cache[cache_key] = json_value

    def get_all_by_component(self, component: str, regime: str | None = None) -> dict[str, Any]:
        """Get all configuration values for a component.

        Args:
            component: Component name (e.g., "fusion", "agent", "causal")
            regime: Optional regime filter

        Returns:
            Dictionary of key → value for all matching config
        """
        result = {}

        for (key, cached_regime), value in self._cache.items():
            # Check if key belongs to component
            if not key.startswith(f"{component}."):
                continue

            # Check regime match
            if regime and cached_regime != regime:
                continue

            # Extract value from JSONB wrapper if needed
            extracted_value = value["value"] if isinstance(value, dict) and "value" in value else value
            result[key] = extracted_value

        # Fill in missing values from defaults
        for key, default_value in DEFAULT_CONFIG.items():
            if key.startswith(f"{component}.") and key not in result:
                result[key] = default_value

        return result

    def get_all_changes(self, limit: int = 50) -> list[dict]:
        """Get recent configuration changes for dashboard.

        Returns:
            List of changes with key, old_value, new_value, reason, delta, timestamp
        """
        # This would query the database for recent changes
        # For now, return empty list - will implement with dashboard
        return []

    async def rollback(self, key: str, regime: str | None = None, to_version: int = None) -> bool:
        """Rollback a configuration change to a previous version.

        Args:
            key: Configuration key
            regime: Optional regime
            to_version: Target version (or None for default)

        Returns:
            True if successful
        """
        # This would query version history and restore previous value
        # For now, restore to default
        if key in DEFAULT_CONFIG:
            await self.set(
                key=key,
                value=DEFAULT_CONFIG[key],
                regime=regime,
                reason=f"Rollback to default (version {to_version})" if to_version else "Rollback to default",
                updated_by="rollback_system"
            )
            return True
        return False


# Global singleton instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
