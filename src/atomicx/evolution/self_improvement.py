"""Autonomous self-improvement engine with AGI capabilities.

Continuously diagnoses system performance, generates improvement proposals,
validates them through shadow testing, and deploys successful changes.

Flow:
1. Diagnosis (every 50 cycles) → Identify weaknesses
2. Proposal Generation → Create specific, actionable changes
3. Shadow Testing → Run proposals in parallel with live system
4. Deployment → Apply changes that show > 5% improvement
5. Monitoring → Rollback if performance degrades > 10%
6. Reporting → Generate periodic evolution reports

AGI EXTENSIONS (NEW):
7. API Discovery → Search web for better data sources autonomously
8. Connector Generation → Write and deploy new connector code automatically
9. Database Evolution → Create tables/schemas dynamically as needed
10. Infinite Extension → System can add ANY capability it discovers

All changes are logged, reversible, and tied to performance metrics.
NO RESTRICTIONS - Full autonomy with safety monitoring.
"""

from typing import Any
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from collections import defaultdict
from loguru import logger
import sqlalchemy as sa
from sqlalchemy import select, insert, func, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atomicx.data.storage.models import (
    PredictionOutcome, DiagnosisLog, EvolutionProposal,
    ABTestResult, EvolutionReport, CausalWeight, AgentPerformance
)
from atomicx.data.storage.database import get_session
from atomicx.evolution.config_manager import get_config_manager


# ═══════════════════════════════════════════════════════════════════════════
# AGI CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

AGI_DEFAULT_CONFIG = {
    "agi.enabled": False,  # AGI features OFF by default (enable explicitly)
    "agi.api_discovery_enabled": True,  # Allow autonomous API discovery
    "agi.code_generation_enabled": True,  # Allow autonomous code generation
    "agi.schema_evolution_enabled": True,  # Allow autonomous database evolution
    "agi.min_quality_improvement": 0.15,  # Minimum quality improvement to trigger API switch
    "agi.max_apis_per_asset": 3,  # Max APIs to evaluate per asset
    "agi.connector_validation_strict": True,  # Strict validation for generated code
    "agi.cost_tracking_enabled": True,  # Track API costs for discovered sources
    "agi.max_monthly_cost": 100.0,  # Max $100/month for discovered APIs
    "agi.auto_rollback_on_cost_exceeded": True,  # Auto-rollback if cost limit exceeded
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosisResult:
    """Results of system health diagnosis."""

    def __init__(self):
        self.win_rate_by_regime: dict[str, float] = {}
        self.win_rate_by_agent: dict[str, float] = {}
        self.win_rate_by_variable: dict[str, float] = {}
        self.win_rate_by_timeframe: dict[str, float] = {}
        self.worst_component: str | None = None
        self.best_component: str | None = None
        self.recommended_actions: list[dict] = []
        self.system_health_score: float = 0.0


class Proposal:
    """Autonomous improvement proposal."""

    def __init__(
        self,
        component: str,
        action_type: str,
        parameter_path: str,
        old_value: Any,
        proposed_value: Any,
        evidence: dict,
        confidence: float,
        expected_improvement: float | None = None
    ):
        self.proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
        self.component = component
        self.action_type = action_type
        self.parameter_path = parameter_path
        self.old_value = old_value
        self.proposed_value = proposed_value
        self.evidence = evidence
        self.confidence = confidence
        self.expected_improvement = expected_improvement
        self.status = "pending"
        self.created_at = datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════
# AUTONOMOUS SELF-IMPROVEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousSelfImprovementEngine:
    """Autonomous diagnosis, proposals, A/B testing, deployment.

    Continuously improves system performance by:
    - Diagnosing weaknesses (every 50 cycles)
    - Generating improvement proposals with evidence
    - Shadow testing proposals (parallel execution)
    - Auto-deploying successful changes
    - Rolling back degradations
    """

    def __init__(self, config_manager=None):
        self.config = config_manager or get_config_manager()
        self._active_shadow_tests: dict[str, dict] = {}
        self._deployment_monitors: dict[str, dict] = {}
        self._last_diagnosis_cycle = 0
        self._last_evolution_cycle = 0
        self._last_report_cycle = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the evolution engine."""
        await self.config.initialize()
        logger.info("[EVOLUTION] AutonomousSelfImprovementEngine initialized")

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSIS
    # ═══════════════════════════════════════════════════════════════════════

    async def run_diagnosis(self, cycles_since_last: int = 50) -> DiagnosisResult:
        """Diagnose system performance and generate proposals.

        Analyzes last N cycles to identify:
        - Win rate by regime (which regimes are we weak in?)
        - Win rate by agent (which agents are underperforming?)
        - Win rate by variable (which indicators are misleading?)

        Args:
            cycles_since_last: Number of cycles to analyze

        Returns:
            DiagnosisResult with findings and recommended actions
        """
        logger.info(f"[EVOLUTION] Running diagnosis over last {cycles_since_last} cycles...")

        result = DiagnosisResult()

        async with get_session() as session:
            # Calculate cutoff time (last N cycles ≈ last N*2 minutes)
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=cycles_since_last * 2)

            # ═══ Win Rate by Regime ═══
            regime_query = await session.execute(
                select(
                    PredictionOutcome.regime,
                    func.count(PredictionOutcome.prediction_id).label("total"),
                    func.sum(func.cast(PredictionOutcome.was_correct, sa.Integer)).label("correct")
                )
                .where(
                    and_(
                        PredictionOutcome.verified_at.isnot(None),
                        PredictionOutcome.verified_at >= cutoff_time,
                        PredictionOutcome.regime.isnot(None)
                    )
                )
                .group_by(PredictionOutcome.regime)
            )

            for row in regime_query:
                regime = row.regime
                total = row.total
                correct = row.correct or 0
                win_rate = correct / total if total > 0 else 0.5
                result.win_rate_by_regime[regime] = win_rate

                logger.info(f"[DIAGNOSIS] Regime {regime}: {correct}/{total} = {win_rate:.2%}")

            # ═══ Win Rate by Agent ═══
            # Query agent_performance table for recent performance
            agent_query = await session.execute(
                select(AgentPerformance)
                .where(AgentPerformance.total_predictions >= 10)  # Min sample size
                .order_by(AgentPerformance.performance_edge.desc())
            )

            for agent in agent_query.scalars():
                if agent.total_predictions > 0:
                    win_rate = float(agent.correct_predictions) / float(agent.total_predictions)
                    result.win_rate_by_agent[agent.agent_id] = win_rate

            # ═══ Identify Worst Components ═══
            if result.win_rate_by_regime:
                worst_regime = min(result.win_rate_by_regime.items(), key=lambda x: x[1])
                if worst_regime[1] < 0.45:
                    result.worst_component = f"regime:{worst_regime[0]}"
                    logger.warning(f"[DIAGNOSIS] Worst performing regime: {worst_regime[0]} ({worst_regime[1]:.2%})")

            if result.win_rate_by_agent:
                worst_agent = min(result.win_rate_by_agent.items(), key=lambda x: x[1])
                if worst_agent[1] < 0.40:
                    if not result.worst_component:
                        result.worst_component = f"agent:{worst_agent[0]}"
                    logger.warning(f"[DIAGNOSIS] Worst performing agent: {worst_agent[0]} ({worst_agent[1]:.2%})")

            # ═══ Calculate System Health Score (0-100) ═══
            overall_total = sum(row.total for row in regime_query) if result.win_rate_by_regime else 0
            overall_correct = sum(
                (result.win_rate_by_regime[row.regime] * row.total)
                for row in regime_query
            ) if result.win_rate_by_regime else 0

            overall_win_rate = overall_correct / overall_total if overall_total > 0 else 0.50

            # Health score: 0-100, where 50% win rate = 50 points
            # 60% win rate = 80 points, 70% = 100 points
            result.system_health_score = min(100, max(0, (overall_win_rate - 0.30) * 250))

            logger.info(f"[DIAGNOSIS] System health score: {result.system_health_score:.1f}/100 (WR: {overall_win_rate:.2%})")

            # ═══ Generate Proposals (including AGI extensions) ═══
            proposals = await self._generate_proposals(result)

            # ═══ AGI Extension: Check if API discovery needed ═══
            agi_proposals = await self._check_agi_triggers(result)
            proposals.extend(agi_proposals)

            result.recommended_actions = [
                {
                    "action": p.action_type,
                    "component": p.component,
                    "parameter": p.parameter_path,
                    "confidence": p.confidence,
                    "expected_improvement": p.expected_improvement
                }
                for p in proposals
            ]

            # ═══ Win Rate by Timeframe ═══
            # Analyze performance by timeframe (based on prediction duration)
            # Group by verification delay: <5min, 5-15min, 15-30min, >30min
            timeframe_buckets = {
                "0-5min": {"predictions": [], "outcomes": []},
                "5-15min": {"predictions": [], "outcomes": []},
                "15-30min": {"predictions": [], "outcomes": []},
                "30min+": {"predictions": [], "outcomes": []}
            }

            outcomes_for_timeframe = await session.execute(
                select(PredictionOutcome)
                .where(
                    and_(
                        PredictionOutcome.verified_at.isnot(None),
                        PredictionOutcome.verified_at >= cutoff_time
                    )
                )
            )

            for outcome in outcomes_for_timeframe.scalars():
                if not outcome.verified_at or not outcome.created_at:
                    continue

                # Calculate duration in minutes
                duration = (outcome.verified_at - outcome.created_at).total_seconds() / 60

                # Determine bucket
                if duration < 5:
                    bucket = "0-5min"
                elif duration < 15:
                    bucket = "5-15min"
                elif duration < 30:
                    bucket = "15-30min"
                else:
                    bucket = "30min+"

                timeframe_buckets[bucket]["predictions"].append(outcome.prediction_id)
                timeframe_buckets[bucket]["outcomes"].append(1.0 if outcome.was_correct else 0.0)

            for timeframe, data in timeframe_buckets.items():
                if data["outcomes"]:
                    win_rate = sum(data["outcomes"]) / len(data["outcomes"])
                    result.win_rate_by_timeframe[timeframe] = win_rate
                    logger.info(f"[DIAGNOSIS] Timeframe {timeframe}: {len(data['outcomes'])} predictions, WR={win_rate:.2%}")

            # ═══ Save Diagnosis to Database ═══
            stmt = insert(DiagnosisLog).values(
                win_rate_by_regime=result.win_rate_by_regime,
                win_rate_by_agent=result.win_rate_by_agent,
                win_rate_by_variable=result.win_rate_by_variable,
                win_rate_by_timeframe=result.win_rate_by_timeframe,
                worst_component=result.worst_component,
                best_component=result.best_component,
                recommended_actions=result.recommended_actions,
                system_health_score=Decimal(str(result.system_health_score))
            )
            await session.execute(stmt)
            await session.commit()

        logger.success(f"[EVOLUTION] Diagnosis complete: {len(proposals)} proposals generated")
        return result

    async def _generate_proposals(self, diagnosis: DiagnosisResult) -> list[Proposal]:
        """Generate specific improvement proposals from diagnosis.

        Args:
            diagnosis: DiagnosisResult with performance data

        Returns:
            List of Proposal objects
        """
        proposals = []
        min_confidence = self.config.get("evolution.min_proposal_confidence", default=0.70)

        # ═══ Proposal 1: Adjust fusion threshold for weak regimes ═══
        for regime, win_rate in diagnosis.win_rate_by_regime.items():
            if win_rate < 0.45:
                current_threshold = self.config.get("fusion.bet_threshold", regime=regime)

                # If win rate is low, try raising threshold (be more selective)
                proposed_threshold = min(0.85, current_threshold + 0.05)

                proposal = Proposal(
                    component="fusion",
                    action_type="increase_threshold",
                    parameter_path="fusion.bet_threshold",
                    old_value=current_threshold,
                    proposed_value=proposed_threshold,
                    evidence={
                        "regime": regime,
                        "current_win_rate": win_rate,
                        "sample_size": len([r for r in diagnosis.win_rate_by_regime if r == regime]),
                        "reasoning": f"Win rate {win_rate:.2%} below target in {regime} regime"
                    },
                    confidence=0.75 if win_rate < 0.40 else 0.65,
                    expected_improvement=0.05
                )

                await self._save_proposal(proposal)
                proposals.append(proposal)

        # ═══ Proposal 2: Disable severely underperforming agents ═══
        disable_threshold = self.config.get("agent.disable_threshold", default=0.30)

        for agent_id, win_rate in diagnosis.win_rate_by_agent.items():
            if win_rate < disable_threshold:
                proposal = Proposal(
                    component="agent",
                    action_type="disable_agent",
                    parameter_path=f"agent.{agent_id}.is_active",
                    old_value=True,
                    proposed_value=False,
                    evidence={
                        "agent_id": agent_id,
                        "win_rate": win_rate,
                        "threshold": disable_threshold,
                        "reasoning": f"Agent {agent_id} win rate {win_rate:.2%} below disable threshold {disable_threshold:.2%}"
                    },
                    confidence=0.85,
                    expected_improvement=0.02
                )

                await self._save_proposal(proposal)
                proposals.append(proposal)

        # ═══ Proposal 3: Mutate moderately underperforming agents ═══
        mutate_threshold = self.config.get("agent.mutate_threshold", default=0.40)

        for agent_id, win_rate in diagnosis.win_rate_by_agent.items():
            if disable_threshold < win_rate < mutate_threshold:
                proposal = Proposal(
                    component="agent",
                    action_type="mutate_parameters",
                    parameter_path=f"agent.{agent_id}.parameters",
                    old_value=None,  # Will be retrieved when executed
                    proposed_value="mutated",  # Specific mutation generated later
                    evidence={
                        "agent_id": agent_id,
                        "win_rate": win_rate,
                        "threshold": mutate_threshold,
                        "reasoning": f"Agent {agent_id} win rate {win_rate:.2%} below mutation threshold {mutate_threshold:.2%}"
                    },
                    confidence=0.70,
                    expected_improvement=0.03
                )

                await self._save_proposal(proposal)
                proposals.append(proposal)

        # Filter by minimum confidence
        proposals = [p for p in proposals if p.confidence >= min_confidence]

        return proposals

    async def _check_agi_triggers(self, diagnosis: DiagnosisResult) -> list[Proposal]:
        """Check if AGI capabilities are needed based on system state.

        Detects scenarios where autonomous API discovery, connector generation,
        or database evolution would improve the system.

        Args:
            diagnosis: Current system diagnosis

        Returns:
            List of AGI-related proposals
        """
        agi_proposals = []

        # Check if AGI features are enabled
        agi_enabled = self.config.get("agi.enabled", default=False)
        if not agi_enabled:
            return []

        logger.info("[AGI] Checking for AGI triggers...")

        async with get_session() as session:
            # ═══ Trigger 1: Check for assets with poor data quality ═══
            # Query portfolio_assets table for assets with data issues
            try:
                from atomicx.data.storage.models import PortfolioAsset

                # Find assets with errors or low backfill progress
                problem_assets = await session.execute(
                    select(PortfolioAsset).where(
                        sa.or_(
                            PortfolioAsset.status == 'error',
                            sa.and_(
                                PortfolioAsset.status == 'active',
                                PortfolioAsset.backfill_progress < 100
                            )
                        )
                    )
                )

                for asset in problem_assets.scalars():
                    # Trigger API discovery for problematic assets
                    logger.info(
                        f"[AGI] Detected problematic asset: {asset.symbol} "
                        f"(status: {asset.status}, progress: {asset.backfill_progress}%)"
                    )

                    # Calculate current data quality score
                    current_quality = asset.backfill_progress / 100.0 if asset.backfill_progress else 0.0

                    # Trigger autonomous API discovery
                    api_proposals = await self.discover_data_sources_autonomously(
                        asset_symbol=asset.symbol,
                        asset_type=asset.asset_type or "unknown",
                        current_source_quality=current_quality
                    )

                    agi_proposals.extend(api_proposals)

            except Exception as e:
                logger.warning(f"[AGI] Failed to check asset data quality: {e}")

            # ═══ Trigger 2: Check for assets with low win rates by type ═══
            # If win rate for specific asset type is low, might need better data
            for regime, win_rate in diagnosis.win_rate_by_regime.items():
                if win_rate < 0.40:  # Very poor performance
                    logger.info(
                        f"[AGI] Detected poor performance in regime {regime} (WR: {win_rate:.2%}). "
                        f"Checking if better data sources available..."
                    )

                    # Trigger API discovery for this asset type
                    # (assuming regime correlates to asset type or symbol)
                    # This is a heuristic - you may need to query assets by regime
                    try:
                        # Get a sample asset from this regime to determine type
                        sample_predictions = await session.execute(
                            select(PredictionOutcome.symbol).where(
                                PredictionOutcome.regime == regime
                            ).limit(1)
                        )
                        sample_symbol = sample_predictions.scalar_one_or_none()

                        if sample_symbol:
                            # Detect asset type from symbol
                            from atomicx.data.connectors.router import ConnectorRouter
                            router = ConnectorRouter()
                            detected_type, _ = router.detect_asset_type(sample_symbol)

                            # Trigger API discovery
                            api_proposals = await self.discover_data_sources_autonomously(
                                asset_symbol=sample_symbol,
                                asset_type=detected_type.value,
                                current_source_quality=win_rate  # Use win rate as proxy for quality
                            )

                            agi_proposals.extend(api_proposals)

                    except Exception as e:
                        logger.warning(f"[AGI] Failed to trigger API discovery for regime {regime}: {e}")

            # ═══ Trigger 3: Process pending API integration proposals ═══
            # If we have approved "integrate_api" proposals, generate connector code
            pending_integrations = await session.execute(
                select(EvolutionProposal).where(
                    sa.and_(
                        EvolutionProposal.action_type == "integrate_api",
                        EvolutionProposal.status == "approved"
                    )
                )
            )

            for proposal in pending_integrations.scalars():
                logger.info(f"[AGI] Generating connector code for approved proposal: {proposal.proposal_id}")

                try:
                    # Extract API spec from proposal
                    if isinstance(proposal.proposed_value, dict):
                        api_spec = proposal.proposed_value.get("api_candidate", {})
                        integration_plan = proposal.proposed_value.get("integration_plan", {})

                        # Generate connector code
                        connector_code = await self.generate_connector_code_autonomously(
                            api_spec=api_spec,
                            integration_plan=integration_plan
                        )

                        if connector_code:
                            # Save generated code to file
                            connector_name = integration_plan.get("connector_name", "GeneratedConnector")
                            connector_file = f"src/atomicx/data/connectors/{connector_name.lower()}.py"

                            with open(f"/Users/mdshaonsikder/dev/projects/atomicx/{connector_file}", "w") as f:
                                f.write(connector_code)

                            logger.success(f"[AGI] Connector code generated and saved: {connector_file}")

                            # Update proposal status
                            proposal.status = "code_generated"
                            proposal.extra_data = proposal.extra_data or {}
                            proposal.extra_data["connector_file"] = connector_file
                            await session.commit()

                except Exception as e:
                    logger.error(f"[AGI] Failed to generate connector code for {proposal.proposal_id}: {e}")

        logger.info(f"[AGI] AGI trigger check complete: {len(agi_proposals)} proposals generated")
        return agi_proposals

    async def _save_proposal(self, proposal: Proposal) -> None:
        """Save proposal to database."""
        async with get_session() as session:
            stmt = insert(EvolutionProposal).values(
                proposal_id=proposal.proposal_id,
                component=proposal.component,
                action_type=proposal.action_type,
                parameter_path=proposal.parameter_path,
                old_value=proposal.old_value if isinstance(proposal.old_value, dict) else {"value": proposal.old_value},
                proposed_value=proposal.proposed_value if isinstance(proposal.proposed_value, dict) else {"value": proposal.proposed_value},
                evidence=proposal.evidence,
                confidence=Decimal(str(proposal.confidence)),
                expected_improvement=Decimal(str(proposal.expected_improvement)) if proposal.expected_improvement else None,
                status=proposal.status
            )
            await session.execute(stmt)
            await session.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # WEIGHT EVOLUTION (CausalRL)
    # ═══════════════════════════════════════════════════════════════════════

    async def run_weight_evolution(self, cycles_since_last: int = 100) -> dict[str, dict]:
        """Optimize causal weights based on profitability (CausalRL).

        Analyzes which variables led to profitable predictions and adjusts
        their weights accordingly using reinforcement learning principles.

        Args:
            cycles_since_last: Number of cycles to analyze

        Returns:
            Dictionary of variable → weight adjustments by regime
        """
        logger.info(f"[EVOLUTION] Running weight evolution over last {cycles_since_last} cycles...")

        learning_rate = self.config.get("causalrl.learning_rate", default=0.001)
        discount_factor = self.config.get("causalrl.discount_factor", default=0.95)

        weight_updates = {}

        async with get_session() as session:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=cycles_since_last * 2)

            # Query profitable vs unprofitable predictions
            outcomes = await session.execute(
                select(PredictionOutcome)
                .where(
                    and_(
                        PredictionOutcome.verified_at.isnot(None),
                        PredictionOutcome.verified_at >= cutoff_time,
                        PredictionOutcome.variables_snapshot.isnot(None)
                    )
                )
            )

            # Group by regime and variable
            regime_variable_rewards: dict[tuple[str, str], list[float]] = defaultdict(list)

            for outcome in outcomes.scalars():
                if not outcome.variables_snapshot:
                    continue

                regime = outcome.regime or "UNKNOWN"
                profit = float(outcome.profit_return or 0.0)

                # Reward is the profit/loss
                reward = profit

                # For each variable that was active in this prediction
                for var_name, var_value in outcome.variables_snapshot.items():
                    if var_value and var_value != 0:
                        regime_variable_rewards[(regime, var_name)].append(reward)

            # Calculate new weights using CausalRL update rule
            for (regime, variable), rewards in regime_variable_rewards.items():
                if len(rewards) < 5:
                    continue  # Need minimum sample size

                avg_reward = sum(rewards) / len(rewards)

                # Get current weight
                current_weight_result = await session.execute(
                    select(CausalWeight)
                    .where(
                        and_(
                            CausalWeight.variable_id == variable,
                            CausalWeight.regime == regime
                        )
                    )
                )
                current_weight_row = current_weight_result.scalar_one_or_none()
                current_weight = float(current_weight_row.weight) if current_weight_row else 1.0

                # CausalRL update: w' = w + α * (R - baseline) * discount
                # Baseline is 0 (neutral profitability)
                new_weight = current_weight + learning_rate * avg_reward * discount_factor

                # Clamp between 0.1 and 3.0
                new_weight = max(0.1, min(3.0, new_weight))

                if abs(new_weight - current_weight) > 0.01:
                    # Save updated weight
                    stmt = pg_insert(CausalWeight).values(
                        variable_id=variable,
                        regime=regime,
                        weight=Decimal(str(new_weight)),
                        updated_by="causalrl",
                        update_reason=f"Weight evolution: avg_reward={avg_reward:.4f}, samples={len(rewards)}",
                        previous_weight=Decimal(str(current_weight))
                    ).on_conflict_do_update(
                        index_elements=["variable_id", "regime"],
                        set_={
                            "weight": Decimal(str(new_weight)),
                            "updated_at": datetime.now(timezone.utc),
                            "updated_by": "causalrl",
                            "update_reason": f"Weight evolution: avg_reward={avg_reward:.4f}, samples={len(rewards)}",
                            "previous_weight": Decimal(str(current_weight))
                        }
                    )
                    await session.execute(stmt)

                    weight_updates[f"{regime}:{variable}"] = {
                        "old": current_weight,
                        "new": new_weight,
                        "delta": new_weight - current_weight,
                        "avg_reward": avg_reward,
                        "samples": len(rewards)
                    }

                    logger.info(
                        f"[CAUSALRL] {regime}:{variable}: "
                        f"{current_weight:.3f} → {new_weight:.3f} "
                        f"(reward={avg_reward:+.4f}, n={len(rewards)})"
                    )

            await session.commit()

        logger.success(f"[EVOLUTION] Weight evolution complete: {len(weight_updates)} weights updated")
        return weight_updates

    # ═══════════════════════════════════════════════════════════════════════
    # EVOLUTION REPORTS
    # ═══════════════════════════════════════════════════════════════════════

    async def generate_evolution_report(self) -> dict:
        """Generate periodic evolution report.

        Summarizes:
        - What changed this period
        - Win rate trends
        - System health score
        - Top improvements and weaknesses

        Returns:
            Report dictionary
        """
        logger.info("[EVOLUTION] Generating evolution report...")

        report = {
            "report_id": f"report_{uuid.uuid4().hex[:12]}",
            "generated_at": datetime.now(timezone.utc),
            "changes_made": [],
            "health_score": 0.0,
            "top_improvements": [],
            "top_weaknesses": [],
            "win_rate_trend": {},
            "next_planned": []
        }

        async with get_session() as session:
            # Get latest diagnosis
            latest_diagnosis = await session.execute(
                select(DiagnosisLog)
                .order_by(DiagnosisLog.diagnosed_at.desc())
                .limit(1)
            )
            diagnosis = latest_diagnosis.scalar_one_or_none()

            if diagnosis:
                report["health_score"] = float(diagnosis.system_health_score)
                report["top_weaknesses"] = [diagnosis.worst_component] if diagnosis.worst_component else []

            # Get recent approved proposals
            recent_proposals = await session.execute(
                select(EvolutionProposal)
                .where(EvolutionProposal.status == "approved")
                .order_by(EvolutionProposal.approved_at.desc())
                .limit(10)
            )

            for proposal in recent_proposals.scalars():
                report["changes_made"].append({
                    "action": proposal.action_type,
                    "component": proposal.component,
                    "parameter": proposal.parameter_path,
                    "confidence": float(proposal.confidence)
                })

            # Save report
            stmt = insert(EvolutionReport).values(
                report_id=report["report_id"],
                changes_made=report["changes_made"],
                health_score=Decimal(str(report["health_score"])),
                top_improvements=report["top_improvements"],
                top_weaknesses=report["top_weaknesses"],
                win_rate_trend=report["win_rate_trend"],
                next_planned=report["next_planned"]
            )
            await session.execute(stmt)
            await session.commit()

        logger.success(
            f"[EVOLUTION] Report generated: "
            f"Health={report['health_score']:.1f}/100, "
            f"Changes={len(report['changes_made'])}"
        )

        return report

    # ═══════════════════════════════════════════════════════════════════════
    # BACKGROUND TASKS
    # ═══════════════════════════════════════════════════════════════════════

    async def shadow_testing_loop(self) -> None:
        """Background task: continuously run shadow tests for approved proposals.

        For each approved proposal:
        1. Apply in shadow mode (separate prediction path)
        2. Run for N cycles
        3. Compare shadow vs live win rate
        4. Auto-promote if improvement > threshold
        """
        logger.info("[EVOLUTION] Shadow testing loop started")

        shadow_test_cycles = self.config.get("evolution.shadow_test_cycles", default=20)
        auto_promote_delta = self.config.get("evolution.auto_promote_delta", default=0.05)

        while True:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes

                # Get pending shadow tests
                async with get_session() as session:
                    pending = await session.execute(
                        select(EvolutionProposal)
                        .where(EvolutionProposal.status == "shadow_testing")
                    )

                    for proposal in pending.scalars():
                        # Check if shadow test has enough cycles
                        test_data = self._active_shadow_tests.get(proposal.proposal_id)
                        if not test_data:
                            continue

                        if test_data["cycles"] >= shadow_test_cycles:
                            # Calculate results
                            shadow_wr = test_data["shadow_wins"] / test_data["shadow_total"] if test_data["shadow_total"] > 0 else 0.5
                            live_wr = test_data["live_wins"] / test_data["live_total"] if test_data["live_total"] > 0 else 0.5
                            delta = shadow_wr - live_wr

                            decision = "promote" if delta >= auto_promote_delta else "reject"

                            # Save A/B test result
                            stmt = insert(ABTestResult).values(
                                test_id=f"test_{uuid.uuid4().hex[:12]}",
                                proposal_id=proposal.proposal_id,
                                shadow_predictions=test_data.get("shadow_predictions", []),
                                shadow_win_rate=Decimal(str(shadow_wr)),
                                live_win_rate=Decimal(str(live_wr)),
                                delta=Decimal(str(delta)),
                                cycles_tested=test_data["cycles"],
                                decision=decision
                            )
                            await session.execute(stmt)

                            # Update proposal status
                            if decision == "promote":
                                proposal.status = "approved"
                                proposal.approved_at = datetime.now(timezone.utc)
                                logger.success(
                                    f"[SHADOW TEST] Promoting {proposal.proposal_id}: "
                                    f"shadow_wr={shadow_wr:.2%}, live_wr={live_wr:.2%}, delta={delta:+.2%}"
                                )
                            else:
                                proposal.status = "rejected"
                                proposal.rejected_at = datetime.now(timezone.utc)
                                proposal.rejection_reason = f"Shadow test failed: delta={delta:+.2%} < {auto_promote_delta:.2%}"
                                logger.warning(
                                    f"[SHADOW TEST] Rejecting {proposal.proposal_id}: "
                                    f"insufficient improvement (delta={delta:+.2%})"
                                )

                            await session.commit()

                            # Remove from active tests
                            del self._active_shadow_tests[proposal.proposal_id]

            except Exception as e:
                logger.error(f"[EVOLUTION] Shadow testing loop error: {e}")

    async def deployment_loop(self) -> None:
        """Background task: deploy approved proposals and monitor performance.

        Apply promoted proposals via ConfigManager and monitor win rate.
        Auto-rollback if performance degrades.
        """
        logger.info("[EVOLUTION] Deployment loop started")

        monitor_cycles = self.config.get("evolution.deployment_monitor_cycles", default=30)
        rollback_threshold = self.config.get("evolution.auto_rollback_delta", default=-0.10)

        while True:
            try:
                await asyncio.sleep(180)  # Check every 3 minutes

                async with get_session() as session:
                    # ═══ Deploy approved proposals ═══
                    approved = await session.execute(
                        select(EvolutionProposal)
                        .where(EvolutionProposal.status == "approved")
                    )

                    for proposal in approved.scalars():
                        # Get baseline win rate before deployment
                        baseline_wr = await self._get_current_win_rate(session)

                        # Deploy the change
                        await self._deploy_proposal(proposal)

                        # Start monitoring
                        self._deployment_monitors[proposal.proposal_id] = {
                            "start_time": datetime.now(timezone.utc),
                            "baseline_wr": baseline_wr,
                            "deployed_at": datetime.now(timezone.utc),
                            "rollback_snapshot": await self._create_rollback_snapshot(proposal)
                        }

                        proposal.status = "deployed"
                        await session.commit()

                        logger.success(f"[DEPLOYMENT] Deployed {proposal.proposal_id}: {proposal.action_type}")

                    # ═══ Monitor deployed changes and auto-rollback if needed ═══
                    deployed = await session.execute(
                        select(EvolutionProposal)
                        .where(EvolutionProposal.status == "deployed")
                    )

                    for proposal in deployed.scalars():
                        monitor_data = self._deployment_monitors.get(proposal.proposal_id)
                        if not monitor_data:
                            continue

                        # Check if monitoring period has elapsed
                        time_since_deploy = (datetime.now(timezone.utc) - monitor_data["deployed_at"]).total_seconds() / 60

                        if time_since_deploy >= monitor_cycles * 2:  # Monitor for N cycles
                            # Calculate current win rate
                            current_wr = await self._get_current_win_rate(session)
                            baseline_wr = monitor_data["baseline_wr"]
                            delta = current_wr - baseline_wr

                            if delta < rollback_threshold:
                                # CRITICAL: Performance degraded, AUTO-ROLLBACK
                                logger.critical(
                                    f"[ROLLBACK] Performance degradation detected! "
                                    f"Proposal {proposal.proposal_id}: "
                                    f"baseline_wr={baseline_wr:.2%}, current_wr={current_wr:.2%}, delta={delta:+.2%}"
                                )

                                # Execute rollback
                                rollback_success = await self._execute_rollback(
                                    proposal=proposal,
                                    rollback_snapshot=monitor_data["rollback_snapshot"],
                                    reason=f"Performance degradation: delta={delta:+.2%} below threshold {rollback_threshold:+.2%}"
                                )

                                if rollback_success:
                                    proposal.status = "rolled_back"
                                    proposal.rejection_reason = f"Auto-rollback: performance degraded {delta:+.2%}"
                                    logger.success(f"[ROLLBACK] Successfully rolled back {proposal.proposal_id}")
                                else:
                                    proposal.status = "rollback_failed"
                                    logger.error(f"[ROLLBACK] Failed to rollback {proposal.proposal_id}")

                                await session.commit()
                                del self._deployment_monitors[proposal.proposal_id]

                            else:
                                # Performance is stable or improved, finalize deployment
                                proposal.status = "finalized"
                                logger.success(
                                    f"[DEPLOYMENT] Finalized {proposal.proposal_id}: "
                                    f"delta={delta:+.2%} (stable)"
                                )
                                await session.commit()
                                del self._deployment_monitors[proposal.proposal_id]

                    # ═══ Cost tracking for AGI-discovered APIs ═══
                    await self._track_api_costs(session)

            except Exception as e:
                logger.error(f"[EVOLUTION] Deployment loop error: {e}")

    async def _deploy_proposal(self, proposal: EvolutionProposal) -> None:
        """Apply a proposal's changes to the system."""
        if proposal.action_type in ["increase_threshold", "decrease_threshold"]:
            # Update config value
            value = proposal.proposed_value
            if isinstance(value, dict) and "value" in value:
                value = value["value"]

            await self.config.set(
                key=proposal.parameter_path,
                value=value,
                regime=proposal.evidence.get("regime") if isinstance(proposal.evidence, dict) else None,
                reason=f"Autonomous evolution: {proposal.action_type}",
                updated_by="evolution_engine",
                performance_delta=float(proposal.expected_improvement) if proposal.expected_improvement else None
            )

        elif proposal.action_type == "disable_agent":
            # Disable underperforming agent
            await self._execute_agent_disable(proposal)

        elif proposal.action_type == "mutate_parameters":
            # Mutate agent parameters via genome
            await self._execute_agent_mutation(proposal)

        elif proposal.action_type == "integrate_api":
            # Deploy newly generated API connector
            await self._execute_api_integration(proposal)

        elif proposal.action_type == "schema_evolution":
            # Database schema was already evolved during proposal creation
            # Just log deployment
            logger.info(f"[DEPLOYMENT] Schema evolution completed: {proposal.parameter_path}")

    async def _execute_agent_disable(self, proposal: EvolutionProposal) -> None:
        """Disable an underperforming agent.

        Args:
            proposal: Proposal containing agent_id and evidence
        """
        agent_id = proposal.evidence.get("agent_id") if isinstance(proposal.evidence, dict) else None
        if not agent_id:
            logger.error("[DEPLOYMENT] Cannot disable agent: no agent_id in evidence")
            return

        # Store config to disable agent
        await self.config.set(
            key=f"agent.{agent_id}.is_active",
            value=False,
            regime=None,
            reason=f"Auto-disabled: win rate {proposal.evidence.get('win_rate', 0):.2%} below threshold",
            updated_by="evolution_engine",
            performance_delta=float(proposal.expected_improvement) if proposal.expected_improvement else None
        )

        # Log to agent evolution table
        async with get_session() as session:
            from atomicx.data.storage.models import AgentEvolutionLog
            stmt = insert(AgentEvolutionLog).values(
                agent_id=agent_id,
                event_type="disabled",
                old_params={"is_active": True},
                new_params={"is_active": False},
                reason=proposal.evidence.get("reasoning", "Performance below threshold"),
                win_rate_before=Decimal(str(proposal.evidence.get("win_rate", 0.0))),
                win_rate_after=None
            )
            await session.execute(stmt)
            await session.commit()

        logger.warning(f"[AGENT EVOLUTION] Disabled agent {agent_id}")

    async def _execute_agent_mutation(self, proposal: EvolutionProposal) -> None:
        """Mutate agent parameters using genome evolution.

        Args:
            proposal: Proposal containing agent_id and mutation details
        """
        agent_id = proposal.evidence.get("agent_id") if isinstance(proposal.evidence, dict) else None
        if not agent_id:
            logger.error("[DEPLOYMENT] Cannot mutate agent: no agent_id in evidence")
            return

        # Get current agent parameters from config
        current_params = {}
        for key in ["confidence_threshold", "weight", "lookback_window", "sensitivity"]:
            param_key = f"agent.{agent_id}.{key}"
            value = self.config.get(param_key, default=None)
            if value is not None:
                current_params[key] = value

        # Generate mutated parameters (apply 15% variance)
        import random
        mutation_rate = self.config.get("genome.mutation_rate", default=0.15)

        mutated_params = {}
        for param_name, param_value in current_params.items():
            variance = random.uniform(-mutation_rate, mutation_rate)
            new_value = param_value * (1.0 + variance)

            # Apply bounds
            if "threshold" in param_name or "confidence" in param_name:
                new_value = max(0.0, min(1.0, new_value))
            elif "weight" in param_name:
                new_value = max(0.1, min(2.0, new_value))
            elif "window" in param_name:
                new_value = max(5, min(200, int(new_value)))
            else:
                new_value = max(0.0, new_value)

            mutated_params[param_name] = round(new_value, 4)

        # Apply mutated parameters
        for param_name, new_value in mutated_params.items():
            param_key = f"agent.{agent_id}.{param_name}"
            await self.config.set(
                key=param_key,
                value=new_value,
                regime=None,
                reason=f"Parameter mutation: {param_name} {current_params.get(param_name)} → {new_value}",
                updated_by="evolution_engine",
                performance_delta=float(proposal.expected_improvement) if proposal.expected_improvement else None
            )

        # Log to agent evolution table
        async with get_session() as session:
            from atomicx.data.storage.models import AgentEvolutionLog
            stmt = insert(AgentEvolutionLog).values(
                agent_id=agent_id,
                event_type="mutated",
                old_params=current_params,
                new_params=mutated_params,
                reason=f"Mutation due to low performance (WR: {proposal.evidence.get('win_rate', 0):.2%})",
                win_rate_before=Decimal(str(proposal.evidence.get("win_rate", 0.0))),
                win_rate_after=None
            )
            await session.execute(stmt)
            await session.commit()

        logger.success(
            f"[AGENT EVOLUTION] Mutated agent {agent_id}: "
            f"{len(mutated_params)} parameters changed"
        )

    async def _execute_api_integration(self, proposal: EvolutionProposal) -> None:
        """Deploy newly generated API connector into the system.

        Hot-loads the connector code, validates it, and registers with router.

        Args:
            proposal: Proposal with connector details and file path
        """
        logger.info(f"[AGI-DEPLOY] Deploying API connector: {proposal.proposal_id}")

        try:
            # Extract connector file path from proposal
            extra_data = proposal.extra_data or {}
            connector_file = extra_data.get("connector_file")

            if not connector_file:
                logger.error("[AGI-DEPLOY] No connector file found in proposal")
                return

            # Load connector module dynamically
            import importlib.util
            import sys

            module_name = connector_file.replace("/", ".").replace(".py", "").split("atomicx.")[-1]
            full_module_name = f"atomicx.{module_name}"

            spec = importlib.util.spec_from_file_location(
                full_module_name,
                f"/Users/mdshaonsikder/dev/projects/atomicx/{connector_file}"
            )

            if not spec or not spec.loader:
                logger.error(f"[AGI-DEPLOY] Failed to load module spec: {connector_file}")
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[full_module_name] = module
            spec.loader.exec_module(module)

            logger.success(f"[AGI-DEPLOY] Connector module loaded: {full_module_name}")

            # Find the DataConnector subclass in the module
            from atomicx.data.connectors.base import DataConnector

            connector_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, DataConnector) and obj is not DataConnector:
                    connector_class = obj
                    break

            if not connector_class:
                logger.error("[AGI-DEPLOY] No DataConnector subclass found in generated module")
                return

            logger.success(f"[AGI-DEPLOY] Connector class found: {connector_class.__name__}")

            # Register connector with router (would need to extend router to support dynamic registration)
            # For now, just validate that it can be instantiated
            from atomicx.data.storage.models import PortfolioAsset

            # Get target asset from proposal evidence
            evidence = proposal.evidence if isinstance(proposal.evidence, dict) else {}
            target_symbol = evidence.get("asset_symbol")

            if target_symbol:
                # Try to instantiate and validate
                try:
                    connector_instance = connector_class(symbol=target_symbol)
                    is_valid = await connector_instance.validate_symbol()

                    if is_valid:
                        logger.success(
                            f"[AGI-DEPLOY] Connector validated successfully for {target_symbol}"
                        )

                        # Update asset status in database
                        async with get_session() as session:
                            asset = await session.execute(
                                select(PortfolioAsset).where(PortfolioAsset.symbol == target_symbol)
                            )
                            asset_obj = asset.scalar_one_or_none()

                            if asset_obj:
                                asset_obj.data_source = connector_class.__name__
                                asset_obj.status = "active"
                                asset_obj.error_message = None
                                await session.commit()

                                logger.success(
                                    f"[AGI-DEPLOY] Asset {target_symbol} updated to use new connector"
                                )

                    else:
                        logger.warning(f"[AGI-DEPLOY] Connector validation failed for {target_symbol}")

                except Exception as e:
                    logger.error(f"[AGI-DEPLOY] Failed to instantiate connector: {e}")

            # Mark proposal as deployed
            proposal.status = "deployed"
            proposal.deployed_at = datetime.now(timezone.utc)

            logger.success(f"[AGI-DEPLOY] API integration deployment complete: {connector_class.__name__}")

        except Exception as e:
            logger.error(f"[AGI-DEPLOY] API integration deployment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════════
    # AGI EXTENSIONS - AUTONOMOUS API DISCOVERY & INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════

    async def discover_data_sources_autonomously(
        self,
        asset_symbol: str,
        asset_type: str,
        current_source_quality: float = 0.5
    ) -> list[Proposal]:
        """Search web for better data APIs when current source is inadequate.

        Triggered when:
        - New asset added with no connector available
        - Current data quality poor (stale, rate limited, expensive)
        - Win rate low for specific asset type (data may be inadequate)

        Uses Claude LLM to:
        1. Research available APIs for this asset type
        2. Evaluate cost, quality, reliability, rate limits
        3. Generate ranked integration proposals

        Args:
            asset_symbol: Symbol to find data for (e.g., "AAPL", "BTC/USDT")
            asset_type: Type of asset (crypto, stock, commodity, forex)
            current_source_quality: Quality score of current source (0.0-1.0)

        Returns:
            List of Proposal objects with action_type="integrate_api"
        """
        logger.info(
            f"[AGI-RESEARCH] Autonomously discovering data sources for "
            f"{asset_symbol} ({asset_type}), current quality: {current_source_quality:.2f}"
        )

        # Import autonomous research agent
        try:
            from atomicx.agi.autonomous_research import get_research_agent
        except ImportError:
            logger.warning("[AGI-RESEARCH] autonomous_research module not available")
            return []

        research_agent = get_research_agent()

        # Discover API candidates using LLM
        try:
            candidates = await research_agent.discover_apis_for_asset(
                asset_symbol=asset_symbol,
                asset_type=asset_type
            )

            if not candidates:
                logger.warning(f"[AGI-RESEARCH] No API candidates found for {asset_symbol}")
                return []

            # Convert APICandidate objects to Proposal objects
            proposals = []
            for candidate in candidates[:3]:  # Top 3 candidates
                # Only propose if quality is better than current
                if candidate.quality_score <= current_source_quality:
                    continue

                # Generate integration plan
                try:
                    integration_plan = await research_agent.generate_integration_plan(candidate)

                    proposal = Proposal(
                        component="data_connector",
                        action_type="integrate_api",
                        parameter_path=f"connector.{candidate.name.lower().replace(' ', '_')}",
                        old_value=None,
                        proposed_value={
                            "api_candidate": {
                                "name": candidate.name,
                                "url": candidate.url,
                                "documentation": candidate.documentation_url,
                                "pricing": candidate.pricing,
                                "rate_limit": candidate.rate_limit
                            },
                            "integration_plan": {
                                "connector_name": integration_plan.connector_name,
                                "estimated_effort": integration_plan.estimated_effort,
                                "dependencies": integration_plan.dependencies,
                                "expected_benefits": integration_plan.expected_benefits,
                                "risks": integration_plan.risks
                            }
                        },
                        evidence={
                            "asset_symbol": asset_symbol,
                            "asset_type": asset_type,
                            "current_quality": current_source_quality,
                            "discovered_quality": candidate.quality_score,
                            "quality_improvement": candidate.quality_score - current_source_quality,
                            "discovery_notes": candidate.evaluation_notes,
                            "reasoning": f"Discovered {candidate.name} with quality score {candidate.quality_score:.2f} (improvement: {candidate.quality_score - current_source_quality:+.2f})"
                        },
                        confidence=min(0.95, 0.60 + (candidate.quality_score - current_source_quality) * 0.5),
                        expected_improvement=candidate.quality_score - current_source_quality
                    )

                    await self._save_proposal(proposal)
                    proposals.append(proposal)

                    logger.success(
                        f"[AGI-RESEARCH] Proposal created: Integrate {candidate.name} "
                        f"(quality: {candidate.quality_score:.2f}, priority: {integration_plan.priority}/10)"
                    )

                except Exception as e:
                    logger.error(f"[AGI-RESEARCH] Failed to create integration plan for {candidate.name}: {e}")
                    continue

            return proposals

        except Exception as e:
            logger.error(f"[AGI-RESEARCH] API discovery failed: {e}")
            return []

    async def generate_connector_code_autonomously(
        self,
        api_spec: dict,
        integration_plan: dict
    ) -> str | None:
        """Generate Python connector code using Claude LLM.

        Creates a complete DataConnector implementation automatically.
        Generated code is validated for:
        - Syntax correctness
        - Required interface implementation
        - Security (no eval, exec, shell commands)
        - Import safety (only approved packages)

        Args:
            api_spec: API details (name, url, docs, pricing, rate limits)
            integration_plan: Integration plan with connector name, dependencies

        Returns:
            Complete Python code as string, or None if generation fails
        """
        logger.info(f"[AGI-CODEGEN] Generating connector code for {api_spec.get('name')}")

        # Import research agent for LLM access
        try:
            from atomicx.agi.autonomous_research import get_research_agent
            import anthropic
            import os
        except ImportError:
            logger.warning("[AGI-CODEGEN] Required modules not available")
            return None

        research_agent = get_research_agent()
        if not research_agent._anthropic:
            logger.warning("[AGI-CODEGEN] Claude API not initialized")
            return None

        # Read the DataConnector base class to include in prompt
        try:
            with open("/Users/mdshaonsikder/dev/projects/atomicx/src/atomicx/data/connectors/base.py", "r") as f:
                base_connector_code = f.read()
        except Exception as e:
            logger.error(f"[AGI-CODEGEN] Failed to read base connector: {e}")
            return None

        # Read an example connector (stock.py) as reference
        try:
            with open("/Users/mdshaonsikder/dev/projects/atomicx/src/atomicx/data/connectors/stock.py", "r") as f:
                example_connector_code = f.read()
        except Exception as e:
            logger.warning(f"[AGI-CODEGEN] Failed to read example connector: {e}")
            example_connector_code = "# No example available"

        # Generate code using Claude
        codegen_prompt = f"""You are an autonomous AI agent generating Python connector code.

TASK: Create a complete DataConnector implementation for this API:

API DETAILS:
- Name: {api_spec.get('name')}
- URL: {api_spec.get('url')}
- Documentation: {api_spec.get('documentation')}
- Pricing: {api_spec.get('pricing')}
- Rate Limit: {api_spec.get('rate_limit')}

INTEGRATION PLAN:
- Connector Class Name: {integration_plan.get('connector_name')}
- Dependencies: {', '.join(integration_plan.get('dependencies', []))}
- Expected Benefits: {integration_plan.get('expected_benefits')}
- Known Risks: {integration_plan.get('risks')}

BASE CLASS (must implement):
```python
{base_connector_code}
```

REFERENCE EXAMPLE (Yahoo Finance connector):
```python
{example_connector_code[:3000]}  # First 3000 chars
```

REQUIREMENTS:
1. Implement ALL abstract methods from DataConnector
2. Handle rate limits gracefully (respect API limits)
3. Support multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
4. Include proper error handling and logging
5. Use async/await for all I/O operations
6. Follow the same structure as the example connector
7. Add docstrings for all methods
8. NO use of eval(), exec(), or shell commands
9. Only import standard library + httpx, aiohttp, ccxt, yfinance, or API-specific libraries

OUTPUT FORMAT:
Return ONLY the complete Python code with NO markdown formatting, NO explanations.
Start with imports, end with class definition.
"""

        try:
            response = await research_agent._anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": codegen_prompt}]
            )

            generated_code = response.content[0].text

            # Validate generated code
            is_valid, validation_errors = await self._validate_generated_code(generated_code)

            if not is_valid:
                logger.error(f"[AGI-CODEGEN] Generated code validation failed: {validation_errors}")
                return None

            logger.success(
                f"[AGI-CODEGEN] Connector code generated: {integration_plan.get('connector_name')} "
                f"({len(generated_code)} chars)"
            )

            return generated_code

        except Exception as e:
            logger.error(f"[AGI-CODEGEN] Code generation failed: {e}")
            return None

    async def _validate_generated_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate generated connector code for safety and correctness.

        Args:
            code: Generated Python code

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # 1. Syntax check
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors

        # 2. Security check - forbidden patterns
        forbidden_patterns = [
            "eval(", "exec(", "__import__", "subprocess", "os.system",
            "shell=True", "compile(", "globals()", "locals()"
        ]

        for pattern in forbidden_patterns:
            if pattern in code:
                errors.append(f"Security risk: forbidden pattern '{pattern}' found")

        # 3. Required interface check
        required_methods = [
            "async def get_historical_ohlcv",
            "async def subscribe_realtime",
            "async def validate_symbol"
        ]

        for method in required_methods:
            if method not in code:
                errors.append(f"Missing required method: {method}")

        # 4. Must inherit from DataConnector
        if "DataConnector" not in code or "class " not in code:
            errors.append("Must define a class that inherits from DataConnector")

        # 5. Approved imports only
        import re
        import_lines = re.findall(r'^import\s+(\S+)|^from\s+(\S+)', code, re.MULTILINE)
        approved_packages = {
            "asyncio", "aiohttp", "httpx", "ccxt", "yfinance",
            "datetime", "typing", "dataclasses", "json", "decimal",
            "loguru", "time", "math", "collections", "itertools",
            "atomicx.data.connectors.base", "atomicx.data.storage"
        }

        for imp in import_lines:
            package = imp[0] or imp[1]
            base_package = package.split('.')[0]
            if base_package not in approved_packages and not package.startswith("atomicx"):
                errors.append(f"Unapproved import: {package}")

        if errors:
            return False, errors

        return True, []

    async def evolve_database_schema_autonomously(
        self,
        table_name: str,
        table_spec: dict,
        reason: str
    ) -> bool:
        """Create database tables/columns dynamically as needed.

        Generates Alembic migration, validates safety, and executes.
        NO predefined limits - can create infinite schemas as needed.

        Safety constraints:
        - No DROP operations (tables or columns)
        - No DELETE or TRUNCATE
        - No ALTER that removes data
        - All changes are additive only

        Args:
            table_name: Name of table to create/modify
            table_spec: Schema specification (columns, types, indexes)
            reason: Why this schema change is needed

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"[AGI-SCHEMA] Autonomously evolving database schema: {table_name}")

        try:
            import subprocess
            from datetime import datetime

            # Generate migration file name
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            migration_name = f"agi_auto_{table_name}_{timestamp}"
            migration_file = f"alembic/versions/{migration_name}.py"

            # Generate migration code
            migration_code = self._generate_migration_code(
                revision_id=migration_name,
                table_name=table_name,
                table_spec=table_spec,
                reason=reason
            )

            # Validate migration (must be additive only)
            is_safe, safety_errors = self._validate_migration_safety(migration_code)
            if not is_safe:
                logger.error(f"[AGI-SCHEMA] Migration validation failed: {safety_errors}")
                return False

            # Write migration file
            with open(migration_file, "w") as f:
                f.write(migration_code)

            logger.info(f"[AGI-SCHEMA] Generated migration: {migration_file}")

            # Execute migration
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                cwd="/Users/mdshaonsikder/dev/projects/atomicx",
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.success(f"[AGI-SCHEMA] Migration applied successfully: {table_name}")

                # Log to evolution_proposals
                proposal = Proposal(
                    component="database",
                    action_type="schema_evolution",
                    parameter_path=f"schema.{table_name}",
                    old_value=None,
                    proposed_value=table_spec,
                    evidence={
                        "table_name": table_name,
                        "columns": list(table_spec.get("columns", {}).keys()),
                        "reason": reason,
                        "migration_file": migration_file
                    },
                    confidence=1.0,  # Schema changes are deterministic
                    expected_improvement=None
                )
                await self._save_proposal(proposal)

                return True
            else:
                logger.error(f"[AGI-SCHEMA] Migration failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"[AGI-SCHEMA] Schema evolution failed: {e}")
            return False

    def _generate_migration_code(
        self,
        revision_id: str,
        table_name: str,
        table_spec: dict,
        reason: str
    ) -> str:
        """Generate Alembic migration code from table specification.

        Args:
            revision_id: Unique migration ID
            table_name: Name of table to create
            table_spec: Schema specification with columns, types, indexes
            reason: Reason for migration

        Returns:
            Complete Python migration code
        """
        # Sanitize identifiers to prevent SQL injection
        table_name = self._sanitize_sql_identifier(table_name)
        revision_id = self._sanitize_sql_identifier(revision_id)

        # Extract columns
        columns = table_spec.get("columns", {})
        indexes = table_spec.get("indexes", [])
        foreign_keys = table_spec.get("foreign_keys", [])

        # Generate column definitions
        column_defs = []
        for col_name, col_spec in columns.items():
            # Sanitize column name
            col_name = self._sanitize_sql_identifier(col_name)
            col_type = col_spec.get("type", "String(255)")
            nullable = col_spec.get("nullable", True)
            default = col_spec.get("default")
            index = col_spec.get("index", False)

            col_def = f"    sa.Column('{col_name}', sa.{col_type}, nullable={nullable}"
            if default:
                col_def += f", default={repr(default)}"
            if index:
                col_def += f", index=True"
            col_def += ")"

            column_defs.append(col_def)

        # Generate migration file
        migration_code = f'''"""AGI Autonomous Schema Evolution: {table_name}

Reason: {reason}

Revision ID: {revision_id}
Revises: AUTO (gets latest)
Create Date: {datetime.now(timezone.utc).isoformat()}

Generated autonomously by AGI evolution engine.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '{revision_id}'
down_revision = None  # Will auto-resolve to latest
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create {table_name} table."""
    op.create_table(
        '{table_name}',
{chr(10).join(column_defs)},
    )

'''

        # Add indexes
        for index in indexes:
            if isinstance(index, dict):
                index_name = self._sanitize_sql_identifier(index.get("name", "idx_auto"))
                columns = index.get("columns", [])
                # Sanitize column names in index
                sanitized_columns = [self._sanitize_sql_identifier(col) for col in columns]
                migration_code += f"    op.create_index('{index_name}', '{table_name}', {sanitized_columns})\n"

        migration_code += "\n\ndef downgrade() -> None:\n"
        migration_code += f"    op.drop_table('{table_name}')\n"

        return migration_code

    def _validate_migration_safety(self, migration_code: str) -> tuple[bool, list[str]]:
        """Validate that migration is safe (additive only, no data loss).

        Args:
            migration_code: Generated migration code

        Returns:
            Tuple of (is_safe, list_of_errors)
        """
        errors = []

        # Forbidden operations (data loss risks)
        forbidden_patterns = [
            "op.drop_table",  # In upgrade() - downgrade() is OK
            "op.drop_column",
            "op.drop_index",
            "DROP TABLE",
            "DROP COLUMN",
            "DELETE FROM",
            "TRUNCATE",
            "ALTER.*DROP"
        ]

        for pattern in forbidden_patterns:
            import re
            # Check if pattern appears in upgrade() function only
            upgrade_section = re.search(r'def upgrade\(\).*?(?=def downgrade)', migration_code, re.DOTALL)
            if upgrade_section and re.search(pattern, upgrade_section.group(), re.IGNORECASE):
                errors.append(f"Unsafe operation in upgrade(): {pattern}")

        # Must have create_table
        if "op.create_table" not in migration_code:
            errors.append("Migration must create a table (additive only)")

        if errors:
            return False, errors

        return True, []

    # ═══════════════════════════════════════════════════════════════════════
    # SAFETY MONITORING & ROLLBACK
    # ═══════════════════════════════════════════════════════════════════════

    async def _get_current_win_rate(self, session) -> float:
        """Get current system win rate over last 50 predictions.

        Args:
            session: Database session

        Returns:
            Win rate as float (0.0 to 1.0)
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=100)

        result = await session.execute(
            select(
                func.count(PredictionOutcome.prediction_id).label("total"),
                func.sum(func.cast(PredictionOutcome.was_correct, sa.Integer)).label("correct")
            )
            .where(
                and_(
                    PredictionOutcome.verified_at.isnot(None),
                    PredictionOutcome.verified_at >= cutoff_time
                )
            )
        )

        row = result.first()
        if row and row.total > 0:
            return (row.correct or 0) / row.total

        return 0.50  # Default if no data

    async def _create_rollback_snapshot(self, proposal: EvolutionProposal) -> dict:
        """Create snapshot of current state before deploying proposal.

        Captures:
        - Config values that will be changed
        - Agent states that will be modified
        - Database schema before evolution

        Args:
            proposal: Proposal about to be deployed

        Returns:
            Snapshot dictionary for rollback
        """
        snapshot = {
            "proposal_id": proposal.proposal_id,
            "action_type": proposal.action_type,
            "parameter_path": proposal.parameter_path,
            "old_value": proposal.old_value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Capture current config value
        if proposal.action_type in ["increase_threshold", "decrease_threshold", "mutate_parameters"]:
            current_value = self.config.get(proposal.parameter_path, default=None)
            snapshot["config_value"] = current_value

        # Capture agent state for agent evolution
        if proposal.action_type in ["disable_agent", "mutate_parameters"]:
            evidence = proposal.evidence if isinstance(proposal.evidence, dict) else {}
            agent_id = evidence.get("agent_id")
            if agent_id:
                snapshot["agent_id"] = agent_id
                snapshot["agent_state"] = {
                    "is_active": self.config.get(f"agent.{agent_id}.is_active", default=True),
                    "parameters": {}
                }

                # Capture agent parameters
                for param in ["confidence_threshold", "weight", "lookback_window", "sensitivity"]:
                    param_key = f"agent.{agent_id}.{param}"
                    value = self.config.get(param_key, default=None)
                    if value is not None:
                        snapshot["agent_state"]["parameters"][param] = value

        # Capture connector file for API integration
        if proposal.action_type == "integrate_api":
            extra_data = proposal.extra_data or {}
            snapshot["connector_file"] = extra_data.get("connector_file")

        return snapshot

    async def _execute_rollback(
        self,
        proposal: EvolutionProposal,
        rollback_snapshot: dict,
        reason: str
    ) -> bool:
        """Execute rollback to restore system to pre-deployment state.

        Args:
            proposal: Proposal to rollback
            rollback_snapshot: State snapshot from before deployment
            reason: Reason for rollback

        Returns:
            True if rollback successful, False otherwise
        """
        logger.warning(f"[ROLLBACK] Rolling back {proposal.proposal_id}: {reason}")

        try:
            action_type = rollback_snapshot.get("action_type")

            # Rollback config changes
            if action_type in ["increase_threshold", "decrease_threshold"]:
                old_value = rollback_snapshot.get("config_value")
                if old_value is not None:
                    await self.config.set(
                        key=rollback_snapshot["parameter_path"],
                        value=old_value,
                        regime=None,
                        reason=f"Rollback: {reason}",
                        updated_by="evolution_engine_rollback",
                        performance_delta=None
                    )
                    logger.info(f"[ROLLBACK] Restored config: {rollback_snapshot['parameter_path']} = {old_value}")

            # Rollback agent changes
            elif action_type in ["disable_agent", "mutate_parameters"]:
                agent_id = rollback_snapshot.get("agent_id")
                agent_state = rollback_snapshot.get("agent_state", {})

                if agent_id and agent_state:
                    # Restore is_active state
                    await self.config.set(
                        key=f"agent.{agent_id}.is_active",
                        value=agent_state.get("is_active", True),
                        regime=None,
                        reason=f"Rollback: {reason}",
                        updated_by="evolution_engine_rollback",
                        performance_delta=None
                    )

                    # Restore agent parameters
                    for param_name, param_value in agent_state.get("parameters", {}).items():
                        await self.config.set(
                            key=f"agent.{agent_id}.{param_name}",
                            value=param_value,
                            regime=None,
                            reason=f"Rollback: {reason}",
                            updated_by="evolution_engine_rollback",
                            performance_delta=None
                        )

                    logger.info(f"[ROLLBACK] Restored agent: {agent_id}")

            # Rollback API integration (remove connector)
            elif action_type == "integrate_api":
                connector_file = rollback_snapshot.get("connector_file")
                if connector_file:
                    import os
                    full_path = f"/Users/mdshaonsikder/dev/projects/atomicx/{connector_file}"
                    if os.path.exists(full_path):
                        # Rename instead of delete (safer)
                        os.rename(full_path, f"{full_path}.rolled_back")
                        logger.info(f"[ROLLBACK] Disabled connector: {connector_file}")

            # Log rollback event
            async with get_session() as session:
                from atomicx.data.storage.models import AgentEvolutionLog
                stmt = insert(AgentEvolutionLog).values(
                    agent_id="system",
                    event_type="rollback",
                    old_params=rollback_snapshot,
                    new_params={},
                    reason=reason,
                    win_rate_before=None,
                    win_rate_after=None
                )
                await session.execute(stmt)
                await session.commit()

            logger.success(f"[ROLLBACK] Rollback complete: {proposal.proposal_id}")
            return True

        except Exception as e:
            logger.error(f"[ROLLBACK] Rollback failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def _track_api_costs(self, session) -> None:
        """Track API costs for AGI-discovered connectors.

        Monitors spending and auto-disables expensive APIs if cost limit exceeded.
        """
        cost_tracking_enabled = self.config.get("agi.cost_tracking_enabled", default=True)
        if not cost_tracking_enabled:
            return

        max_monthly_cost = self.config.get("agi.max_monthly_cost", default=100.0)
        auto_rollback_on_exceeded = self.config.get("agi.auto_rollback_on_cost_exceeded", default=True)

        # Query API integration proposals that are deployed
        deployed_apis = await session.execute(
            select(EvolutionProposal).where(
                and_(
                    EvolutionProposal.action_type == "integrate_api",
                    EvolutionProposal.status.in_(["deployed", "finalized"])
                )
            )
        )

        total_monthly_cost = 0.0

        for proposal in deployed_apis.scalars():
            # Extract cost information from proposal
            if isinstance(proposal.proposed_value, dict):
                api_spec = proposal.proposed_value.get("api_candidate", {})
                pricing = api_spec.get("pricing", "")

                # Parse estimated monthly cost from pricing string
                # This is a heuristic - real implementation would track actual usage
                estimated_cost = self._estimate_monthly_cost(pricing)
                total_monthly_cost += estimated_cost

                # Store cost in proposal extra_data
                if not proposal.extra_data:
                    proposal.extra_data = {}
                proposal.extra_data["estimated_monthly_cost"] = estimated_cost
                proposal.extra_data["last_cost_check"] = datetime.now(timezone.utc).isoformat()

        # Check if cost limit exceeded
        if total_monthly_cost > max_monthly_cost:
            logger.critical(
                f"[COST TRACKING] Monthly cost limit exceeded: "
                f"${total_monthly_cost:.2f} > ${max_monthly_cost:.2f}"
            )

            if auto_rollback_on_exceeded:
                # Find most expensive API and rollback
                most_expensive = None
                max_cost = 0.0

                for proposal in deployed_apis.scalars():
                    cost = proposal.extra_data.get("estimated_monthly_cost", 0.0) if proposal.extra_data else 0.0
                    if cost > max_cost:
                        max_cost = cost
                        most_expensive = proposal

                if most_expensive:
                    logger.warning(
                        f"[COST TRACKING] Auto-rolling back most expensive API: "
                        f"{most_expensive.proposal_id} (${max_cost:.2f}/month)"
                    )

                    # Execute rollback
                    monitor_data = self._deployment_monitors.get(most_expensive.proposal_id)
                    if monitor_data:
                        await self._execute_rollback(
                            proposal=most_expensive,
                            rollback_snapshot=monitor_data.get("rollback_snapshot", {}),
                            reason=f"Cost limit exceeded: ${total_monthly_cost:.2f} > ${max_monthly_cost:.2f}"
                        )

                        most_expensive.status = "rolled_back"
                        most_expensive.rejection_reason = "Cost limit exceeded"
                        await session.commit()

        else:
            logger.info(f"[COST TRACKING] Monthly API cost: ${total_monthly_cost:.2f} / ${max_monthly_cost:.2f}")

        await session.commit()

    def _sanitize_sql_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier to prevent injection.

        Only allows alphanumeric characters and underscores.
        Replaces invalid characters with underscores.

        Args:
            identifier: Table name, column name, or other SQL identifier

        Returns:
            Sanitized identifier safe for SQL use
        """
        import re

        # Remove any non-alphanumeric characters except underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', identifier)

        # Ensure it starts with a letter (SQL requirement)
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'tbl_' + sanitized

        # Ensure not empty
        if not sanitized:
            sanitized = 'generated_table'

        # Limit length to 63 characters (PostgreSQL limit)
        return sanitized[:63]

    def _estimate_monthly_cost(self, pricing_string: str) -> float:
        """Estimate monthly cost from pricing description.

        This is a heuristic parser. Real implementation would track actual API usage.

        Args:
            pricing_string: Pricing description (e.g., "free", "$0.01 per call", "$50/month")

        Returns:
            Estimated monthly cost in USD
        """
        pricing_lower = pricing_string.lower()

        # Check for "free" BUT not in context of paid tier (e.g., "free tier, $20 for pro")
        if "free" in pricing_lower and "freemium" not in pricing_lower:
            # If there's also a dollar amount, it's freemium with paid tier
            import re
            dollar_matches = re.findall(r'\$(\d+(?:\.\d+)?)', pricing_string)
            if not dollar_matches:
                return 0.0
            # Has both "free" and price → freemium, use the price

        # Parse dollar amounts
        import re
        dollar_matches = re.findall(r'\$(\d+(?:\.\d+)?)', pricing_string)

        if dollar_matches:
            amount = float(dollar_matches[0])

            # Check if it's per month
            if "/month" in pricing_lower or "per month" in pricing_lower or "monthly" in pricing_lower:
                return amount

            # If per call/request, estimate based on typical usage
            if "per call" in pricing_lower or "per request" in pricing_lower:
                # Assume 10,000 calls per month (aggressive usage)
                return amount * 10000

            # If just a flat fee with no context, assume it's monthly
            return amount

        # Default: assume moderate cost
        return 10.0
