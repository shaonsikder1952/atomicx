"""Variable Registry — manages variable lifecycle, metadata, and performance tracking.

The registry is the single source of truth for what variables exist,
how they're configured, and how they're performing. It enables:
- Dynamic registration of new variables via config (VAR-05)
- Weight and performance tracking per variable (VAR-04)
- Auto-demotion of underperforming variables (CAUS-04)
- Querying variable metadata and history
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from atomicx.data.storage.database import get_session_factory
from atomicx.variables.models import (
    ComputedVariable,
    VariableRegistryEntry,
    VariableWeightHistory,
)
from atomicx.variables.types import VariableDefinition, VariableStatus, VariableValue


class VariableRegistry:
    """Manages the complete lifecycle of variables in the system.

    Thread-safe, backed by PostgreSQL for persistence.
    In-memory cache for fast lookups during computation.
    """

    def __init__(self) -> None:
        self._session_factory = get_session_factory()
        self._cache: dict[str, VariableDefinition] = {}

    async def register(self, variable: VariableDefinition) -> None:
        """Register a new variable or update an existing one."""
        async with self._session_factory() as session:
            stmt = pg_insert(VariableRegistryEntry).values(
                variable_id=variable.id,
                name=variable.name,
                description=variable.description,
                domain=variable.domain.value,
                category=variable.category,
                tags=variable.tags,
                source=variable.source,
                update_frequency=variable.update_frequency.value,
                causal_half_life=variable.causal_half_life,
                reliability_score=variable.reliability_score,
                lookback_periods=variable.lookback_periods,
                params=variable.params,
                symbol_specific=variable.symbol_specific,
                status=variable.status.value,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["variable_id"],
                set_={
                    "name": stmt.excluded.name,
                    "description": stmt.excluded.description,
                    "params": stmt.excluded.params,
                    "reliability_score": stmt.excluded.reliability_score,
                    "causal_half_life": stmt.excluded.causal_half_life,
                },
            )
            await session.execute(stmt)
            await session.commit()

        self._cache[variable.id] = variable
        logger.debug(f"Registered variable: {variable.id}")

    async def register_batch(self, variables: list[VariableDefinition]) -> None:
        """Register multiple variables at once."""
        for v in variables:
            await self.register(v)
        logger.info(f"Registered {len(variables)} variables")

    async def get(self, variable_id: str) -> VariableDefinition | None:
        """Get a variable definition by ID (cache-first)."""
        if variable_id in self._cache:
            return self._cache[variable_id]

        async with self._session_factory() as session:
            result = await session.execute(
                select(VariableRegistryEntry).where(
                    VariableRegistryEntry.variable_id == variable_id
                )
            )
            entry = result.scalar_one_or_none()
            if entry:
                var_def = self._entry_to_definition(entry)
                self._cache[variable_id] = var_def
                return var_def
        return None

    async def list_active(self) -> list[VariableDefinition]:
        """Get all active variables."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(VariableRegistryEntry).where(
                    VariableRegistryEntry.status == "active"
                )
            )
            entries = result.scalars().all()
            return [self._entry_to_definition(e) for e in entries]

    async def list_by_domain(self, domain: str) -> list[VariableDefinition]:
        """Get all variables in a specific domain."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(VariableRegistryEntry).where(
                    VariableRegistryEntry.domain == domain,
                    VariableRegistryEntry.status == "active",
                )
            )
            entries = result.scalars().all()
            return [self._entry_to_definition(e) for e in entries]

    async def store_value(self, value: VariableValue) -> None:
        """Store a computed variable value."""
        async with self._session_factory() as session:
            await session.execute(
                ComputedVariable.__table__.insert().values(
                    timestamp=value.timestamp,
                    variable_id=value.variable_id,
                    symbol=value.symbol,
                    timeframe=value.timeframe.value,
                    value=value.value,
                    confidence=value.confidence,
                    metadata=value.metadata,
                )
            )
            await session.commit()

    async def store_values_batch(self, values: list[VariableValue]) -> None:
        """Batch store computed variable values."""
        if not values:
            return
        async with self._session_factory() as session:
            await session.execute(
                ComputedVariable.__table__.insert(),
                [
                    {
                        "timestamp": v.timestamp,
                        "variable_id": v.variable_id,
                        "symbol": v.symbol,
                        "timeframe": v.timeframe.value,
                        "value": v.value,
                        "confidence": v.confidence,
                        "metadata": v.metadata,
                    }
                    for v in values
                ],
            )
            await session.commit()

    async def update_weight(
        self,
        variable_id: str,
        weight: float,
        performance_edge: float,
        prediction_count: int,
        reason: str = "",
    ) -> None:
        """Update a variable's weight and log the change."""
        now = datetime.now(tz=timezone.utc)

        async with self._session_factory() as session:
            # Update registry
            await session.execute(
                update(VariableRegistryEntry)
                .where(VariableRegistryEntry.variable_id == variable_id)
                .values(
                    current_weight=weight,
                    performance_edge=performance_edge,
                    prediction_count=prediction_count,
                )
            )
            # Log to history
            await session.execute(
                VariableWeightHistory.__table__.insert().values(
                    timestamp=now,
                    variable_id=variable_id,
                    weight=weight,
                    performance_edge=performance_edge,
                    prediction_count=prediction_count,
                    reason=reason,
                )
            )
            await session.commit()

        logger.info(
            f"Variable {variable_id} weight updated: {weight:.3f} "
            f"(edge: {performance_edge:.2%}, n={prediction_count})"
        )

    async def check_demotion(
        self,
        variable_id: str,
        min_edge: float = 0.02,
        min_predictions: int = 200,
    ) -> bool:
        """Check if a variable should be auto-demoted (CAUS-04).

        Returns True if the variable was demoted.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(VariableRegistryEntry).where(
                    VariableRegistryEntry.variable_id == variable_id
                )
            )
            entry = result.scalar_one_or_none()

            if not entry:
                return False

            if (
                entry.prediction_count >= min_predictions
                and entry.performance_edge < min_edge
            ):
                await session.execute(
                    update(VariableRegistryEntry)
                    .where(VariableRegistryEntry.variable_id == variable_id)
                    .values(status="demoted")
                )
                await session.commit()

                # Remove from cache
                self._cache.pop(variable_id, None)

                logger.warning(
                    f"Variable {variable_id} AUTO-DEMOTED: "
                    f"edge={entry.performance_edge:.2%} < {min_edge:.2%} "
                    f"after {entry.prediction_count} predictions"
                )
                return True

        return False

    async def get_performance_summary(self) -> list[dict]:
        """Get performance summary for all variables."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(VariableRegistryEntry).order_by(
                    VariableRegistryEntry.performance_edge.desc()
                )
            )
            entries = result.scalars().all()
            return [
                {
                    "id": e.variable_id,
                    "name": e.name,
                    "domain": e.domain,
                    "weight": e.current_weight,
                    "edge": e.performance_edge,
                    "predictions": e.prediction_count,
                    "status": e.status,
                }
                for e in entries
            ]

    def _entry_to_definition(self, entry: VariableRegistryEntry) -> VariableDefinition:
        """Convert a DB entry to a VariableDefinition."""
        from atomicx.variables.types import VariableDomain, VariableTimeframe

        return VariableDefinition(
            id=entry.variable_id,
            name=entry.name,
            description=entry.description or "",
            domain=VariableDomain(entry.domain),
            category=entry.category,
            tags=entry.tags or [],
            source=entry.source,
            update_frequency=VariableTimeframe(entry.update_frequency),
            causal_half_life=entry.causal_half_life,
            reliability_score=entry.reliability_score,
            lookback_periods=entry.lookback_periods,
            params=entry.params or {},
            symbol_specific=entry.symbol_specific,
            status=VariableStatus(entry.status),
        )
