"""Causal Discovery Engine — orchestrates all causal discovery algorithms.

Runs NOTEARS, PC Algorithm, and Granger Causality on variable data,
combines their results into a unified causal DAG, and maintains
a persistent causal graph that evolves with new data.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select

from atomicx.causal.algorithms import (
    GrangerCausalityDiscovery,
    NOTEARSDiscovery,
    PCAlgorithmDiscovery,
)
from atomicx.causal.types import (
    CausalChain,
    CausalDAG,
    CausalDirection,
    CausalEdge,
    CausalStrength,
)
from atomicx.data.storage.database import get_session_factory
from atomicx.variables.models import ComputedVariable


class CausalDiscoveryEngine:
    """Orchestrates causal discovery across all algorithms.

    Workflow:
    1. Load computed variable data from TimescaleDB
    2. Run NOTEARS, PC Algorithm, and Granger causality
    3. Combine results — edges confirmed by 2+ algorithms are stronger
    4. Build causal chains (multi-hop paths to price)
    5. Prune weak edges (< 2% contribution after 200 predictions)
    6. Store results for the agent hierarchy to consume
    """

    DEFAULT_WEIGHTS = {
        "RSI_14": 0.65, "RSI_7": 0.58,
        "STOCH_RSI_K": 0.62, "STOCH_RSI_D": 0.60,
        "MACD_LINE": 0.63, "MACD_HISTOGRAM": 0.60, "MACD_SIGNAL": 0.55,
        "EMA_9": 0.50, "EMA_21": 0.52, "EMA_50": 0.48, "EMA_200": 0.45,
        "VWAP": 0.58, "OBV": 0.52, "ADX": 0.60,
        "FUNDING_RATE": 0.72, "FUNDING_ZSCORE": 0.70,
        "REL_VOLUME": 0.55, "BB_PERCENT_B": 0.50,
        "ATR_14": 0.45, "VOL_RATIO": 0.48,
    }

    def __init__(self) -> None:
        self._session_factory = get_session_factory()
        self._notears = NOTEARSDiscovery(threshold=0.05)
        self._pc = PCAlgorithmDiscovery(alpha=0.05, max_cond_set_size=3)
        self._granger = GrangerCausalityDiscovery(max_lag=12, alpha=0.05)
        self._current_dag: CausalDAG | None = None
        self.weights = self.DEFAULT_WEIGHTS.copy()

    def initialize_with_defaults(self) -> None:
        """Initialize the engine with pre-computed causal assumptions."""
        self.weights = self.DEFAULT_WEIGHTS.copy()
        logger.info(f"Causal Engine initialized with {len(self.weights)} default weights")

    def get_weights(self) -> dict[str, float]:
        """Return the current set of causal weights."""
        return self.weights

    def get_weight(self, variable_id: str) -> float:
        """Get the weight for a specific variable."""
        return self.weights.get(variable_id, 0.5)

    async def discover(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        target_variable: str = "close",
        max_variables: int = 30,
    ) -> CausalDAG:
        """Run full causal discovery pipeline.

        Args:
            symbol: Trading pair to analyze
            timeframe: Data timeframe
            target_variable: The outcome variable (usually 'close' or 'returns')
            max_variables: Max variables to include (computational constraint)

        Returns:
            CausalDAG with all discovered relationships
        """
        logger.info(f"Starting causal discovery for {symbol} @ {timeframe}")

        # Step 1: Load data
        df = await self._load_variable_matrix(symbol, timeframe, max_variables)
        if df.empty or len(df.columns) < 3:
            logger.warning("Not enough data for causal discovery")
            return CausalDAG()

        logger.info(f"Loaded {len(df)} observations × {len(df.columns)} variables")

        # Step 2: Run all algorithms
        all_edges: list[CausalEdge] = []

        # NOTEARS
        try:
            notears_edges = self._notears.discover(df)
            all_edges.extend(notears_edges)
        except Exception as e:
            logger.error(f"NOTEARS failed: {e}")

        # PC Algorithm
        try:
            pc_edges = self._pc.discover(df)
            all_edges.extend(pc_edges)
        except Exception as e:
            logger.error(f"PC Algorithm failed: {e}")

        # Granger Causality
        try:
            granger_edges = self._granger.discover(df)
            all_edges.extend(granger_edges)
        except Exception as e:
            logger.error(f"Granger causality failed: {e}")

        # Step 3: Combine results
        dag = self._combine_edges(all_edges)
        logger.info(
            f"Combined DAG: {len(dag.edges)} edges "
            f"({sum(1 for e in dag.edges if e.strength == CausalStrength.STRONG)} strong, "
            f"{sum(1 for e in dag.edges if e.strength == CausalStrength.MODERATE)} moderate)"
        )

        # Step 4: Build causal chains
        dag.chains = self._build_chains(dag, target_variable)
        logger.info(f"Built {len(dag.chains)} causal chains to {target_variable}")

        # Step 5: Prune weak edges
        pruned = dag.prune_weak_edges(min_weight=0.02)
        if pruned > 0:
            logger.info(f"Pruned {pruned} weak edges")

        dag.variable_count = len(df.columns)
        dag.algorithm_versions = {
            "notears": "v1.0",
            "pc": "v1.0",
            "granger": "v1.0",
        }

        self._current_dag = dag
        return dag

    def _combine_edges(self, all_edges: list[CausalEdge]) -> CausalDAG:
        """Combine edges from multiple algorithms.

        Edges confirmed by 2+ algorithms get higher strength.
        """
        edge_map: dict[tuple[str, str], list[CausalEdge]] = {}

        for edge in all_edges:
            key = (edge.source, edge.target)
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(edge)

        combined_edges = []
        for (source, target), edges in edge_map.items():
            # Average weight across algorithms
            avg_weight = np.mean([e.weight for e in edges])
            n_algorithms = len(set(e.algorithm for e in edges))

            # Boost weight for edges confirmed by multiple algorithms
            if n_algorithms >= 3:
                strength = CausalStrength.STRONG
                avg_weight = min(avg_weight * 1.5, 1.0)
            elif n_algorithms >= 2:
                strength = CausalStrength.MODERATE
                avg_weight = min(avg_weight * 1.2, 1.0)
            else:
                strength = CausalStrength.WEAK

            # Determine direction (majority vote)
            directions = [e.direction for e in edges]
            pos_count = sum(1 for d in directions if d == CausalDirection.POSITIVE)
            neg_count = sum(1 for d in directions if d == CausalDirection.NEGATIVE)
            direction = CausalDirection.POSITIVE if pos_count >= neg_count else CausalDirection.NEGATIVE

            # Best p-value across algorithms
            p_values = [e.p_value for e in edges if e.p_value is not None]
            best_p = min(p_values) if p_values else None

            # Max lag from Granger
            lags = [e.lag_periods for e in edges if e.lag_periods > 0]
            best_lag = max(lags) if lags else 0

            combined_edges.append(
                CausalEdge(
                    source=source,
                    target=target,
                    weight=float(avg_weight),
                    direction=direction,
                    strength=strength,
                    algorithm=",".join(set(e.algorithm for e in edges)),
                    p_value=best_p,
                    lag_periods=best_lag,
                    metadata={
                        "n_algorithms": n_algorithms,
                        "algorithms": list(set(e.algorithm for e in edges)),
                    },
                )
            )

        return CausalDAG(edges=combined_edges)

    def _build_chains(
        self,
        dag: CausalDAG,
        target: str,
        max_depth: int = 4,
    ) -> list[CausalChain]:
        """Build causal chains — multi-hop paths from any variable to the target.

        Uses DFS to find all paths of length <= max_depth.
        """
        # Build adjacency for fast lookup
        adj: dict[str, list[CausalEdge]] = {}
        for edge in dag.edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge)

        chains: list[CausalChain] = []
        chain_id = 0

        def dfs(node: str, path: list[CausalEdge], visited: set[str]) -> None:
            nonlocal chain_id
            if len(path) > max_depth:
                return
            if node == target and len(path) > 0:
                total_strength = 1.0
                for edge in path:
                    total_strength *= edge.weight

                reasoning = " → ".join(
                    [path[0].source] + [e.target for e in path]
                )
                chains.append(
                    CausalChain(
                        chain_id=f"chain_{chain_id}",
                        edges=list(path),
                        total_strength=total_strength,
                        reasoning=f"Causal path: {reasoning}",
                    )
                )
                chain_id += 1
                return

            if node in adj:
                for edge in adj[node]:
                    if edge.target not in visited:
                        visited.add(edge.target)
                        path.append(edge)
                        dfs(edge.target, path, visited)
                        path.pop()
                        visited.discard(edge.target)

        # Start DFS from every variable
        all_vars = set()
        for edge in dag.edges:
            all_vars.add(edge.source)
            all_vars.add(edge.target)

        for var in all_vars:
            if var != target:
                dfs(var, [], {var})

        # Sort by total strength
        chains.sort(key=lambda c: c.total_strength, reverse=True)
        return chains[:100]  # Top 100 chains

    async def _load_variable_matrix(
        self,
        symbol: str,
        timeframe: str,
        max_variables: int,
    ) -> pd.DataFrame:
        """Load computed variables as a wide matrix for causal analysis."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ComputedVariable)
                .where(
                    ComputedVariable.symbol == symbol,
                    ComputedVariable.timeframe == timeframe,
                )
                .order_by(ComputedVariable.timestamp)
            )
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"timestamp": r.timestamp, "variable_id": r.variable_id, "value": r.value}
            for r in rows
        ])

        # Pivot to wide format
        wide = df.pivot_table(index="timestamp", columns="variable_id", values="value", aggfunc="last")

        # Drop columns with too many NaN values
        threshold = 0.5 * len(wide)
        wide = wide.dropna(axis=1, thresh=int(threshold))

        # Fill remaining NaN with forward fill
        wide = wide.ffill().dropna()

        # Limit variables if needed
        if len(wide.columns) > max_variables:
            # Keep variables with highest variance (most informative)
            variances = wide.var().sort_values(ascending=False)
            keep_cols = variances.head(max_variables).index
            wide = wide[keep_cols]

        return wide

    @property
    def current_dag(self) -> CausalDAG | None:
        """Get the most recently computed causal DAG."""
        return self._current_dag

    def get_causes_of_price(self, top_k: int = 20) -> list[dict[str, Any]]:
        """Get the top causes of price movement from the current DAG."""
        if not self._current_dag:
            return []

        causes = self._current_dag.get_causes_of("close")
        causes.sort(key=lambda e: e.weight, reverse=True)

        return [
            {
                "variable": e.source,
                "weight": e.weight,
                "direction": e.direction.value,
                "strength": e.strength.value,
                "algorithm": e.algorithm,
                "lag": e.lag_periods,
            }
            for e in causes[:top_k]
        ]

    def apply_weights(self, new_weights: dict[str, float]) -> None:
        """Apply CausalRL-adjusted weights to the engine.

        Args:
            new_weights: Dictionary of variable_id -> weight mappings

        Updates the internal weights dictionary with RL-optimized values
        that will be used in the next prediction cycle.
        """
        if not new_weights:
            return

        updated_count = 0
        for var_id, weight in new_weights.items():
            if var_id in self.weights:
                old_weight = self.weights[var_id]
                self.weights[var_id] = weight
                updated_count += 1
                logger.debug(
                    f"[CAUSAL] Updated {var_id}: {old_weight:.3f} → {weight:.3f}"
                )

        if updated_count > 0:
            logger.info(
                f"[CAUSAL] Applied {updated_count} CausalRL weight adjustments"
            )
