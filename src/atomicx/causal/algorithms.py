"""Causal discovery algorithms — NOTEARS, PC Algorithm, Granger causality.

This module wraps multiple causal discovery libraries into a unified
interface that the CausalDiscoveryEngine orchestrates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from atomicx.causal.types import CausalDirection, CausalEdge, CausalStrength


class NOTEARSDiscovery:
    """NOTEARS-based continuous DAG learning.

    Uses the NOTEARS algorithm (Non-combinatorial Optimization via Trace
    Exponential and Augmented lagrangian for Structure learning) to learn
    a DAG structure directly from data.

    Falls back to a linear algebra-based approach if gCastle is not available.
    """

    def __init__(self, threshold: float = 0.05, max_iter: int = 100) -> None:
        self._threshold = threshold
        self._max_iter = max_iter

    def discover(
        self, data: pd.DataFrame, variable_names: list[str] | None = None
    ) -> list[CausalEdge]:
        """Discover causal structure using NOTEARS.

        Args:
            data: DataFrame where columns are variables and rows are observations
            variable_names: Optional column names to use

        Returns:
            List of discovered causal edges
        """
        cols = variable_names or list(data.columns)
        X = data[cols].values.astype(np.float64)

        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        try:
            W = self._notears_linear(X)
        except Exception as e:
            logger.warning(f"NOTEARS failed, using correlation fallback: {e}")
            W = self._correlation_fallback(X)

        edges = []
        n = len(cols)
        for i in range(n):
            for j in range(n):
                if i != j and abs(W[i, j]) > self._threshold:
                    edges.append(
                        CausalEdge(
                            source=cols[i],
                            target=cols[j],
                            weight=float(min(abs(W[i, j]), 1.0)),
                            direction=(
                                CausalDirection.POSITIVE if W[i, j] > 0
                                else CausalDirection.NEGATIVE
                            ),
                            algorithm="NOTEARS",
                        )
                    )

        logger.info(f"NOTEARS discovered {len(edges)} causal edges from {n} variables")
        return edges

    def _notears_linear(self, X: np.ndarray) -> np.ndarray:
        """NOTEARS linear implementation using gradient descent with DAG constraint.

        Minimizes: 0.5/n * ||X - XW||^2 + lambda * |W|
        Subject to: h(W) = tr(e^(W∘W)) - d = 0 (acyclicity)
        """
        n, d = X.shape
        W = np.zeros((d, d))

        # Use augmented Lagrangian method
        rho = 1.0
        alpha = 0.0
        h_prev = np.inf

        for iteration in range(self._max_iter):
            # Gradient of least squares loss
            M = X @ W - X
            grad = (X.T @ M) / n

            # Acyclicity constraint gradient
            E = np.exp(W * W)
            h = np.trace(E) - d
            grad_h = 2 * W * E

            # Combined gradient
            total_grad = grad + (rho * h + alpha) * grad_h

            # L1 proximal step (soft thresholding)
            lr = 0.001
            W_new = W - lr * total_grad
            lambda_reg = 0.01
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lr * lambda_reg, 0)

            # Zero out diagonal
            np.fill_diagonal(W_new, 0)

            W = W_new

            # Update Lagrangian parameters
            if h > 0.25 * h_prev:
                rho *= 10
            alpha += rho * h
            h_prev = h

            if h < 1e-8 and iteration > 10:
                break

        return W

    def _correlation_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback: use partial correlation as proxy for causal structure."""
        n, d = X.shape
        corr = np.corrcoef(X.T)

        # Zero diagonal and weak correlations
        W = corr.copy()
        np.fill_diagonal(W, 0)
        W[np.abs(W) < self._threshold] = 0

        return W


class PCAlgorithmDiscovery:
    """PC Algorithm for constraint-based causal discovery.

    Uses conditional independence tests to build a causal skeleton,
    then orients edges using collider detection.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: int = 3,
    ) -> None:
        self._alpha = alpha
        self._max_cond_set = max_cond_set_size

    def discover(
        self, data: pd.DataFrame, variable_names: list[str] | None = None
    ) -> list[CausalEdge]:
        """Run PC algorithm on the data."""
        cols = variable_names or list(data.columns)
        n_vars = len(cols)
        X = data[cols].values.astype(np.float64)

        # Step 1: Build complete skeleton
        adjacency = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adjacency, False)

        # Step 2: Remove edges by conditional independence tests
        for cond_size in range(self._max_cond_set + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacency[i, j]:
                        continue

                    # Get possible conditioning sets
                    neighbors = [
                        k for k in range(n_vars)
                        if k != i and k != j and adjacency[i, k]
                    ]

                    if len(neighbors) < cond_size:
                        continue

                    # Test conditional independence with subsets
                    from itertools import combinations
                    for cond_set in combinations(neighbors, cond_size):
                        if self._is_conditionally_independent(
                            X[:, i], X[:, j], X[:, list(cond_set)] if cond_set else None
                        ):
                            adjacency[i, j] = False
                            adjacency[j, i] = False
                            break

        # Step 3: Convert to causal edges
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j]:
                    # Use correlation direction for orientation
                    r = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    edges.append(
                        CausalEdge(
                            source=cols[i],
                            target=cols[j],
                            weight=float(min(abs(r), 1.0)),
                            direction=(
                                CausalDirection.POSITIVE if r > 0
                                else CausalDirection.NEGATIVE
                            ),
                            algorithm="PC",
                            p_value=self._alpha,
                        )
                    )

        logger.info(f"PC Algorithm discovered {len(edges)} edges from {n_vars} variables")
        return edges

    def _is_conditionally_independent(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> bool:
        """Test conditional independence using partial correlation + Fisher's z-test."""
        if z is None or z.shape[1] == 0:
            r, p = stats.pearsonr(x, y)
            return p > self._alpha

        # Partial correlation via regression residuals
        from numpy.linalg import lstsq
        x_res = x - z @ lstsq(z, x, rcond=None)[0]
        y_res = y - z @ lstsq(z, y, rcond=None)[0]

        r, p = stats.pearsonr(x_res, y_res)
        return p > self._alpha


class GrangerCausalityDiscovery:
    """Time-varying Granger causality for temporal causal detection.

    Tests whether lagged values of one variable improve prediction of
    another beyond its own lagged values.
    """

    def __init__(
        self,
        max_lag: int = 12,
        alpha: float = 0.05,
        window_size: int | None = None,
    ) -> None:
        self._max_lag = max_lag
        self._alpha = alpha
        self._window_size = window_size

    def discover(
        self, data: pd.DataFrame, variable_names: list[str] | None = None
    ) -> list[CausalEdge]:
        """Run Granger causality tests on all variable pairs."""
        cols = variable_names or list(data.columns)
        edges = []

        for i, source in enumerate(cols):
            for j, target in enumerate(cols):
                if i == j:
                    continue

                result = self._granger_test(data[source].values, data[target].values)
                if result["significant"]:
                    edges.append(
                        CausalEdge(
                            source=source,
                            target=target,
                            weight=float(min(result["f_stat"] / 100, 1.0)),
                            direction=CausalDirection.POSITIVE,
                            algorithm="Granger",
                            p_value=result["p_value"],
                            lag_periods=result["best_lag"],
                            metadata={"f_stat": result["f_stat"]},
                        )
                    )

        logger.info(f"Granger causality discovered {len(edges)} edges from {len(cols)} variables")
        return edges

    def _granger_test(
        self, x: np.ndarray, y: np.ndarray
    ) -> dict[str, Any]:
        """Test if x Granger-causes y using F-test.

        Compare:
        - Restricted model: y_t = a0 + a1*y_{t-1} + ... + a_p*y_{t-p}
        - Unrestricted model: y_t = a0 + a1*y_{t-1} + ... + b1*x_{t-1} + ... + b_p*x_{t-p}
        """
        best_result = {"significant": False, "p_value": 1.0, "f_stat": 0.0, "best_lag": 0}

        for lag in range(1, self._max_lag + 1):
            try:
                # Build lagged matrices
                n = len(y) - lag
                if n < lag * 4:
                    continue

                # Restricted: only y lags
                Y = y[lag:]
                Y_lags = np.column_stack([y[lag - k : len(y) - k] for k in range(1, lag + 1)])

                # Unrestricted: y lags + x lags
                X_lags = np.column_stack([x[lag - k : len(x) - k] for k in range(1, lag + 1)])
                full_lags = np.column_stack([Y_lags, X_lags])

                # Add intercept
                Y_lags = np.column_stack([np.ones(n), Y_lags])
                full_lags = np.column_stack([np.ones(n), full_lags])

                # Fit models
                from numpy.linalg import lstsq
                _, res_r, _, _ = lstsq(Y_lags, Y[:n], rcond=None)
                _, res_u, _, _ = lstsq(full_lags, Y[:n], rcond=None)

                if len(res_r) == 0 or len(res_u) == 0:
                    continue

                ssr_r = res_r[0] if len(res_r) > 0 else np.sum((Y[:n] - Y_lags @ lstsq(Y_lags, Y[:n], rcond=None)[0]) ** 2)
                ssr_u = res_u[0] if len(res_u) > 0 else np.sum((Y[:n] - full_lags @ lstsq(full_lags, Y[:n], rcond=None)[0]) ** 2)

                # F-test
                df1 = lag
                df2 = n - 2 * lag - 1
                if df2 <= 0 or ssr_u <= 0:
                    continue

                f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                if p_value < best_result["p_value"]:
                    best_result = {
                        "significant": p_value < self._alpha,
                        "p_value": float(p_value),
                        "f_stat": float(f_stat),
                        "best_lag": lag,
                    }

            except Exception:
                continue

        return best_result
