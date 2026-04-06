"""Kernel Correlation Engine — Renaissance-Style Hidden Data.

Uses kernel methods and non-linear correlation to find hidden relationships
between seemingly unrelated alternative data sources and crypto price action.

Examples of "hidden" connections:
- Global Electricity Grid Stress → Bitcoin mining profitability → miner sell pressure
- GPU Shipping Delays → AI compute costs → institutional crypto allocation shifts
- Cloud Cover in Tokyo → Solar energy output → Japanese exchange activity patterns
- Port Congestion Index → Global supply chain stress → risk-off sentiment
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field


class AlternativeDataSource(BaseModel):
    """An alternative data stream tracked by the Kernel Engine."""
    source_id: str = Field(default_factory=lambda: f"alt-{uuid.uuid4().hex[:8]}")
    name: str
    category: str  # "energy", "supply_chain", "weather", "hardware", "macro"
    current_value: float = 0.0
    historical_values: list[float] = Field(default_factory=list)
    kernel_correlation_to_btc: float = 0.0  # Non-linear correlation score [-1, 1]
    last_updated: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    significance: str = "unknown"  # "strong", "moderate", "weak", "unknown"


class KernelCorrelationEngine:
    """Discovers hidden non-linear relationships between alternative data and crypto.
    
    Renaissance Tech's edge: they found correlations invisible to linear analysis.
    Our edge: we use the same math but on publicly available alternative data.
    """
    
    def __init__(self) -> None:
        self.logger = logger.bind(module="titan.kernel")
        self.sources: dict[str, AlternativeDataSource] = {}
        self.correlation_matrix: dict[str, dict[str, float]] = {}
        
        # Pre-register the core alternative data streams
        self._register_default_sources()
        
    def _register_default_sources(self) -> None:
        """Register the default set of alternative data streams to track."""
        defaults = [
            ("Global Electricity Price Index", "energy"),
            ("GPU Shipping Volume (TSMC/NVIDIA)", "hardware"),
            ("Bitcoin Hash Rate", "energy"),
            ("US Natural Gas Futures", "energy"),
            ("Baltic Dry Index", "supply_chain"),
            ("Global Port Congestion Index", "supply_chain"),
            ("Fed Reverse Repo Facility Balance", "macro"),
            ("Stablecoin Net Flow (USDT+USDC)", "macro"),
            ("Crypto Fear & Greed Index", "macro"),
            ("Reddit WSB Mention Velocity", "sentiment"),
        ]
        for name, category in defaults:
            self.sources[name] = AlternativeDataSource(name=name, category=category)
            
        self.logger.info(f"[KERNEL] Registered {len(self.sources)} alternative data sources")
        
    def ingest_data_point(self, source_name: str, value: float) -> None:
        """Feed a new data point into an alternative data stream."""
        if source_name not in self.sources:
            self.sources[source_name] = AlternativeDataSource(
                name=source_name, category="unknown"
            )
            
        src = self.sources[source_name]
        src.current_value = value
        src.historical_values.append(value)
        src.last_updated = datetime.now(tz=timezone.utc)
        
        # Keep only the last 1000 data points
        if len(src.historical_values) > 1000:
            src.historical_values = src.historical_values[-1000:]
            
    def compute_kernel_correlations(self, btc_prices: list[float]) -> dict[str, float]:
        """Compute non-linear correlations between all sources and BTC.
        
        Uses a Radial Basis Function (RBF) kernel-based approach:
        1. Standardize both series
        2. Compute kernel matrix using RBF kernel
        3. Extract HSIC-like dependence measure
        
        Requires at least 30 data points for meaningful results.
        """
        import numpy as np
        
        results = {}
        
        if len(btc_prices) < 30:
            self.logger.debug("[KERNEL] Not enough BTC price data for correlation analysis")
            return results
        
        btc_arr = np.array(btc_prices[-200:], dtype=float)
        # Standardize BTC prices
        btc_std = (btc_arr - btc_arr.mean()) / (btc_arr.std() + 1e-8)
        
        for name, src in self.sources.items():
            if len(src.historical_values) < 30:
                continue
            
            # Align lengths
            alt_arr = np.array(src.historical_values[-len(btc_arr):], dtype=float)
            if len(alt_arr) != len(btc_arr):
                min_len = min(len(alt_arr), len(btc_arr))
                alt_arr = alt_arr[-min_len:]
                btc_slice = btc_std[-min_len:]
            else:
                btc_slice = btc_std
                
            # Standardize alt data
            alt_std = (alt_arr - alt_arr.mean()) / (alt_arr.std() + 1e-8)
            
            # RBF Kernel: K(x,y) = exp(-gamma * ||x-y||^2)
            # We compute a kernel-based dependence measure:
            # For each time step, compute RBF similarity and aggregate
            gamma = 1.0 / (2.0 * max(alt_std.var(), 1e-8))
            
            # Compute pairwise distances and kernel values
            diff = alt_std - btc_slice
            kernel_vals = np.exp(-gamma * diff ** 2)
            
            # Dependence measure: deviation of kernel mean from expected under independence
            # Under independence, E[K] ≈ 1/sqrt(1+2*gamma*var)
            kernel_mean = kernel_vals.mean()
            independence_baseline = 1.0 / np.sqrt(1 + 2 * gamma)
            
            # Signed correlation: positive if co-movement, negative if inverse
            # Use Pearson on standardized series for direction
            pearson = np.corrcoef(alt_std, btc_slice)[0, 1]
            
            # Combine: magnitude from kernel, direction from Pearson
            kernel_strength = min(abs(kernel_mean - independence_baseline) * 5, 1.0)
            kernel_corr = float(kernel_strength * np.sign(pearson))
            kernel_corr = max(-1.0, min(1.0, kernel_corr))
            
            src.kernel_correlation_to_btc = kernel_corr
            src.significance = (
                "strong" if abs(kernel_corr) > 0.7 else
                "moderate" if abs(kernel_corr) > 0.4 else
                "weak"
            )
            results[name] = kernel_corr
            
        # Log discoveries
        strong = {k: v for k, v in results.items() if abs(v) > 0.7}
        if strong:
            self.logger.success(f"[KERNEL] STRONG hidden correlations discovered: {strong}")
            
        return results
        
    def get_hidden_signals(self) -> list[dict[str, Any]]:
        """Return currently active hidden signals from alternative data."""
        signals = []
        for name, src in self.sources.items():
            if src.significance == "strong":
                direction = "bullish" if src.kernel_correlation_to_btc > 0 else "bearish"
                signals.append({
                    "source": name,
                    "category": src.category,
                    "correlation": src.kernel_correlation_to_btc,
                    "direction": direction,
                    "current_value": src.current_value,
                })
        return signals
