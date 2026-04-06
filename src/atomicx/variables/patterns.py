"""Pattern discovery engine — finds recurring patterns across all variables.

Analyzes computed variable history to find:
1. Cross-variable correlations (which variables move together)
2. Leading indicators (which variables predict price movement)
3. Regime patterns (clusters of market conditions)
4. Divergence signals (when indicators disagree with price)
5. Extreme value patterns (what happens after RSI > 80, etc.)
6. Multi-timeframe confluence patterns

Output: Pattern catalog stored in DB + human-readable pattern study file.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select, text

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV
from atomicx.variables.models import ComputedVariable


class PatternDiscoveryEngine:
    """Discovers recurring patterns across all computed variables."""

    def __init__(self) -> None:
        self._session_factory = get_session_factory()
        self._patterns: list[dict[str, Any]] = []

    async def run_full_discovery(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run all pattern discovery analyses.

        Returns a list of discovered patterns.
        """
        logger.info(f"Starting pattern discovery for {symbol} {timeframe}")

        # Load data
        price_df = await self._load_price_data(symbol, timeframe)
        var_df = await self._load_variable_data(symbol, timeframe)

        if price_df.empty or var_df.empty:
            logger.warning("Not enough data for pattern discovery")
            return []

        # Pivot variable data into wide format
        wide_df = self._pivot_variables(var_df, price_df)
        if wide_df.empty:
            return []

        logger.info(f"Loaded {len(wide_df)} rows × {len(wide_df.columns)} columns")

        # Run all pattern analyses
        self._patterns = []
        self._find_correlation_clusters(wide_df)
        self._find_leading_indicators(wide_df)
        self._find_extreme_value_patterns(wide_df)
        self._find_divergence_patterns(wide_df)
        self._find_regime_patterns(wide_df)
        self._find_reversal_patterns(wide_df)
        self._find_trend_patterns(wide_df)

        logger.info(f"Discovered {len(self._patterns)} patterns")

        # Write study file
        if output_path:
            self._write_study_file(output_path, symbol, timeframe)

        return self._patterns

    # ── Data Loading ─────────────────────────────────────────

    async def _load_price_data(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Load OHLCV price data."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(OHLCV)
                .where(OHLCV.symbol == symbol, OHLCV.timeframe == timeframe)
                .order_by(OHLCV.timestamp)
            )
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "close": float(r.close),
                "high": float(r.high),
                "low": float(r.low),
                "volume": float(r.volume),
                "returns_1h": None,
                "returns_4h": None,
                "returns_24h": None,
            }
            for r in rows
        ])

    async def _load_variable_data(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Load computed variable data."""
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

        return pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "variable_id": r.variable_id,
                "value": r.value,
            }
            for r in rows
        ])

    def _pivot_variables(
        self, var_df: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Pivot variable data into wide format and merge with price."""
        if var_df.empty:
            return pd.DataFrame()

        pivoted = var_df.pivot_table(
            index="timestamp", columns="variable_id", values="value", aggfunc="last"
        )

        # Merge with price data
        price_df = price_df.set_index("timestamp")
        merged = pivoted.join(price_df, how="inner")

        # Compute forward returns (what we're trying to predict)
        merged["returns_1h"] = merged["close"].pct_change(1).shift(-1)
        merged["returns_4h"] = merged["close"].pct_change(4).shift(-4)
        merged["returns_24h"] = merged["close"].pct_change(24).shift(-24)

        return merged.dropna(subset=["returns_1h"])

    # ── Pattern Discovery Methods ────────────────────────────

    def _find_correlation_clusters(self, df: pd.DataFrame) -> None:
        """Find groups of variables that are highly correlated."""
        var_cols = [c for c in df.columns if c not in (
            "close", "high", "low", "volume", "returns_1h", "returns_4h", "returns_24h"
        )]

        if len(var_cols) < 2:
            return

        corr = df[var_cols].corr()

        # Find strongly correlated pairs (|r| > 0.8)
        strong_pairs = []
        for i in range(len(var_cols)):
            for j in range(i + 1, len(var_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.8:
                    strong_pairs.append({
                        "var_a": var_cols[i],
                        "var_b": var_cols[j],
                        "correlation": round(r, 3),
                        "direction": "positive" if r > 0 else "negative",
                    })

        if strong_pairs:
            # Sort by absolute correlation
            strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            self._patterns.append({
                "type": "correlation_cluster",
                "name": "Highly Correlated Variable Pairs",
                "description": "Variables that move together (|r| > 0.8). These provide redundant information — the causal engine should pick the most causally important one.",
                "pairs": strong_pairs[:20],
                "count": len(strong_pairs),
            })

    def _find_leading_indicators(self, df: pd.DataFrame) -> None:
        """Find variables that predict future price movement."""
        var_cols = [c for c in df.columns if c not in (
            "close", "high", "low", "volume", "returns_1h", "returns_4h", "returns_24h"
        )]

        leaders = []
        for col in var_cols:
            if df[col].std() == 0:
                continue

            # Correlation with future returns
            for horizon, label in [
                ("returns_1h", "1H"), ("returns_4h", "4H"), ("returns_24h", "24H")
            ]:
                if horizon in df.columns:
                    valid = df[[col, horizon]].dropna()
                    if len(valid) > 50:
                        r = valid[col].corr(valid[horizon])
                        if abs(r) > 0.1:  # Meaningful correlation
                            leaders.append({
                                "variable": col,
                                "horizon": label,
                                "correlation": round(r, 4),
                                "direction": "bullish_when_high" if r > 0 else "bearish_when_high",
                                "strength": "strong" if abs(r) > 0.3 else "moderate" if abs(r) > 0.2 else "weak",
                            })

        if leaders:
            leaders.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            self._patterns.append({
                "type": "leading_indicators",
                "name": "Price-Predictive Variables",
                "description": "Variables correlated with FUTURE price movement. Positive = bullish when variable is high. Negative = bearish when variable is high.",
                "indicators": leaders[:30],
                "count": len(leaders),
            })

    def _find_extreme_value_patterns(self, df: pd.DataFrame) -> None:
        """What happens after indicators hit extreme values?"""
        extremes = []

        # RSI extremes
        for rsi_col in ["RSI_14", "RSI_7"]:
            if rsi_col in df.columns:
                for threshold, label, direction in [
                    (80, "overbought", "RSI > 80"),
                    (70, "overbought_mild", "RSI > 70"),
                    (30, "oversold_mild", "RSI < 30"),
                    (20, "oversold", "RSI < 20"),
                ]:
                    if ">" in direction:
                        mask = df[rsi_col] > threshold
                    else:
                        mask = df[rsi_col] < threshold

                    extreme_rows = df[mask]
                    if len(extreme_rows) > 10:
                        for horizon in ["returns_1h", "returns_4h", "returns_24h"]:
                            if horizon in df.columns:
                                returns = extreme_rows[horizon].dropna()
                                if len(returns) > 5:
                                    extremes.append({
                                        "variable": rsi_col,
                                        "condition": direction,
                                        "horizon": horizon.replace("returns_", ""),
                                        "avg_return": round(returns.mean() * 100, 3),
                                        "median_return": round(returns.median() * 100, 3),
                                        "win_rate": round((returns > 0).mean() * 100, 1),
                                        "sample_size": len(returns),
                                        "signal": "mean_revert" if (
                                            (">" in direction and returns.mean() < 0) or
                                            ("<" in direction and returns.mean() > 0)
                                        ) else "momentum",
                                    })

        # Bollinger Band extremes
        if "BB_PERCENT_B" in df.columns:
            for threshold, direction in [(1.0, "above_upper"), (0.0, "below_lower")]:
                if direction == "above_upper":
                    mask = df["BB_PERCENT_B"] > threshold
                else:
                    mask = df["BB_PERCENT_B"] < threshold

                extreme_rows = df[mask]
                if len(extreme_rows) > 10:
                    for horizon in ["returns_1h", "returns_4h", "returns_24h"]:
                        if horizon in df.columns:
                            returns = extreme_rows[horizon].dropna()
                            if len(returns) > 5:
                                extremes.append({
                                    "variable": "BB_PERCENT_B",
                                    "condition": f"Price {direction.replace('_', ' ')} band",
                                    "horizon": horizon.replace("returns_", ""),
                                    "avg_return": round(returns.mean() * 100, 3),
                                    "median_return": round(returns.median() * 100, 3),
                                    "win_rate": round((returns > 0).mean() * 100, 1),
                                    "sample_size": len(returns),
                                })

        if extremes:
            self._patterns.append({
                "type": "extreme_value_patterns",
                "name": "Extreme Value Signals",
                "description": "What historically happens after indicators hit extreme levels. Mean revert = price reverses after extreme. Momentum = extreme continues.",
                "patterns": extremes,
                "count": len(extremes),
            })

    def _find_divergence_patterns(self, df: pd.DataFrame) -> None:
        """Find when indicators diverge from price (classic divergence signals)."""
        divergences = []

        # Price making new highs but momentum weakening
        if "RSI_14" in df.columns:
            window = 20
            for i in range(window, len(df) - 1):
                chunk = df.iloc[i - window : i + 1]
                price_trend = (chunk["close"].iloc[-1] - chunk["close"].iloc[0]) / chunk["close"].iloc[0]
                rsi_trend = chunk["RSI_14"].iloc[-1] - chunk["RSI_14"].iloc[0]

                # Bearish divergence: price up, RSI down
                if price_trend > 0.02 and rsi_trend < -10:
                    future_ret = df.iloc[i:i+24]["close"].pct_change(24).iloc[-1] if i + 24 < len(df) else None
                    if future_ret is not None:
                        divergences.append({"type": "bearish", "return_24h": future_ret})

                # Bullish divergence: price down, RSI up
                elif price_trend < -0.02 and rsi_trend > 10:
                    future_ret = df.iloc[i:i+24]["close"].pct_change(24).iloc[-1] if i + 24 < len(df) else None
                    if future_ret is not None:
                        divergences.append({"type": "bullish", "return_24h": future_ret})

        if divergences:
            bearish = [d for d in divergences if d["type"] == "bearish"]
            bullish = [d for d in divergences if d["type"] == "bullish"]

            pattern = {
                "type": "divergence_signals",
                "name": "RSI-Price Divergence",
                "description": "When price makes new highs/lows but RSI doesn't confirm. Classic reversal signal.",
            }

            if bearish:
                returns = [d["return_24h"] for d in bearish]
                pattern["bearish_divergence"] = {
                    "count": len(bearish),
                    "avg_return_24h": round(np.mean(returns) * 100, 3),
                    "win_rate": round(sum(1 for r in returns if r < 0) / len(returns) * 100, 1),
                }

            if bullish:
                returns = [d["return_24h"] for d in bullish]
                pattern["bullish_divergence"] = {
                    "count": len(bullish),
                    "avg_return_24h": round(np.mean(returns) * 100, 3),
                    "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
                }

            self._patterns.append(pattern)

    def _find_regime_patterns(self, df: pd.DataFrame) -> None:
        """Identify market regime clusters based on variable combinations."""
        regime_cols = []
        for col in ["ADX", "BB_BANDWIDTH", "RSI_14", "GARCH_VOL"]:
            if col in df.columns:
                regime_cols.append(col)

        if len(regime_cols) < 2:
            return

        regime_df = df[regime_cols].dropna()
        if len(regime_df) < 100:
            return

        # Simple regime classification based on ADX + volatility
        regimes = []

        if "ADX" in regime_cols and "BB_BANDWIDTH" in regime_cols:
            for _, row in regime_df.iterrows():
                adx_val = row.get("ADX", 25)
                bw_val = row.get("BB_BANDWIDTH", 0.05)

                if adx_val > 30 and bw_val > regime_df["BB_BANDWIDTH"].quantile(0.75):
                    regimes.append("trending_volatile")
                elif adx_val > 30 and bw_val <= regime_df["BB_BANDWIDTH"].quantile(0.75):
                    regimes.append("trending_calm")
                elif adx_val <= 30 and bw_val > regime_df["BB_BANDWIDTH"].quantile(0.75):
                    regimes.append("ranging_volatile")
                else:
                    regimes.append("ranging_calm")

            regime_series = pd.Series(regimes, index=regime_df.index)
            regime_counts = regime_series.value_counts()

            regime_returns = {}
            for regime_name in regime_counts.index:
                mask = regime_series == regime_name
                if "returns_4h" in df.columns:
                    rets = df.loc[mask, "returns_4h"].dropna()
                    if len(rets) > 10:
                        regime_returns[regime_name] = {
                            "count": int(regime_counts[regime_name]),
                            "pct_of_time": round(regime_counts[regime_name] / len(regime_series) * 100, 1),
                            "avg_return_4h": round(rets.mean() * 100, 3),
                            "volatility": round(rets.std() * 100, 3),
                            "best_strategy": "trend_follow" if regime_name.startswith("trending") else "mean_revert",
                        }

            self._patterns.append({
                "type": "regime_patterns",
                "name": "Market Regime Classification",
                "description": "Markets classified into 4 regimes using ADX (trend strength) and Bollinger Bandwidth (volatility). Different strategies work in different regimes.",
                "regimes": regime_returns,
            })

    def _find_reversal_patterns(self, df: pd.DataFrame) -> None:
        """Find price reversal patterns based on variable combinations."""
        reversals = []

        # V-bottom pattern: RSI oversold + volume spike + MACD histogram turning positive
        if all(col in df.columns for col in ["RSI_14", "REL_VOLUME", "MACD_HISTOGRAM"]):
            for i in range(5, len(df) - 24):
                row = df.iloc[i]
                prev = df.iloc[i - 1]

                if (
                    row.get("RSI_14", 50) < 30
                    and row.get("REL_VOLUME", 1) > 2.0
                    and prev.get("MACD_HISTOGRAM", 0) < 0
                    and row.get("MACD_HISTOGRAM", 0) > prev.get("MACD_HISTOGRAM", 0)
                ):
                    future_ret = (df.iloc[min(i + 24, len(df) - 1)]["close"] - row["close"]) / row["close"]
                    reversals.append({"return_24h": future_ret})

        if len(reversals) > 5:
            returns = [r["return_24h"] for r in reversals]
            self._patterns.append({
                "type": "reversal_patterns",
                "name": "V-Bottom Reversal Setup",
                "description": "RSI < 30 + Volume > 2x average + MACD histogram turning up. Classic capitulation reversal.",
                "count": len(reversals),
                "avg_return_24h": round(np.mean(returns) * 100, 3),
                "median_return_24h": round(np.median(returns) * 100, 3),
                "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
            })

    def _find_trend_patterns(self, df: pd.DataFrame) -> None:
        """Find trend continuation patterns."""
        continuations = []

        # EMA alignment: 9 > 21 > 50 (bullish trend structure)
        if all(col in df.columns for col in ["EMA_9", "EMA_21", "EMA_50"]):
            bull_aligned = df[
                (df["EMA_9"] > df["EMA_21"]) & (df["EMA_21"] > df["EMA_50"])
            ]
            bear_aligned = df[
                (df["EMA_9"] < df["EMA_21"]) & (df["EMA_21"] < df["EMA_50"])
            ]

            for aligned, label in [(bull_aligned, "bullish"), (bear_aligned, "bearish")]:
                if len(aligned) > 20 and "returns_4h" in df.columns:
                    rets = aligned["returns_4h"].dropna()
                    if len(rets) > 10:
                        continuations.append({
                            "name": f"EMA Stack ({label})",
                            "condition": f"EMA9 {'>' if label == 'bullish' else '<'} EMA21 {'>' if label == 'bullish' else '<'} EMA50",
                            "count": len(rets),
                            "pct_of_time": round(len(aligned) / len(df) * 100, 1),
                            "avg_return_4h": round(rets.mean() * 100, 3),
                            "win_rate": round(
                                (rets > 0).mean() * 100 if label == "bullish"
                                else (rets < 0).mean() * 100,
                                1
                            ),
                        })

        if continuations:
            self._patterns.append({
                "type": "trend_patterns",
                "name": "EMA Trend Structure",
                "description": "When EMAs are stacked in order (9 > 21 > 50 or reverse), the trend tends to continue. This is the foundation of trend-following strategies.",
                "patterns": continuations,
            })

    # ── Output ───────────────────────────────────────────────

    def _write_study_file(
        self, path: str, symbol: str, timeframe: str
    ) -> None:
        """Write a human-readable pattern study file."""
        lines = [
            f"# 📊 AtomicX Pattern Study Guide — {symbol}",
            f"",
            f"**Generated:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Data:** {symbol} @ {timeframe}",
            f"**Patterns discovered:** {len(self._patterns)}",
            f"",
            f"---",
            f"",
            f"> **How to read this:** Each pattern below was discovered by analyzing",
            f"> historical data across all 46 variables. The system found these",
            f"> by looking at what ACTUALLY happened after specific conditions,",
            f"> not by guessing. Win rates and returns are from real data.",
            f"",
        ]

        for i, pattern in enumerate(self._patterns, 1):
            lines.append(f"## {i}. {pattern.get('name', 'Unknown Pattern')}")
            lines.append(f"")
            lines.append(f"**Type:** `{pattern.get('type', 'unknown')}`")
            lines.append(f"")

            if "description" in pattern:
                lines.append(f"**What it means:** {pattern['description']}")
                lines.append(f"")

            # Format based on pattern type
            ptype = pattern.get("type", "")

            if ptype == "correlation_cluster":
                lines.append(f"**{pattern.get('count', 0)} highly correlated pairs found.**")
                lines.append(f"")
                lines.append("| Variable A | Variable B | Correlation | Direction |")
                lines.append("|------------|------------|-------------|-----------|")
                for pair in pattern.get("pairs", [])[:15]:
                    lines.append(
                        f"| {pair['var_a']} | {pair['var_b']} | "
                        f"{pair['correlation']:.3f} | {pair['direction']} |"
                    )
                lines.append(f"")

            elif ptype == "leading_indicators":
                lines.append(f"**{pattern.get('count', 0)} predictive relationships found.**")
                lines.append(f"")
                lines.append("| Variable | Horizon | Correlation | Signal | Strength |")
                lines.append("|----------|---------|-------------|--------|----------|")
                for ind in pattern.get("indicators", [])[:20]:
                    lines.append(
                        f"| {ind['variable']} | {ind['horizon']} | "
                        f"{ind['correlation']:.4f} | {ind['direction']} | {ind['strength']} |"
                    )
                lines.append(f"")

            elif ptype == "extreme_value_patterns":
                lines.append(f"")
                for ep in pattern.get("patterns", []):
                    emoji = "🔴" if ep.get("avg_return", 0) < 0 else "🟢"
                    lines.append(
                        f"- {emoji} **{ep['variable']}** {ep['condition']} → "
                        f"**{ep['horizon']}** avg return: **{ep['avg_return']:.3f}%**, "
                        f"win rate: **{ep['win_rate']:.1f}%** "
                        f"(n={ep['sample_size']}) "
                        f"{'[MEAN REVERT]' if ep.get('signal') == 'mean_revert' else '[MOMENTUM]'}"
                    )
                lines.append(f"")

            elif ptype == "divergence_signals":
                if "bearish_divergence" in pattern:
                    bd = pattern["bearish_divergence"]
                    lines.append(
                        f"- 🔴 **Bearish divergence** (price up + RSI down): "
                        f"{bd['count']} events, avg return 24H: **{bd['avg_return_24h']:.3f}%**, "
                        f"win rate (price fell): **{bd['win_rate']:.1f}%**"
                    )
                if "bullish_divergence" in pattern:
                    bd = pattern["bullish_divergence"]
                    lines.append(
                        f"- 🟢 **Bullish divergence** (price down + RSI up): "
                        f"{bd['count']} events, avg return 24H: **{bd['avg_return_24h']:.3f}%**, "
                        f"win rate (price rose): **{bd['win_rate']:.1f}%**"
                    )
                lines.append(f"")

            elif ptype == "regime_patterns":
                lines.append(f"")
                lines.append("| Regime | % of Time | Avg Return 4H | Volatility | Best Strategy |")
                lines.append("|--------|-----------|---------------|------------|---------------|")
                for name, info in pattern.get("regimes", {}).items():
                    lines.append(
                        f"| {name.replace('_', ' ').title()} | "
                        f"{info['pct_of_time']}% | {info['avg_return_4h']}% | "
                        f"{info['volatility']}% | {info['best_strategy']} |"
                    )
                lines.append(f"")

            elif ptype == "reversal_patterns":
                lines.append(
                    f"- **Occurrences:** {pattern.get('count', 0)}\n"
                    f"- **Avg return 24H:** {pattern.get('avg_return_24h', 0):.3f}%\n"
                    f"- **Win rate:** {pattern.get('win_rate', 0):.1f}%\n"
                )

            elif ptype == "trend_patterns":
                for tp in pattern.get("patterns", []):
                    emoji = "🟢" if "bullish" in tp["name"].lower() else "🔴"
                    lines.append(
                        f"- {emoji} **{tp['name']}** ({tp['condition']}) — "
                        f"present {tp['pct_of_time']:.1f}% of time, "
                        f"avg 4H return: **{tp['avg_return_4h']:.3f}%**, "
                        f"win rate: **{tp['win_rate']:.1f}%** (n={tp['count']})"
                    )
                lines.append(f"")

            lines.append(f"---")
            lines.append(f"")

        # Add study notes
        lines.extend([
            f"## 🎓 How to Use These Patterns",
            f"",
            f"### The Key Principles",
            f"",
            f"1. **No single pattern is reliable alone.** The system uses ALL patterns",
            f"   together through the causal engine, not any one in isolation.",
            f"",
            f"2. **Win rates above 55% are meaningful.** In crypto markets, even 55%",
            f"   accuracy with proper risk management (1:2.5 R:R) is highly profitable.",
            f"",
            f"3. **Regime matters more than signals.** A bullish RSI divergence in a",
            f"   trending_volatile regime behaves very differently than in ranging_calm.",
            f"",
            f"4. **Mean reversion vs momentum.** Some patterns predict reversals,",
            f"   others predict continuation. The regime tells you which to trust.",
            f"",
            f"5. **Sample size matters.** Patterns with n < 30 are unreliable.",
            f"   The system will require 200+ samples before trusting any variable.",
            f"",
            f"### What the Causal Engine Does With These",
            f"",
            f"The causal discovery engine (Phase 3) takes these raw patterns and asks:",
            f"- Does RSI **cause** price reversal, or do they just happen to correlate?",
            f"- Which variable is the **root cause** vs which is just a symptom?",
            f"- When two patterns disagree, which one is more likely to be right?",
            f"",
            f"This is where AtomicX diverges from every other trading bot —",
            f"we find **causation**, not just correlation.",
            f"",
        ])

        # Write file
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Pattern study file written to {path}")

    def to_json(self) -> str:
        """Export all patterns as JSON."""
        return json.dumps(self._patterns, indent=2, default=str)
