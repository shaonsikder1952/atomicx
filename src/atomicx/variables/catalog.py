"""Default variable catalog — all variables that ship with AtomicX.

Defines the initial ~50+ variables across all 7 domains.
New variables can be added here or via YAML config (VAR-05).
"""

from __future__ import annotations

from atomicx.variables.types import VariableDefinition, VariableDomain, VariableTimeframe


def get_default_variables() -> list[VariableDefinition]:
    """Return the full catalog of default variables."""
    return [
        *_economic_trend_variables(),
        *_economic_momentum_variables(),
        *_economic_volatility_variables(),
        *_economic_volume_variables(),
        *_physical_microstructure_variables(),
        *_behavioral_funding_variables(),
        *_social_sentiment_variables(),
        *_temporal_variables(),
    ]


# ── Economic Domain: Trend ────────────────────────────────────


def _economic_trend_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="EMA_9", name="EMA (9-period)", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", params={"period": 9},
            causal_half_life=6.0, lookback_periods=9,
        ),
        VariableDefinition(
            id="EMA_21", name="EMA (21-period)", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", params={"period": 21},
            causal_half_life=12.0, lookback_periods=21,
        ),
        VariableDefinition(
            id="EMA_50", name="EMA (50-period)", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", params={"period": 50},
            causal_half_life=48.0, lookback_periods=50,
        ),
        VariableDefinition(
            id="EMA_200", name="EMA (200-period)", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", params={"period": 200},
            causal_half_life=168.0, lookback_periods=200,
        ),
        VariableDefinition(
            id="SMA_20", name="SMA (20-period)", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", params={"period": 20},
            causal_half_life=24.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="SMA_50", name="SMA (50-period)", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", params={"period": 50},
            causal_half_life=48.0, lookback_periods=50,
        ),
        VariableDefinition(
            id="VWAP", name="Volume Weighted Average Price", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", causal_half_life=8.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="DAILY_CHANGE_PERCENT", name="Daily Change %", domain=VariableDomain.ECONOMIC,
            category="trend", source="binance", causal_half_life=1.0, lookback_periods=2,
        ),
    ]


# ── Economic Domain: Momentum ──────────────────────────────


def _economic_momentum_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="RSI_14", name="RSI (14-period)", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", params={"period": 14},
            causal_half_life=12.0, lookback_periods=14,
        ),
        VariableDefinition(
            id="RSI_7", name="RSI (7-period)", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", params={"period": 7},
            causal_half_life=6.0, lookback_periods=7,
        ),
        VariableDefinition(
            id="MACD_LINE", name="MACD Line", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance",
            params={"fast": 12, "slow": 26, "signal": 9},
            causal_half_life=24.0, lookback_periods=26,
        ),
        VariableDefinition(
            id="MACD_SIGNAL", name="MACD Signal", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", causal_half_life=24.0, lookback_periods=35,
        ),
        VariableDefinition(
            id="MACD_HISTOGRAM", name="MACD Histogram", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", causal_half_life=12.0, lookback_periods=35,
        ),
        VariableDefinition(
            id="STOCH_RSI_K", name="Stochastic RSI %K", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", params={"rsi_period": 14, "stoch_period": 14},
            causal_half_life=8.0, lookback_periods=28,
        ),
        VariableDefinition(
            id="STOCH_RSI_D", name="Stochastic RSI %D", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", causal_half_life=8.0, lookback_periods=31,
        ),
        VariableDefinition(
            id="ROC_12", name="Rate of Change (12-period)", domain=VariableDomain.ECONOMIC,
            category="momentum", source="binance", params={"period": 12},
            causal_half_life=12.0, lookback_periods=12,
        ),
    ]


# ── Economic Domain: Volatility ──────────────────────────────


def _economic_volatility_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="BB_UPPER", name="Bollinger Band Upper", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance",
            params={"period": 20, "std_dev": 2.0},
            causal_half_life=24.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="BB_LOWER", name="Bollinger Band Lower", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance", causal_half_life=24.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="BB_PERCENT_B", name="Bollinger %B", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance", causal_half_life=12.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="BB_BANDWIDTH", name="Bollinger Bandwidth", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance", causal_half_life=24.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="ATR_14", name="Average True Range (14-period)", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance", params={"period": 14},
            causal_half_life=24.0, lookback_periods=14,
        ),
        VariableDefinition(
            id="GARCH_VOL", name="GARCH Volatility", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance", params={"window": 20},
            causal_half_life=48.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="GK_VOL", name="Garman-Klass Volatility", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance", params={"window": 20},
            causal_half_life=48.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="VOL_RATIO", name="Short/Long Volatility Ratio", domain=VariableDomain.ECONOMIC,
            category="volatility", source="binance",
            params={"short_window": 24, "long_window": 168},
            causal_half_life=48.0, lookback_periods=168,
        ),
    ]


# ── Economic Domain: Volume ──────────────────────────────────


def _economic_volume_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="OBV", name="On-Balance Volume", domain=VariableDomain.ECONOMIC,
            category="volume", source="binance", causal_half_life=24.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="REL_VOLUME", name="Relative Volume (20-period)", domain=VariableDomain.ECONOMIC,
            category="volume", source="binance", params={"period": 20},
            causal_half_life=8.0, lookback_periods=20,
        ),
        VariableDefinition(
            id="VPT", name="Volume Price Trend", domain=VariableDomain.ECONOMIC,
            category="volume", source="binance", causal_half_life=24.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="ADX", name="Average Directional Index", domain=VariableDomain.ECONOMIC,
            category="trend_strength", source="binance", params={"period": 14},
            causal_half_life=24.0, lookback_periods=28,
        ),
        VariableDefinition(
            id="PLUS_DI", name="Positive DI", domain=VariableDomain.ECONOMIC,
            category="trend_strength", source="binance", causal_half_life=24.0, lookback_periods=28,
        ),
        VariableDefinition(
            id="MINUS_DI", name="Negative DI", domain=VariableDomain.ECONOMIC,
            category="trend_strength", source="binance", causal_half_life=24.0, lookback_periods=28,
        ),
    ]


# ── Physical Domain: Market Microstructure ────────────────────


def _physical_microstructure_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="OB_IMBALANCE", name="Order Book Imbalance", domain=VariableDomain.PHYSICAL,
            category="microstructure", source="binance",
            update_frequency=VariableTimeframe.TICK,
            causal_half_life=1.0, lookback_periods=1,
            description="Buy/sell volume ratio in top 10 levels of order book",
        ),
        VariableDefinition(
            id="OB_DEPTH_05", name="Depth Ratio 0.5%", domain=VariableDomain.PHYSICAL,
            category="microstructure", source="binance",
            update_frequency=VariableTimeframe.TICK,
            causal_half_life=1.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="OB_DEPTH_1", name="Depth Ratio 1%", domain=VariableDomain.PHYSICAL,
            category="microstructure", source="binance",
            update_frequency=VariableTimeframe.TICK,
            causal_half_life=2.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="SPREAD_BPS", name="Bid-Ask Spread (bps)", domain=VariableDomain.PHYSICAL,
            category="microstructure", source="binance",
            update_frequency=VariableTimeframe.TICK,
            causal_half_life=0.5, lookback_periods=1,
        ),
        VariableDefinition(
            id="TRADE_FLOW", name="Trade Flow Imbalance", domain=VariableDomain.PHYSICAL,
            category="microstructure", source="binance",
            update_frequency=VariableTimeframe.TICK,
            causal_half_life=2.0, lookback_periods=50,
        ),
        VariableDefinition(
            id="WHALE_WALLS", name="Whale Wall Count", domain=VariableDomain.PHYSICAL,
            category="microstructure", source="binance",
            update_frequency=VariableTimeframe.M5,
            causal_half_life=4.0, lookback_periods=1,
        ),
    ]


# ── Behavioral Domain: Funding & Leverage ─────────────────────


def _behavioral_funding_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="FUNDING_RATE", name="Funding Rate", domain=VariableDomain.BEHAVIORAL,
            category="leverage", source="binance",
            update_frequency=VariableTimeframe.H1,
            causal_half_life=8.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="FUNDING_ZSCORE", name="Funding Rate Z-Score", domain=VariableDomain.BEHAVIORAL,
            category="leverage", source="binance",
            update_frequency=VariableTimeframe.H1,
            params={"lookback": 168},
            causal_half_life=24.0, lookback_periods=168,
        ),
        VariableDefinition(
            id="FUNDING_MOMENTUM", name="Funding Rate Momentum", domain=VariableDomain.BEHAVIORAL,
            category="leverage", source="binance",
            update_frequency=VariableTimeframe.H1,
            causal_half_life=12.0, lookback_periods=24,
        ),
        VariableDefinition(
            id="FUNDING_DIVERGENCE", name="Funding Rate Divergence", domain=VariableDomain.BEHAVIORAL,
            category="leverage", source="binance",
            update_frequency=VariableTimeframe.H1,
            causal_half_life=24.0, lookback_periods=168,
        ),
    ]


# ── Social Domain: Sentiment ─────────────────────────────────


def _social_sentiment_variables() -> list[VariableDefinition]:
    """Social variables — placeholders for Phase 8 (Narrative Tracker)."""
    return [
        VariableDefinition(
            id="MARKET_CAP_RANK", name="Market Cap Rank", domain=VariableDomain.SOCIAL,
            category="market_overview", source="coingecko",
            update_frequency=VariableTimeframe.H1,
            causal_half_life=168.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="VOLUME_24H_CHANGE", name="24H Volume Change %", domain=VariableDomain.SOCIAL,
            category="market_overview", source="coingecko",
            update_frequency=VariableTimeframe.H1,
            causal_half_life=24.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="PRICE_CHANGE_1H", name="Price Change 1H %", domain=VariableDomain.SOCIAL,
            category="market_overview", source="coingecko",
            update_frequency=VariableTimeframe.H1,
            causal_half_life=2.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="BTC_DOMINANCE", name="BTC Dominance %", domain=VariableDomain.SOCIAL,
            category="market_overview", source="coingecko", symbol_specific=False,
            update_frequency=VariableTimeframe.H1,
            causal_half_life=168.0, lookback_periods=1,
        ),
    ]


# ── Temporal Domain: Cycles ──────────────────────────────────


def _temporal_variables() -> list[VariableDefinition]:
    return [
        VariableDefinition(
            id="HOUR_OF_DAY", name="Hour of Day (UTC)", domain=VariableDomain.TEMPORAL,
            category="time_cycle", source="system", symbol_specific=False,
            update_frequency=VariableTimeframe.H1,
            causal_half_life=1.0, lookback_periods=1,
            description="Cyclical encoding of UTC hour (0-23)",
        ),
        VariableDefinition(
            id="DAY_OF_WEEK", name="Day of Week", domain=VariableDomain.TEMPORAL,
            category="time_cycle", source="system", symbol_specific=False,
            update_frequency=VariableTimeframe.D1,
            causal_half_life=24.0, lookback_periods=1,
        ),
        VariableDefinition(
            id="DAYS_FROM_ATH", name="Days Since All-Time High", domain=VariableDomain.TEMPORAL,
            category="market_cycle", source="coingecko",
            update_frequency=VariableTimeframe.D1,
            causal_half_life=720.0, lookback_periods=1,
        ),
    ]
