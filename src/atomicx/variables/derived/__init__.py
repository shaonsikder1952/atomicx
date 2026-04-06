"""Advanced derived variable calculators (VAR-02).

These are higher-order variables derived from market microstructure
rather than simple price/volume indicators. They capture structural
signals that standard indicators miss:

- Funding rate term structure (perp vs spot pricing tension)
- Order-book imbalance (buy/sell pressure asymmetry)
- Liquidation pressure estimates
- Open interest change rate
- Basis spread (spot vs futures divergence)
- Whale flow indicators
- Volume-weighted bid/ask depth ratio
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


# ── Funding Rate Variables ────────────────────────────────────


def funding_rate_zscore(
    funding_rates: pd.Series, lookback: int = 168
) -> pd.Series:
    """Z-score of funding rate vs rolling history.

    High positive = market extremely overleveraged long → bearish signal
    High negative = market extremely overleveraged short → bullish signal
    """
    mean = funding_rates.rolling(window=lookback).mean()
    std = funding_rates.rolling(window=lookback).std()
    return (funding_rates - mean) / std.replace(0, np.nan)


def funding_rate_momentum(funding_rates: pd.Series, period: int = 24) -> pd.Series:
    """Rate of change in funding rate — detects acceleration of leverage."""
    return funding_rates.diff(period)


def funding_rate_term_structure(
    spot_price: pd.Series,
    futures_price: pd.Series,
    funding_rate: pd.Series,
) -> dict[str, pd.Series]:
    """Funding rate term structure analysis.

    Compares current funding vs implied funding from spot-futures basis.
    Divergence signals forced positioning or arbitrage pressure.
    """
    implied_rate = (futures_price - spot_price) / spot_price
    divergence = funding_rate - implied_rate
    return {
        "implied_funding": implied_rate,
        "funding_divergence": divergence,
        "divergence_zscore": funding_rate_zscore(divergence),
    }


# ── Order Book Variables ──────────────────────────────────────


def orderbook_imbalance(
    bids: list[list[float]], asks: list[list[float]], depth_levels: int = 10
) -> dict[str, float]:
    """Order book imbalance — measures buy/sell pressure asymmetry.

    Returns multiple granularities of imbalance.
    """
    bid_volume = sum(qty for _, qty in bids[:depth_levels])
    ask_volume = sum(qty for _, qty in asks[:depth_levels])
    total_volume = bid_volume + ask_volume

    if total_volume == 0:
        return {"imbalance": 0.0, "bid_ratio": 0.5, "pressure": "neutral"}

    imbalance = (bid_volume - ask_volume) / total_volume  # -1 to +1
    bid_ratio = bid_volume / total_volume

    if imbalance > 0.3:
        pressure = "strong_buy"
    elif imbalance > 0.1:
        pressure = "mild_buy"
    elif imbalance < -0.3:
        pressure = "strong_sell"
    elif imbalance < -0.1:
        pressure = "mild_sell"
    else:
        pressure = "neutral"

    return {
        "imbalance": imbalance,
        "bid_ratio": bid_ratio,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "pressure": pressure,
    }


def orderbook_depth_ratio(
    bids: list[list[float]], asks: list[list[float]], pct_levels: list[float] | None = None
) -> dict[str, float]:
    """Depth ratio at various price levels from mid-price.

    Measures the ratio of bid volume to ask volume at 0.5%, 1%, 2% from mid.
    """
    if not bids or not asks:
        return {}

    mid_price = (bids[0][0] + asks[0][0]) / 2
    levels = pct_levels or [0.005, 0.01, 0.02, 0.05]
    result = {}

    for pct in levels:
        bid_vol = sum(qty for price, qty in bids if price >= mid_price * (1 - pct))
        ask_vol = sum(qty for price, qty in asks if price <= mid_price * (1 + pct))
        total = bid_vol + ask_vol
        result[f"depth_ratio_{pct:.1%}"] = bid_vol / total if total > 0 else 0.5

    return result


def whale_wall_detector(
    bids: list[list[float]],
    asks: list[list[float]],
    threshold_multiplier: float = 5.0,
) -> dict[str, Any]:
    """Detect whale walls — abnormally large orders in the book.

    A wall is defined as an order > threshold_multiplier × median order size.
    """
    all_quantities = [qty for _, qty in bids + asks]
    if not all_quantities:
        return {"bid_walls": [], "ask_walls": []}

    median_qty = float(np.median(all_quantities))
    threshold = median_qty * threshold_multiplier

    bid_walls = [
        {"price": price, "quantity": qty, "size_ratio": qty / median_qty}
        for price, qty in bids
        if qty >= threshold
    ]
    ask_walls = [
        {"price": price, "quantity": qty, "size_ratio": qty / median_qty}
        for price, qty in asks
        if qty >= threshold
    ]

    return {"bid_walls": bid_walls, "ask_walls": ask_walls, "wall_threshold": threshold}


# ── Market Microstructure Variables ───────────────────────────


def spread_analysis(
    bids: list[list[float]], asks: list[list[float]]
) -> dict[str, float]:
    """Spread and liquidity analysis."""
    if not bids or not asks:
        return {}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2

    return {
        "spread_absolute": best_ask - best_bid,
        "spread_bps": ((best_ask - best_bid) / mid) * 10000 if mid > 0 else 0,
        "mid_price": mid,
    }


def trade_flow_imbalance(
    trades: pd.DataFrame, window: int = 50
) -> pd.Series:
    """Net trade flow — volume-weighted buy vs sell pressure.

    Positive = aggressive buying dominating.
    Negative = aggressive selling dominating.
    """
    signed_volume = trades["quantity"].copy()
    signed_volume[trades["is_buyer_maker"]] *= -1  # Seller-initiated
    return signed_volume.rolling(window=window).sum()


def price_impact_estimate(
    bids: list[list[float]], asks: list[list[float]], order_size: float
) -> dict[str, float]:
    """Estimate price impact for a given order size.

    Walks the order book to determine fill price at various sizes.
    Critical for the Tradability Agent (Phase 10).
    """
    def walk_book(levels: list[list[float]], size: float) -> float:
        filled = 0.0
        cost = 0.0
        for price, qty in levels:
            fill_qty = min(qty, size - filled)
            cost += fill_qty * price
            filled += fill_qty
            if filled >= size:
                break
        return cost / filled if filled > 0 else 0.0

    avg_buy_price = walk_book(asks, order_size)
    avg_sell_price = walk_book(bids, order_size)
    mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0

    return {
        "buy_impact_bps": ((avg_buy_price - mid) / mid) * 10000 if mid > 0 else 0,
        "sell_impact_bps": ((mid - avg_sell_price) / mid) * 10000 if mid > 0 else 0,
        "avg_buy_fill": avg_buy_price,
        "avg_sell_fill": avg_sell_price,
    }


# ── Volatility Regime Variables ───────────────────────────────


def realized_vs_implied_vol(
    returns: pd.Series,
    short_window: int = 24,
    long_window: int = 168,
) -> dict[str, pd.Series]:
    """Realized volatility at multiple horizons.

    Divergence between short and long-term vol signals regime transitions.
    """
    short_vol = returns.rolling(window=short_window).std() * np.sqrt(365 * 24)
    long_vol = returns.rolling(window=long_window).std() * np.sqrt(365 * 24)
    vol_ratio = short_vol / long_vol.replace(0, np.nan)

    return {
        "vol_short": short_vol,
        "vol_long": long_vol,
        "vol_ratio": vol_ratio,  # >1 = vol expanding, <1 = vol contracting
    }


def garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass volatility — more efficient than close-to-close."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window=window).mean().apply(np.sqrt) * np.sqrt(252)
