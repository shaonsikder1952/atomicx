"""Order Book Reconstruction and Microstructure Features.

Full L2 order book reconstruction from WebSocket updates with advanced
microstructure feature extraction:
- Support/resistance wall detection
- Spoofing detection (fake walls)
- Iceberg order inference
- Toxicity metrics (Kyle's lambda)
- Market making behavior analysis
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    volume: float
    orders: int = 1  # Number of orders at this level


@dataclass
class MicrostructureFeatures:
    """Extracted microstructure features from order book."""
    # Basic features
    bid_wall_strength: float  # Max bid depth / avg bid depth
    ask_wall_strength: float  # Max ask depth / avg ask depth
    spread_bps: float  # Spread in basis points
    ob_imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)

    # Advanced features
    toxicity: float  # Kyle's lambda (informed trading pressure)
    spoofing_score: float  # Probability of fake walls (0-1)
    iceberg_score: float  # Probability of hidden orders (0-1)
    pressure_score: float  # Net buying/selling pressure

    # Wall detection
    bid_walls: List[Tuple[float, float]]  # [(price, volume)]
    ask_walls: List[Tuple[float, float]]

    # Microstructure metadata
    bid_depth_total: float
    ask_depth_total: float
    mid_price: float
    timestamp: Optional[float] = None


class OrderBookReconstructor:
    """Reconstructs full L2 order book from WebSocket updates.

    Maintains up to 100 levels on each side with automatic cleanup.

    Usage:
        ob = OrderBookReconstructor(symbol="BTC/USDT", depth=100)

        # Process L2 updates
        ob.process_update({
            'bids': [[52000, 15.5], [51950, 80.2], ...],
            'asks': [[52050, 20.1], [52100, 50.0], ...],
        })

        # Extract features
        features = ob.get_microstructure_features()

        if features.spoofing_score > 0.7:
            logger.warning("Possible spoofing detected!")

        if features.bid_wall_strength > 3.0:
            logger.info("Strong bid wall at support")
    """

    def __init__(
        self,
        symbol: str,
        depth: int = 100,
        history_window: int = 1000,
    ):
        self.symbol = symbol
        self.depth = depth

        # Order book state
        self.bids: Dict[float, OrderBookLevel] = {}  # price -> level
        self.asks: Dict[float, OrderBookLevel] = {}

        # History for analysis
        self.spread_history = deque(maxlen=history_window)
        self.imbalance_history = deque(maxlen=history_window)
        self.volume_history = deque(maxlen=history_window)
        self.mid_price_history = deque(maxlen=history_window)

        # Spoofing detection
        self.wall_history: Dict[float, List[float]] = {}  # price -> [volumes over time]

        self.last_update_time = 0.0
        self.update_count = 0

        logger.info(f"[ORDERBOOK] Initialized for {symbol} (depth={depth})")

    def process_update(self, update: Dict[str, List[List[float]]]):
        """Process L2 order book update.

        Args:
            update: {'bids': [[price, volume], ...], 'asks': [[price, volume], ...]}
        """
        self.update_count += 1

        # Update bids
        for price, volume in update.get('bids', []):
            if volume == 0:
                # Remove level
                self.bids.pop(price, None)
            else:
                # Update level
                self.bids[price] = OrderBookLevel(price, volume)

        # Update asks
        for price, volume in update.get('asks', []):
            if volume == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = OrderBookLevel(price, volume)

        # Trim to depth
        self._trim_to_depth()

        # Update histories
        self._update_histories()

        if self.update_count % 1000 == 0:
            logger.debug(f"[ORDERBOOK] Processed {self.update_count} updates for {self.symbol}")

    def _trim_to_depth(self):
        """Keep only top N levels on each side."""
        if len(self.bids) > self.depth:
            # Keep highest bids
            top_bids = sorted(self.bids.keys(), reverse=True)[:self.depth]
            self.bids = {p: self.bids[p] for p in top_bids}

        if len(self.asks) > self.depth:
            # Keep lowest asks
            top_asks = sorted(self.asks.keys())[:self.depth]
            self.asks = {p: self.asks[p] for p in top_asks}

    def _update_histories(self):
        """Update historical data for feature calculation."""
        if not self.bids or not self.asks:
            return

        # Spread
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        spread = best_ask - best_bid
        self.spread_history.append(spread)

        # Mid price
        mid_price = (best_bid + best_ask) / 2
        self.mid_price_history.append(mid_price)

        # Imbalance
        bid_vol = sum(level.volume for level in self.bids.values())
        ask_vol = sum(level.volume for level in self.asks.values())
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        self.imbalance_history.append(imbalance)

        # Total volume
        total_vol = bid_vol + ask_vol
        self.volume_history.append(total_vol)

    def get_microstructure_features(self) -> MicrostructureFeatures:
        """Extract comprehensive microstructure features."""
        if not self.bids or not self.asks:
            return self._empty_features()

        # Basic features
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000

        # Order book depth
        bid_volumes = [level.volume for level in self.bids.values()]
        ask_volumes = [level.volume for level in self.asks.values()]

        bid_depth_total = sum(bid_volumes)
        ask_depth_total = sum(ask_volumes)

        # Imbalance
        ob_imbalance = (bid_depth_total - ask_depth_total) / (bid_depth_total + ask_depth_total + 1e-10)

        # Wall detection
        bid_walls = self._detect_walls(self.bids, side="bid")
        ask_walls = self._detect_walls(self.asks, side="ask")

        avg_bid_volume = np.mean(bid_volumes) if bid_volumes else 1.0
        avg_ask_volume = np.mean(ask_volumes) if ask_volumes else 1.0

        bid_wall_strength = max(bid_volumes) / avg_bid_volume if bid_volumes else 1.0
        ask_wall_strength = max(ask_volumes) / avg_ask_volume if ask_volumes else 1.0

        # Advanced features
        toxicity = self._calculate_toxicity()
        spoofing_score = self._calculate_spoofing_score(bid_walls, ask_walls)
        iceberg_score = self._calculate_iceberg_score()
        pressure_score = self._calculate_pressure_score()

        features = MicrostructureFeatures(
            bid_wall_strength=float(bid_wall_strength),
            ask_wall_strength=float(ask_wall_strength),
            spread_bps=float(spread_bps),
            ob_imbalance=float(ob_imbalance),
            toxicity=float(toxicity),
            spoofing_score=float(spoofing_score),
            iceberg_score=float(iceberg_score),
            pressure_score=float(pressure_score),
            bid_walls=bid_walls,
            ask_walls=ask_walls,
            bid_depth_total=float(bid_depth_total),
            ask_depth_total=float(ask_depth_total),
            mid_price=float(mid_price),
        )

        return features

    def _detect_walls(
        self, levels: Dict[float, OrderBookLevel], side: str, threshold_multiplier: float = 2.5
    ) -> List[Tuple[float, float]]:
        """Detect support/resistance walls in order book.

        A wall is a price level with significantly higher volume than average.

        Args:
            levels: Order book levels
            side: "bid" or "ask"
            threshold_multiplier: How many times avg volume to qualify as wall

        Returns:
            List of (price, volume) tuples for detected walls
        """
        if not levels:
            return []

        volumes = [level.volume for level in levels.values()]
        avg_volume = np.mean(volumes)
        threshold = avg_volume * threshold_multiplier

        walls = []
        for price, level in levels.items():
            if level.volume >= threshold:
                walls.append((price, level.volume))

        # Sort walls by volume (strongest first)
        walls.sort(key=lambda x: x[1], reverse=True)

        return walls[:5]  # Return top 5 walls

    def _calculate_toxicity(self) -> float:
        """Calculate Kyle's lambda (market toxicity).

        Measures informed trading pressure. High toxicity = informed traders.
        Formula: lambda = dP / dV (price impact per unit volume)
        """
        if len(self.mid_price_history) < 10 or len(self.volume_history) < 10:
            return 0.5  # Default neutral

        # Calculate price changes and volume changes
        price_changes = np.diff(list(self.mid_price_history)[-20:])
        volume_changes = np.diff(list(self.volume_history)[-20:])

        # Avoid division by zero
        volume_changes = np.where(volume_changes == 0, 1e-10, volume_changes)

        # Kyle's lambda = covariance(dP, dV) / variance(dV)
        if volume_changes.std() > 0:
            lambda_kyle = np.cov(price_changes, volume_changes)[0, 1] / np.var(volume_changes)
            toxicity = min(abs(lambda_kyle) * 1000, 1.0)  # Normalize to 0-1
        else:
            toxicity = 0.5

        return float(toxicity)

    def _calculate_spoofing_score(
        self, bid_walls: List[Tuple[float, float]], ask_walls: List[Tuple[float, float]]
    ) -> float:
        """Calculate probability of spoofing (fake walls).

        Spoofing indicators:
        - Large walls that appear/disappear frequently
        - Walls far from best bid/ask (not executing)
        - Asymmetric walls (one side only)
        """
        if not bid_walls and not ask_walls:
            return 0.0

        spoofing_indicators = 0.0
        num_indicators = 0

        # Indicator 1: Wall volatility (walls that change frequently)
        for price, volume in bid_walls + ask_walls:
            if price in self.wall_history:
                wall_vols = self.wall_history[price]
                if len(wall_vols) >= 5:
                    volatility = np.std(wall_vols) / np.mean(wall_vols) if np.mean(wall_vols) > 0 else 0
                    if volatility > 0.5:  # High volatility = suspicious
                        spoofing_indicators += 1
                num_indicators += 1
            else:
                self.wall_history[price] = [volume]

        # Indicator 2: Walls far from best price
        if self.bids and self.asks:
            best_bid = max(self.bids.keys())
            best_ask = min(self.asks.keys())

            for price, _ in bid_walls:
                distance = (best_bid - price) / best_bid
                if distance > 0.01:  # More than 1% away
                    spoofing_indicators += 0.5
                num_indicators += 1

            for price, _ in ask_walls:
                distance = (price - best_ask) / best_ask
                if distance > 0.01:
                    spoofing_indicators += 0.5
                num_indicators += 1

        # Indicator 3: Asymmetric walls
        bid_wall_total = sum(v for _, v in bid_walls)
        ask_wall_total = sum(v for _, v in ask_walls)
        if bid_wall_total > 0 or ask_wall_total > 0:
            asymmetry = abs(bid_wall_total - ask_wall_total) / (bid_wall_total + ask_wall_total + 1e-10)
            if asymmetry > 0.7:  # Highly asymmetric
                spoofing_indicators += 1
            num_indicators += 1

        spoofing_score = spoofing_indicators / max(num_indicators, 1)
        return float(np.clip(spoofing_score, 0.0, 1.0))

    def _calculate_iceberg_score(self) -> float:
        """Calculate probability of iceberg orders (hidden large orders).

        Indicators:
        - Trade volume >> visible order book depth
        - Repeated execution at same price without book update
        """
        # Simple heuristic: if total visible depth is low relative to historical volume
        if len(self.volume_history) < 10:
            return 0.5

        current_visible = sum(level.volume for level in self.bids.values()) + \
                          sum(level.volume for level in self.asks.values())

        avg_historical_vol = np.mean(list(self.volume_history))

        if current_visible < avg_historical_vol * 0.5:
            # Visible depth much lower than normal = possible icebergs
            iceberg_score = 0.7
        else:
            iceberg_score = 0.3

        return float(iceberg_score)

    def _calculate_pressure_score(self) -> float:
        """Calculate net buying/selling pressure.

        Positive = buying pressure, Negative = selling pressure
        Uses order book imbalance + recent price changes
        """
        if len(self.imbalance_history) < 5 or len(self.mid_price_history) < 5:
            return 0.0

        # Recent imbalance (positive = bid-heavy = buying pressure)
        recent_imbalance = np.mean(list(self.imbalance_history)[-10:])

        # Recent price momentum
        recent_prices = list(self.mid_price_history)[-10:]
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0

        # Combine (50/50 weight)
        pressure = (recent_imbalance + np.sign(price_momentum) * min(abs(price_momentum) * 100, 1.0)) / 2

        return float(np.clip(pressure, -1.0, 1.0))

    def _empty_features(self) -> MicrostructureFeatures:
        """Return default features when order book is empty."""
        return MicrostructureFeatures(
            bid_wall_strength=1.0,
            ask_wall_strength=1.0,
            spread_bps=10.0,
            ob_imbalance=0.0,
            toxicity=0.5,
            spoofing_score=0.0,
            iceberg_score=0.5,
            pressure_score=0.0,
            bid_walls=[],
            ask_walls=[],
            bid_depth_total=0.0,
            ask_depth_total=0.0,
            mid_price=0.0,
        )

    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices."""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return best_bid, best_ask

    def get_depth_at_price(self, price: float, side: str) -> float:
        """Get total depth up to a given price."""
        if side == "bid":
            return sum(level.volume for p, level in self.bids.items() if p >= price)
        else:  # ask
            return sum(level.volume for p, level in self.asks.items() if p <= price)

    def get_status(self) -> Dict:
        """Get order book status."""
        best_bid, best_ask = self.get_best_bid_ask()

        return {
            "symbol": self.symbol,
            "update_count": self.update_count,
            "num_bid_levels": len(self.bids),
            "num_ask_levels": len(self.asks),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": (best_ask - best_bid) if (best_bid and best_ask) else None,
        }
