"""Retail Flow Anticipator — Citadel-Style "Dumb Money Meter."

Citadel processes retail order flow to predict crowd behavior.
We replicate this by combining social sentiment vectorization with
whale activity monitoring to detect "Liquidity Traps."

The core insight: When retail screams "Bitcoin is dead" but whales
are quietly accumulating, it's a Liquidity Trap → buy before the
retail crowd gains courage to jump back in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field


class RetailSentimentSnapshot(BaseModel):
    """A snapshot of retail trader behavior."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    fear_greed_index: float = 50.0        # 0 = extreme fear, 100 = extreme greed
    social_negativity: float = 0.0        # 0-1 scale of "Bitcoin is dead" sentiment
    retail_buy_pressure: float = 0.5      # 0-1 scale from exchange retail flow
    retail_leverage_ratio: float = 1.0    # How leveraged retail is (1.0 = no leverage)
    dumb_money_meter: float = 0.5         # 0 = max fear/capitulation, 1 = max euphoria


class WhaleActivitySnapshot(BaseModel):
    """A snapshot of whale/institutional behavior."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    large_tx_volume_24h: float = 0.0      # Volume of transactions > $100k
    exchange_net_flow: float = 0.0        # Negative = withdrawals (bullish), Positive = deposits (bearish)
    whale_accumulation_score: float = 0.5  # 0 = dumping, 1 = aggressive accumulation
    

class LiquidityTrapSignal(BaseModel):
    """A detected Liquidity Trap — the Citadel-style edge."""
    trap_type: str       # "capitulation_buy", "euphoria_sell", "squeeze_incoming"
    confidence: float
    retail_state: str    # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
    whale_state: str     # "accumulating", "distributing", "quiet"
    recommended_action: str  # "buy_before_retail", "sell_before_retail", "wait"
    reasoning: str


class RetailFlowAnticipator:
    """Detects divergences between retail sentiment and whale behavior.
    
    Citadel's GMI Framework: retail at 95th percentile risk-on often
    "leans into weakness." We detect when retail is wrong and whales
    are positioning opposite — that's our entry signal.
    """
    
    def __init__(self) -> None:
        self.logger = logger.bind(module="titan.retail_flow")
        self.retail_history: list[RetailSentimentSnapshot] = []
        self.whale_history: list[WhaleActivitySnapshot] = []
        
    def ingest_retail_data(
        self,
        fear_greed: float,
        social_negativity: float,
        buy_pressure: float,
        leverage_ratio: float = 1.0
    ) -> RetailSentimentSnapshot:
        """Process the latest retail sentiment data."""
        # Calculate the Dumb Money Meter
        # High fear + high negativity + low buy pressure = capitulation (meter → 0)
        # High greed + low negativity + high buy pressure = euphoria (meter → 1)
        dumb_money = (fear_greed / 100 * 0.4) + ((1 - social_negativity) * 0.3) + (buy_pressure * 0.3)
        
        snapshot = RetailSentimentSnapshot(
            fear_greed_index=fear_greed,
            social_negativity=social_negativity,
            retail_buy_pressure=buy_pressure,
            retail_leverage_ratio=leverage_ratio,
            dumb_money_meter=dumb_money,
        )
        self.retail_history.append(snapshot)
        
        # Keep last 500 snapshots
        if len(self.retail_history) > 500:
            self.retail_history = self.retail_history[-500:]
            
        return snapshot
        
    def ingest_whale_data(
        self,
        large_tx_volume: float,
        exchange_net_flow: float,
    ) -> WhaleActivitySnapshot:
        """Process the latest whale/institutional activity data."""
        # Accumulation score: high volume + negative net flow (withdrawals) = bullish
        accumulation = 0.5
        if exchange_net_flow < 0:  # Withdrawals from exchanges
            accumulation = min(1.0, 0.5 + abs(exchange_net_flow) / 10000)
        elif exchange_net_flow > 0:  # Deposits to exchanges (ready to sell)
            accumulation = max(0.0, 0.5 - exchange_net_flow / 10000)
            
        snapshot = WhaleActivitySnapshot(
            large_tx_volume_24h=large_tx_volume,
            exchange_net_flow=exchange_net_flow,
            whale_accumulation_score=accumulation,
        )
        self.whale_history.append(snapshot)
        
        if len(self.whale_history) > 500:
            self.whale_history = self.whale_history[-500:]
            
        return snapshot
        
    def detect_liquidity_traps(self) -> LiquidityTrapSignal | None:
        """The core algorithm: find divergences between retail and whale behavior."""
        if not self.retail_history or not self.whale_history:
            return None
            
        retail = self.retail_history[-1]
        whale = self.whale_history[-1]
        
        retail_state = self._classify_retail_state(retail.dumb_money_meter)
        whale_state = self._classify_whale_state(whale.whale_accumulation_score)
        
        # TRAP 1: Retail capitulation + Whale accumulation → BUY BEFORE RETAIL
        if retail_state in ("extreme_fear", "fear") and whale_state == "accumulating":
            confidence = (1 - retail.dumb_money_meter) * whale.whale_accumulation_score
            trap = LiquidityTrapSignal(
                trap_type="capitulation_buy",
                confidence=confidence,
                retail_state=retail_state,
                whale_state=whale_state,
                recommended_action="buy_before_retail",
                reasoning=(
                    f"Retail Dumb Money Meter at {retail.dumb_money_meter:.2f} (capitulating), "
                    f"but whales accumulating at {whale.whale_accumulation_score:.2f}. "
                    f"Classic Liquidity Trap — buy before retail gains courage."
                )
            )
            self.logger.success(f"[CITADEL] 🎯 LIQUIDITY TRAP: {trap.reasoning}")
            return trap
            
        # TRAP 2: Retail euphoria + Whale distribution → SELL BEFORE RETAIL
        if retail_state in ("extreme_greed", "greed") and whale_state == "distributing":
            confidence = retail.dumb_money_meter * (1 - whale.whale_accumulation_score)
            trap = LiquidityTrapSignal(
                trap_type="euphoria_sell",
                confidence=confidence,
                retail_state=retail_state,
                whale_state=whale_state,
                recommended_action="sell_before_retail",
                reasoning=(
                    f"Retail euphoric (DMM: {retail.dumb_money_meter:.2f}), "
                    f"whales dumping (acc: {whale.whale_accumulation_score:.2f}). "
                    f"Distribution trap — exit before retail bag-holds."
                )
            )
            self.logger.warning(f"[CITADEL] 🎯 DISTRIBUTION TRAP: {trap.reasoning}")
            return trap
            
        # TRAP 3: Retail max leverage + Any whale activity → SQUEEZE INCOMING
        if retail.retail_leverage_ratio > 3.0:
            trap = LiquidityTrapSignal(
                trap_type="squeeze_incoming",
                confidence=min(1.0, retail.retail_leverage_ratio / 5.0),
                retail_state=retail_state,
                whale_state=whale_state,
                recommended_action="wait",
                reasoning=(
                    f"Retail leverage at {retail.retail_leverage_ratio:.1f}x — "
                    f"liquidation cascade imminent. Wait for the squeeze."
                )
            )
            self.logger.warning(f"[CITADEL] ⚠️ SQUEEZE WARNING: {trap.reasoning}")
            return trap
            
        return None
        
    def _classify_retail_state(self, dumb_money: float) -> str:
        if dumb_money < 0.15: return "extreme_fear"
        if dumb_money < 0.35: return "fear"
        if dumb_money < 0.65: return "neutral"
        if dumb_money < 0.85: return "greed"
        return "extreme_greed"
        
    def _classify_whale_state(self, accumulation: float) -> str:
        if accumulation > 0.7: return "accumulating"
        if accumulation < 0.3: return "distributing"
        return "quiet"
