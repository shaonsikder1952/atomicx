"""Strategic Actor Layer — Jiang Framework (Phase 7).

Models market actors' incentives, detects traps, and builds
entity decision genomes. This layer answers: "Who is doing what
and why? Are they setting a trap?"
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class ActorType(str, Enum):
    """Market actor categories."""
    WHALE = "whale"
    MINER = "miner"
    INSTITUTION = "institution"
    RETAIL = "retail"
    MARKET_MAKER = "market_maker"
    EXCHANGE = "exchange"


class IncentiveVector(BaseModel):
    """An actor's current self-interest direction."""
    actor_type: ActorType
    direction: str = "neutral"  # bullish, bearish, neutral
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    signals: list[str] = Field(default_factory=list)


class TrapTemplate(BaseModel):
    """A known historical trap pattern."""
    trap_id: str
    name: str
    description: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    historical_examples: list[str] = Field(default_factory=list)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


class EscalationStep(BaseModel):
    """A step in an escalation path."""
    step: int
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    price_impact: str = ""


class EntityGenome(BaseModel):
    """Decision genome for a tracked entity (e.g., specific whale address)."""
    entity_id: str
    actor_type: ActorType
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    discovered_rules: list[str] = Field(default_factory=list)
    confidence: float = 0.0


# ── Historical Trap Database ─────────────────────────────────

HISTORICAL_TRAPS: list[TrapTemplate] = [
    TrapTemplate(
        trap_id="ftx_collapse", name="Exchange Insolvency Cascade",
        description="Exchange becomes insolvent, triggers mass withdrawals and contagion",
        conditions={"exchange_outflows": ">3x_average", "social_panic": True, "funding_extreme": True},
        historical_examples=["FTX Nov 2022", "Mt. Gox 2014"],
        severity=0.95,
    ),
    TrapTemplate(
        trap_id="etf_buy_rumor", name="ETF Buy-the-Rumor Sell-the-News",
        description="Price pumps into ETF decision, then dumps regardless of outcome",
        conditions={"pre_event_rally": ">15%", "funding_rate": ">0.05%", "retail_euphoria": True},
        historical_examples=["BTC ETF Jan 2024", "ETH ETF speculation 2024"],
        severity=0.7,
    ),
    TrapTemplate(
        trap_id="miner_capitulation", name="Miner Capitulation Trap",
        description="Miners forced to sell, creating temporary oversupply before recovery",
        conditions={"hash_rate_drop": ">10%", "miner_outflows": "elevated", "difficulty_adjustment": "pending"},
        historical_examples=["Post-halving 2020", "China ban 2021"],
        severity=0.6,
    ),
    TrapTemplate(
        trap_id="whale_accumulation", name="Whale Accumulation Disguise",
        description="Whales accumulate while price appears to be declining",
        conditions={"ob_imbalance": "sell_heavy_surface", "large_limit_bids": True, "exchange_outflows": "high"},
        historical_examples=["BTC $16k-$20k range 2022-2023"],
        severity=0.5,
    ),
    TrapTemplate(
        trap_id="short_squeeze", name="Engineered Short Squeeze",
        description="Accumulated short positions get forced out by deliberate price spike",
        conditions={"funding_rate": "<-0.03%", "oi_high": True, "whale_buy_walls": True},
        historical_examples=["BTC Jan 2023 squeeze", "Frequent perp liquidation cascades"],
        severity=0.65,
    ),
    TrapTemplate(
        trap_id="long_squeeze", name="Engineered Long Squeeze",
        description="Overleveraged longs get liquidated by deliberate dump",
        conditions={"funding_rate": ">0.05%", "oi_high": True, "whale_sell_walls": True},
        historical_examples=["Multiple BTC flash crashes", "May 2021 cascade"],
        severity=0.65,
    ),
]


class IncentiveMapAgent:
    """Tracks live self-interest vectors for each actor type (STRT-01)."""

    def analyze(self, variables: dict[str, float]) -> list[IncentiveVector]:
        """Compute incentive vectors from current market data."""
        vectors = []

        funding = variables.get("FUNDING_RATE")
        ob_imbalance = variables.get("OB_IMBALANCE")
        rel_volume = variables.get("REL_VOLUME")
        cvd = variables.get("CVD")  # Alternative signal when funding/volume missing

        # Priority 2B: Honest strategic grounding check
        # Need at least ONE strategic variable to avoid hallucination
        # Changed from requiring ALL THREE to requiring ANY ONE
        live_signals = [v for v in [funding, ob_imbalance, rel_volume, cvd] if v is not None]
        if not live_signals:
            return []

        # Proceed with real data - graceful degradation with available signals
        rsi = variables.get("RSI_14", 50)

        # Use available data with sensible defaults
        f_val = funding if funding is not None else 0.0
        obi_val = ob_imbalance if ob_imbalance is not None else 0.0
        rv_val = rel_volume if rel_volume is not None else 1.0
        cvd_val = cvd if cvd is not None else 0.0

        # Whale incentives - use OB_IMBALANCE + (REL_VOLUME or CVD)
        if obi_val > 0.3:
            if rv_val > 2 or abs(cvd_val) > 100:  # High volume OR large CVD
                vectors.append(IncentiveVector(
                    actor_type=ActorType.WHALE, direction="bullish", intensity=0.7,
                    reasoning="Large buy walls + high volume/CVD = whale accumulation",
                    signals=[f"OB_IMBALANCE={obi_val:.2f}", f"REL_VOLUME={rv_val:.2f}", f"CVD={cvd_val:.1f}"],
                ))
        elif obi_val < -0.3:
            if rv_val > 2 or abs(cvd_val) > 100:  # High volume OR large CVD
                vectors.append(IncentiveVector(
                    actor_type=ActorType.WHALE, direction="bearish", intensity=0.7,
                    reasoning="Large sell walls + high volume/CVD = whale distribution",
                    signals=[f"OB_IMBALANCE={obi_val:.2f}", f"REL_VOLUME={rv_val:.2f}", f"CVD={cvd_val:.1f}"],
                ))

        # Alternative whale detection using CVD alone when orderbook data weak
        elif abs(cvd_val) > 200:  # Extreme CVD = whale activity
            direction = "bullish" if cvd_val > 0 else "bearish"
            vectors.append(IncentiveVector(
                actor_type=ActorType.WHALE, direction=direction, intensity=0.6,
                reasoning=f"Extreme CVD ({cvd_val:.1f}) indicates whale {direction} aggression",
                signals=[f"CVD={cvd_val:.1f}"],
            ))

        # Retail incentives (contrarian signal)
        if f_val > 0.03:
            vectors.append(IncentiveVector(
                actor_type=ActorType.RETAIL, direction="bullish", intensity=0.8,
                reasoning="Extreme positive funding = retail overleveraged long (contrarian bearish)",
                signals=["FUNDING > 0.03"],
            ))
        elif f_val < -0.03:
            vectors.append(IncentiveVector(
                actor_type=ActorType.RETAIL, direction="bearish", intensity=0.8,
                reasoning="Extreme negative funding = retail overleveraged short (contrarian bullish)",
                signals=["FUNDING < -0.03"],
            ))

        # Market maker incentives - detect balanced book or tight control
        if abs(obi_val) < 0.05:
            vectors.append(IncentiveVector(
                actor_type=ActorType.MARKET_MAKER, direction="neutral", intensity=0.5,
                reasoning="Balanced book = market makers in control, range-bound likely",
                signals=[f"OB_IMBALANCE={obi_val:.3f}"],
            ))
        elif abs(obi_val) < 0.15 and abs(cvd_val) < 50:
            # Tight orderbook with low CVD = market makers suppressing volatility
            vectors.append(IncentiveVector(
                actor_type=ActorType.MARKET_MAKER, direction="neutral", intensity=0.6,
                reasoning="Tight book + low CVD = market makers suppressing move, expect compression",
                signals=[f"OB_IMBALANCE={obi_val:.3f}", f"CVD={cvd_val:.1f}"],
            ))

        return vectors


class StrategicTrapDetector:
    """Matches current state against historical trap templates (STRT-02)."""

    def __init__(self) -> None:
        self.traps = HISTORICAL_TRAPS

    def detect(self, variables: dict[str, float], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Check current conditions against all trap templates."""
        matches = []
        funding = variables.get("FUNDING_RATE", 0)
        rel_volume = variables.get("REL_VOLUME", 1)
        rsi = variables.get("RSI_14", 50)

        for trap in self.traps:
            match_score = 0.0
            total_conditions = max(len(trap.conditions), 1)

            if trap.trap_id == "short_squeeze" and funding < -0.03:
                match_score += 1.0
            if trap.trap_id == "long_squeeze" and funding > 0.05:
                match_score += 1.0
            if "retail_euphoria" in trap.conditions and rsi > 80:
                match_score += 0.5
            if "funding_extreme" in trap.conditions and abs(funding) > 0.05:
                match_score += 0.5

            normalized = match_score / total_conditions
            if normalized > 0.2:
                matches.append({
                    "trap": trap.name,
                    "match_score": round(normalized, 2),
                    "severity": trap.severity,
                    "description": trap.description,
                    "risk_level": "HIGH" if normalized > 0.6 else "MODERATE" if normalized > 0.3 else "LOW",
                })

        return sorted(matches, key=lambda x: x["match_score"], reverse=True)


class EscalationLadder:
    """Produces probabilistic escalation paths (STRT-03)."""

    def generate(self, trap_name: str, variables: dict[str, float]) -> list[EscalationStep]:
        """Generate escalation path for a detected trap."""
        if "squeeze" in trap_name.lower():
            return [
                EscalationStep(step=1, description="Initial price movement triggers liquidations", probability=0.7, price_impact="-2 to -5%"),
                EscalationStep(step=2, description="Liquidation cascade accelerates move", probability=0.5, price_impact="-5 to -10%"),
                EscalationStep(step=3, description="Forced selling exhaustion → reversal", probability=0.4, price_impact="Recovery 3-7%"),
            ]
        elif "capitulation" in trap_name.lower():
            return [
                EscalationStep(step=1, description="Miners begin selling holdings", probability=0.6, price_impact="-3 to -8%"),
                EscalationStep(step=2, description="Hash rate continues dropping", probability=0.4, price_impact="-5 to -15%"),
                EscalationStep(step=3, description="Weak miners exit, difficulty adjusts, recovery begins", probability=0.6, price_impact="Recovery over weeks"),
            ]
        return [EscalationStep(step=1, description=f"Monitor {trap_name} conditions", probability=0.5)]


class StrategicActorLayer:
    """Orchestrates all strategic analysis components."""

    def __init__(self) -> None:
        self.incentive_map = IncentiveMapAgent()
        self.trap_detector = StrategicTrapDetector()
        self.escalation = EscalationLadder()

    def analyze(self, variables: dict[str, float]) -> dict[str, Any]:
        """Run full strategic analysis."""
        incentives = self.incentive_map.analyze(variables)
        traps = self.trap_detector.detect(variables)

        escalations = []
        for trap in traps[:3]:
            steps = self.escalation.generate(trap["trap"], variables)
            escalations.append({"trap": trap["trap"], "steps": [s.model_dump() for s in steps]})

        # Strategic signal
        whale_vectors = [v for v in incentives if v.actor_type == ActorType.WHALE]
        retail_vectors = [v for v in incentives if v.actor_type == ActorType.RETAIL]

        strategic_direction = "neutral"
        strategic_confidence = 0.0

        if whale_vectors:
            strategic_direction = whale_vectors[0].direction
            strategic_confidence = whale_vectors[0].intensity * 0.7

        # Contrarian to retail extreme
        if retail_vectors and retail_vectors[0].intensity > 0.6:
            if retail_vectors[0].direction == "bullish":
                strategic_direction = "bearish"
                strategic_confidence = max(strategic_confidence, 0.5)
            elif retail_vectors[0].direction == "bearish":
                strategic_direction = "bullish"
                strategic_confidence = max(strategic_confidence, 0.5)

        # Reduce confidence if trap detected
        if traps and traps[0]["match_score"] > 0.5:
            strategic_confidence *= 0.6

        return {
            "direction": strategic_direction,
            "confidence": strategic_confidence,
            "incentives": [v.model_dump() for v in incentives],
            "traps": traps,
            "escalations": escalations,
        }
