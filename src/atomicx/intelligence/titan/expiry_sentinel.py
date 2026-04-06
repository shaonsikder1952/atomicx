"""Expiry Day Sentinel — Jane Street-Style Manipulation Detection.

Jane Street moves the cash market to profit on massive options positions.
They "Mark the Close" by dumping or pumping in the final 60 minutes.

This sentinel monitors:
1. Options expiry schedules (BTC/ETH weekly, monthly, quarterly)
2. Open interest concentration at specific strike prices
3. Unusual price action in the final 60 minutes of expiry day
4. Absence of fundamental news during the move (= manipulation signal)

The edge: Don't fight the manipulation — ride it.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field


class ExpiryEvent(BaseModel):
    """A tracked options/futures expiry event."""
    event_id: str
    asset: str  # "BTC", "ETH"
    expiry_type: str  # "weekly", "monthly", "quarterly"
    expiry_datetime: datetime
    open_interest_usd: float = 0.0
    max_pain_price: float = 0.0  # The price where most options expire worthless
    dominant_side: str = "neutral"  # "calls" or "puts" — which side has more OI


class ManipulationSignal(BaseModel):
    """A detected Jane Street-style manipulation event."""
    signal_type: str  # "mark_the_close_dump", "mark_the_close_pump", "pin_to_max_pain"
    asset: str
    expiry_type: str
    confidence: float
    price_move_pct: float  # How much price moved in the manipulation window
    has_fundamental_reason: bool  # False = likely manipulation
    recommended_action: str  # "ride_dump", "ride_pump", "fade_after_expiry"
    reasoning: str


class ExpiryDaySentinel:
    """Monitors options expiry events for institutional manipulation patterns."""
    
    def __init__(self) -> None:
        self.logger = logger.bind(module="titan.expiry_sentinel")
        self.upcoming_expiries: list[ExpiryEvent] = []
        self.detected_manipulations: list[ManipulationSignal] = []
        
        # Pre-load known expiry schedule patterns
        self._load_expiry_schedule()
        
    def _load_expiry_schedule(self) -> None:
        """Load the known options/futures expiry schedule."""
        now = datetime.now(tz=timezone.utc)
        
        # Generate next 4 weekly expiries (every Friday 08:00 UTC)
        for i in range(1, 5):
            days_until_friday = (4 - now.weekday()) % 7
            if days_until_friday == 0 and i > 1:
                days_until_friday = 7
            expiry_date = now + timedelta(days=days_until_friday + (i - 1) * 7)
            expiry_date = expiry_date.replace(hour=8, minute=0, second=0, microsecond=0)
            
            self.upcoming_expiries.append(ExpiryEvent(
                event_id=f"btc-weekly-{i}",
                asset="BTC",
                expiry_type="weekly" if i < 4 else "monthly",
                expiry_datetime=expiry_date,
                open_interest_usd=500_000_000 * (2 if i == 4 else 1),  # Monthly has 2x OI
                max_pain_price=95000.0,
                dominant_side="calls",
            ))
            
        self.logger.info(f"[SENTINEL] Loaded {len(self.upcoming_expiries)} upcoming expiry events")
        
    def is_expiry_window(self, current_time: datetime | None = None) -> ExpiryEvent | None:
        """Check if we're currently in a manipulation-risk window (final 4 hours before expiry)."""
        now = current_time or datetime.now(tz=timezone.utc)
        
        for event in self.upcoming_expiries:
            time_to_expiry = (event.expiry_datetime - now).total_seconds() / 3600  # Hours
            if 0 < time_to_expiry <= 4:  # Within 4 hours of expiry
                self.logger.warning(
                    f"[SENTINEL] ⚠️ EXPIRY WINDOW ACTIVE: {event.asset} {event.expiry_type} "
                    f"expires in {time_to_expiry:.1f}h | OI: ${event.open_interest_usd:,.0f} | "
                    f"Max Pain: ${event.max_pain_price:,.0f}"
                )
                return event
        return None
        
    def analyze_price_action(
        self,
        current_price: float,
        price_1h_ago: float,
        has_news_catalyst: bool,
        expiry_event: ExpiryEvent
    ) -> ManipulationSignal | None:
        """Analyze if current price action is organic or manipulation.
        
        The Jane Street Tell: A sharp move in the final hour WITHOUT
        a news catalyst is likely institutional manipulation to pin
        the price to max pain or profit on options positions.
        """
        price_move_pct = ((current_price - price_1h_ago) / price_1h_ago) * 100
        
        # No significant move? No manipulation.
        if abs(price_move_pct) < 0.5:
            return None
            
        # Check if the move is toward max pain (pinning behavior)
        moving_toward_max_pain = (
            (current_price > price_1h_ago and current_price <= expiry_event.max_pain_price) or
            (current_price < price_1h_ago and current_price >= expiry_event.max_pain_price)
        )
        
        if not has_news_catalyst:
            # No news + sharp move = manipulation signal
            if price_move_pct < -1.0:
                signal = ManipulationSignal(
                    signal_type="mark_the_close_dump",
                    asset=expiry_event.asset,
                    expiry_type=expiry_event.expiry_type,
                    confidence=min(0.95, abs(price_move_pct) / 3.0),
                    price_move_pct=price_move_pct,
                    has_fundamental_reason=False,
                    recommended_action="ride_dump" if abs(price_move_pct) > 2.0 else "fade_after_expiry",
                    reasoning=(
                        f"{price_move_pct:.1f}% dump in final hour with NO news catalyst. "
                        f"OI: ${expiry_event.open_interest_usd:,.0f}. "
                        f"Likely Jane Street-style manipulation to profit on {expiry_event.dominant_side} positions."
                    )
                )
                self.detected_manipulations.append(signal)
                self.logger.warning(f"[SENTINEL] 🎯 MANIPULATION DETECTED: {signal.reasoning}")
                return signal
                
            elif price_move_pct > 1.0:
                signal = ManipulationSignal(
                    signal_type="mark_the_close_pump",
                    asset=expiry_event.asset,
                    expiry_type=expiry_event.expiry_type,
                    confidence=min(0.95, abs(price_move_pct) / 3.0),
                    price_move_pct=price_move_pct,
                    has_fundamental_reason=False,
                    recommended_action="ride_pump" if abs(price_move_pct) > 2.0 else "fade_after_expiry",
                    reasoning=(
                        f"+{price_move_pct:.1f}% pump in final hour with NO news catalyst. "
                        f"OI: ${expiry_event.open_interest_usd:,.0f}. "
                        f"Likely mark-the-close pump for options settlement."
                    )
                )
                self.detected_manipulations.append(signal)
                self.logger.warning(f"[SENTINEL] 🎯 MANIPULATION DETECTED: {signal.reasoning}")
                return signal
                
        if moving_toward_max_pain and not has_news_catalyst:
            signal = ManipulationSignal(
                signal_type="pin_to_max_pain",
                asset=expiry_event.asset,
                expiry_type=expiry_event.expiry_type,
                confidence=0.7,
                price_move_pct=price_move_pct,
                has_fundamental_reason=False,
                recommended_action="fade_after_expiry",
                reasoning=(
                    f"Price converging on max pain ${expiry_event.max_pain_price:,.0f}. "
                    f"Classic options market-maker pinning behavior."
                )
            )
            self.detected_manipulations.append(signal)
            self.logger.info(f"[SENTINEL] Max pain pinning detected: {signal.reasoning}")
            return signal
            
        return None
        
    def get_dashboard_alerts(self) -> list[dict[str, Any]]:
        """Return alerts for the Dashboard (Phase 18) to highlight."""
        alerts = []
        
        # Upcoming expiry warnings
        now = datetime.now(tz=timezone.utc)
        for event in self.upcoming_expiries:
            hours_left = (event.expiry_datetime - now).total_seconds() / 3600
            if 0 < hours_left <= 24:
                alerts.append({
                    "type": "expiry_warning",
                    "severity": "high" if hours_left <= 4 else "medium",
                    "message": (
                        f"{event.asset} {event.expiry_type} expiry in {hours_left:.0f}h | "
                        f"OI: ${event.open_interest_usd:,.0f} | Max Pain: ${event.max_pain_price:,.0f}"
                    ),
                })
                
        # Recent manipulation signals
        for signal in self.detected_manipulations[-5:]:
            alerts.append({
                "type": "manipulation_detected",
                "severity": "critical",
                "message": signal.reasoning,
                "action": signal.recommended_action,
            })
            
        return alerts
