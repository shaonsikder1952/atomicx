"""Command Receiver Module.

Receives Intents from the Brain and validates them against the hard
neurosymbolic guardrails before allowing them into the Action Engine.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from atomicx.brain.decider import DecisionIntent
from atomicx.guardrails import TradabilityGuardrails


class ReceivedCommand(BaseModel):
    """A wrapper for Intents that have entered the Action Engine."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    command_id: str = Field(default_factory=lambda: f"cmd-{uuid.uuid4().hex[:8]}")
    received_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    intent: DecisionIntent
    symbol: str
    price: float
    status: str = "received"
    validation_reason: str = ""


class CommandReceiver:
    """The intake valve for the Action Engine."""
    
    def __init__(self, guardrails: TradabilityGuardrails) -> None:
        self.guardrails = guardrails
        self.logger = logger.bind(module="action.receiver")

    def process_intent(self, symbol: str, price: float, intent: DecisionIntent) -> ReceivedCommand | None:
        """Validate an Intent from the Brain against hard guardrails."""
        command = ReceivedCommand(intent=intent, symbol=symbol, price=price)
        
        self.logger.info(f"[{symbol}] RECEIVER INTAKE: Intent='{intent.intent_type}' | Direction='{intent.direction}'")

        # Mock Guardrail Verification
        # In full production we pass the intent into TradabilityGuardrails
        # For Phase 15 skeleton, we simulate the hard block check:
        if intent.direction == "stay_out" or not intent.is_actionable():
            command.status = "rejected"
            command.validation_reason = "Intent is non-actionable or stay_out."
            self.logger.warning(f"[{symbol}] RECEIVER REJECTED: {command.validation_reason}")
            return None
            
        # Example hard guardrail check (simulated)
        if self.guardrails.get_active_risk() > 0.05:  # e.g., max 5% total account risk
            command.status = "rejected"
            command.validation_reason = "Hard guardrail violation: Max account risk exceeded."
            self.logger.error(f"[{symbol}] RECEIVER REJECTED: {command.validation_reason}")
            return None

        command.status = "validated"
        self.logger.success(f"[{symbol}] RECEIVER VALIDATED: Passed all hard guardrails. Forwarding to Orchestrator.")
        return command
