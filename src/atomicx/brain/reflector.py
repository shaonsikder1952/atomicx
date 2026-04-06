"""Recursive Reflector — The Self-Awareness Module.

Runs continuous meta-loops analyzing why the system made its last decisions.
Generates a natural-language internal monologue that is fully logged and auditable.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import json
from loguru import logger
from pydantic import BaseModel, Field
from atomicx.config import get_settings


class ReflectorMonologue(BaseModel):
    """Structured output of the Reflector's internal reasoning."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    cycle_id: str
    dominant_layer: str
    regime_alignment: float
    reasoning: str
    regime: str = "unknown"
    action_item: str | None = None


class RecursiveReflector:
    """Provides meta-awareness by analyzing the Meta-Orchestrator's state and outputs."""

    def __init__(self, memory: Any | None = None, history_limit: int = 100) -> None:
        self.monologues: list[ReflectorMonologue] = []
        self._history_limit = history_limit
        self.logger = logger.bind(module="brain.reflector")
        self._settings = get_settings()
        self.memory = memory

    async def reflect(
        self,
        cycle_id: str,
        regime: str,
        layer_states: dict[str, Any],
        orchestrator_decision: dict[str, Any]
    ) -> ReflectorMonologue:
        """Analyze the current state and decision to generate an internal monologue.
        
        This mimics self-awareness by identifying which layer drove the decision,
        whether that aligns with the current regime, and if any trust weights should change.
        """
        # 1. Prepare context for LLM
        prompt = f"""You are the Recursive Reflector, the self-awareness module of AtomicX.
Current Analysis:
- Symbol: {cycle_id.split('-')[-1]} (ID: {cycle_id})
- Detected Regime: {regime}
- Layer States: {json.dumps(layer_states)}
- Planned Decision: {json.dumps(orchestrator_decision)}

Task: Generate a 1-2 sentence 'Internal Monologue' reflecting on why this decision was made and if we should adjust trust weights (e.g., 'reduce_swarm_weight').
Return JSON format: {{"reasoning": "...", "action": "..."}}"""

        reasoning = f"The {regime} cycle is active. Standard weightings applied."
        action = None
        alignment = 1.0

        if self._settings.aws_bearer_token:
            try:
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": prompt}],
                }
                base_url = self._settings.aws_endpoint_url or f"https://bedrock-runtime.{self._settings.aws_region_name}.amazonaws.com"
                url = f"{base_url}/model/{self._settings.bedrock_model_id}/invoke"
                
                headers = {
                    "Authorization": f"Bearer {self._settings.aws_bearer_token}",
                    "x-api-key": self._settings.aws_bearer_token,
                    "Content-Type": "application/json",
                }
                
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
                    if resp.status_code == 200:
                        res = resp.json()
                        text = res["content"][0]["text"]

                        # Robust JSON extraction from LLM response
                        from atomicx.common.json_utils import extract_json_from_llm_text

                        data = extract_json_from_llm_text(
                            text,
                            expected_keys=["reasoning", "action"],
                            default=None
                        )

                        if data:
                            reasoning = data.get("reasoning", reasoning)
                            action = data.get("action", action)
                        else:
                            self.logger.warning(
                                f"Reflector: Failed to extract JSON from response: {text[:100]}"
                            )
                    else:
                        self.logger.warning(f"Bedrock reflection HTTP error {resp.status_code}: {resp.text}")
            except Exception as e:
                self.logger.warning(f"Bedrock reflection failed: {e}. Using deterministic fallback.")

        monologue = ReflectorMonologue(
            cycle_id=cycle_id,
            dominant_layer=regime,
            regime_alignment=alignment,
            reasoning=reasoning,
            regime=regime,
            action_item=action
        )
        
        # Store in-memory for immediate access
        self.monologues.append(monologue)
        if len(self.monologues) > self._history_limit:
            self.monologues.pop(0)

        # Store in Infinite Memory (Qdrant) if available
        if self.memory:
            from atomicx.memory.service import MemoryEntry, MemoryType
            try:
                await self.memory.store(MemoryEntry(
                    memory_type=MemoryType.SEMANTIC,
                    content=f"INTERNAL MONOLOGUE: {monologue.reasoning}",
                    symbol=cycle_id.split('-')[-1] if '-' in cycle_id else "GLOBAL",
                    metadata={
                        "cycle_id": cycle_id,
                        "regime": regime,
                        "action": action or "none",
                        "alignment": alignment
                    }
                ))
            except Exception as e:
                self.logger.warning(f"Failed to persist monologue to Qdrant: {e}")

        self.logger.info(f"INTERNAL MONOLOGUE [{cycle_id}]: {monologue.reasoning}")
        return monologue

    async def get_recent_monologues(self, count: int = 5) -> list[ReflectorMonologue]:
        """Fetch recent reflections from memory or cache."""
        # For now, we prefer the in-memory cache for speed during a live session,
        # but we could fall back or merge with Qdrant here.
        return self.monologues[-count:]
