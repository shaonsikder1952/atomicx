"""Internal Debate Chamber for AtomicX.

Sub-agents analyze the Orchestrator's sensory input from differing philosophical
standpoints. The Debate Chamber produces a consensus debate summary before
the Decider Core makes a final intent determination.

Powered by Claude 3.5 via AWS Bedrock for real cognitive diversity.
"""

from __future__ import annotations
import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import json
from typing import Any

import boto3
import httpx

try:
    import anthropic
except ImportError:
    anthropic = None

from loguru import logger
from pydantic import BaseModel, Field

from atomicx.config import get_settings


class DebateArgument(BaseModel):
    """An argument put forward by a sub-agent."""

    agent_name: str
    stance: str  # "bullish", "bearish", "neutral", "stay_out"
    conviction: float
    reasoning: str


class DebateSummary(BaseModel):
    """The final summarized outcome of a debate cycle."""

    dominant_stance: str
    overall_conviction: float
    conflict_detected: bool
    arguments: list[DebateArgument]
    synthesis: str


class CausalPurist:
    """Agent that only cares about structural/causal links.
    Focuses on technical indicators and structural breaks.
    """

    def __init__(self, chamber: DebateChamber) -> None:
        self.chamber = chamber

    async def argue(self, senses: dict[str, Any], variables: dict[str, Any]) -> DebateArgument:
        # Filter to only variables with real values
        live_vars = {k: v for k, v in variables.items() if v is not None and v != 0}

        if not live_vars:
            return DebateArgument(
                agent_name="CausalPurist",
                stance="neutral",
                conviction=0.0,
                reasoning="No live variable data available.",
            )

        pattern_data = senses.get("pattern", {})

        prompt = f"""You are CausalPurist. Analyze ONLY structural/causal signals.
Ignore sentiment and social data entirely.

Live variables:
{json.dumps(live_vars, indent=2)}

Statistical Pattern Momentum (Use ONLY as a tie-breaker if causal data is deadlocked):
{json.dumps(pattern_data, indent=2)}

Return JSON only:
{{"stance": "bullish|bearish|neutral", "conviction": 0.0-1.0, "reasoning": "one sentence"}}
"""
        result = await self.chamber._call_llm(self.__class__.__name__, prompt)
        return DebateArgument(
            agent_name="CausalPurist",
            stance=result.get("stance", "neutral"),
            conviction=result.get("conviction", 0.0),
            reasoning=result.get("reasoning", "No reasoning provided."),
        )


class NarrativeHeretic:
    """Agent that prioritizes social momentum and trend narrative over fundamentals."""

    def __init__(self, chamber: DebateChamber) -> None:
        self.chamber = chamber

    async def argue(self, senses: dict[str, Any], variables: dict[str, Any]) -> DebateArgument:
        narr = senses.get("narrative", {})

        # GATE: if no narrative data, return honest neutral
        if not narr or not narr.get("direction"):
            return DebateArgument(
                agent_name="NarrativeHeretic",
                stance="neutral",
                conviction=0.0,
                reasoning="Narrative layer offline. No social/news data connected.",
            )

        prompt = f"""You are NarrativeHeretic. Analyze social momentum and narrative only.

Narrative data:
{json.dumps(narr, indent=2)}

Return JSON only:
{{"stance": "bullish|bearish|neutral", "conviction": 0.0-1.0, "reasoning": "one sentence"}}
"""
        result = await self.chamber._call_llm(self.__class__.__name__, prompt)
        return DebateArgument(
            agent_name="NarrativeHeretic",
            stance=result.get("stance", "neutral"),
            conviction=result.get("conviction", 0.0),
            reasoning=result.get("reasoning", "No reasoning provided."),
        )


class RiskParanoid:
    """Agent that constantly searches for traps and tail risks."""

    def __init__(self, chamber: DebateChamber) -> None:
        self.chamber = chamber

    async def argue(self, senses: dict[str, Any], variables: dict[str, Any]) -> DebateArgument:
        strategic = senses.get("strategic", {})

        # GATE: if no swarm/strategic data, return honest neutral
        if not strategic or not strategic.get("direction"):
            return DebateArgument(
                agent_name="RiskParanoid",
                stance="neutral",
                conviction=0.0,
                reasoning="Swarm layer offline. No strategic trap data available.",
            )

        prompt = f"""You are RiskParanoid. Search only for traps and tail risks.
Bias toward STAY_OUT. Only approve if evidence is clean.

Strategic data:
{json.dumps(strategic, indent=2)}

Return JSON only:
{{"stance": "bullish|bearish|neutral|stay_out", "conviction": 0.0-1.0, "reasoning": "one sentence"}}
"""
        result = await self.chamber._call_llm(self.__class__.__name__, prompt)
        return DebateArgument(
            agent_name="RiskParanoid",
            stance=result.get("stance", "neutral"),
            conviction=result.get("conviction", 0.0),
            reasoning=result.get("reasoning", "No reasoning provided."),
        )


class DebateChamber:
    """Orchestrates internal debate between cognitive sub-agents."""

    def __init__(self) -> None:
        self.agents = [CausalPurist(self), NarrativeHeretic(self), RiskParanoid(self)]
        self.llm_connected = True  # flip this to False to disable all LLM calls
        self.logger = logger.bind(module="brain.debate")
        self._settings = get_settings()
        self._bedrock = None
        self._anthropic = None

        # Strategy 1: Anthropic SDK (User Preferred)
        if self._settings.anthropic_api_key and anthropic:
            try:
                self._anthropic = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
                self.logger.info("Debate Chamber connected via Anthropic Direct SDK")
            except Exception as e:
                self.logger.error(f"Failed to init Anthropic client: {e}")

        # Strategy 2: Bedrock / Proxy
        if not self._anthropic and self.llm_connected and self._settings.aws_access_key_id:
            try:
                self._bedrock = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self._settings.aws_region_name,
                    aws_access_key_id=self._settings.aws_access_key_id,
                    aws_secret_access_key=self._settings.aws_secret_access_key,
                )
            except Exception as e:
                self.logger.error(f"Failed to init Bedrock client: {e}")
                self.llm_connected = False

    async def _call_llm(self, agent_name: str, prompt: str) -> dict[str, Any]:
        """Internal helper to route requests to Anthropic, Bedrock, or Proxy."""
        if not self.llm_connected:
            return {"stance": "neutral", "conviction": 0.0, "reasoning": "LLM disconnected."}

        try:
            # 1. Try Anthropic SDK (Primary)
            if self._anthropic:
                response = await self._anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                    system=f"You are part of the AtomicX Debate Chamber. Acting as {agent_name}.",
                )

                # Safely extract text from Anthropic SDK response
                if hasattr(response, 'content') and len(response.content) > 0:
                    text = response.content[0].text
                else:
                    raise ValueError(f"Unexpected Anthropic SDK response format: {response}")

            # 2. Try Bedrock Proxy or API Key (v5.1 standard)
            elif self._settings.aws_bearer_token:
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                    "system": f"You are part of the AtomicX Debate Chamber. Acting as {agent_name}.",
                }
                # Default to regional runtime endpoint if no proxy URL provided
                base_url = self._settings.aws_endpoint_url or f"https://bedrock-runtime.{self._settings.aws_region_name}.amazonaws.com"
                url = f"{base_url}/model/{self._settings.bedrock_model_id}/invoke"
                
                headers = {
                    "Authorization": f"Bearer {self._settings.aws_bearer_token}",
                    "x-api-key": self._settings.aws_bearer_token, # Support both common formats
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
                    resp.raise_for_status()
                    result = resp.json()

                # Safely extract text from Bedrock API response
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        text = content[0].get("text", "")
                    else:
                        raise ValueError(f"Unexpected Bedrock response format: {result}")
                else:
                    raise ValueError(f"Invalid Bedrock response structure: {result}")

            # 3. Try Direct Bedrock (Fallback)
            elif self._bedrock:
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                    "system": f"You are part of the AtomicX Debate Chamber. Acting as {agent_name}.",
                }
                response = self._bedrock.invoke_model(
                    modelId=self._settings.bedrock_model_id, body=json.dumps(payload)
                )
                result = json.loads(response.get("body").read())

                # Safely extract text from Bedrock API response
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        text = content[0].get("text", "")
                    else:
                        raise ValueError(f"Unexpected Bedrock response format: {result}")
                else:
                    raise ValueError(f"Invalid Bedrock response structure: {result}")
            else:
                return {"stance": "neutral", "conviction": 0.0, "reasoning": "No LLM interface."}

            # Robust JSON extraction from LLM response
            from atomicx.common.json_utils import extract_json_from_llm_text

            result = extract_json_from_llm_text(
                text,
                expected_keys=["stance", "conviction", "reasoning"],
                default={
                    "stance": "neutral",
                    "conviction": 0.0,
                    "reasoning": f"Failed to parse {agent_name} response"
                }
            )

            # Validate stance value (allow stay_out and map it)
            if result["stance"] not in ("bullish", "bearish", "neutral", "stay_out"):
                self.logger.warning(
                    f"{agent_name} returned invalid stance '{result['stance']}', defaulting to neutral"
                )
                result["stance"] = "neutral"
            # Map stay_out to neutral for consensus (they mean the same in debate context)
            elif result["stance"] == "stay_out":
                result["stance"] = "neutral"
                self.logger.debug(f"{agent_name} stance 'stay_out' mapped to 'neutral' for debate")

            # Validate conviction is numeric and in range
            try:
                result["conviction"] = float(result["conviction"])
                result["conviction"] = max(0.0, min(1.0, result["conviction"]))
            except (ValueError, TypeError):
                self.logger.warning(f"{agent_name} returned invalid conviction, defaulting to 0.0")
                result["conviction"] = 0.0

            return result

        except Exception as e:
            self.logger.error(f"LLM call for {agent_name} failed: {e}")
            return {"stance": "neutral", "conviction": 0.0, "reasoning": f"LLM error: {str(e)[:50]}"}

    async def debate(self, brain_state: dict[str, Any]) -> DebateSummary:
        """Run a single debate cycle and return the synthesized summary."""
        senses = brain_state.get("senses", {})
        variables = brain_state.get("variables", {})
        trust_weights = brain_state.get("trust_weights", {})

        arguments: list[DebateArgument] = []
        for agent in self.agents:
            arg = await agent.argue(senses, variables)
            # Apply dynamic trust weights if mapped
            weight_key = self._map_agent_to_weight(agent.__class__.__name__)
            if weight_key in trust_weights:
                arg.conviction *= trust_weights[weight_key]
            arguments.append(arg)

        return self._synthesize(arguments)

    def _map_agent_to_weight(self, agent_name: str) -> str:
        if agent_name == "CausalPurist":
            return "causal"
        if agent_name == "NarrativeHeretic":
            return "narrative"
        if agent_name == "RiskParanoid":
            return "strategic"
        return "default"

    def _synthesize(self, arguments: list[DebateArgument]) -> DebateSummary:
        """Resolve the arguments into a single consensus summary."""
        stances = {"bullish": 0.0, "bearish": 0.0, "stay_out": 0.0, "neutral": 0.0}

        conflict_detected = False
        highest_conviction = 0.0
        dominant = "neutral"

        # Look for hard vetos first
        for arg in arguments:
            if arg.stance == "stay_out" and arg.conviction > 0.8:
                return DebateSummary(
                    dominant_stance="stay_out",
                    overall_conviction=arg.conviction,
                    conflict_detected=True,
                    arguments=arguments,
                    synthesis=f"HARD VETO from {arg.agent_name}: {arg.reasoning}",
                )
            if arg.stance in stances:
                stances[arg.stance] += arg.conviction

        # Check for conflict (e.g. strong bull AND strong bear)
        if stances["bullish"] > 0.5 and stances["bearish"] > 0.5:
            conflict_detected = True

        for stance, score in stances.items():
            if score > highest_conviction:
                highest_conviction = score
                dominant = stance

        # FIX: Calculate conviction as average strength (not consensus ratio)
        # This ensures weak consensus (3 agents @ 0.3) ≠ strong breakout (3 agents @ 0.9)
        dominant_args = [arg for arg in arguments if arg.stance == dominant]
        avg_conviction = sum(arg.conviction for arg in dominant_args) / len(dominant_args) if dominant_args else 0.0

        # Also track consensus ratio for transparency
        total_score = sum(stances.values())
        consensus_ratio = (highest_conviction / total_score) if total_score > 0 else 0.0

        # Use average conviction as overall_conviction (reflects signal strength)
        overall_conviction = avg_conviction

        # FIX: Upgrade to "strongly_*" when both high consensus AND high conviction
        # This unlocks the strong conviction execution path in DualConfirmationEngine
        if dominant in ("bullish", "bearish"):
            if consensus_ratio > 0.80 and avg_conviction > 0.75:
                dominant = f"strongly_{dominant}"
                self.logger.success(
                    f"[DEBATE] 🔥 Strong conviction detected: {dominant} "
                    f"(strength={avg_conviction:.1%}, alignment={consensus_ratio:.1%})"
                )

        synthesis = f"Debate concluded. Consensus is {dominant} ({overall_conviction:.0%} strength, {consensus_ratio:.0%} alignment). Conflict: {conflict_detected}."
        self.logger.info(synthesis)

        return DebateSummary(
            dominant_stance=dominant,
            overall_conviction=overall_conviction,
            conflict_detected=conflict_detected,
            arguments=arguments,
            synthesis=synthesis,
        )
