"""Autonomous Research Agent - Discovers and evaluates new data sources.

This agent can:
1. Search the web for new APIs
2. Read and understand documentation
3. Evaluate API quality, cost, and reliability
4. Decide which APIs to integrate
5. Generate integration plans

NO HUMAN APPROVAL REQUIRED - fully autonomous decision making.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from dataclasses import dataclass

from loguru import logger
import httpx


@dataclass
class APICandidate:
    """A discovered API that could be integrated."""
    name: str
    url: str
    description: str
    asset_types: list[str]  # crypto, stocks, commodities, forex
    pricing: str  # free, freemium, paid
    rate_limit: str  # e.g., "5 calls/min", "unlimited"
    quality_score: float  # 0.0 to 1.0
    documentation_url: str
    integration_complexity: str  # simple, medium, complex
    discovered_at: datetime
    evaluation_notes: str


@dataclass
class IntegrationPlan:
    """Plan for integrating a new API."""
    api_candidate: APICandidate
    connector_name: str  # e.g., "AlphaVantageConnector"
    estimated_effort: str  # hours
    priority: int  # 1-10, 10 = highest
    dependencies: list[str]  # e.g., ["alpha_vantage_api_key"]
    expected_benefits: str
    risks: str


class AutonomousResearchAgent:
    """Autonomous agent that discovers and evaluates new data sources.

    This is the "eyes and ears" of the AGI system - constantly searching
    for better data sources and deciding what to integrate.
    """

    def __init__(self):
        self.logger = logger.bind(module="agi.research")
        self._discovered_apis: list[APICandidate] = []
        self._integration_plans: list[IntegrationPlan] = []
        self._anthropic_api_key = None
        self._web_search_available = False

        # Initialize LLM client for research
        try:
            import anthropic
            import os
            self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if self._anthropic_api_key:
                self._anthropic = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)
                self.logger.success("[AGI] Claude API initialized for autonomous research")
            else:
                self.logger.warning("[AGI] No ANTHROPIC_API_KEY - research capability limited")
                self._anthropic = None
        except ImportError:
            self.logger.warning("[AGI] anthropic package not installed")
            self._anthropic = None

    async def discover_apis_for_asset(self, asset_symbol: str, asset_type: str) -> list[APICandidate]:
        """Autonomously discover APIs that can provide data for this asset.

        Uses LLM to:
        1. Determine what kind of data is needed
        2. Research available APIs
        3. Evaluate quality and cost
        4. Return ranked candidates

        Args:
            asset_symbol: e.g., "AAPL", "BTC/USDT", "EURUSD"
            asset_type: crypto, stock, commodity, forex

        Returns:
            List of API candidates ranked by quality score
        """
        self.logger.info(f"[AGI-RESEARCH] Discovering APIs for {asset_symbol} ({asset_type})")

        if not self._anthropic:
            self.logger.warning("[AGI] Cannot discover APIs without Claude API")
            return []

        # Ask Claude to research data sources
        research_prompt = f"""You are an autonomous AI agent researching financial data APIs.

TASK: Find the BEST data APIs for this asset:
- Symbol: {asset_symbol}
- Type: {asset_type}

REQUIREMENTS:
1. Find 3-5 APIs that provide real-time and historical data
2. Evaluate: Cost (free/paid), Rate limits, Data quality, Reliability
3. Provide: API name, URL, documentation link, pricing details
4. Rank by quality score (0.0-1.0)

OUTPUT FORMAT (JSON):
{{
  "apis": [
    {{
      "name": "API Name",
      "url": "https://...",
      "description": "What it provides",
      "asset_types": ["crypto", "stocks"],
      "pricing": "free/freemium/paid - details",
      "rate_limit": "X calls per minute/day",
      "quality_score": 0.85,
      "documentation_url": "https://docs...",
      "integration_complexity": "simple/medium/complex",
      "evaluation_notes": "Why this is good/bad"
    }}
  ]
}}

Research real APIs that exist today. Be thorough."""

        try:
            response = await self._anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": research_prompt}]
            )

            # Extract JSON from response
            response_text = response.content[0].text

            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)

                # Convert to APICandidate objects
                candidates = []
                for api_data in data.get("apis", []):
                    candidate = APICandidate(
                        name=api_data["name"],
                        url=api_data["url"],
                        description=api_data["description"],
                        asset_types=api_data["asset_types"],
                        pricing=api_data["pricing"],
                        rate_limit=api_data["rate_limit"],
                        quality_score=float(api_data["quality_score"]),
                        documentation_url=api_data["documentation_url"],
                        integration_complexity=api_data["integration_complexity"],
                        discovered_at=datetime.now(timezone.utc),
                        evaluation_notes=api_data["evaluation_notes"]
                    )
                    candidates.append(candidate)
                    self._discovered_apis.append(candidate)

                    self.logger.success(
                        f"[AGI-RESEARCH] Discovered: {candidate.name} "
                        f"(quality: {candidate.quality_score:.2f}, {candidate.pricing})"
                    )

                # Sort by quality score
                candidates.sort(key=lambda x: x.quality_score, reverse=True)
                return candidates

        except Exception as e:
            self.logger.error(f"[AGI-RESEARCH] API discovery failed: {e}")
            return []

    async def generate_integration_plan(self, api_candidate: APICandidate) -> IntegrationPlan:
        """Generate a detailed plan for integrating this API.

        Uses LLM to analyze the API and create implementation plan.
        """
        self.logger.info(f"[AGI-RESEARCH] Generating integration plan for {api_candidate.name}")

        if not self._anthropic:
            # Fallback: generate basic plan without LLM
            return IntegrationPlan(
                api_candidate=api_candidate,
                connector_name=f"{api_candidate.name.replace(' ', '')}Connector",
                estimated_effort="4-8 hours",
                priority=5,
                dependencies=["api_key"],
                expected_benefits="Additional data source",
                risks="Unknown API stability"
            )

        planning_prompt = f"""You are an autonomous AI agent planning API integration.

API TO INTEGRATE:
- Name: {api_candidate.name}
- URL: {api_candidate.url}
- Documentation: {api_candidate.documentation_url}
- Pricing: {api_candidate.pricing}
- Rate Limit: {api_candidate.rate_limit}
- Complexity: {api_candidate.integration_complexity}
- Quality Score: {api_candidate.quality_score}

TASK: Create integration plan with:
1. Python class name for connector (e.g., "AlphaVantageConnector")
2. Estimated development time
3. Priority (1-10, where 10 = critical, 1 = nice-to-have)
4. Required dependencies (API keys, packages, etc.)
5. Expected benefits (what this enables)
6. Potential risks (rate limits, cost, reliability issues)

OUTPUT FORMAT (JSON):
{{
  "connector_name": "ClassName",
  "estimated_effort": "X hours",
  "priority": 8,
  "dependencies": ["package_name", "api_key"],
  "expected_benefits": "Detailed benefits...",
  "risks": "Potential issues..."
}}

Be realistic and thorough."""

        try:
            response = await self._anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": planning_prompt}]
            )

            response_text = response.content[0].text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)

                plan = IntegrationPlan(
                    api_candidate=api_candidate,
                    connector_name=data["connector_name"],
                    estimated_effort=data["estimated_effort"],
                    priority=int(data["priority"]),
                    dependencies=data["dependencies"],
                    expected_benefits=data["expected_benefits"],
                    risks=data["risks"]
                )

                self._integration_plans.append(plan)

                self.logger.success(
                    f"[AGI-RESEARCH] Integration plan created: {plan.connector_name} "
                    f"(priority: {plan.priority}/10, effort: {plan.estimated_effort})"
                )

                return plan

        except Exception as e:
            self.logger.error(f"[AGI-RESEARCH] Plan generation failed: {e}")
            raise

    async def autonomous_research_cycle(self, needed_asset_types: list[str]) -> list[IntegrationPlan]:
        """Run a complete autonomous research cycle.

        Discovers APIs, evaluates them, creates integration plans,
        and returns ranked recommendations.

        Args:
            needed_asset_types: Types of data needed (crypto, stocks, etc.)

        Returns:
            List of integration plans ranked by priority
        """
        self.logger.info(
            f"[AGI-RESEARCH] Starting autonomous research cycle for: {needed_asset_types}"
        )

        all_plans = []

        for asset_type in needed_asset_types:
            # Discover APIs for this asset type
            candidates = await self.discover_apis_for_asset(
                asset_symbol=f"EXAMPLE_{asset_type.upper()}",
                asset_type=asset_type
            )

            # Generate integration plans for top candidates
            for candidate in candidates[:2]:  # Top 2 per asset type
                if candidate.quality_score >= 0.7:  # Only high-quality APIs
                    try:
                        plan = await self.generate_integration_plan(candidate)
                        all_plans.append(plan)
                    except Exception as e:
                        self.logger.error(f"Failed to plan {candidate.name}: {e}")

        # Sort plans by priority
        all_plans.sort(key=lambda x: x.priority, reverse=True)

        self.logger.success(
            f"[AGI-RESEARCH] Research cycle complete. "
            f"Found {len(all_plans)} integration opportunities."
        )

        return all_plans

    def get_discovered_apis(self) -> list[APICandidate]:
        """Get all discovered APIs."""
        return self._discovered_apis

    def get_integration_plans(self) -> list[IntegrationPlan]:
        """Get all integration plans."""
        return self._integration_plans


# Global singleton
_research_agent: AutonomousResearchAgent | None = None


def get_research_agent() -> AutonomousResearchAgent:
    """Get the global research agent instance."""
    global _research_agent
    if _research_agent is None:
        _research_agent = AutonomousResearchAgent()
    return _research_agent
