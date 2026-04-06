"""Future Impact Prediction System — The Crystal Ball.

Chain-reaction prediction: ingest a story, identify related entities,
create a sentiment variable, simulate the cascade via the actual Swarm
simulator, and assess whether to act before the story goes viral.

v2: Uses REAL swarm simulation instead of hardcoded probability heuristics.
"""

from __future__ import annotations

import uuid
from typing import Any
from loguru import logger

from atomicx.intelligence.scanner import NewsItem
from atomicx.intelligence.browser_agent import BrowserAgent
from atomicx.intelligence.knowledge_graph import KnowledgeGraph
from atomicx.swarm import SwarmSimulator


class ImpactPredictor:
    """The 5-step cascade prediction pipeline.
    
    Uses real swarm simulation for cascade probability estimation.
    """
    
    def __init__(self, browser: BrowserAgent, graph: KnowledgeGraph) -> None:
        self.browser = browser
        self.graph = graph
        self.swarm = SwarmSimulator(seed=99)  # Dedicated swarm for impact simulation
        self.logger = logger.bind(module="intelligence.impact")
        self.dynamic_variables: dict[str, dict[str, Any]] = {}
        
    async def predict_impact(self, news_item: NewsItem) -> dict[str, Any]:
        """Execute the full 5-step Crystal Ball pipeline."""
        self.logger.info(f"[CRYSTAL BALL] === Impact Prediction Pipeline START ===")
        self.logger.info(f"[CRYSTAL BALL] Story: '{news_item.title}'")
        
        # Step 1: Ingest Story (already done by NewsScanner)
        
        # Step 2: Context Search — Deep-dive for related people
        research = await self.browser.deep_dive(news_item)
        self.graph.ingest_research(research)
        
        # Step 3: Variable Creation — Create dynamic sentiment variables
        dynamic_vars = []
        for person in research.get("related_people", []):
            name = person.get("name", "unknown")
            var_name = f"{name.replace(' ', '_')}_Sentiment_Momentum"
            impact = self.graph.query_person_impact(name)
            
            self.dynamic_variables[var_name] = {
                "variable_id": f"dyn-{uuid.uuid4().hex[:8]}",
                "person": name,
                "historical_avg_impact": impact.get("avg_market_impact", "0%"),
                "current_sentiment": research.get("sentiment_signal", "neutral"),
                "story_context": news_item.title,
                "data_quality": research.get("data_quality", "none"),
            }
            dynamic_vars.append(var_name)
            self.logger.info(f"[CRYSTAL BALL] Created dynamic variable: {var_name}")
        
        # Step 4: Swarm Projection — Run REAL swarm simulation for cascade estimate
        cascade_prediction = self._simulate_swarm_cascade(research, dynamic_vars)
        
        # Step 5: Decision — Should the Brain act before virality?
        decision = self._make_decision(cascade_prediction, news_item)
        
        self.logger.info(f"[CRYSTAL BALL] === Pipeline COMPLETE === Decision: {decision['action']}")
        
        return {
            "story": news_item.title,
            "dynamic_variables_created": dynamic_vars,
            "cascade_prediction": cascade_prediction,
            "decision": decision,
        }
        
    def _simulate_swarm_cascade(
        self, research: dict[str, Any], dynamic_vars: list[str]
    ) -> dict[str, Any]:
        """Run the actual OASIS Swarm Simulator to estimate cascade probability.
        
        Constructs a synthetic variable snapshot from the research context,
        then runs the swarm to see how agents react to the injected signal.
        """
        sentiment = research.get("sentiment_signal", "neutral")
        data_quality = research.get("data_quality", "none")
        num_people = len(research.get("related_people", []))
        urgency = research.get("urgency", "low")
        
        # Construct a synthetic variable snapshot that represents the news impact
        # These values are derived from the research, not hardcoded
        sentiment_to_momentum = {
            "bullish": 0.03, "dovish": 0.02,
            "bearish": -0.03, "hawkish": -0.02,
            "neutral": 0.0,
        }
        momentum = sentiment_to_momentum.get(sentiment, 0.0)
        
        # Scale momentum based on data quality (more data = higher confidence)
        quality_mult = {"full": 1.0, "partial": 0.7, "headline_only": 0.4, "none": 0.1}
        momentum *= quality_mult.get(data_quality, 0.1)
        
        # Build a minimal variable snapshot for the swarm
        synthetic_vars = {
            "RSI_14": 50 + (momentum * 500),  # Center around 50, perturb by momentum
            "MACD_HISTOGRAM": momentum * 10,
            "REL_VOLUME": 1.5 + (0.5 if urgency == "high" else 0.0),
            "FUNDING_RATE": momentum * 0.01,
        }
        
        # Use a base price of 67000 (approximate BTC — this is just for the sim)
        base_price = 67000.0
        
        # Run the REAL swarm simulation
        tier = "medium" if urgency == "high" else "fast"
        result = self.swarm.simulate(
            current_price=base_price,
            variables=synthetic_vars,
            tier=tier,
            steps=200,  # 200 steps for cascade analysis
        )
        
        # Derive cascade probability from swarm results
        # High consensus strength + high price change = likely cascade
        price_change_pct = abs(result.metadata.get("price_change_pct", 0.0))
        cascade_probability = min(0.95, (
            result.consensus_strength * 0.4 +
            min(price_change_pct / 5.0, 1.0) * 0.3 +
            result.regime_shift_probability * 0.2 +
            (0.1 if num_people >= 2 else 0.0)
        ))
        
        # Estimate impact based on swarm price trajectory
        estimated_impact_pct = price_change_pct
        direction = "+" if result.consensus_direction == "bullish" else "-"
        
        prediction = {
            "cascade_probability": round(cascade_probability, 3),
            "estimated_price_impact_pct": f"{direction}{estimated_impact_pct:.1f}%",
            "time_to_viral_hours": 4 if urgency == "high" else 8,
            "confidence": "high" if cascade_probability >= 0.6 else "medium" if cascade_probability >= 0.3 else "low",
            "swarm_consensus": result.consensus_direction,
            "swarm_agents_used": result.agent_count,
            "simulation_tier": result.simulation_tier,
            "data_quality": data_quality,
        }
        
        self.logger.info(
            f"[CRYSTAL BALL] Swarm Projection ({result.agent_count} agents, {tier}): "
            f"{cascade_probability:.0%} cascade probability, "
            f"est. impact: {prediction['estimated_price_impact_pct']}, "
            f"consensus: {result.consensus_direction}"
        )
        
        return prediction
        
    def _make_decision(self, cascade: dict[str, Any], news_item: NewsItem) -> dict[str, Any]:
        """Decide whether to act before the story goes viral."""
        prob = cascade.get("cascade_probability", 0)
        data_quality = cascade.get("data_quality", "none")
        
        # Require at least partial data quality for fast-path decisions
        if data_quality in ("none", "headline_only") and prob >= 0.7:
            return {
                "action": "ALERT_BRAIN",
                "reason": f"High cascade probability ({prob:.0%}) but low data quality ({data_quality}). "
                          f"Need more research before acting.",
            }
        
        if prob >= 0.7:
            return {
                "action": "FAST_PATH_INTENT",
                "intent_type": "pre_viral_accumulation",
                "conviction": prob,
                "reason": f"High cascade probability ({prob:.0%}) for: {news_item.title}",
                "time_pressure": "act_within_30_minutes",
            }
        elif prob >= 0.5:
            return {
                "action": "ALERT_BRAIN",
                "reason": f"Moderate cascade probability ({prob:.0%}) — Brain should debate",
            }
        else:
            return {
                "action": "MONITOR_ONLY",
                "reason": f"Low cascade probability ({prob:.0%}) — continue watching",
            }
