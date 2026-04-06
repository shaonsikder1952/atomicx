"""God Mode Cognitive Cache - High-Speed Background Data Store.

This cache solves the "Thundering Herd" problem by decoupling API requests
from computation. The CognitiveLoop updates the cache in the background,
and the API serves data instantly from memory.

Performance:
- API latency: < 5ms (was 60,000+ms)
- No blocking compute on request
- Single source of truth
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from loguru import logger


class GodModeCognitiveCache:
    """High-speed in-memory cache for God Mode dashboard data.

    Updated by: CognitiveLoop (background, every cycle)
    Read by: FastAPI endpoints (on-demand, instant)
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Dict[str, datetime] = {}
        logger.info("[GOD-MODE-CACHE] Cognitive cache initialized")

    async def update(
        self,
        symbol: str,
        prediction: Any,
        variables: Dict[str, float],
        swarm_result: Optional[Dict] = None,
        narrative_state: Optional[Dict] = None,
        causal_links: Optional[list] = None,
        all_agents: Optional[list] = None,
    ) -> None:
        """Update cache with latest cognitive cycle data.

        Called by: CognitiveLoop (after each cycle)
        Latency: ~1ms (async lock + dict write)
        """
        async with self._lock:
            # Extract individual agent predictions from all_agents or hierarchy signal
            individual_predictions = {}

            # PRIORITY: Use all_agents from brain_state if available (has all 46+ atomic agents)
            if all_agents:
                logger.info(f"[GOD-MODE-CACHE] Found {len(all_agents)} agents from brain_state")
                for agent in all_agents:
                    agent_name = agent.get("name", "unknown")
                    # Convert dashboard signal format to cache format
                    sig = agent.get("sig", "NEUTRAL")
                    direction = "bullish" if sig == "BULLISH" else ("bearish" if sig == "BEARISH" else "neutral")

                    # Parse confidence from "XX%" format
                    acc_str = agent.get("acc", "0%")
                    try:
                        confidence = float(acc_str.replace("%", "")) / 100.0
                    except (ValueError, AttributeError):
                        confidence = 0.0

                    individual_predictions[agent_name] = {
                        "name": agent_name,
                        "direction": direction,
                        "confidence": confidence,
                        "reasoning": agent.get("logic", ""),
                        "data_source": "REAL"
                    }
            else:
                # FALLBACK: Try to extract from prediction metadata
                try:
                    if hasattr(prediction, 'metadata') and prediction.metadata:
                        hierarchy_raw = prediction.metadata.get("hierarchy_signal_raw")
                        if hierarchy_raw and isinstance(hierarchy_raw, dict):
                            child_signals = hierarchy_raw.get("child_signals", [])
                            logger.debug(f"[GOD-MODE-CACHE] Found {len(child_signals)} child signals from metadata")
                            for child in child_signals:
                                agent_type = child.get("agent_type", "unknown")
                                # Clean up agent type name
                                model_name = agent_type.replace("_agent", "").replace("_", " ").title()
                                individual_predictions[agent_type] = {
                                    "name": model_name,
                                    "direction": child.get("direction", "neutral"),
                                    "confidence": float(child.get("confidence", 0.0)),
                                    "data_source": "REAL"
                                }
                        else:
                            logger.warning("[GOD-MODE-CACHE] hierarchy_signal_raw is None or not a dict, and no all_agents provided")
                except Exception as e:
                    logger.error(f"[GOD-MODE-CACHE] Failed to extract child signals: {e}")

            # Count votes from actual predictions
            bullish_votes = sum(1 for p in individual_predictions.values() if p["direction"] == "bullish")
            bearish_votes = sum(1 for p in individual_predictions.values() if p["direction"] == "bearish")
            neutral_votes = sum(1 for p in individual_predictions.values() if p["direction"] == "neutral")
            total_votes = bullish_votes + bearish_votes + neutral_votes

            # Calculate agreement
            if total_votes > 0:
                max_votes = max(bullish_votes, bearish_votes, neutral_votes)
                agreement = max_votes / total_votes
            else:
                agreement = 0.0

            self._cache[symbol] = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),

                # Ensemble prediction from FusionNode
                "ensemble": {
                    "direction": prediction.direction,
                    "confidence": prediction.confidence,
                    "action": prediction.action.value if hasattr(prediction.action, 'value') else str(prediction.action),
                    "regime": prediction.regime.value if hasattr(prediction.regime, 'value') else str(prediction.regime),
                    "agreement": agreement,
                    "models_bullish": bullish_votes,
                    "models_bearish": bearish_votes,
                    "models_neutral": neutral_votes,
                    "data_source": "REAL"
                },

                # Individual model predictions
                "predictions": {
                    "ensemble": {
                        "name": "Ensemble (Fusion)",
                        "direction": prediction.direction,
                        "confidence": prediction.confidence,
                        "action": prediction.action.value if hasattr(prediction.action, 'value') else str(prediction.action),
                        "regime": prediction.regime.value if hasattr(prediction.regime, 'value') else str(prediction.regime),
                        "data_source": "REAL"
                    },

                    # Add individual agent predictions
                    **individual_predictions,

                    # Swarm simulation
                    "swarm": swarm_result if swarm_result else {
                        "name": "MARL Swarm",
                        "direction": "neutral",
                        "confidence": 0.0,
                        "agent_count": 0,
                        "bullish_agents": 0,
                        "data_source": "UNAVAILABLE"
                    },

                    # Alternative data (narrative tracker)
                    "altdata": {
                        "name": "Alternative Data",
                        "direction": narrative_state.get("social_momentum", "neutral") if narrative_state else "neutral",
                        "confidence": narrative_state.get("social_confidence", 0.0) if narrative_state else 0.0,
                        "sentiment_score": narrative_state.get("sentiment", 0.5) if narrative_state else 0.5,
                        "news_count": narrative_state.get("recent_signals", 0) if narrative_state else 0,
                        "data_source": "REAL" if narrative_state else "UNAVAILABLE"
                    },

                    # Causal links
                    "causal": {
                        "name": "Causal Graph",
                        "direction": "neutral",  # Can derive from causal strength
                        "confidence": 0.0,
                        "causal_links": causal_links if causal_links else [],
                        "link_count": len(causal_links) if causal_links else 0,
                        "data_source": "REAL" if causal_links else "UNAVAILABLE"
                    },
                },

                # Raw variables
                "variables": variables,

                # Current price
                "current_price": prediction.entry_price if prediction.entry_price else variables.get("PRICE", 0.0),

                # Vote breakdown
                "votes": {
                    "bullish": bullish_votes,
                    "neutral": neutral_votes,
                    "bearish": bearish_votes
                },

                # Metadata
                "models_active": len(individual_predictions) + self._count_active_models(swarm_result, narrative_state, causal_links),
                "total_models": len(individual_predictions) + 3,  # individual agents + swarm + altdata + causal
                "cache_updated_at": datetime.now(timezone.utc).isoformat()
            }

            self._last_update[symbol] = datetime.now(timezone.utc)

            logger.info(
                f"[GOD-MODE-CACHE] Updated {symbol}: "
                f"{prediction.direction} @ {prediction.confidence:.2%} "
                f"({self._cache[symbol]['models_active']}/{self._cache[symbol]['total_models']} models)"
            )

    async def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached data for symbol (instant, no compute).

        Called by: API endpoints
        Latency: < 1ms (async lock + dict read)
        """
        async with self._lock:
            data = self._cache.get(symbol)

            if data:
                # Add staleness indicator
                last_update = self._last_update.get(symbol)
                if last_update:
                    staleness_seconds = (datetime.now(timezone.utc) - last_update).total_seconds()
                    data["staleness_seconds"] = staleness_seconds

                    if staleness_seconds > 300:  # 5 minutes
                        logger.warning(f"[GOD-MODE-CACHE] Stale data for {symbol} ({staleness_seconds:.0f}s old)")

                return data

            logger.warning(f"[GOD-MODE-CACHE] No cached data for {symbol}")
            return None

    def _count_active_models(
        self,
        swarm_result: Optional[Dict],
        narrative_state: Optional[Dict],
        causal_links: Optional[list]
    ) -> int:
        """Count how many models returned real data."""
        count = 1  # Ensemble always present
        if swarm_result:
            count += 1
        if narrative_state:
            count += 1
        if causal_links:
            count += 1
        return count

    async def get_all_symbols(self) -> list[str]:
        """Get list of all cached symbols."""
        async with self._lock:
            return list(self._cache.keys())

    async def clear(self):
        """Clear all cached data."""
        async with self._lock:
            self._cache.clear()
            self._last_update.clear()
            logger.info("[GOD-MODE-CACHE] Cache cleared")


# Global singleton instance
_god_mode_cache: Optional[GodModeCognitiveCache] = None


def get_god_mode_cache() -> GodModeCognitiveCache:
    """Get the global God Mode cache instance."""
    global _god_mode_cache
    if _god_mode_cache is None:
        _god_mode_cache = GodModeCognitiveCache()
    return _god_mode_cache
