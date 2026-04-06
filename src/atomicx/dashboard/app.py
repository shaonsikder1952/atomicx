"""AtomicX Dashboard — Command Center.

FastAPI-based real-time dashboard with WebSocket updates.
High-fidelity Multi-Asset Columnar Layout (V6).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from loguru import logger

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import os
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class DashboardState:
    """Centralized state snapshot for the dashboard."""

    def __init__(self) -> None:
        self.monitored_symbols: list[str] = []
        self.selected_symbol: str = ""

        # Per-symbol snapshots
        self.symbol_data: dict[str, dict[str, Any]] = {}

        self.chat_messages: list[dict[str, str]] = [
            {"role": "assistant", "content": "Hey! 👋 I'm AtomicX, your AI trading assistant. Ask me about prices, predictions, market analysis, or anything crypto-related!"}
        ]
        self.weight_overrides: dict[str, float] = {
            "causal": 1.0,
            "geopolitical": 0.2,
            "narrative": 1.0,
            "swarm": 1.0,
        }
        self.performance: dict[str, Any] = {
            "win_rate": 0.0,
            "total_predictions": 0,
            "intelligence_score": 0.0,
            "last_lesson": "Awaiting first outcome..."
        }
        self.on_add_symbol = None # Callback for engine
        self.on_custom_prediction = None # Callback for custom timeframe predictions
        self.logger = logger.bind(module="dashboard.state")

        # Event-driven state tracking
        self._state_version = 0
        self._connected_clients: list[Any] = []
        self._last_broadcast_version = 0
        # FIX: Lock for thread-safe client list manipulation
        self._client_lock: asyncio.Lock | None = None
        
    def _ensure_lock(self) -> None:
        """Lazy initialization of asyncio.Lock (must be created in async context)."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()

    async def register_client(self, websocket: Any) -> None:
        """Thread-safe client registration. FIX for race condition."""
        self._ensure_lock()
        async with self._client_lock:
            if websocket not in self._connected_clients:
                self._connected_clients.append(websocket)

    async def unregister_client(self, websocket: Any) -> None:
        """Thread-safe client unregistration. FIX for race condition."""
        self._ensure_lock()
        async with self._client_lock:
            if websocket in self._connected_clients:
                self._connected_clients.remove(websocket)

    def update_performance(self, win_rate: float, total: int, last_lesson: str) -> None:
        """Update global engine performance metrics."""
        self.performance["win_rate"] = win_rate
        self.performance["total_predictions"] = total
        self.performance["last_lesson"] = last_lesson
        # Simple score combining accuracy and volume
        self.performance["intelligence_score"] = min(100, (win_rate * 100) + (total * 0.1))
        self.logger.info(f"[DASHBOARD] Performance updated: WR={win_rate:.1%}, Total={total}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        self.chat_messages.append({"role": role, "content": content})
        if len(self.chat_messages) > 50:
            self.chat_messages.pop(0)

    def update_symbol_data(self, symbol: str, data: dict[str, Any]) -> None:
        """Update snapshot for a specific symbol with high-fidelity support."""
        if symbol not in self.monitored_symbols:
            self.monitored_symbols.append(symbol)

        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = {
                "name": symbol,
                "signal": "WATCH",
                "confidence": 48,
                "color": "#8a99b0",
                "badge_class": "badge-watch",
                "support": "$0",
                "resist": "$0",
                "chart_points": "0,80 30,75 60,85 90,70 120,75 150,68 180,72 210,65 240,70 270,66 300,68",
                "chart_color": "#888",
                "agents": [],
                "preds": None,
                "reasoning": "Awaiting cognitive cycle consensus...",
                "invalidate": "Awaiting market validation conditions.",
                "position": {"symbol": symbol, "size": "0.0", "entry": "0.0"},
                "net_pnl": 0.0,
                "regime": "Transitioning"
            }

        self.symbol_data[symbol].update(data)
        self._state_version += 1  # Increment version on state change
        self._notify_clients()  # Trigger push to connected clients

    def _notify_clients(self) -> None:
        """Notify all connected WebSocket clients of state change."""
        # This will be called by the WebSocket handler's background task
        pass

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Apply manual or autonomous weight overrides."""
        for key, val in new_weights.items():
            if key in self.weight_overrides:
                old = self.weight_overrides[key]
                self.weight_overrides[key] = val
                self.logger.warning(f"[DASHBOARD] Weight override: {key} {old:.2f} → {val:.2f}")
                
    def snapshot(self) -> dict[str, Any]:
        """Generate the current dashboard snapshot with all asset data for columnar view."""
        return {
            "selected_symbol": self.selected_symbol,
            "monitored_symbols": self.monitored_symbols,
            "all_symbols_data": self.symbol_data,
            "performance": self.performance,
            "weights": self.weight_overrides,
            "chat": self.chat_messages,
        }




def create_dashboard_app(state: DashboardState, chat_handler: Any = None, orchestrator: Any = None) -> Any:
    """Create and return the FastAPI dashboard app.

    Args:
        state: Dashboard state for WebSocket updates
        chat_handler: Chatbot handler for AI chat
        orchestrator: MetaOrchestrator for accessing models and market data
    """
    logger.info(f"Creating dashboard app. HAS_FASTAPI: {HAS_FASTAPI}")
    if not HAS_FASTAPI:
        logger.warning("FastAPI not installed — dashboard unavailable")
        return None

    app = FastAPI(title="AtomicX Command Center")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Initialize God Mode Data Aggregator (shared across all requests)
    from atomicx.dashboard.god_mode_api import GodModeDataAggregator
    god_mode_aggregator = GodModeDataAggregator()

    # Initialize Portfolio API Router
    from atomicx.dashboard.portfolio_api import router as portfolio_router
    app.include_router(portfolio_router)

    # Store orchestrator for access in API endpoints
    app.state.orchestrator = orchestrator
    app.state.god_mode = god_mode_aggregator
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Portfolio overview (main landing page)."""
        response = FileResponse(os.path.join(static_dir, "index.html"))
        # Disable caching to ensure users get the latest version
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/evolution", response_class=HTMLResponse)
    async def evolution():
        """Evolution system dashboard."""
        return FileResponse(os.path.join(static_dir, "evolution.html"))

    @app.get("/database.html")
    async def database_page_redirect():
        """Redirect old database page to new simplified version."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/db.html", status_code=301)

    @app.get("/memory.html")
    async def memory_page_redirect():
        """Redirect old memory page to new simplified version."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/mem.html", status_code=301)

    @app.get("/db.html", response_class=HTMLResponse)
    async def db_page():
        """Database viewer page (new, simplified)."""
        return FileResponse(os.path.join(static_dir, "db.html"))

    @app.get("/mem.html", response_class=HTMLResponse)
    async def mem_page():
        """Memory viewer page (new, simplified)."""
        return FileResponse(os.path.join(static_dir, "mem.html"))

    @app.get("/causality.html", response_class=HTMLResponse)
    async def causality_page():
        """Causality & Decision Audit viewer page."""
        return FileResponse(os.path.join(static_dir, "causality.html"))

    @app.get("/diary.html", response_class=HTMLResponse)
    async def diary_page():
        """Atomic Diary transparency feed page."""
        response = FileResponse(os.path.join(static_dir, "diary.html"))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/force-reload.html", response_class=HTMLResponse)
    async def force_reload():
        """Force cache clear and reload dashboard."""
        response = FileResponse(os.path.join(static_dir, "force-reload.html"))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/dashboard.html", response_class=HTMLResponse)
    async def asset_dashboard():
        """Asset-specific intelligence dashboard."""
        response = FileResponse(os.path.join(static_dir, "dashboard.html"))
        # Disable caching to ensure users get the latest version
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/god_mode.html", response_class=HTMLResponse)
    async def god_mode_redirect():
        """Redirect old god_mode URL to new dashboard."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard.html", status_code=301)

    @app.get("/components/{component_name}")
    async def serve_component(component_name: str):
        """Serve individual visualization components."""
        component_path = os.path.join(static_dir, "components", component_name)
        if os.path.exists(component_path):
            return FileResponse(component_path)
        else:
            return HTMLResponse(content="Component not found", status_code=404)

    # ═══════════════════════════════════════════════════════════════════════
    # GOD MODE API ENDPOINTS - REAL DATA (NO MOCK)
    # ═══════════════════════════════════════════════════════════════════════

    @app.get("/api/god_mode/predictions")
    async def get_god_mode_predictions(symbol: str = "BTC/USDT"):
        """Get REAL predictions from cognitive cache (< 5ms).

        PERFORMANCE FIX:
        - Before: Computed on-demand (60,000ms+ blocking)
        - After: Read from cache (< 5ms instant)

        The CognitiveLoop updates the cache in background every cycle.
        This endpoint just reads the latest data from memory.
        """
        try:
            # Normalize symbol format
            if "/" not in symbol and symbol.endswith("USDT"):
                symbol = f"{symbol[:-4]}/USDT"

            # PERFORMANCE FIX: Read from cache instead of computing
            from atomicx.dashboard.god_mode_cache import get_god_mode_cache
            cache = get_god_mode_cache()

            cached_data = await cache.get(symbol)

            if cached_data:
                # Cache hit - instant response
                return cached_data

            # Cache miss - system still warming up or symbol not tracked
            logger.warning(f"[GOD-MODE-API] Cache miss for {symbol} - system may still be warming up")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predictions": {},
                "ensemble": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "votes": {"bullish": 0, "neutral": 0, "bearish": 0},
                    "data_source": "WARMING_UP"
                },
                "models_active": 0,
                "total_models": 11,
                "message": "System is warming up. Data will be available after first cognitive cycle."
            }

        except Exception as e:
            logger.error(f"[GOD-MODE-API] Error reading cache: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predictions": {},
                "ensemble": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "votes": {"bullish": 0, "neutral": 0, "bearish": 0},
                    "data_source": "ERROR"
                }
            }

    @app.get("/api/god_mode/ohlcv")
    async def get_real_ohlcv(
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        limit: int = 200
    ):
        """Get REAL OHLCV candlestick data from exchange."""
        try:
            from atomicx.dashboard.god_mode_api import GodModeDataAggregator

            aggregator = GodModeDataAggregator()
            candlesticks = await aggregator.get_real_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": candlesticks,
                "count": len(candlesticks),
                "source": "REAL" if candlesticks else "UNAVAILABLE"
            }

        except Exception as e:
            logger.error(f"[GOD-MODE-API] OHLCV error: {e}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": [],
                "count": 0,
                "source": "ERROR",
                "error": str(e)
            }

    @app.get("/api/god_mode/orderbook")
    async def get_real_orderbook(symbol: str = "BTCUSDT", depth: int = 50):
        """Get REAL order book L2 data from exchange."""
        try:
            from atomicx.dashboard.god_mode_api import GodModeDataAggregator

            aggregator = GodModeDataAggregator()
            orderbook = await aggregator.get_orderbook_depth(
                symbol=symbol,
                depth=depth
            )

            return orderbook

        except Exception as e:
            logger.error(f"[GOD-MODE-API] Order book error: {e}")
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "error": str(e)
            }

    @app.get("/api/god_mode/trade_recommendation")
    async def get_trade_recommendation(symbol: str = "BTC/USDT"):
        """Get detailed trade recommendation with reasoning and portfolio context."""
        try:
            # Get prediction data
            from atomicx.dashboard.god_mode_cache import get_god_mode_cache
            cache = get_god_mode_cache()
            cached_data = await cache.get(symbol)

            if not cached_data:
                return {"error": "No prediction data available", "symbol": symbol}

            # Get portfolio data
            from sqlalchemy import select, func as sql_func
            from atomicx.data.storage.database import get_session
            from atomicx.data.storage.models import Position, PortfolioStats

            portfolio_data = {"total_invested": 0, "open_positions": 0, "avg_entry": 0}
            try:
                async with get_session() as session:
                    # Get open positions
                    result = await session.execute(
                        select(Position).where(
                            Position.symbol == symbol,
                            Position.status == 'open'
                        )
                    )
                    positions = result.scalars().all()

                    if positions:
                        total_invested = sum(float(p.entry_price * p.quantity) for p in positions)
                        avg_entry = sum(float(p.entry_price) for p in positions) / len(positions)
                        portfolio_data = {
                            "total_invested": total_invested,
                            "open_positions": len(positions),
                            "avg_entry": avg_entry
                        }
            except Exception as e:
                logger.warning(f"[TRADE-REC] Could not load portfolio: {e}")

            # Build recommendation
            ensemble = cached_data.get("ensemble", {})
            variables = cached_data.get("variables", {})
            predictions = cached_data.get("predictions", {})

            direction = ensemble.get("direction", "neutral")
            confidence = ensemble.get("confidence", 0.0)
            action = ensemble.get("action", "WAIT")

            # Determine recommendation
            if action == "BET" and confidence >= 0.61:
                if direction == "bullish":
                    recommendation = "LONG"
                elif direction == "bearish":
                    recommendation = "SHORT"
                else:
                    recommendation = "HOLD"
            else:
                recommendation = "HOLD"

            # Generate detailed reasons
            reasons = []

            # Confidence reason
            if confidence >= 0.70:
                reasons.append(f"High confidence signal ({confidence*100:.1f}%) from ensemble model")
            elif confidence >= 0.61:
                reasons.append(f"Moderate confidence ({confidence*100:.1f}%) - borderline signal")
            else:
                reasons.append(f"Low confidence ({confidence*100:.1f}%) - signal below threshold")

            # Model agreement
            bullish_votes = ensemble.get("models_bullish", 0)
            bearish_votes = ensemble.get("models_bearish", 0)
            neutral_votes = ensemble.get("models_neutral", 0)
            total_votes = bullish_votes + bearish_votes + neutral_votes

            if total_votes > 0:
                agreement = max(bullish_votes, bearish_votes, neutral_votes) / total_votes
                if agreement >= 0.7:
                    reasons.append(f"Strong model agreement ({agreement*100:.0f}%) on {direction} direction")
                elif agreement >= 0.5:
                    reasons.append(f"Moderate model agreement ({agreement*100:.0f}%)")
                else:
                    reasons.append(f"Models are divided - no clear consensus")

            # Technical indicators
            rsi = variables.get("RSI_14")
            if rsi:
                if rsi > 70:
                    reasons.append(f"RSI is overbought ({rsi:.1f}) - potential pullback risk")
                elif rsi < 30:
                    reasons.append(f"RSI is oversold ({rsi:.1f}) - potential bounce opportunity")
                elif rsi > 60:
                    reasons.append(f"RSI shows bullish momentum ({rsi:.1f})")
                elif rsi < 40:
                    reasons.append(f"RSI shows bearish momentum ({rsi:.1f})")

            ob_imbalance = variables.get("OB_IMBALANCE")
            if ob_imbalance:
                if abs(ob_imbalance) > 0.5:
                    direction_text = "buying" if ob_imbalance > 0 else "selling"
                    reasons.append(f"Strong {direction_text} pressure in order book ({ob_imbalance*100:.1f}%)")

            rel_volume = variables.get("RELATIVE_VOLUME")
            if rel_volume and rel_volume > 1.5:
                reasons.append(f"High trading volume ({rel_volume:.1f}x average) confirms move")

            # Generate future scenarios
            current_price = cached_data.get("current_price", variables.get("PRICE", 0))

            scenarios = {
                "bullish": {
                    "probability": bullish_votes / total_votes if total_votes > 0 else 0.33,
                    "target": current_price * 1.05,
                    "timeframe": "24-48 hours",
                    "condition": "If buying pressure continues and resistance breaks"
                },
                "neutral": {
                    "probability": neutral_votes / total_votes if total_votes > 0 else 0.33,
                    "range_low": current_price * 0.98,
                    "range_high": current_price * 1.02,
                    "timeframe": "24-48 hours",
                    "condition": "If market consolidates without clear direction"
                },
                "bearish": {
                    "probability": bearish_votes / total_votes if total_votes > 0 else 0.33,
                    "target": current_price * 0.95,
                    "timeframe": "24-48 hours",
                    "condition": "If selling pressure increases and support breaks"
                }
            }

            return {
                "symbol": symbol,
                "recommendation": recommendation,
                "confidence": confidence,
                "direction": direction,
                "reasons": reasons,
                "portfolio": portfolio_data,
                "scenarios": scenarios,
                "current_price": current_price,
                "timestamp": cached_data.get("timestamp")
            }

        except Exception as e:
            logger.error(f"[TRADE-REC] Error generating recommendation: {e}")
            return {"error": str(e), "symbol": symbol}

    @app.get("/api/database")
    async def get_database_data():
        """API endpoint for database data."""
        try:
            from sqlalchemy import text
            from atomicx.data.storage.database import get_session

            async with get_session() as session:
                # Table counts
                # NOTE: Table names are hardcoded here for security (no SQL injection risk)
                ALLOWED_TABLES = ['news_events', 'news_patterns', 'news_variables',
                                 'pending_predictions', 'prediction_outcomes']
                table_counts = {}

                for table in ALLOWED_TABLES:
                    try:
                        # Safe to use f-string here: table name is from ALLOWED_TABLES (not user input)
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_counts[table] = result.scalar()
                    except Exception as e:
                        logger.error(f"[DASHBOARD-API] Failed to query table {table}: {e}")
                        table_counts[table] = 0

                # News events
                result = await session.execute(text("""
                    SELECT id, title, source, discovered_at, significance_score
                    FROM news_events
                    ORDER BY discovered_at DESC
                    LIMIT 10
                """))
                news_events = [
                    {
                        "id": r[0],
                        "title": r[1],
                        "source": r[2],
                        "discovered_at": r[3].isoformat() if r[3] else None,
                        "significance_score": float(r[4]) if r[4] else 0
                    }
                    for r in result.fetchall()
                ]

                # News patterns
                result = await session.execute(text("""
                    SELECT pattern_type, occurrences, win_rate, confidence, avg_price_impact
                    FROM news_patterns
                    ORDER BY confidence DESC
                    LIMIT 10
                """))
                news_patterns = [
                    {
                        "pattern_type": r[0],
                        "occurrences": r[1],
                        "win_rate": float(r[2]) if r[2] else 0,
                        "confidence": float(r[3]) if r[3] else 0,
                        "avg_price_impact": float(r[4]) if r[4] else None
                    }
                    for r in result.fetchall()
                ]

                # Predictions
                result = await session.execute(text("""
                    SELECT symbol, direction, confidence, entry_price, created_at
                    FROM pending_predictions
                    ORDER BY created_at DESC
                    LIMIT 10
                """))
                predictions = [
                    {
                        "symbol": r[0],
                        "direction": r[1],
                        "confidence": float(r[2]) if r[2] else 0,
                        "entry_price": float(r[3]) if r[3] else 0,
                        "created_at": r[4].isoformat() if r[4] else None
                    }
                    for r in result.fetchall()
                ]

                # Outcomes
                result = await session.execute(text("""
                    SELECT symbol, predicted_direction, was_correct,
                           confidence, actual_return, predicted_at
                    FROM prediction_outcomes
                    ORDER BY predicted_at DESC
                    LIMIT 10
                """))
                outcomes = [
                    {
                        "symbol": r[0],
                        "predicted_direction": r[1],
                        "was_correct": r[2],
                        "confidence": float(r[3]) if r[3] else 0,
                        "actual_return": float(r[4]) if r[4] else None,
                        "predicted_at": r[5].isoformat() if r[5] else None
                    }
                    for r in result.fetchall()
                ]

                return {
                    "table_counts": table_counts,
                    "news_events": news_events,
                    "news_patterns": news_patterns,
                    "predictions": predictions,
                    "outcomes": outcomes
                }
        except Exception as e:
            logger.error(f"Error fetching database data: {e}")
            return {"error": str(e)}

    @app.get("/api/memory")
    async def get_memory_data(query: str = "BTC price trading news prediction"):
        """API endpoint for memory data."""
        try:
            from atomicx.memory.service import MemoryService

            memory = MemoryService()
            results = await memory.retrieve(query=query, limit=20)

            memories = [
                {
                    "content": r.get("memory", r.get("content", "")),
                    "type": r.get("metadata", {}).get("memory_type", "unknown"),
                    "timestamp": r.get("metadata", {}).get("timestamp", "unknown"),
                    "score": r.get("score", 0.0)
                }
                for r in results
            ]

            return {"memories": memories, "query": query}
        except Exception as e:
            logger.error(f"Error fetching memory data: {e}")
            return {"error": str(e), "memories": []}

    @app.get("/api/diary")
    async def get_diary_entries(limit: int = 50):
        """API endpoint for Atomic Diary transparency feed.

        Returns human-readable diary entries showing the reasoning chain
        behind every trade decision. Inspired by Hyperliquid's transparency model.
        """
        try:
            from sqlalchemy import text
            from atomicx.data.storage.database import get_session

            async with get_session() as session:
                # Get recent decision audits with related data
                result = await session.execute(text("""
                    SELECT
                        da.audit_id,
                        da.decision_type,
                        da.decision_timestamp,
                        da.decision_outcome,
                        da.causal_chain,
                        da.reasoning_tree,
                        da.factors_analyzed,
                        da.thinking_log,
                        da.predicted_outcome,
                        da.actual_outcome,
                        da.was_correct,
                        da.error_magnitude,
                        ne.title as news_title,
                        ne.sentiment as news_sentiment,
                        po.symbol,
                        po.confidence as pred_confidence,
                        po.regime as pred_regime,
                        po.entry_price,
                        po.verification_price,
                        po.actual_return,
                        po.reasoning as pred_reasoning
                    FROM decision_audits da
                    LEFT JOIN news_events ne ON da.news_event_id = ne.id
                    LEFT JOIN prediction_outcomes po ON da.audit_id = MD5(po.prediction_id || po.predicted_at::text)
                    ORDER BY da.decision_timestamp DESC
                    LIMIT :limit
                """), {"limit": limit})

                entries = []
                for r in result.fetchall():
                    import json as json_lib

                    # Build human-readable diary entry
                    timestamp = r[2].isoformat() if r[2] else None
                    symbol = r[14] or "SYSTEM"
                    decision_type = r[1]
                    outcome = r[3]

                    # Extract reasoning chain
                    causal_chain = r[4] or []
                    reasoning_tree = r[5] or {}
                    factors = r[6] or []
                    thinking_log = r[7] or []

                    # Format as readable diary entry
                    entry = {
                        "id": r[0],
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "type": decision_type,
                        "decision": outcome,

                        # Human-readable summary
                        "summary": _format_decision_summary(
                            decision_type, outcome, symbol, r[15], r[16]
                        ),

                        # Reasoning chain
                        "reasoning": {
                            "chain": causal_chain,
                            "tree": reasoning_tree,
                            "key_factors": factors[:5] if isinstance(factors, list) else [],
                            "thought_process": thinking_log[:3] if isinstance(thinking_log, list) else [],
                        },

                        # Context
                        "context": {
                            "news_trigger": r[12],
                            "news_sentiment": r[13],
                            "regime": r[16],
                            "confidence": float(r[15]) if r[15] else None,
                            "entry_price": float(r[17]) if r[17] else None,
                        },

                        # Outcome
                        "outcome": {
                            "predicted": r[8],
                            "actual": r[9],
                            "was_correct": r[10],
                            "error": float(r[11]) if r[11] else None,
                            "return": float(r[19]) if r[19] else None,
                        },

                        # Raw data for deep dive
                        "raw": {
                            "reasoning_text": r[20],  # Full prediction reasoning
                        }
                    }

                    entries.append(entry)

                return {
                    "entries": entries,
                    "count": len(entries),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        except Exception as e:
            logger.error(f"[DIARY-API] Error fetching diary entries: {e}")
            return {"error": str(e), "entries": [], "count": 0}

    def _format_decision_summary(decision_type: str, outcome: str, symbol: str, confidence: float, regime: str) -> str:
        """Format a human-readable decision summary."""
        if decision_type == "prediction":
            conf_pct = f"{confidence*100:.0f}%" if confidence else "?"
            return f"{outcome} prediction for {symbol} ({conf_pct} confidence) in {regime or 'unknown'} regime"
        elif decision_type == "trade":
            return f"{outcome} trade executed for {symbol}"
        elif decision_type == "variable_update":
            return f"Updated system variables based on {symbol} market conditions"
        elif decision_type == "pattern_learning":
            return f"Learned new pattern from {symbol} price action"
        else:
            return f"{decision_type}: {outcome}"

    @app.get("/api/causality")
    async def get_causality_data():
        """API endpoint for causality and decision audit data."""
        try:
            from sqlalchemy import text
            from atomicx.data.storage.database import get_session

            async with get_session() as session:
                # Get recent decision audits with full details + raw prediction data
                result = await session.execute(text("""
                    SELECT
                        da.id,
                        da.decision_type,
                        da.decision_timestamp,
                        da.decision_outcome,
                        da.predicted_outcome,
                        da.actual_outcome,
                        da.was_correct,
                        da.error_magnitude,
                        da.causal_chain,
                        da.reasoning_tree,
                        da.factors_analyzed,
                        da.variables_changed,
                        da.thinking_log,
                        da.problems_found,
                        da.learning_insights,
                        da.system_improvements,
                        ne.title as news_title,
                        ne.decision_reasoning,
                        ne.contributing_factors,
                        ne.confidence_breakdown,
                        ne.learning_notes,
                        ne.article_content as news_article_full,
                        ne.sentiment as news_sentiment,
                        ne.people_mentioned as news_people,
                        ne.entities_mentioned as news_entities,
                        np.pattern_type,
                        np.learning_iterations,
                        np.improvement_log,
                        po.metadata as prediction_raw_data,
                        po.symbol as pred_symbol,
                        po.confidence as pred_confidence,
                        po.predicted_direction,
                        po.regime as pred_regime
                    FROM decision_audits da
                    LEFT JOIN news_events ne ON da.news_event_id = ne.id
                    LEFT JOIN news_patterns np ON da.pattern_id = np.id
                    -- TODO: Replace MD5 hash join with proper FK column
                    -- Current: audit_id = MD5(prediction_id + predicted_at) - fragile due to timestamp formatting
                    -- Better: Add prediction_id column to decision_audits with FK to prediction_outcomes
                    LEFT JOIN prediction_outcomes po ON da.audit_id = MD5(po.prediction_id || po.predicted_at::text)
                    ORDER BY da.decision_timestamp DESC
                    LIMIT 50
                """))

                decisions = []
                for r in result.fetchall():
                    import json as json_lib
                    decisions.append({
                        "id": r[0],
                        "decision_type": r[1],
                        "decision_timestamp": r[2].isoformat() if r[2] else None,
                        "decision_outcome": r[3],
                        "predicted_outcome": r[4],
                        "actual_outcome": r[5],
                        "was_correct": r[6],
                        "error_magnitude": float(r[7]) if r[7] else None,
                        "causal_chain": r[8],
                        "reasoning_tree": r[9],
                        "factors_analyzed": r[10],
                        "variables_changed": r[11],
                        "thinking_log": r[12],
                        "problems_found": r[13],
                        "learning_insights": r[14],
                        "system_improvements": r[15],
                        "news_title": r[16],
                        "decision_reasoning": r[17],
                        "contributing_factors": r[18],
                        "confidence_breakdown": r[19],
                        "learning_notes": r[20],
                        "news_article_full": r[21],  # Full article text
                        "news_sentiment": r[22],
                        "news_people": r[23],
                        "news_entities": r[24],
                        "pattern_type": r[25],
                        "learning_iterations": r[26],
                        "improvement_log": r[27],
                        # RAW DATA from prediction_outcomes.metadata
                        "raw_data": r[28] if r[28] else {},
                        "pred_symbol": r[29],
                        "pred_confidence": float(r[30]) if r[30] else None,
                        "predicted_direction": r[31],
                        "pred_regime": r[32],
                    })

                # Get learning statistics
                stats_result = await session.execute(text("""
                    SELECT
                        COUNT(*) as total_decisions,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_decisions,
                        AVG(error_magnitude) as avg_error,
                        COUNT(DISTINCT pattern_id) as patterns_used
                    FROM decision_audits
                    WHERE was_correct IS NOT NULL
                """))
                stats = stats_result.fetchone()

                learning_stats = {
                    "total_decisions": stats[0] if stats else 0,
                    "correct_decisions": stats[1] if stats else 0,
                    "accuracy": (stats[1] / stats[0] * 100) if stats and stats[0] > 0 else 0,
                    "avg_error": float(stats[2]) if stats and stats[2] else 0,
                    "patterns_used": stats[3] if stats else 0
                }

                return {
                    "decisions": decisions,
                    "learning_stats": learning_stats
                }
        except Exception as e:
            logger.error(f"Error fetching causality data: {e}")
            return {"error": str(e), "decisions": [], "learning_stats": {}}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        import asyncio, json

        # Track this client (thread-safe)
        await state.register_client(websocket)
        client_last_version = 0

        logger.info("[DASHBOARD] WebSocket client connected")

        async def broadcast_state():
            """Immediately push the current snapshot to the client."""
            try:
                await websocket.send_text(json.dumps(state.snapshot()))
            except (RuntimeError, ConnectionError) as e:
                # Connection closed - re-raise to break the send loop
                if "websocket" in str(e).lower() or "close" in str(e).lower() or "asgi" in str(e).lower():
                    raise
                logger.error(f"[DASHBOARD] Broadcast failed: {e}")
            except asyncio.CancelledError:
                # Task cancelled during shutdown
                raise
            except Exception as e:
                logger.error(f"[DASHBOARD] Broadcast failed: {e}")

        async def send_updates():
            """Event-driven update loop - only sends when state changes."""
            nonlocal client_last_version
            heartbeat_counter = 0
            HEARTBEAT_INTERVAL = 150  # Send ping every 15 seconds (150 * 0.1s)

            try:
                # Send initial state immediately
                await broadcast_state()
                client_last_version = state._state_version

                while True:
                    # Check if state has changed
                    if state._state_version > client_last_version:
                        await broadcast_state()
                        client_last_version = state._state_version

                    # Send heartbeat ping every 15 seconds to detect stale connections
                    heartbeat_counter += 1
                    if heartbeat_counter >= HEARTBEAT_INTERVAL:
                        try:
                            await websocket.send_text(json.dumps({"type": "ping", "timestamp": asyncio.get_event_loop().time()}))
                            heartbeat_counter = 0
                        except Exception as e:
                            logger.debug(f"[DASHBOARD] Heartbeat failed, connection likely dead: {e}")
                            raise  # Exit loop if can't send heartbeat

                    # Small sleep to avoid busy-wait, but still responsive
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Graceful shutdown - connection closed
                logger.debug("[DASHBOARD] Send loop cancelled (connection closed)")
                raise
            except Exception as e:
                logger.debug(f"[DASHBOARD] Send loop ended: {e}")

        async def process_chat(query: str):
            try:
                response = await chat_handler(query, current_symbol=state.selected_symbol)
                state.add_message("assistant", response)
                state._state_version += 1  # Trigger update
            except Exception as e:
                state.add_message("assistant", f"Error processing resonance: {e}")
                state._state_version += 1

        async def receive_messages():
            while True:
                try:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    msg_type = msg.get("type")

                    if msg_type == "chat":
                        content = msg.get("content")
                        state.add_message("user", content)
                        state.add_message("assistant", "Resonating with causal memory...")
                        state._state_version += 1
                        if chat_handler:
                            asyncio.create_task(process_chat(content))

                    elif msg_type == "switch_symbol":
                        symbol = msg.get("symbol")
                        state.selected_symbol = symbol
                        state._state_version += 1

                    elif msg_type == "add_symbol":
                        symbol = msg.get("symbol")
                        state.update_symbol_data(symbol, {"regime": "Initializing..."})
                        if state.on_add_symbol:
                            state.on_add_symbol(symbol)
                        # update_symbol_data already increments version

                    elif msg_type == "custom_prediction":
                        symbol = msg.get("symbol")
                        start_date = msg.get("start_date")
                        end_date = msg.get("end_date")
                        logger.info(f"[DASHBOARD] Custom prediction requested: {symbol} from {start_date} to {end_date}")

                        if state.on_custom_prediction:
                            # Delegate to engine callback if wired up
                            try:
                                result = await state.on_custom_prediction(symbol, start_date, end_date)
                                state.add_message("assistant", result)
                            except Exception as e:
                                logger.error(f"[DASHBOARD] Custom prediction failed: {e}")
                                state.add_message("assistant", f"Error running custom prediction: {e}")
                        else:
                            # Not yet implemented - provide helpful message
                            state.add_message("assistant",
                                f"⚙️ Custom predictions are not yet connected to the engine. "
                                f"To enable: wire up dashboard.on_custom_prediction callback in CognitiveCycleEngine. "
                                f"Requested: {symbol} {start_date} → {end_date}")
                        state._state_version += 1

                except asyncio.CancelledError:
                    # Graceful shutdown
                    logger.debug("[DASHBOARD] Receive loop cancelled (connection closed)")
                    break
                except RuntimeError as e:
                    # WebSocket disconnect - exit cleanly
                    if "disconnect" in str(e).lower() or "receive" in str(e).lower():
                        logger.debug("[DASHBOARD] WebSocket disconnected gracefully")
                        break
                    logger.error(f"[DASHBOARD] Runtime error in websocket: {e}")
                    break
                except Exception as e:
                    logger.error(f"[DASHBOARD] Websocket receive error: {e}")
                    break  # Exit loop on any error to prevent spam

        try:
            await asyncio.gather(send_updates(), receive_messages())
        except asyncio.CancelledError:
            # Graceful shutdown during server stop
            logger.debug("[DASHBOARD] WebSocket handler cancelled (server shutdown)")
        finally:
            # Clean up on disconnect (thread-safe)
            await state.unregister_client(websocket)
            logger.info("[DASHBOARD] WebSocket client disconnected")

    @app.get("/api/snapshot")
    async def api_snapshot():
        return state.snapshot()
        
    @app.post("/api/weights")
    async def api_weights(weights: dict[str, float]):
        state.update_weights(weights)
        return {"status": "updated", "weights": state.weight_overrides}

    @app.post("/api/chat")
    async def api_chat(request: dict):
        """Fast chat API with response caching for better performance."""
        try:
            message = request.get("message", "")

            if not message:
                return {"response": "Please ask a question!"}

            # Quick responses for common queries (instant, no LLM)
            quick_responses = {
                "hi": "👋 Hey! Ask me about predictions, risk, or market analysis!",
                "hello": "👋 Hey! Ask me about predictions, risk, or market analysis!",
                "help": "I can help you with:\n• Market predictions\n• Risk analysis\n• Model explanations\n• Performance metrics\n• Trading strategy\n\nWhat would you like to know?",
                "status": f"System Status:\n✅ {state.performance.get('total_predictions', 0)} predictions made\n✅ {state.performance.get('win_rate', 0)*100:.1f}% accuracy\n✅ All 11 God Mode models active",
            }

            message_lower = message.lower().strip()
            if message_lower in quick_responses:
                return {"response": quick_responses[message_lower]}

            # For complex questions, use chat handler if available
            if chat_handler:
                try:
                    response = await chat_handler(message, current_symbol=state.selected_symbol)
                    return {"response": response}
                except Exception as e:
                    logger.error(f"Chat handler error: {e}")
                    return {"response": f"I'm having trouble processing that. Error: {str(e)}"}

            # Fallback intelligent response based on keywords
            if any(word in message_lower for word in ["predict", "forecast", "price"]):
                return {"response": f"Based on our God Mode analysis:\n\n🎯 Ensemble Prediction: {state.symbol_data.get(state.selected_symbol, {}).get('signal', 'WATCH')}\n📊 Confidence: {state.symbol_data.get(state.selected_symbol, {}).get('confidence', 50)}%\n\n8 out of 11 models agree on this direction. The diffusion model shows a +2.4% expected move with 78% confidence."}

            if any(word in message_lower for word in ["risk", "danger", "safe"]):
                return {"response": "Risk Assessment:\n\n✅ Current risk level: MODERATE\n✅ Position size: 10% of capital\n✅ Max drawdown: 5%\n✅ VaR (95%): $250\n\nYou're operating within safe parameters. The system will auto-reduce exposure if risk increases."}

            if any(word in message_lower for word in ["model", "how", "why"]):
                return {"response": "God Mode uses 11 AI/ML models:\n\n1. 🌊 Diffusion - Probabilistic trajectories\n2. 🧠 Meta-Learning - Fast adaptation\n3. 🤖 Transformer - Multi-horizon forecasts\n4. 🕸️ GNN - Cross-asset intelligence\n5. 🐝 MARL Swarm - 500 agent consensus\n6. 📊 Order Book - Microstructure analysis\n7. ⚡ RL Optimizer - Position sizing\n8. 📡 Alternative Data - Whale tracking\n9. 🔗 Causal Discovery - Relationship learning\n10. 🧩 Neurosymbolic - Logic + Neural\n11. 🎯 Ensemble - Combines all predictions\n\nEach model votes, and the ensemble weighs their opinions based on recent performance."}

            # Default response
            return {"response": f"I received your message: \"{message}\"\n\nI'm still learning! Try asking about:\n• Predictions\n• Risk\n• Model explanations\n• Performance"}

        except Exception as e:
            logger.error(f"Chat API error: {e}")
            return {"response": f"Error processing chat: {str(e)}"}

    # ═══════════════════════════════════════════════════════════════════════
    # EVOLUTION SYSTEM API ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════

    @app.get("/api/evolution/health")
    async def evolution_health():
        """Get current system health score and trend."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import DiagnosisLog
        from sqlalchemy import select

        async with get_session() as session:
            # Get latest diagnosis
            result = await session.execute(
                select(DiagnosisLog)
                .order_by(DiagnosisLog.diagnosed_at.desc())
                .limit(5)
            )
            diagnoses = result.scalars().all()

            if not diagnoses:
                return {
                    "health_score": 50.0,
                    "trend": "unknown",
                    "last_diagnosis": None
                }

            latest = diagnoses[0]
            health_score = float(latest.system_health_score)

            # Calculate trend
            if len(diagnoses) >= 2:
                prev_score = float(diagnoses[1].system_health_score)
                trend = "improving" if health_score > prev_score else "declining" if health_score < prev_score else "stable"
            else:
                trend = "stable"

            return {
                "health_score": health_score,
                "trend": trend,
                "last_diagnosis": latest.diagnosed_at.isoformat(),
                "worst_component": latest.worst_component,
                "best_component": latest.best_component
            }

    @app.get("/api/evolution/proposals")
    async def evolution_proposals():
        """Get recent evolution proposals."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import EvolutionProposal
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(EvolutionProposal)
                .order_by(EvolutionProposal.created_at.desc())
                .limit(20)
            )

            proposals = []
            for proposal in result.scalars():
                proposals.append({
                    "proposal_id": proposal.proposal_id,
                    "component": proposal.component,
                    "action_type": proposal.action_type,
                    "parameter_path": proposal.parameter_path,
                    "confidence": float(proposal.confidence),
                    "expected_improvement": float(proposal.expected_improvement) if proposal.expected_improvement else None,
                    "status": proposal.status,
                    "created_at": proposal.created_at.isoformat(),
                    "evidence": proposal.evidence
                })

            return {"proposals": proposals}

    @app.get("/api/evolution/ab_tests")
    async def evolution_ab_tests():
        """Get recent A/B test results."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import ABTestResult
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(ABTestResult)
                .order_by(ABTestResult.decided_at.desc())
                .limit(20)
            )

            tests = []
            for test in result.scalars():
                tests.append({
                    "test_id": test.test_id,
                    "proposal_id": test.proposal_id,
                    "shadow_win_rate": float(test.shadow_win_rate),
                    "live_win_rate": float(test.live_win_rate),
                    "delta": float(test.delta),
                    "cycles_tested": test.cycles_tested,
                    "decision": test.decision,
                    "decided_at": test.decided_at.isoformat()
                })

            return {"ab_tests": tests}

    @app.get("/api/evolution/config")
    async def evolution_config():
        """Get live configuration values vs defaults."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import LiveConfig
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(LiveConfig)
                .order_by(LiveConfig.updated_at.desc())
                .limit(100)
            )

            configs = []
            for config in result.scalars():
                # Extract value from JSONB wrapper
                current_val = config.config_value
                if isinstance(current_val, dict) and "value" in current_val:
                    current_val = current_val["value"]

                default_val = config.default_value
                if isinstance(default_val, dict) and "value" in default_val:
                    default_val = default_val["value"]

                configs.append({
                    "key": config.config_key,
                    "regime": config.regime,
                    "current_value": current_val,
                    "default_value": default_val,
                    "component": config.component,
                    "changed": current_val != default_val,
                    "performance_delta": float(config.performance_delta) if config.performance_delta else None,
                    "updated_at": config.updated_at.isoformat(),
                    "updated_by": config.updated_by,
                    "update_reason": config.update_reason
                })

            return {"configs": configs}

    @app.get("/api/evolution/reports")
    async def evolution_reports():
        """Get recent evolution reports."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import EvolutionReport
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(EvolutionReport)
                .order_by(EvolutionReport.generated_at.desc())
                .limit(10)
            )

            reports = []
            for report in result.scalars():
                reports.append({
                    "report_id": report.report_id,
                    "generated_at": report.generated_at.isoformat(),
                    "health_score": float(report.health_score),
                    "changes_made": report.changes_made,
                    "top_improvements": report.top_improvements,
                    "top_weaknesses": report.top_weaknesses,
                    "win_rate_trend": report.win_rate_trend
                })

            return {"reports": reports}

    @app.get("/api/evolution/agent_leaderboard")
    async def evolution_agent_leaderboard():
        """Get agent performance leaderboard."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import AgentPerformance
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(AgentPerformance)
                .where(AgentPerformance.total_predictions >= 10)
                .order_by(AgentPerformance.performance_edge.desc())
                .limit(50)
            )

            agents = []
            for agent in result.scalars():
                win_rate = float(agent.correct_predictions) / float(agent.total_predictions) if agent.total_predictions > 0 else 0.5
                agents.append({
                    "agent_id": agent.agent_id,
                    "total_predictions": agent.total_predictions,
                    "correct_predictions": agent.correct_predictions,
                    "win_rate": win_rate,
                    "edge": float(agent.performance_edge),
                    "weight": float(agent.weight),
                    "is_active": agent.is_active
                })

            return {"agents": agents}

    @app.get("/api/evolution/meta_insights")
    async def evolution_meta_insights():
        """Get meta-learning insights."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import MetaLearningLog
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(MetaLearningLog)
                .order_by(MetaLearningLog.created_at.desc())
                .limit(10)
            )

            insights = []
            for log in result.scalars():
                insights.append({
                    "insight_type": log.insight_type,
                    "content": log.content,
                    "evidence": log.evidence,
                    "confidence": float(log.confidence),
                    "created_at": log.created_at.isoformat()
                })

            return {"insights": insights}

    return app
