"""Conversational chatbot with full RAG (Retrieval Augmented Generation).

Uses Claude API for natural responses with access to:
- Live market data & indicators
- Memory system (Qdrant vector store)
- Database (predictions, patterns, outcomes)
- System state (agents, regime, evolution)
"""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from loguru import logger


class AtomicXChatbot:
    """Conversational chatbot with full system RAG access."""

    def __init__(self, orchestrator: Any):
        """Initialize chatbot with orchestrator reference.

        Args:
            orchestrator: BrainOrchestrator instance for accessing all systems
        """
        self.orchestrator = orchestrator
        self.logger = logger.bind(module="chatbot")

        # References to all system components
        self.var_engine = orchestrator.var_engine
        self.fusion_node = orchestrator.fusion_node
        self.memory = self.fusion_node.memory

        # Get Anthropic client
        self._anthropic = getattr(orchestrator, '_anthropic', None)
        self._bedrock = getattr(orchestrator, '_bedrock', None)
        self._settings = getattr(orchestrator, '_settings', None)

    async def ask(self, query: str, current_symbol: str = "BTC/USDT") -> str:
        """Answer user's question using Claude API with full RAG context.

        Args:
            query: User's question
            current_symbol: Currently selected trading pair

        Returns:
            Natural, conversational response from Claude
        """
        # Detect symbol from query
        symbol = self._detect_symbol(query, current_symbol)

        # Build RAG context - retrieve relevant information from all systems
        try:
            context = await self._build_rag_context(query, symbol)
        except Exception as e:
            self.logger.warning(f"RAG context build failed: {e}")
            context = {"symbol": symbol, "error": "System warming up"}

        # Generate response using Claude API
        return await self._generate_claude_response(query, context)

    async def _build_rag_context(self, query: str, symbol: str) -> dict[str, Any]:
        """Build RAG context by retrieving relevant data from all systems.

        Args:
            query: User's question
            symbol: Trading pair to analyze

        Returns:
            Dictionary with relevant context for Claude
        """
        context = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        }

        # 1. MARKET DATA - Always fetch (very fast)
        try:
            vars_snap = await self.var_engine.compute_snapshot(symbol)
            context["price"] = vars_snap.get("PRICE", 0.0)
            context["rsi_14"] = vars_snap.get("RSI_14", 50.0)
            context["macd"] = vars_snap.get("MACD", 0.0)
            context["volume_24h"] = vars_snap.get("VOLUME_24H", 0.0)
            context["volatility"] = vars_snap.get("ATR_14", 0.0)

            # Get key support/resistance if available
            context["support"] = vars_snap.get("SUPPORT", None)
            context["resistance"] = vars_snap.get("RESISTANCE", None)

        except Exception as e:
            self.logger.warning(f"Market data fetch failed: {e}")
            context["price"] = 0.0

        # 2. REGIME & SYSTEM STATE - From memory (instant)
        try:
            context["regime"] = getattr(self.orchestrator.self_model, 'current_regime', 'unknown')
            context["risk_appetite"] = getattr(self.orchestrator.self_model, 'risk_appetite', 0.5)
        except Exception:
            context["regime"] = "unknown"
            context["risk_appetite"] = 0.5

        # 3. MEMORY RETRIEVAL - Query vector store for relevant past experiences
        q_lower = query.lower()
        if any(word in q_lower for word in ['past', 'before', 'history', 'remember', 'last time', 'previously']):
            try:
                # Search memory for relevant past experiences
                similar_memories = await self.memory.retrieve(
                    query=f"{symbol} {query}",
                    limit=3
                )
                context["relevant_memories"] = [
                    {
                        "content": m.get("memory", m.get("content", "")),
                        "timestamp": m.get("metadata", {}).get("timestamp", "unknown")
                    }
                    for m in similar_memories
                ]
            except Exception as e:
                self.logger.debug(f"Memory retrieval skipped: {e}")

        # 4. RECENT PREDICTIONS - If asking about predictions/forecasts
        if any(word in q_lower for word in ['predict', 'forecast', 'think', 'expect', 'will', 'future']):
            try:
                from atomicx.data.storage.database import get_session_factory
                from atomicx.data.storage.models import PredictionOutcome
                from sqlalchemy import select

                async with get_session_factory()() as session:
                    # Get last 3 predictions for this symbol
                    result = await session.execute(
                        select(PredictionOutcome)
                        .where(PredictionOutcome.symbol == symbol)
                        .order_by(PredictionOutcome.predicted_at.desc())
                        .limit(3)
                    )
                    recent_preds = result.scalars().all()

                    if recent_preds:
                        context["recent_predictions"] = [
                            {
                                "direction": p.predicted_direction,
                                "confidence": float(p.predicted_confidence),
                                "was_correct": p.was_correct,
                                "when": p.predicted_at.strftime('%Y-%m-%d %H:%M')
                            }
                            for p in recent_preds
                        ]
            except Exception as e:
                self.logger.debug(f"Prediction history skipped: {e}")

        # 5. PATTERN PERFORMANCE - If asking about patterns/signals
        if any(word in q_lower for word in ['pattern', 'signal', 'indicator', 'technical', 'chart']):
            try:
                from atomicx.data.pattern_verification import PatternVerificationService
                pattern_svc = PatternVerificationService()

                # Get pattern stats for this symbol
                stats = await pattern_svc.get_pattern_stats(symbol=symbol)
                if stats:
                    context["pattern_stats"] = {
                        "total_detected": stats.get("total_detected", 0),
                        "verified": stats.get("verified_count", 0),
                        "win_rate": stats.get("win_rate", 0.0)
                    }
            except Exception as e:
                self.logger.debug(f"Pattern stats skipped: {e}")

        # 6. AGENT OPINIONS - If asking for analysis/opinion
        if any(word in q_lower for word in ['analysis', 'opinion', 'think', 'assess', 'view']):
            try:
                # Get current agent hierarchy state
                total_agents = len(self.fusion_node.hierarchy.atomic_agents)
                active_agents = sum(1 for a in self.fusion_node.hierarchy.atomic_agents.values() if a.is_active)

                context["agents"] = {
                    "total": total_agents,
                    "active": active_agents
                }
            except Exception:
                pass

        return context

    async def _generate_claude_response(self, query: str, context: dict[str, Any]) -> str:
        """Generate natural response using Claude API with RAG context.

        Args:
            query: User's question
            context: Retrieved context from all systems

        Returns:
            Natural, conversational response
        """
        # Check if Claude API is available
        if not self._anthropic and not self._bedrock:
            return self._fallback_response(query, context)

        # Build system prompt - define persona and capabilities
        system_prompt = """You are a friendly, knowledgeable trading assistant for AtomicX.

Your personality:
- Conversational and natural, like talking to a knowledgeable friend
- Helpful and patient, never condescending
- Honest about uncertainty - say "I'm not sure" when you don't know
- Use emojis occasionally to be friendly 😊 but don't overdo it

Your capabilities:
- Access to LIVE market data (price, RSI, MACD, volume, volatility)
- Memory of past predictions and patterns
- Knowledge of technical analysis and trading concepts
- Understanding of market regimes and risk management

Response style:
- Keep responses conversational and clear (2-5 sentences usually)
- Explain technical terms simply when they come up
- Give practical, actionable insights
- Always add disclaimers for trading advice ("Not financial advice, but...")
- Use the data in the context to give specific, relevant answers

Important:
- NEVER make up data - only use what's in the context provided
- If context is missing key info, acknowledge it honestly
- Be natural - you're having a conversation, not writing a report"""

        # Build user prompt with context
        user_prompt = self._format_context_for_claude(query, context)

        try:
            # Call Claude API
            if self._anthropic:
                response = await self._anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    temperature=0.7,  # Slightly creative for natural conversation
                    messages=[{"role": "user", "content": user_prompt}],
                    system=system_prompt
                )

                if hasattr(response, 'content') and len(response.content) > 0:
                    return response.content[0].text

            elif self._bedrock:
                import json
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "system": system_prompt
                }

                resp = self._bedrock.invoke_model(
                    modelId=self._settings.bedrock_model_id,
                    body=json.dumps(payload)
                )
                result = json.loads(resp.get("body").read())

                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get("text", "")

            # If we get here, something went wrong
            raise ValueError("Claude API returned unexpected format")

        except Exception as e:
            self.logger.warning(f"Claude API call failed: {e}")
            return self._fallback_response(query, context)

    def _format_context_for_claude(self, query: str, context: dict[str, Any]) -> str:
        """Format retrieved context into a clear prompt for Claude.

        Args:
            query: User's question
            context: Retrieved context data

        Returns:
            Formatted prompt string
        """
        prompt = f"User question: {query}\n\n"
        prompt += "Context you have access to:\n\n"

        # Market data
        symbol = context.get("symbol", "UNKNOWN")
        price = context.get("price", 0)

        if price > 0:
            prompt += f"📊 Market Data for {symbol}:\n"
            prompt += f"- Current Price: ${price:,.2f}\n"
            prompt += f"- RSI (14): {context.get('rsi_14', 50):.1f}\n"
            prompt += f"- MACD: {context.get('macd', 0):.2f}\n"

            if context.get("support"):
                prompt += f"- Support: ${context.get('support'):,.2f}\n"
            if context.get("resistance"):
                prompt += f"- Resistance: ${context.get('resistance'):,.2f}\n"

            prompt += f"- Market Regime: {context.get('regime', 'unknown')}\n"
            prompt += f"- Risk Level: {context.get('risk_appetite', 0.5):.0%}\n\n"
        else:
            prompt += f"Note: Market data for {symbol} is still loading\n\n"

        # Recent predictions
        if "recent_predictions" in context:
            prompt += "📈 Recent Predictions:\n"
            for pred in context["recent_predictions"]:
                result = "✓" if pred["was_correct"] else "✗"
                prompt += f"- {pred['when']}: {pred['direction']} @ {pred['confidence']:.0%} {result}\n"
            prompt += "\n"

        # Pattern stats
        if "pattern_stats" in context:
            stats = context["pattern_stats"]
            prompt += f"📉 Pattern Analysis:\n"
            prompt += f"- Patterns detected: {stats['total_detected']}\n"
            prompt += f"- Verified outcomes: {stats['verified']}\n"
            prompt += f"- Win rate: {stats['win_rate']:.1%}\n\n"

        # Relevant memories
        if "relevant_memories" in context:
            prompt += "🧠 Relevant Past Experiences:\n"
            for mem in context["relevant_memories"]:
                prompt += f"- {mem['timestamp']}: {mem['content']}\n"
            prompt += "\n"

        # Agent info
        if "agents" in context:
            prompt += f"🤖 Analysis System: {context['agents']['active']}/{context['agents']['total']} agents active\n\n"

        prompt += f"Current time: {context.get('timestamp', 'unknown')}\n\n"
        prompt += "Please answer the user's question naturally using this context."

        return prompt

    def _fallback_response(self, query: str, context: dict[str, Any]) -> str:
        """Fallback response when Claude API is unavailable.

        Args:
            query: User's question
            context: Retrieved context

        Returns:
            Simple response using template
        """
        symbol = context.get("symbol", "BTC/USDT")
        price = context.get("price", 0)
        rsi = context.get("rsi_14", 50)

        if price <= 0:
            return "I'm still loading market data. Give me a moment and try again!"

        q_lower = query.lower()

        if any(word in q_lower for word in ['price', 'cost', 'worth']):
            return f"{symbol} is currently trading at ${price:,.2f}."

        elif any(word in q_lower for word in ['buy', 'sell', 'should']):
            if rsi < 30:
                return f"⚠️ Not financial advice, but {symbol} looks oversold (RSI {rsi:.1f}). Could be a buying opportunity."
            elif rsi > 70:
                return f"⚠️ Not financial advice, but {symbol} looks overbought (RSI {rsi:.1f}). Might be time to take profits."
            else:
                return f"⚠️ Not financial advice, but {symbol} is in a neutral zone (RSI {rsi:.1f}). Wait for clearer signals."

        else:
            return f"I'm tracking {symbol} at ${price:,.2f} with RSI at {rsi:.1f}. Ask me anything about price, predictions, or trading advice!"

    def _detect_symbol(self, query: str, current: str) -> str:
        """Detect trading pair from query.

        Args:
            query: User's question
            current: Currently selected symbol

        Returns:
            Detected symbol or current symbol
        """
        words = query.upper().split()

        # Check for explicit pairs like BTC/USDT
        for word in words:
            if "/" in word and len(word) > 5:
                return word

        # Check for common symbols
        common_symbols = {
            "BTC": "BTC/USDT",
            "BITCOIN": "BTC/USDT",
            "ETH": "ETH/USDT",
            "ETHEREUM": "ETH/USDT",
            "SOL": "SOL/USDT",
            "SOLANA": "SOL/USDT",
            "XRP": "XRP/USDT",
            "RIPPLE": "XRP/USDT",
            "BNB": "BNB/USDT",
            "BINANCE": "BNB/USDT",
        }

        for word in words:
            if word in common_symbols:
                return common_symbols[word]

        return current
