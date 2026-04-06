"""Agentic chatbot with tool use - Claude decides what to do dynamically.

The chatbot has access to tools and Claude decides which ones to use:
- get_market_data: Fetch live price, indicators
- search_memory: Query Qdrant vector store
- query_predictions: Get recent predictions from database
- query_patterns: Get pattern statistics
- run_analysis: Trigger brain to analyze a symbol
- web_search: Search online for news/info
- get_system_state: Get current system status
"""

from __future__ import annotations

from typing import Any, Callable
from datetime import datetime, timezone
from loguru import logger
import json


class AgenticChatbot:
    """Agentic chatbot - Claude decides what tools to use."""

    def __init__(self, orchestrator: Any):
        """Initialize agentic chatbot.

        Args:
            orchestrator: BrainOrchestrator with all system access
        """
        self.orchestrator = orchestrator
        self.logger = logger.bind(module="chatbot_agent")

        # System components
        self.var_engine = orchestrator.var_engine
        self.fusion_node = orchestrator.fusion_node
        self.memory = self.fusion_node.memory

        # Claude API
        self._anthropic = getattr(orchestrator, '_anthropic', None)
        self._bedrock = getattr(orchestrator, '_bedrock', None)
        self._settings = getattr(orchestrator, '_settings', None)

        # Check API availability
        has_api = (
            self._anthropic or
            self._bedrock or
            (self._settings and self._settings.aws_bearer_token)
        )

        if not has_api:
            self.logger.warning(
                "[CHATBOT] No Claude API available! "
                "Set ANTHROPIC_API_KEY or AWS credentials in .env"
            )
        else:
            if self._anthropic:
                api_type = "Anthropic SDK"
            elif self._bedrock:
                api_type = "Bedrock SDK"
            elif self._settings and self._settings.aws_bearer_token:
                api_type = "Bedrock Bearer Token"
            else:
                api_type = "Unknown"
            self.logger.success(f"[CHATBOT] Claude API ready via {api_type}")

        # Register all available tools
        self.tools = self._register_tools()

    def _register_tools(self) -> list[dict]:
        """Register all tools that Claude can use.

        Returns:
            List of tool definitions for Claude API
        """
        return [
            {
                "name": "get_market_data",
                "description": "Get live market data for a symbol including price, RSI, MACD, volume, support/resistance. Use this when user asks about current price, indicators, or market state.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair like BTC/USDT, ETH/USDT"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "search_memory",
                "description": "Search the system's memory (Qdrant vector store) for past experiences, patterns, or lessons. Use when user asks about history, past events, 'last time', 'remember when', or similar contexts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max number of results",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "query_predictions",
                "description": "Query recent predictions from database with their outcomes. Use when user asks about forecast accuracy, past predictions, or 'how did you do'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of predictions to fetch",
                            "default": 5
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "query_patterns",
                "description": "Get pattern detection statistics including win rates and accuracy. Use when user asks about signal quality, pattern accuracy, or technical analysis performance.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair, or 'all' for system-wide stats"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "run_analysis",
                "description": "Trigger the brain orchestrator to run a full analysis on a symbol. This runs all agents, patterns, and prediction engines. Use when user asks for 'analysis', 'what do you think', or wants a fresh prediction.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair to analyze"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "web_search",
                "description": "Search online for news, articles, or information. Use when user asks about external events, news, 'what's happening', or information not in the system.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_system_state",
                "description": "Get current system status including regime, active agents, evolution cycles, database stats. Use when user asks 'how are you', 'system status', or wants to know about system health.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "query_news_patterns",
                "description": "Query learned news patterns and their causal effects on price movements. Shows historical patterns like 'when Powell mentions X, BTC moves Y%'. Use when user asks about news impact, Fed decisions, regulatory events, or 'what happens when'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern_type": {
                            "type": "string",
                            "description": "Type of pattern (e.g., 'fed_rate_decision', 'regulatory_action', 'hack', 'all' for all patterns)",
                            "default": "all"
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence level (0.0-1.0)",
                            "default": 0.6
                        }
                    },
                    "required": []
                }
            }
        ]

    async def ask(self, query: str, current_symbol: str = "BTC/USDT") -> str:
        """Answer user query using agentic tool calling.

        Claude decides which tools to use and in what order.

        Args:
            query: User's question
            current_symbol: Currently selected symbol

        Returns:
            Natural conversational response
        """
        self.logger.info(f"[CHATBOT] Received query: '{query[:80]}...' (symbol: {current_symbol})")

        has_api = (
            self._anthropic or
            self._bedrock or
            (self._settings and self._settings.aws_bearer_token)
        )

        if not has_api:
            return (
                "⚠️ Claude API not configured!\n\n"
                "To enable the AI chatbot, add one of these to your .env file:\n\n"
                "**Option 1 (Anthropic Direct)**:\n"
                "ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "**Option 2 (AWS Bedrock)**:\n"
                "AWS_ACCESS_KEY_ID=...\n"
                "AWS_SECRET_ACCESS_KEY=...\n"
                "AWS_REGION_NAME=us-east-1\n\n"
                "**Option 3 (Bedrock Bearer Token)**:\n"
                "AWS_BEARER_TOKEN_BEDROCK=...\n"
                "AWS_REGION_NAME=us-east-1\n\n"
                "Then restart the system. Check logs for confirmation!"
            )

        # System prompt for the agent
        system_prompt = """You are AtomicX, a helpful AI trading assistant with access to powerful tools.

Your personality:
- Friendly, conversational, natural (like talking to a knowledgeable friend)
- Helpful and patient
- Honest when you don't know something
- Use emojis occasionally 😊 but not excessively

Your capabilities (via tools):
- Access live market data (price, indicators, volume)
- Search system memory for past experiences
- Query prediction history and accuracy
- Get pattern statistics and win rates
- Trigger full market analysis
- Search online for news and events
- Check system status

How to work:
1. Understand what the user wants
2. Decide which tools you need (you can use multiple tools)
3. Call the appropriate tools to gather information
4. Give a natural, conversational answer based on the data

Important:
- Always use tools to get real data - never make up information
- You can call multiple tools if needed
- Keep responses conversational (2-5 sentences usually)
- Include trading disclaimers when giving advice
- Be honest if you can't find information

Current context:
- User is looking at: {symbol}
- Current time: {timestamp}"""

        # Build initial message
        messages = [
            {
                "role": "user",
                "content": f"User is viewing {current_symbol}. Their question: {query}"
            }
        ]

        try:
            # Agentic loop - Claude can call tools multiple times
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                # Call Claude with tools (supports multiple API methods)
                response = await self._call_claude_api(
                    messages=messages,
                    system_prompt=system_prompt.format(
                        symbol=current_symbol,
                        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    ),
                    tools=self.tools
                )

                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use":
                    # Extract tool calls
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_input = block.input
                            tool_id = block.id

                            self.logger.info(f"[AGENT] Calling tool: {tool_name} with {tool_input}")

                            # Execute the tool
                            try:
                                result = await self._execute_tool(tool_name, tool_input)
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": json.dumps(result)
                                })
                            except Exception as e:
                                self.logger.error(f"Tool {tool_name} failed: {e}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": json.dumps({"error": str(e)})
                                })

                    # Add assistant response and tool results to conversation
                    # Convert ContentBlock objects to dicts for JSON serialization
                    assistant_content = []
                    for block in response.content:
                        if block.type == "tool_use":
                            assistant_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "text" and block.text:
                            assistant_content.append({
                                "type": "text",
                                "text": block.text
                            })

                    messages.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                    iteration += 1

                elif response.stop_reason == "end_turn":
                    # Claude is done, extract final text response
                    self.logger.info(f"[CHATBOT] Extracting response from {len(response.content)} content blocks")
                    for block in response.content:
                        self.logger.debug(f"[CHATBOT] Block type: {block.type}, has text: {hasattr(block, 'text')}")
                        if hasattr(block, 'text') and block.text:
                            self.logger.success(f"[CHATBOT] Returning response: {block.text[:100]}...")
                            return block.text

                    self.logger.warning("[CHATBOT] No text found in response blocks")
                    return "I'm having trouble formulating a response. Could you rephrase your question?"

                else:
                    # Unexpected stop reason
                    return f"Unexpected response from AI (stop_reason: {response.stop_reason})"

            # Max iterations reached
            return "I tried to gather information but ran into complexity. Could you ask a more specific question?"

        except Exception as e:
            self.logger.error(f"Agentic chat failed: {e}")
            return "I encountered an error while processing your question. Please try again!"

    async def _call_claude_api(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict]
    ) -> Any:
        """Call Claude API using available method (Anthropic SDK, Bedrock, or Bearer Token).

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            tools: Available tools

        Returns:
            Claude API response object
        """
        # Helper classes for non-SDK responses
        class ContentBlock:
            def __init__(self, block_data):
                self.type = block_data.get("type")
                self.text = block_data.get("text", "")
                self.name = block_data.get("name")
                self.input = block_data.get("input", {})
                self.id = block_data.get("id")

        class APIResponse:
            def __init__(self, data):
                self.content = [ContentBlock(b) for b in data.get("content", [])]
                self.stop_reason = data.get("stop_reason")

        # Method 1: Anthropic SDK (direct)
        if self._anthropic:
            return await self._anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=messages,
                tools=tools
            )

        # Method 2: Bearer Token (HTTP)
        elif self._settings and self._settings.aws_bearer_token:
            import httpx

            # Build request payload
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.7,
                "system": system_prompt,
                "messages": messages,
                "tools": tools
            }

            # Endpoint URL
            base_url = (
                self._settings.aws_endpoint_url or
                f"https://bedrock-runtime.{self._settings.aws_region_name}.amazonaws.com"
            )
            url = f"{base_url}/model/{self._settings.bedrock_model_id}/invoke"

            # Headers
            headers = {
                "Authorization": f"Bearer {self._settings.aws_bearer_token}",
                "x-api-key": self._settings.aws_bearer_token,
                "Content-Type": "application/json",
            }

            # Make HTTP call
            self.logger.info(f"[CHATBOT] Calling Claude via Bearer Token: {url}")
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                result = resp.json()
            self.logger.success(f"[CHATBOT] Claude API response received (stop_reason: {result.get('stop_reason')})")

            return APIResponse(result)

        # Method 3: Bedrock SDK
        elif self._bedrock:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.7,
                "system": system_prompt,
                "messages": messages,
                "tools": tools
            }

            resp = self._bedrock.invoke_model(
                modelId=self._settings.bedrock_model_id,
                body=json.dumps(payload)
            )
            result = json.loads(resp.get("body").read())

            return APIResponse(result)

        else:
            raise ValueError("No Claude API method available")

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool and return results.

        Args:
            tool_name: Name of tool to execute
            tool_input: Tool parameters

        Returns:
            Tool execution results
        """
        if tool_name == "get_market_data":
            return await self._tool_get_market_data(tool_input.get("symbol"))

        elif tool_name == "search_memory":
            return await self._tool_search_memory(
                tool_input.get("query"),
                tool_input.get("limit", 3)
            )

        elif tool_name == "query_predictions":
            return await self._tool_query_predictions(
                tool_input.get("symbol"),
                tool_input.get("limit", 5)
            )

        elif tool_name == "query_patterns":
            return await self._tool_query_patterns(
                tool_input.get("symbol")
            )

        elif tool_name == "run_analysis":
            return await self._tool_run_analysis(
                tool_input.get("symbol")
            )

        elif tool_name == "web_search":
            return await self._tool_web_search(
                tool_input.get("query")
            )

        elif tool_name == "get_system_state":
            return await self._tool_get_system_state()

        elif tool_name == "query_news_patterns":
            return await self._tool_query_news_patterns(
                tool_input.get("pattern_type", "all"),
                tool_input.get("min_confidence", 0.6)
            )

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # ═══════════════════════════════════════════════════════════════════
    # TOOL IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════

    async def _tool_get_market_data(self, symbol: str) -> dict:
        """Get live market data."""
        try:
            vars_snap = await self.var_engine.compute_snapshot(symbol)

            return {
                "symbol": symbol,
                "price": vars_snap.get("PRICE", 0.0),
                "rsi_14": vars_snap.get("RSI_14", 50.0),
                "macd": vars_snap.get("MACD", 0.0),
                "volume_24h": vars_snap.get("VOLUME_24H", 0.0),
                "atr_14": vars_snap.get("ATR_14", 0.0),
                "support": vars_snap.get("SUPPORT"),
                "resistance": vars_snap.get("RESISTANCE"),
                "ema_20": vars_snap.get("EMA_20"),
                "regime": getattr(self.orchestrator.self_model, 'current_regime', 'unknown')
            }
        except Exception as e:
            return {"error": f"Failed to fetch market data: {e}"}

    async def _tool_search_memory(self, query: str, limit: int) -> dict:
        """Search memory system."""
        try:
            results = await self.memory.retrieve(query=query, limit=limit)

            return {
                "query": query,
                "results": [
                    {
                        "content": r.get("memory", r.get("content", "")),
                        "timestamp": r.get("metadata", {}).get("timestamp", "unknown"),
                        "relevance": r.get("score", 0.0)
                    }
                    for r in results
                ]
            }
        except Exception as e:
            return {"error": f"Memory search failed: {e}"}

    async def _tool_query_predictions(self, symbol: str, limit: int) -> dict:
        """Query recent predictions."""
        try:
            from atomicx.data.storage.database import get_session_factory
            from atomicx.data.storage.models import PredictionOutcome
            from sqlalchemy import select

            async with get_session_factory()() as session:
                result = await session.execute(
                    select(PredictionOutcome)
                    .where(PredictionOutcome.symbol == symbol)
                    .order_by(PredictionOutcome.predicted_at.desc())
                    .limit(limit)
                )
                predictions = result.scalars().all()

                return {
                    "symbol": symbol,
                    "predictions": [
                        {
                            "direction": p.predicted_direction,
                            "confidence": float(p.predicted_confidence),
                            "was_correct": p.was_correct,
                            "predicted_at": p.predicted_at.isoformat(),
                            "outcome_return": float(p.outcome_return) if p.outcome_return else None
                        }
                        for p in predictions
                    ],
                    "total": len(predictions)
                }
        except Exception as e:
            return {"error": f"Prediction query failed: {e}"}

    async def _tool_query_patterns(self, symbol: str) -> dict:
        """Get pattern statistics."""
        try:
            from atomicx.data.pattern_verification import PatternVerificationService
            pattern_svc = PatternVerificationService()

            if symbol == "all":
                stats = await pattern_svc.get_pattern_performance_stats()
                return {
                    "scope": "all",
                    "patterns": stats if stats else []
                }
            else:
                stats = await pattern_svc.get_pattern_stats(symbol=symbol)
                return {
                    "symbol": symbol,
                    "total_detected": stats.get("total_detected", 0),
                    "verified": stats.get("verified_count", 0),
                    "win_rate": stats.get("win_rate", 0.0)
                }
        except Exception as e:
            return {"error": f"Pattern query failed: {e}"}

    async def _tool_run_analysis(self, symbol: str) -> dict:
        """Trigger brain to analyze symbol."""
        try:
            # Run full cognitive cycle
            brain_state = await self.orchestrator.observe_and_reflect(symbol)

            return {
                "symbol": symbol,
                "price": brain_state.get("price", 0.0),
                "regime": brain_state.get("regime", "unknown"),
                "agents_active": len(brain_state.get("all_agents", [])),
                "analysis_complete": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

    async def _tool_web_search(self, query: str) -> dict:
        """Search online for information."""
        try:
            # Use intelligence scanner if available
            if hasattr(self.orchestrator, 'news_scanner'):
                stories = await self.orchestrator.news_scanner.scan_cycle()

                # Filter relevant stories
                relevant = [
                    s for s in stories
                    if any(word.lower() in s.get("title", "").lower() for word in query.split())
                ][:3]

                return {
                    "query": query,
                    "results": [
                        {
                            "title": s.get("title"),
                            "summary": s.get("summary", ""),
                            "source": s.get("source"),
                            "significance": s.get("significance", 0.0)
                        }
                        for s in relevant
                    ]
                }
            else:
                return {"error": "Web search not available"}
        except Exception as e:
            return {"error": f"Web search failed: {e}"}

    async def _tool_get_system_state(self) -> dict:
        """Get current system status."""
        try:
            return {
                "regime": getattr(self.orchestrator.self_model, 'current_regime', 'unknown'),
                "risk_appetite": getattr(self.orchestrator.self_model, 'risk_appetite', 0.5),
                "total_agents": len(self.fusion_node.hierarchy.atomic_agents),
                "active_agents": sum(
                    1 for a in self.fusion_node.hierarchy.atomic_agents.values()
                    if a.is_active
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "operational"
            }
        except Exception as e:
            return {"error": f"System state query failed: {e}"}

    async def _tool_query_news_patterns(self, pattern_type: str, min_confidence: float) -> dict:
        """Query learned news patterns."""
        try:
            from atomicx.intelligence.news_intelligence import NewsIntelligence
            from atomicx.data.models.news import NewsPattern
            from atomicx.data.storage.database import get_session
            from sqlalchemy import select, and_

            news_intel = NewsIntelligence()
            stats = await news_intel.get_stats()

            # Query patterns
            async with get_session() as session:
                query = select(NewsPattern).where(
                    and_(
                        NewsPattern.confidence >= min_confidence,
                        NewsPattern.occurrences >= 3
                    )
                )

                if pattern_type != "all":
                    query = query.where(NewsPattern.pattern_type == pattern_type)

                query = query.order_by(NewsPattern.confidence.desc()).limit(10)
                result = await session.execute(query)
                patterns = result.scalars().all()

                return {
                    "pattern_type_filter": pattern_type,
                    "min_confidence": min_confidence,
                    "stats": {
                        "total_events": stats["total_events"],
                        "tracked_outcomes": stats["tracked_outcomes"],
                        "total_patterns": stats["total_patterns"],
                        "high_confidence_patterns": stats["high_confidence_patterns"]
                    },
                    "patterns": [
                        {
                            "type": p.pattern_type,
                            "avg_impact": f"{p.avg_price_impact:+.1f}%" if p.avg_price_impact else "N/A",
                            "confidence": f"{p.confidence:.0%}",
                            "win_rate": f"{p.win_rate:.0%}",
                            "occurrences": p.occurrences,
                            "examples": p.examples[-2:] if p.examples else []
                        }
                        for p in patterns
                    ]
                }
        except Exception as e:
            return {"error": f"News pattern query failed: {e}"}
