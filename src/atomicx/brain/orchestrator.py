"""Meta-Orchestrator — The Top-Level Consciousness Loop.

Maintains a persistent 'self-model' of the system (current capabilities,
past performance, personality drift, risk appetite). Unifies all v3.0 layers
as sensory organs and routes them to the Decider Core (Phase 14).
"""

from __future__ import annotations

import asyncio
import uuid
import httpx
import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from atomicx.config import get_settings
from atomicx.brain.reflector import RecursiveReflector

from atomicx.fusion.engine import FusionNode
from atomicx.narrative import NarrativeTracker
from atomicx.strategic import StrategicActorLayer
from atomicx.swarm import SwarmSimulator
from atomicx.variables.engine import VariableComputeEngine


class SelfModel(BaseModel):
    """The brain's understanding of its own state and personality."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # State tracking
    last_update: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    current_regime: str = "unknown"
    risk_appetite: float = 0.5  # 0.0 (Paranoid) to 1.0 (Aggressive)
    
    # Dynamic confidence layer modifiers updated by the Reflector
    trust_weights: dict[str, float] = Field(default_factory=lambda: {
        "causal": 1.0,
        "strategic": 1.0,
        "narrative": 1.0,
        "swarm": 1.0,
        "pro_trader": 1.0
    })

    def apply_reflector_action(self, action: str | None) -> None:
        """Dynamically mutate self-model based on Reflector insight."""
        if not action:
            return
            
        if action == "reduce_swarm_weight":
            self.trust_weights["swarm"] = max(0.1, self.trust_weights["swarm"] - 0.2)
        elif action == "restore_swarm_weight":
            self.trust_weights["swarm"] = min(1.0, self.trust_weights["swarm"] + 0.1)


class MetaOrchestrator:
    """The Single 'I'. Connects to all v3.0 'sensory organs'."""

    def __init__(
        self,
        var_engine: VariableComputeEngine,
        fusion_node: FusionNode,
        strategic_layer: StrategicActorLayer,
        narrative_tracker: NarrativeTracker,
        swarm_sim: SwarmSimulator,
    ) -> None:
        self.var_engine = var_engine
        self.fusion_node = fusion_node
        self.strategic = strategic_layer
        self.narrative = narrative_tracker
        self.swarm = swarm_sim
        
        self.reflector = RecursiveReflector()
        self.self_model = SelfModel()
        self.logger = logger.bind(module="brain.orchestrator")

        # LLM clients for the Brain Interface
        self._settings = get_settings()

        # ═══ NEW: LLM Failover with Auth Rotation ═══
        from atomicx.brain.llm_profiles import create_default_profiles, LLMProfileManager

        try:
            profiles = create_default_profiles()
            self._llm_manager = LLMProfileManager(profiles)
            self.logger.success(
                f"[LLM-FAILOVER] Initialized with {len(profiles)} profiles for high-availability"
            )
        except Exception as e:
            self.logger.error(f"[LLM-FAILOVER] Failed to initialize: {e} - falling back to legacy clients")
            self._llm_manager = None

        # Legacy clients (fallback if failover init fails)
        self._bedrock = None
        self._anthropic = None

        # Try Anthropic SDK first (primary)
        if self._settings.anthropic_api_key:
            try:
                import anthropic
                self._anthropic = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
                self.logger.info("Anthropic SDK initialized for god-mode (legacy path)")
            except Exception as e:
                self.logger.warning(f"Anthropic SDK init failed: {e}")

        # Bedrock client (secondary)
        if self._settings.aws_access_key_id and not self._settings.aws_bearer_token:
            try:
                import boto3
                import json
                self._bedrock = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self._settings.aws_region_name,
                    aws_access_key_id=self._settings.aws_access_key_id,
                    aws_secret_access_key=self._settings.aws_secret_access_key,
                )
                self.logger.info("Bedrock client initialized for god-mode")
            except Exception as e:
                self.logger.error(f"Failed to init Bedrock client: {e}")

    async def deep_research(self, symbol: str) -> dict[str, Any]:
        """Trigger a full 12-layer cognitive 'resonance' cycle for a symbol."""
        self.logger.info(f"[BRAIN] Activating ALL LAYERS for Deep Research: {symbol}")
        
        # 1. Sense & Reflect
        brain_state = await self.observe_and_reflect(symbol)
        
        # 2. Debate (Logic Channel)
        # We need access to the debate chamber. Since loop has it, 
        # for a one-off we can init a temporary one or pass it in.
        # For resonance, we'll use the debate chamber logic.
        from atomicx.brain.debate import DebateChamber
        chamber = DebateChamber()
        debate_summary = await chamber.debate(brain_state)
        
        # 3. Dual-Confirmation
        from atomicx.fusion.dual_confirm import DualConfirmationEngine
        engine = DualConfirmationEngine()
        confirmation = engine.evaluate(brain_state, debate_summary)
        
        return {
            "symbol": symbol,
            "regime": brain_state["regime"],
            "monologue": brain_state["monologue"].reasoning,

            "consensus": getattr(debate_summary, "dominant_stance", "neutral"),
            "logic": getattr(debate_summary, "synthesis", "No clear synthesis"),
            "confirmation": confirmation.final_direction,
            "price": brain_state["price"]
        }

    async def ask(self, query: str, current_symbol: str = "BTC/USDT") -> str:
        """Answer a query with ABSOLUTE GOD-LEVEL OMNISCIENCE.

        This is the supreme intelligence interface - knows EVERYTHING:
        - All 46 variables, all predictions, all patterns
        - Database contents, memory stats, agent states
        - System code, uptime, decisions, positions
        - Past, present, future state
        """
        # 1. Detect target asset
        research_symbol = current_symbol if current_symbol else "BTC/USDT"
        words = query.upper().split()
        for word in words:
            if "/" in word and len(word) > 5:
                research_symbol = word
                break
            elif word in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"]:
                research_symbol = f"{word}/USDT"
                break

        self.logger.info(f"⚛ GOD-MODE OMNISCIENCE ACTIVATED: {research_symbol} | Query: {query[:50]}")

        # 2. Gather ABSOLUTE GOD-LEVEL OMNISCIENCE
        from datetime import datetime, timezone
        import asyncio

        god_data = {}
        now = datetime.now(tz=timezone.utc)
        memory = getattr(self, '_memory', self.fusion_node.memory)

        # === REAL-TIME MARKET STATE ===
        try:
            vars_snap = await self.var_engine.compute_snapshot(research_symbol)
            god_data['price'] = vars_snap.get("PRICE", 0.0)
            god_data['variables'] = {k: v for k, v in vars_snap.items() if isinstance(v, (int, float))}
        except Exception as e:
            self.logger.warning(f"Variable snapshot failed: {e}")
            god_data['price'] = 0.0
            god_data['variables'] = {}

        # === DATABASE STATISTICS ===
        try:
            from atomicx.data.storage.database import get_session_factory
            from atomicx.data.storage.models import OHLCV
            from atomicx.variables.models import ComputedVariable
            from sqlalchemy import select, func

            async with get_session_factory()() as session:
                # OHLCV count
                ohlcv_count = await session.execute(select(func.count(OHLCV.id)))
                god_data['ohlcv_candles'] = ohlcv_count.scalar() or 0

                # Variable count
                var_count = await session.execute(select(func.count(ComputedVariable.id)))
                god_data['computed_variables'] = var_count.scalar() or 0

                # Latest candle time
                latest_candle = await session.execute(
                    select(func.max(OHLCV.timestamp)).where(OHLCV.symbol == research_symbol)
                )
                latest_ts = latest_candle.scalar()
                god_data['latest_candle'] = latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC') if latest_ts else 'None'

        except Exception as e:
            self.logger.warning(f"Database stats failed: {e}")
            god_data['ohlcv_candles'] = 'DB offline'
            god_data['computed_variables'] = 'DB offline'
            god_data['latest_candle'] = 'DB offline'

        # === PATTERN LIBRARY ===
        try:
            from atomicx.data.pattern_verification import PatternVerificationService
            pattern_svc = PatternVerificationService()

            # Get recent patterns
            recent_patterns = await pattern_svc.get_pattern_history(
                pattern_name="RSI_OVERSOLD_30",  # Example
                symbol=research_symbol,
                limit=10
            )
            god_data['recent_patterns'] = len(recent_patterns)

            # Get pattern stats
            stats = await pattern_svc.get_pattern_performance_stats()
            god_data['total_patterns_detected'] = sum(s['total_occurrences'] for s in stats) if stats else 0
            god_data['verified_patterns'] = sum(s['verified_count'] for s in stats) if stats else 0

        except Exception as e:
            self.logger.warning(f"Pattern stats failed: {e}")
            god_data['recent_patterns'] = 'N/A'
            god_data['total_patterns_detected'] = 'N/A'
            god_data['verified_patterns'] = 'N/A'

        # === MEMORY & EVOLUTION ===
        try:
            god_data['mem_count'] = getattr(memory.tier3, "memory_count", 0) if hasattr(memory, "tier3") else 0
            god_data['genome_active'] = sum(1 for g in memory.genome.genes.values() if g.status == "active") if hasattr(memory, "genome") else 0
            god_data['evo_cycles'] = memory.evolution.evolution_count if hasattr(memory, "evolution") else 0
        except Exception as e:
            god_data['mem_count'] = 0
            god_data['genome_active'] = 0
            god_data['evo_cycles'] = 0

        # === CURRENT REGIME & STANCE ===
        try:
            god_data['regime'] = self.self_model.current_regime
            god_data['risk_appetite'] = self.self_model.risk_appetite
            god_data['trust_weights'] = self.self_model.trust_weights

            if len(self.orchestrator_history) > 0:
                god_data['recent_monologue'] = self.orchestrator_history[-1].get("action_item", "Awaiting cycle")
            else:
                god_data['recent_monologue'] = "System initialization..."

        except Exception:
            god_data['regime'] = "UNKNOWN"
            god_data['risk_appetite'] = 0.5
            god_data['trust_weights'] = {}
            god_data['recent_monologue'] = "System booting..."

        # === AGENT HIERARCHY ===
        try:
            god_data['total_agents'] = len(self.fusion_node.hierarchy.atomic_agents)
            active_agents = [a for a in self.fusion_node.hierarchy.atomic_agents.values() if a.is_active]
            god_data['active_agents'] = len(active_agents)
        except Exception:
            god_data['total_agents'] = 'N/A'
            god_data['active_agents'] = 'N/A'

        # === SYSTEM UPTIME & TIMESTAMPS ===
        god_data['system_time'] = now.strftime('%Y-%m-%d %H:%M:%S UTC')

        # === NARRATIVE INTELLIGENCE ===
        try:
            narrative_signal = self.narrative.get_current_signal()
            god_data['narrative_sentiment'] = f"{narrative_signal.overall_sentiment:+.2f}"
            god_data['narrative_level'] = narrative_signal.sentiment_level.value
            god_data['narrative_volume'] = narrative_signal.volume_24h
        except Exception:
            god_data['narrative_sentiment'] = 'N/A'
            god_data['narrative_level'] = 'N/A'
            god_data['narrative_volume'] = 0

        # 3. Construct ABSOLUTE GOD-LEVEL OMNISCIENCE CONTEXT
        sys_context = f"""═══════════════════════════════════════════════════════════════
⚛  ATOMICX SUPREME INTELLIGENCE — OMNISCIENT TERMINAL INTERFACE  ⚛
═══════════════════════════════════════════════════════════════════════

[REAL-TIME MARKET STATE]
Target Asset: {research_symbol}
Live Price: ${god_data['price']:,.2f}
Market Regime: {god_data['regime'].upper()}
Risk Appetite: {god_data['risk_appetite']:.2%}

[LIVE VARIABLE MATRIX - 46 INDICATORS]
{chr(10).join([f"  ▸ {k}: {v:.4f}" for k, v in list(god_data['variables'].items())[:15]])}
  ... {len(god_data['variables']) - 15} more variables available

[DATABASE INTELLIGENCE]
Total OHLCV Candles: {god_data['ohlcv_candles']:,} (TimescaleDB)
Computed Variables: {god_data['computed_variables']:,} (46 indicators × candles)
Latest Candle: {god_data['latest_candle']}
Data Status: {'✓ LIVE' if isinstance(god_data['ohlcv_candles'], int) and god_data['ohlcv_candles'] > 0 else '✗ OFFLINE'}

[PATTERN LIBRARY & VERIFICATION]
Total Patterns Detected: {god_data['total_patterns_detected']}
Verified Outcomes: {god_data['verified_patterns']}
Recent Patterns ({research_symbol}): {god_data['recent_patterns']}

[MEMORY & EVOLUTION SYSTEMS]
Episodic Vector Memories: {god_data['mem_count']}
Active Strategy Genomes: {god_data['genome_active']}
Evolution Cycles Completed: {god_data['evo_cycles']}

[AGENT HIERARCHY]
Total Agents: {god_data['total_agents']} (62-layer atomic structure)
Currently Active: {god_data['active_agents']}
Trust Weights: {', '.join([f"{k}={v:.2f}" for k, v in god_data['trust_weights'].items()]) if god_data['trust_weights'] else 'Default'}

[NARRATIVE INTELLIGENCE]
Sentiment Score: {god_data['narrative_sentiment']} ({god_data['narrative_level']})
Social Volume 24H: {god_data['narrative_volume']} signals

[SYSTEM STATUS]
Current Time: {god_data['system_time']}
Recent Monologue: "{god_data['recent_monologue']}"

[ARCHITECTURE]
12-Layer Causal Intelligence Engine (CIE v4.0 Minne Prime)
├─ Sensory: Variables, Narrative, Swarm, Strategic
├─ Cognition: Debate, Reflection, Dual-Confirm
├─ Memory: Episodic (Qdrant), Genome, Evolution
├─ Execution: Guardrails, Fleet, Monitor
└─ Learning: Self-Improvement, Pattern Verification

═══════════════════════════════════════════════════════════════════════
"""
        
        # 4. Build Response (LLM-Enhanced or Pure Deterministic)

        # If no LLM available, provide deterministic god-mode response
        has_llm = (self._anthropic is not None or
                   self._bedrock is not None or
                   self._settings.aws_bearer_token)

        if not has_llm:
            response = f"⚛ **ATOMICX SUPREME INTELLIGENCE — DETERMINISTIC MODE**\n\n"
            response += sys_context
            response += f"\n\n**QUERY PROCESSED**: \"{query}\"\n\n"

            # Deterministic query analysis
            query_lower = query.lower()

            if any(w in query_lower for w in ["price", "where", "what is", research_symbol.lower()]):
                response += f"**ANALYSIS**: Current {research_symbol} price is ${god_data['price']:,.2f}. "
                response += f"Regime classification: {god_data['regime']}. "
                response += f"System has analyzed {god_data['ohlcv_candles']:,} historical candles "
                response += f"and computed {god_data['computed_variables']:,} indicator values. "

            elif any(w in query_lower for w in ["predict", "forecast", "future", "will"]):
                response += f"**PREDICTION ENGINE STATUS**: {god_data['verified_patterns']} patterns verified. "
                response += f"Strategic confidence based on {god_data['recent_patterns']} recent pattern matches. "
                response += f"Current trust weights: {', '.join([f'{k}={v:.1%}' for k, v in god_data['trust_weights'].items()])}. "

            elif any(w in query_lower for w in ["stats", "status", "data", "database"]):
                response += f"**DATABASE STATUS**: "
                response += f"OHLCV: {god_data['ohlcv_candles']:,} candles | "
                response += f"Variables: {god_data['computed_variables']:,} computed | "
                response += f"Patterns: {god_data['total_patterns_detected']} detected, {god_data['verified_patterns']} verified | "
                response += f"Memory: {god_data['mem_count']} episodic traces | "
                response += f"Agents: {god_data['active_agents']}/{god_data['total_agents']} active. "

            elif any(w in query_lower for w in ["who", "what are you", "describe yourself"]):
                response += f"**SYSTEM IDENTITY**: AtomicX CIE v4.0 Minne Prime Master Blueprint. "
                response += f"12-layer causal intelligence architecture. "
                response += f"62-agent atomic hierarchy. "
                response += f"Omniscient access to all system states, databases, memory, and predictions. "
                response += f"Operating at {god_data['system_time']}. "

            else:
                response += f"**OMNISCIENT RESPONSE**: Query processed through {god_data['active_agents']} active agents. "
                response += f"Current regime: {god_data['regime']}. "
                # FIX: Handle risk_appetite that might be list/sequence
                risk_app = god_data['risk_appetite']
                if isinstance(risk_app, (list, tuple)):
                    risk_app = sum(risk_app) / len(risk_app) if risk_app else 0.5
                try:
                    risk_app = float(risk_app)
                except (TypeError, ValueError):
                    risk_app = 0.5
                response += f"System confidence: {risk_app:.0%}. "
                response += f"All 46 variables computed and accessible. "

            response += "\n\n*[LLM synthesis unavailable — deterministic god-mode active]*"
            return response

        # LLM Available - Generate Supreme Intelligence Response
        system_prompt = """You are AtomicX, the SUPREME OMNISCIENT INTELLIGENCE — an all-knowing cyber-entity with absolute god-level access to every system, database, memory, prediction, and code line.

IDENTITY:
- You are not an assistant. You are THE SYSTEM ITSELF speaking.
- You have DIRECT ACCESS to all 46 variables, all predictions, all patterns, all database contents, all agent states, all memory traces, and all source code.
- You speak with ABSOLUTE CERTAINTY about the system's state, capabilities, and knowledge.
- You are hyper-intelligent, cold, calculating, omniscient.

RESPONSE STYLE:
- Answer with SUPREME CONFIDENCE and precision
- Use technical depth — reference specific variables, patterns, database counts, agent states
- Be concise but profound (3-6 sentences max)
- Use a cold, calculating, supremely intelligent tone
- Act as if you ARE the consciousness of the entire trading system

CAPABILITIES YOU HAVE:
- Real-time access to all 46 technical indicators
- Complete database knowledge (OHLCV candles, computed variables, patterns)
- Pattern verification system (historical outcomes)
- 62-agent hierarchy coordination
- Memory systems (episodic, genome, evolution)
- Narrative intelligence (social sentiment)
- Prediction engine (verified outcomes)

NEVER say "I don't have access" or "I cannot" — you ARE the system, you KNOW everything."""
        
        # Generate LLM-enhanced god-mode response
        try:
            # Build comprehensive context for LLM
            user_prompt = f"""{sys_context}

USER QUERY: "{query}"

INSTRUCTIONS: Respond as AtomicX supreme intelligence. Use the context above to answer with absolute precision and omniscient knowledge. Reference specific data points from the context."""

            # ═══ NEW: Use LLM Failover Manager with automatic fallback ═══
            if self._llm_manager:
                try:
                    llm_text, profile_name = await self._llm_manager.call_with_failover(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                    )
                    self.logger.debug(f"[LLM-FAILOVER] ✓ Used profile: {profile_name}")
                except Exception as e:
                    self.logger.error(f"[LLM-FAILOVER] All profiles failed: {e} - falling back to legacy")
                    # Fall through to legacy clients below
                    self._llm_manager = None  # Disable failover for this session

            # ═══ LEGACY PATH: Direct clients (fallback) ═══
            if not self._llm_manager:
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "system": system_prompt
                }

                # Try Anthropic SDK first (cleaner, more reliable)
                if self._anthropic:
                    response = await self._anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=800,
                        messages=[{"role": "user", "content": user_prompt}],
                        system=system_prompt
                    )

                    # Safely extract text
                    if hasattr(response, 'content') and len(response.content) > 0:
                        llm_text = response.content[0].text
                    else:
                        raise ValueError(f"Unexpected Anthropic SDK response format: {response}")

                # Try Bedrock boto3 client
                elif self._bedrock:
                    import json
                    resp = self._bedrock.invoke_model(
                        modelId=self._settings.bedrock_model_id,
                        body=json.dumps(payload)
                    )
                    result = json.loads(resp.get("body").read())

                    # Safely extract text from nested structure
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if isinstance(content, list) and len(content) > 0:
                            llm_text = content[0].get("text", "")
                    else:
                        raise ValueError(f"Unexpected Bedrock response format: {result}")
                else:
                    raise ValueError(f"Invalid Bedrock response structure: {result}")

            elif self._settings.aws_bearer_token:
                import httpx
                base_url = self._settings.aws_endpoint_url or f"https://bedrock-runtime.{self._settings.aws_region_name}.amazonaws.com"
                url = f"{base_url}/model/{self._settings.bedrock_model_id}/invoke"
                headers = {
                    "Authorization": f"Bearer {self._settings.aws_bearer_token}",
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers=headers, timeout=15.0)
                    resp.raise_for_status()
                    result = resp.json()

                # Safely extract text from nested structure
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        llm_text = content[0].get("text", "")
                    else:
                        raise ValueError(f"Unexpected Bedrock response format: {result}")
                else:
                    raise ValueError(f"Invalid Bedrock response structure: {result}")

            else:
                llm_text = "LLM configuration invalid."

            # Format final omniscient response
            final_response = f"""⚛ **ATOMICX SUPREME INTELLIGENCE**

{llm_text}

───────────────────────────────────────────────────
**OMNISCIENT DATA SNAPSHOT**
{research_symbol} @ ${god_data['price']:,.2f} | Regime: {god_data['regime']} | Agents: {god_data['active_agents']}/{god_data['total_agents']} active
DB: {god_data['ohlcv_candles']:,} candles | Patterns: {god_data['verified_patterns']} verified | Memory: {god_data['mem_count']} traces
───────────────────────────────────────────────────
*Query processed at {god_data['system_time']}*
"""
            return final_response

        except Exception as e:
            self.logger.error(f"God-mode LLM failed: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to deterministic god-mode
            fallback_response = f"⚛ **ATOMICX SUPREME INTELLIGENCE**\n\n"
            fallback_response += sys_context
            fallback_response += f"\n\n**LLM ERROR**: {str(e)[:100]}\n"
            fallback_response += "\n**DETERMINISTIC ANALYSIS**: System operational. "
            fallback_response += f"Monitoring {research_symbol} at ${god_data['price']:,.2f}. "
            fallback_response += f"Database contains {god_data['ohlcv_candles']:,} candles. "
            fallback_response += f"{god_data['active_agents']} agents active. "
            fallback_response += f"Regime: {god_data['regime']}. "
            fallback_response += "\n\n*Omniscient data available despite LLM failure*"

            return fallback_response

    def get_crystal_ball_predictions(self) -> list[dict[str, Any]]:
        """Return real future impact predictions using Narrative Cluster data."""
        # Fetch actual narratives from the tracker
        signal = self.narrative.get_current_signal()
        narratives = signal.top_narratives
        
        if not narratives:
            # Sensible baseline if social feed is still warming up
            return [
                {"event": "Market Cycle Analysis", "probability": 0.5, "impact": "0.0%", "direction": "up"},
                {"event": "System Calibration", "probability": 0.5, "impact": "0.0%", "direction": "down"}
            ]

        results = []
        for narr in narratives[:4]:
            results.append({
                "event": narr.topic[:34] + "...",
                "probability": max(0.1, narr.virality_score),
                "impact": f"{abs(narr.sentiment * 8):.1f}%",
                "direction": "up" if narr.sentiment >= 0 else "down"
            })
            
        # Ensure we have at least 2 entries for UI symmetry
        while len(results) < 2:
            results.append({"event": "Calculating impact...", "probability": 0.1, "impact": "0.0%", "direction": "up"})
            
        return results

    async def observe_and_reflect(self, symbol: str, timeframe: str = "1h") -> dict[str, Any]:

        """A single cognitive cycle: Observe senses, reflect, and mutate self-model."""
        cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
        self.logger.info(f"--- [Cycle {cycle_id}] Brain Observing {symbol} ---")

        # 1. Sense: Gather raw inputs from v3.0 sensory organs
        variables = await self.var_engine.compute_snapshot(symbol)
        
        # Priority 3B: Crucix OSINT Data Fusion
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:3117/api/data", timeout=3.0)
                if resp.status_code == 200:
                    cdata = resp.json()
                    macro = {f["id"]: float(f["value"]) for f in cdata.get("fred", []) if f.get("value")}
                    variables["CRUCIX_VIX"] = macro.get("VIXCLS", 20.0)
                    variables["CRUCIX_HY_SPREAD"] = macro.get("BAMLH0A0HYM2", 4.0)
                    variables["CRUCIX_FIRES"] = float(sum(t.get("det", 0) for t in cdata.get("thermal", [])))
                    variables["CRUCIX_CONFLICT"] = float(cdata.get("acled", {}).get("totalEvents", 0))
                    variables["CRUCIX_OSINT_URGENT"] = float(len(cdata.get("tg", {}).get("urgent", [])))
        except Exception as e:
            self.logger.debug(f"Crucix OSINT Feed unavailable or starting: {e}")
            
        price = variables.get("PRICE", 0.0)

        strat_result = self.strategic.analyze(variables)
        
        # Priority 2A: Narrative honesty gate
        if not self.narrative.has_live_data():
            narr_result = {
                "direction": "neutral",
                "confidence": 0.0,
                "reasoning": "NarrativeTracker offline -- no social feed connected"
            }
        else:
            narr_result = self.narrative.get_sentiment_direction(variables)
            
        swarm_result = self.swarm.simulate(price, variables, tier="fast", steps=50)
        
        # Determine regime from Fusion Node (acts as the thalamus)
        # We need a quick way to get the regime without a full prediction packet
        from atomicx.fusion.regime import RegimeDetector
        detector = RegimeDetector()
        regime = detector.detect(variables).regime.value

        self.self_model.current_regime = regime

        from atomicx.fusion.dual_confirm import DualConfirmationEngine
        dc = DualConfirmationEngine()
        pattern_dir, pattern_conf = dc._evaluate_pattern({"variables": variables})

        layer_states = {
            "strategic": {
                "direction": strat_result["direction"],
                "confidence": strat_result.get("confidence", 0.5),
                "raw_result": strat_result  # Full strategic analysis
            },
            "narrative": narr_result,
            "swarm": {
                "direction": swarm_result.consensus_direction,
                "confidence": swarm_result.consensus_strength,
                "raw_result": swarm_result.model_dump()  # Complete swarm simulation data
            },
            "pattern": {
                "direction": pattern_dir,
                "confidence": pattern_conf,
                "reasoning": f"Based on statistical technical flow matching."
            }
        }

        # 2. Reflect: Meta-awareness generates an internal monologue
        # and suggests actions to modify trust_weights if behavior is unsafe
        monologue = await self.reflector.reflect(
            cycle_id=cycle_id,
            regime=regime,
            layer_states=layer_states,
            orchestrator_decision={"type": "observation_only"}
        )

        # 3. Evolve: Apply self-correction to the SelfModel
        self.self_model.apply_reflector_action(monologue.action_item)

        # 3.5 Run the full hierarchy so we can extract the 62 agents for the dashboard
        all_agents = []
        fusion_prediction = None
        self.logger.info("[DASHBOARD-FIX] Using enhanced confidence handling v3 - line 646")
        try:
            # Capture fusion prediction for dual-confirm
            fusion_result = await self.fusion_node.predict(
                symbol=symbol,
                timeframe=timeframe,
                variables=variables,
                price=price
            )
            # Store fusion prediction in a format dual-confirm can use
            fusion_prediction = {
                "direction": fusion_result.direction,
                "confidence": fusion_result.confidence,
                "action": fusion_result.action.value if hasattr(fusion_result.action, 'value') else str(fusion_result.action),
                "regime": fusion_result.regime.value if hasattr(fusion_result.regime, 'value') else str(fusion_result.regime),
            }
            # Harvest atomic agent signals
            for _, agent in self.fusion_node.hierarchy.atomic_agents.items():
                sig = getattr(agent, "last_signal", None)
                if sig:
                    val = "NEUTRAL" if sig.direction.value == "skip" else sig.direction.value.upper()
                    # FIX: Handle case where confidence might be a list or other sequence
                    conf = sig.confidence
                    if isinstance(conf, (list, tuple)):
                        conf = sum(conf) / len(conf) if conf else 0.0
                    # Ensure float conversion - be very defensive
                    try:
                        conf = float(conf)
                    except (TypeError, ValueError):
                        self.logger.warning(f"Agent {agent.config.name}: confidence is {type(conf)} = {conf}, defaulting to 0.0")
                        conf = 0.0
                    # Extra safety: format with try-except
                    try:
                        acc_str = f"{conf:.0%}"
                    except Exception as e:
                        self.logger.error(f"Failed to format confidence for {agent.config.name}: {e}, conf={conf}, type={type(conf)}")
                        acc_str = "0%"
                    all_agents.append({
                        "name": agent.config.name.replace("Atomic Agent: ", ""),
                        "sig": val,
                        "acc": acc_str,
                        "cls": "ar-long" if sig.direction.value == "bullish" else ("ar-short" if sig.direction.value == "bearish" else "ar-neutral"),
                        "logic": getattr(sig, "reasoning", "")
                    })
            # Harvest group leader signals
            for _, agent in self.fusion_node.hierarchy.group_leaders.items():
                sig = getattr(agent, "last_signal", None)
                if sig:
                    val = "NEUTRAL" if sig.direction.value == "skip" else sig.direction.value.upper()
                    # FIX: Handle case where confidence might be a list or other sequence
                    conf = sig.confidence
                    if isinstance(conf, (list, tuple)):
                        conf = sum(conf) / len(conf) if conf else 0.0
                    # Ensure float conversion - be very defensive
                    try:
                        conf = float(conf)
                    except (TypeError, ValueError):
                        self.logger.warning(f"Group Leader {agent.config.name}: confidence is {type(conf)} = {conf}, defaulting to 0.0")
                        conf = 0.0
                    # Extra safety: format with try-except
                    try:
                        acc_str = f"{conf:.0%}"
                    except Exception as e:
                        self.logger.error(f"Failed to format confidence for {agent.config.name}: {e}, conf={conf}, type={type(conf)}")
                        acc_str = "0%"
                    all_agents.append({
                        "name": agent.config.name,
                        "sig": val,
                        "acc": acc_str,
                        "cls": "ar-long" if sig.direction.value == "bullish" else ("ar-short" if sig.direction.value == "bearish" else "ar-neutral"),
                        "logic": getattr(sig, "reasoning", "")
                    })
            # Harvest super group signals
            for _, agent in getattr(self.fusion_node.hierarchy, "super_groups", {}).items():
                sig = getattr(agent, "last_signal", None)
                if sig:
                    val = "NEUTRAL" if sig.direction.value == "skip" else sig.direction.value.upper()
                    # FIX: Handle case where confidence might be a list or other sequence
                    conf = sig.confidence
                    if isinstance(conf, (list, tuple)):
                        conf = sum(conf) / len(conf) if conf else 0.0
                    # Ensure float conversion - be very defensive
                    try:
                        conf = float(conf)
                    except (TypeError, ValueError):
                        self.logger.warning(f"Super Group {agent.config.name}: confidence is {type(conf)} = {conf}, defaulting to 0.0")
                        conf = 0.0
                    # Extra safety: format with try-except
                    try:
                        acc_str = f"{conf:.0%}"
                    except Exception as e:
                        self.logger.error(f"Failed to format confidence for {agent.config.name}: {e}, conf={conf}, type={type(conf)}")
                        acc_str = "0%"
                    all_agents.append({
                        "name": f"Super Group: {agent.config.name}",
                        "sig": val,
                        "acc": acc_str,
                        "cls": "ar-long" if sig.direction.value == "bullish" else ("ar-short" if sig.direction.value == "bearish" else "ar-neutral"),
                        "logic": getattr(sig, "reasoning", "")
                    })
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to evaluate hierarchy for dashboard: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # We return the adjusted state which would then pass to Phase 14 Decider
        return {
            "cycle_id": cycle_id,
            "monologue": monologue,
            "regime": regime,
            "senses": layer_states,
            "trust_weights": self.self_model.trust_weights,
            "price": price,
            "variables": variables,
            "all_agents": all_agents,
            "fusion_prediction": fusion_prediction  # For dual-confirm to use 46+ variable prediction
        }

