"""Cognitive Loop — The Asynchronous Main Event Loop of the Brain.

Replaces the linear run script with an Observe -> Simulate -> Debate -> Decide -> Act -> Reflect loop.
v5.2: Autonomous Evolution Loop — predict, verify, learn, mutate, repeat.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any
import pandas as pd
from loguru import logger

from atomicx.brain.orchestrator import MetaOrchestrator
from atomicx.brain.debate import DebateChamber
from atomicx.brain.decider import DeciderCore
from atomicx.brain.evolver import EvolverAgent
from atomicx.execution.receiver import CommandReceiver
from atomicx.execution.orchestrator import ActionOrchestrator
from atomicx.execution.manager import FleetManager
from atomicx.execution.monitor import LiveMonitor
from atomicx.guardrails import TradabilityGuardrails
from atomicx.memory.orchestrator import MemoryOrchestrator

# Milestone 3: Intelligence Expansion
from atomicx.intelligence.scanner import NewsScanner
from atomicx.intelligence.browser_agent import BrowserAgent
from atomicx.intelligence.knowledge_graph import KnowledgeGraph
from atomicx.intelligence.impact import ImpactPredictor
from atomicx.intelligence.news_intelligence import NewsIntelligence
from atomicx.dashboard.app import DashboardState
from atomicx.fusion.dual_confirm import DualConfirmationEngine
from atomicx.fusion.prediction import PredictionPacket

# Milestone 4: Titan-Killer
from atomicx.intelligence.titan.kernel_engine import KernelCorrelationEngine
from atomicx.intelligence.titan.retail_flow import RetailFlowAnticipator
from atomicx.intelligence.titan.expiry_sentinel import ExpiryDaySentinel


class PendingPrediction:
    """A prediction awaiting verification."""
    __slots__ = ('packet', 'entry_price', 'symbol', 'regime', 'created_at')

    def __init__(self, packet: PredictionPacket, entry_price: float, symbol: str, regime: str) -> None:
        self.packet = packet
        self.entry_price = entry_price
        self.symbol = symbol
        self.regime = regime
        self.created_at = datetime.now(tz=timezone.utc)


class CognitiveLoop:
    """The continuous heartbeat of the Minne Prime architecture.
    
    v5.2 Evolution Loop:
    Each cycle: Observe → Predict → Save → Verify Past → Learn → Evolve
    """
    
    # How long to wait before verifying a prediction (seconds)
    VERIFICATION_DELAY_SECONDS = 15 * 60  # 15 minutes
    # FIXED: Directional correctness - if predicted UP and it went UP (any amount) = correct
    # No minimum threshold - direction matters, not magnitude
    CORRECTNESS_THRESHOLD = 0.0  # Any movement in predicted direction counts
    
    def __init__(self, orchestrator: MetaOrchestrator, symbols: list[str], self_improvement: Any = None, ws_client: Any = None) -> None:
        self.orchestrator = orchestrator
        self.symbols = symbols
        self.self_improvement = self_improvement
        self.ws_client = ws_client
        self._running = False
        self.logger = logger.bind(module="brain.loop")

        # Phase 14 Cognition Modules
        self.debate_chamber = DebateChamber()
        self.decider = DeciderCore()

        # Inject MemoryService into Reflector for Phase 14/16 integration
        from atomicx.memory.service import MemoryService
        self.memory_service = MemoryService()
        self.orchestrator.reflector.memory = self.memory_service

        self.evolver = EvolverAgent(reflector=self.orchestrator.reflector)
        self.cycle_count = 0

        # Phase 15 Action Engine Modules
        self.guardrails = TradabilityGuardrails()
        self.fleet_manager = FleetManager()
        self.command_receiver = CommandReceiver(guardrails=self.guardrails)
        self.action_orchestrator = ActionOrchestrator(fleet_manager=self.fleet_manager)
        self.live_monitor = LiveMonitor(fleet_manager=self.fleet_manager)
        
        # Phase 16 Dynamic Memory Nexus
        self.memory = MemoryOrchestrator()
        
        # Phase 17: Autonomous Web Intelligence
        self.news_scanner = NewsScanner()
        self.news_intelligence = NewsIntelligence()
        self.browser_agent = BrowserAgent()
        self.knowledge_graph = KnowledgeGraph()
        
        # Phase 18: Dashboard State
        self.dashboard = DashboardState()
        
        # Phase 20: Dual-Confirmation Engine (50/50 Pattern vs Logic)
        self.dual_confirm = DualConfirmationEngine()
        
        # Phase 21: Impact Predictor (Crystal Ball)
        self.impact_predictor = ImpactPredictor(
            browser=self.browser_agent, graph=self.knowledge_graph
        )
        
        # Phase 22-24: Titan-Killer Modules
        self.kernel_engine = KernelCorrelationEngine()
        self.retail_flow = RetailFlowAnticipator()
        self.expiry_sentinel = ExpiryDaySentinel()

        # Phase 25: Pattern Library & Verification
        from atomicx.data.pattern_verification import PatternVerificationService
        self.pattern_verifier = PatternVerificationService()

        # ═══ EVOLUTION LOOP STATE ═══
        # Predictions waiting to be verified against actual price
        self.pending_predictions: list[PendingPrediction] = []
        # History of verified outcomes (for dashboard + analysis)
        # FIX: Use deque with max length to prevent unbounded memory growth
        self.prediction_history: deque = deque(maxlen=1000)
        # Track the last known price per symbol for verification
        self._last_prices: dict[str, float] = {}
        # ═══ FIX: Track regime per symbol for change detection ═══
        self._last_regimes: dict[str, str] = {}

        # ═══ FIX: Initialize persistence manager for predictions + regimes ═══
        from atomicx.brain.persistence import PredictionPersistence
        self.persistence = PredictionPersistence()
        self.logger.info("Prediction persistence manager initialized")

        # ═══ AUTONOMOUS EVOLUTION SYSTEM ═══
        from atomicx.evolution.config_manager import get_config_manager
        from atomicx.evolution.self_improvement import AutonomousSelfImprovementEngine
        from atomicx.evolution.code_evolution import CodeEvolutionEngine
        from atomicx.evolution.meta_learning import MetaLearningEngine

        self.config_manager = get_config_manager()
        self.evolution_engine = AutonomousSelfImprovementEngine(config_manager=self.config_manager)
        self.code_evolution = CodeEvolutionEngine()
        self.meta_learning = MetaLearningEngine()
        self.logger.info("Autonomous evolution system initialized")
        
    async def start(self) -> None:
        """Begin continuous autonomous operations.

        ═══ FIX: Now loads all persisted state on startup ═══
        """
        self._running = True
        self.logger.info("INITIATING COGNITIVE LOOP (AtomicX v5.2 — Autonomous Evolution)")

        # Initialize memory service
        await self.memory_service.initialize()

        # ═══ FIX: Initialize persistence and load state ═══
        await self.persistence.initialize()

        # ═══ EVOLUTION: Initialize autonomous evolution system ═══
        await self.config_manager.initialize()
        await self.evolution_engine.initialize()

        # Start evolution background tasks (with error handling)
        self._safe_create_task(
            self.evolution_engine.shadow_testing_loop(),
            "evolution_shadow_testing"
        )
        self._safe_create_task(
            self.evolution_engine.deployment_loop(),
            "evolution_deployment"
        )
        if self.code_evolution.is_enabled():
            self._safe_create_task(
                self.code_evolution.monitor_and_rollback_loop(),
                "code_evolution_monitor"
            )
            self.logger.warning("[EVOLUTION] Code evolution is ENABLED - system can modify its own code")

        self.logger.success("[EVOLUTION] Autonomous evolution system started")

        # Load agent performance from database
        self.logger.info("[PERSISTENCE] Loading agent performance...")
        for agent in self.orchestrator.fusion_node.hierarchy.get_all_agents():
            await agent.load_performance()

        # Load strategy genome from database
        self.logger.info("[PERSISTENCE] Loading strategy genome...")
        genome_count = await self.memory.genome.load_genes()
        if genome_count > 0:
            self.logger.success(f"[PERSISTENCE] Restored {genome_count} strategy genes")
        else:
            self.logger.info(f"[PERSISTENCE] No existing genes found (new system)")

        # Load pending predictions from Redis
        self.logger.info("[PERSISTENCE] Loading pending predictions...")
        pending_data = await self.persistence.load_pending_predictions()
        for pred_data in pending_data:
            try:
                # Reconstruct PendingPrediction object
                from atomicx.fusion.prediction import PredictionPacket
                packet = PredictionPacket.model_validate_json(pred_data["packet_json"])

                pending_pred = PendingPrediction(
                    packet=packet,
                    entry_price=pred_data["entry_price"],
                    symbol=pred_data["symbol"],
                    regime=pred_data["regime"],
                )
                # Restore created_at timestamp
                pending_pred.created_at = datetime.fromisoformat(pred_data["created_at"])

                self.pending_predictions.append(pending_pred)
            except Exception as e:
                self.logger.warning(f"Failed to restore pending prediction: {e}")

        if self.pending_predictions:
            self.logger.success(
                f"[PERSISTENCE] Restored {len(self.pending_predictions)} pending predictions"
            )

        # Load recent outcomes from database
        self.logger.info("[PERSISTENCE] Loading recent outcomes...")
        outcomes = await self.persistence.load_recent_outcomes(limit=100)
        # FIX: Convert to deque (maxlen already set in __init__)
        self.prediction_history.extend(outcomes)
        if self.prediction_history:
            self.logger.success(
                f"[PERSISTENCE] Restored {len(self.prediction_history)} verified outcomes"
            )

        # Load last regime for each symbol
        from atomicx.brain.persistence import load_last_regime
        for symbol in self.symbols:
            last_regime = await load_last_regime(symbol)
            if last_regime:
                self._last_regimes[symbol] = last_regime
                self.logger.info(f"[PERSISTENCE] Restored last regime for {symbol}: {last_regime}")

        # ═══ FIX: Scan for orphaned predictions and verify them ═══
        self.logger.info("[ORPHAN-SCAN] Scanning for orphaned predictions...")
        orphaned = await self.persistence.scan_orphaned_predictions()

        if orphaned:
            self.logger.warning(
                f"[ORPHAN-RECOVERY] Found {len(orphaned)} orphaned predictions - verifying now..."
            )

            # Fetch current prices for symbols
            current_prices = {}
            try:
                for symbol in set(pred["symbol"] for pred in orphaned):
                    vars_snapshot = await self.orchestrator.var_engine.compute_snapshot(symbol)
                    current_prices[symbol] = vars_snapshot.get("PRICE", 0.0)
            except Exception as e:
                self.logger.error(f"[ORPHAN-RECOVERY] Failed to fetch current prices: {e}")

            # Verify each orphaned prediction
            recovered_count = 0
            for pred in orphaned:
                symbol = pred["symbol"]
                current_price = current_prices.get(symbol, 0.0)

                if current_price > 0:
                    try:
                        was_correct, actual_return = await self.persistence.verify_orphaned_prediction(
                            prediction_id=pred["prediction_id"],
                            symbol=symbol,
                            predicted_direction=pred["predicted_direction"],
                            entry_price=pred["entry_price"],
                            current_price=current_price,
                            predicted_at=pred["predicted_at"],
                        )

                        # Update database with verified outcome
                        await self.persistence.save_prediction_outcome(
                            prediction_id=pred["prediction_id"],
                            symbol=symbol,
                            timeframe=pred["timeframe"],
                            predicted_direction=pred["predicted_direction"],
                            confidence=pred["confidence"],
                            entry_price=pred["entry_price"],
                            verification_price=current_price,
                            was_correct=was_correct,
                            actual_return=actual_return,
                            profit_return=actual_return if was_correct else 0.0,
                            regime=pred["regime"],
                            predicted_at=pred["predicted_at"],
                            verified_at=datetime.now(tz=timezone.utc),
                            reasoning=None,
                            variable_snapshot=None,
                        )

                        recovered_count += 1
                    except Exception as e:
                        self.logger.error(
                            f"[ORPHAN-RECOVERY] Failed to verify {pred['prediction_id']}: {e}"
                        )

            if recovered_count > 0:
                self.logger.success(
                    f"[ORPHAN-RECOVERY] Successfully recovered {recovered_count}/{len(orphaned)} orphaned predictions"
                )

                # TODO: Update agent performance metrics based on recovered predictions
                # This requires mapping prediction_id back to which agents participated
                self.logger.warning(
                    "[ORPHAN-RECOVERY] Agent performance metrics NOT updated - need to implement agent mapping"
                )

        self.logger.success("[PERSISTENCE] All state loaded successfully")

        # Pre-populate dashboard so UI loads immediately before LLM debate finishes
        for symbol in self.symbols:
            self.dashboard.update_symbol_data(symbol, {"regime": "Initializing..."})
            
        while self._running:
            cycle_start = time.monotonic()
            
            for symbol in self.symbols:
                await self._run_cycle(symbol)
                
            self.cycle_count += 1
            
            # ═══ EVOLUTION: Verify past predictions ═══
            await self._verify_pending_predictions()

            # ═══ PATTERN LIBRARY: Verify pattern outcomes ═══
            if self.cycle_count % 10 == 0:  # Every 10 cycles
                await self._verify_pattern_outcomes()

            # ═══ EVOLUTION: Push stats to dashboard ═══
            self._sync_dashboard_stats()

            # ═══ EVOLUTION: System diagnosis (every 50 cycles) ═══
            if self.cycle_count % 50 == 0 and self.cycle_count > 0:
                diagnosis_interval = self.config_manager.get("evolution.diagnosis_interval_cycles", default=50)
                if self.cycle_count % int(diagnosis_interval) == 0:
                    self.logger.info("--- Running Autonomous Diagnosis ---")
                    await self.evolution_engine.run_diagnosis(cycles_since_last=50)

            # ═══ FIX: Causal discovery - learn real causal relationships ═══
            if self.cycle_count % 100 == 0 and self.cycle_count > 0:  # Every 100 cycles (~8 hours)
                self.logger.info("--- Running Causal Discovery ---")
                await self._run_causal_discovery()

                # ═══ EVOLUTION: Weight evolution + report generation (every 100 cycles) ═══
                self.logger.info("--- Running Autonomous Weight Evolution ---")
                await self.evolution_engine.run_weight_evolution(cycles_since_last=100)

                self.logger.info("--- Generating Evolution Report ---")
                await self.evolution_engine.generate_evolution_report()

                # ═══ META-LEARNING: Analyze evolution patterns (every 100 cycles) ═══
                if self.cycle_count % 200 == 0:
                    self.logger.info("--- Running Meta-Learning Analysis ---")
                    await self.meta_learning.analyze_proposal_patterns()
                    await self.meta_learning.analyze_overconfidence()
                    await self.meta_learning.optimize_evolution_parameters()
                    await self.meta_learning.analyze_regime_learning_patterns()

            # ═══ FIX: CausalRL periodic weight adjustment (now integrated with evolution) ═══
            if self.cycle_count % 50 == 0 and self.cycle_count > 0:  # Every 50 cycles (~4 hours)
                self.logger.info("--- Running CausalRL Weight Adjustment ---")
                await self._adjust_causal_weights()

            if self.cycle_count % 10 == 0:
                self.logger.info("--- Running Evolver Analysis ---")
                await self.evolver.analyze_and_propose()

                # ═══ FIX: Enact evolution proposals ═══
                await self._enact_evolution_proposals()
                
            # Tick the Action Engine Fleet with real price
            for sym, price in self._last_prices.items():
                if price > 0:
                    await self.fleet_manager.tick_fleet(current_price=price, liquidity_profile={"bid_depth": 100000})
                    break  # tick once with primary symbol
            self.live_monitor.generate_report()
            
            # Pulse the Dynamic Memory Nexus (Phase 16)
            await self.memory.cycle_maintenance(memory_service=self.memory_service)
            
            # Intelligence Scan (Phase 17): Every 5 cycles
            if self.cycle_count % 5 == 0:
                await self._intelligence_scan()

            # Log comprehensive health report every 20 cycles
            if self.cycle_count % 20 == 0:
                self._log_narrative_health_report()
            
            # Titan-Killer Checks (Phases 22-24)
            self._run_titan_checks()
            
            # Log cycle timing
            cycle_ms = (time.monotonic() - cycle_start) * 1000
            if self.cycle_count % 5 == 0:
                self.logger.info(
                    f"[CYCLE #{self.cycle_count}] {cycle_ms:.0f}ms | "
                    f"Pending: {len(self.pending_predictions)} | "
                    f"Verified: {len(self.prediction_history)} | "
                    f"Win Rate: {self._get_win_rate():.1%}"
                )
            
            # FIX: Use dominant regime from dict (not singular variable that flips)
            self.dashboard.active_regime = self._get_dominant_regime()
            
            # Brain sleep
            await asyncio.sleep(5)
            
    def stop(self) -> None:
        """Halt continuous operations."""
        self._running = False
        self.fleet_manager.emergency_kill_switch()
        self.logger.info(f"HALTING COGNITIVE LOOP — {len(self.prediction_history)} predictions verified, {len(self.pending_predictions)} still pending")

    def add_symbol(self, symbol: str) -> None:
        """Dynamically add a new symbol to the brain's monitoring list."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.logger.success(f"DYNAMIC ASSET ADDED: {symbol}. Initializing resonance cycles...")
            if self.ws_client:
                self._safe_create_task(
                    self.ws_client.subscribe(symbol),
                    f"websocket_subscribe_{symbol}"
                )
            if hasattr(self.orchestrator.var_engine, '_symbols'):
                if symbol not in self.orchestrator.var_engine._symbols:
                    self.orchestrator.var_engine._symbols.append(symbol)

    async def _run_cycle(self, symbol: str) -> None:
        """Execute one complete cognitive cycle for a symbol."""
        cycle_start = time.monotonic()

        # Phase 17: News Scanning & Social Logic
        news_items = await self.news_scanner.scan_cycle()
        social_items = await self.news_scanner.scan_social()
        all_news = news_items + social_items

        # Monitor scanner health
        if self.cycle_count % 10 == 0:  # Check every 10 cycles
            scanner_health = self.news_scanner.get_health_status()
            narrative_health = self.orchestrator.narrative.get_health_status()

            if not scanner_health["overall_healthy"]:
                self.logger.error(
                    f"[NARRATIVE] NewsScanner UNHEALTHY: {scanner_health['healthy_sources']}/{scanner_health['total_sources']} sources working"
                )
            elif scanner_health["healthy_sources"] < scanner_health["total_sources"]:
                self.logger.warning(
                    f"[NARRATIVE] NewsScanner DEGRADED: {scanner_health['healthy_sources']}/{scanner_health['total_sources']} sources working"
                )

            if narrative_health["is_stale"]:
                self.logger.error(
                    f"[NARRATIVE] NarrativeTracker data STALE: {narrative_health['seconds_since_ingest']:.0f}s since last update"
                )

        for item in all_news:
            # ═══ FIX: Real NLP sentiment analysis ═══
            from atomicx.common.sentiment import analyze_sentiment

            sentiment_result = analyze_sentiment(item.title)
            # Convert [-1, +1] to [0, 1] for NarrativeTracker
            sentiment_score = (sentiment_result["score"] + 1.0) / 2.0

            self.logger.debug(
                f"[SENTIMENT] {item.source}: '{item.title[:60]}...' → "
                f"{sentiment_result['category']} (score: {sentiment_result['score']:.2f}, "
                f"confidence: {sentiment_result['confidence']:.2f})"
            )

            # Ingest into NarrativeTracker to ensure the social layer stays online
            self.orchestrator.narrative.ingest_signal(
                source=item.source,
                text=item.title,
                sentiment=sentiment_score
            )
            
            if item.deep_dive_triggered:
                research = await self.browser_agent.deep_dive(item)
                from atomicx.memory.service import MemoryEntry, MemoryType
                try:
                    await self.memory_service.store(MemoryEntry(
                        memory_type=MemoryType.EPISODIC,
                        content=f"NEWS EVENT: {item.title}. Analysis: {research.get('sentiment_signal')}",
                        symbol=symbol,
                        metadata={"source": item.source, "url": item.url}
                    ))
                except Exception as e:
                    self.logger.warning(f"Failed to persist news fact to Qdrant: {e}")

        try:
            # 1. Observe & Reflect (Phase 13)
            brain_state = await self.orchestrator.observe_and_reflect(symbol)
            price_curr = brain_state.get("price", 0.0)
            if price_curr <= 0:
                self.logger.warning(f"No live price for {symbol} — skipping cycle")
                return

            # ═══ FIX: Data Staleness Gate - Detect "Ghost Market" ═══
            # Check when the last price update occurred (from sensory cache or OHLCV)
            try:
                from atomicx.common.cache import get_sensory_cache
                cache = get_sensory_cache()
                last_update_time = cache.get_timestamp(symbol, "LAST_PRICE")

                if last_update_time:
                    staleness_seconds = (datetime.now(tz=timezone.utc) - last_update_time).total_seconds()

                    if staleness_seconds > 300:  # 5 minutes stale = CRITICAL
                        self.logger.error(
                            f"[STALENESS-GATE] 🧟 GHOST MARKET DETECTED! Price for {symbol} is {staleness_seconds:.0f}s stale "
                            f"(last update: {last_update_time.strftime('%H:%M:%S UTC')}). ABORTING ALL TRADES!"
                        )
                        return  # Skip this cycle entirely
                    elif staleness_seconds > 60:  # 1 minute stale = WARNING
                        self.logger.warning(
                            f"[STALENESS-GATE] ⚠️ Price data {staleness_seconds:.0f}s old. Trading at reduced confidence."
                        )
            except Exception as e:
                self.logger.warning(f"[STALENESS-GATE] Failed to check data staleness: {e}")

            # Track price for verification
            self._last_prices[symbol] = price_curr
            current_regime = brain_state.get("regime", "TRANSITION")

            # ═══ FIX: Detect regime change and retrieve lessons ═══
            previous_regime = self._last_regimes.get(symbol)
            if previous_regime and previous_regime != current_regime:
                self.logger.warning(
                    f"[REGIME SHIFT] {symbol}: {previous_regime} → {current_regime}"
                )

                # ═══ FIX: Persist regime change to database ═══
                try:
                    from atomicx.brain.persistence import save_regime_change
                    regime_confidence = brain_state.get("regime_confidence", 0.0)
                    await save_regime_change(
                        symbol=symbol,
                        old_regime=previous_regime,
                        new_regime=current_regime,
                        confidence=regime_confidence,
                        drift_score=None,
                        trigger_reason=f"Regime detector: {previous_regime} → {current_regime}",
                        metadata={"variables": brain_state.get("variables", {})},
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to persist regime change: {e}")

                try:
                    regime_lessons = await self.memory_service.get_lessons_for_regime(
                        current_regime, limit=10
                    )
                    if regime_lessons:
                        lesson_summary = " | ".join([
                            m.get("content", m.get("memory", ""))[:80]
                            for m in regime_lessons[:3]
                        ])
                        self.logger.info(
                            f"[MEMORY] Retrieved {len(regime_lessons)} lessons for {current_regime}: "
                            f"{lesson_summary}..."
                        )
                        # Store lessons in brain_state for debate chamber
                        brain_state["regime_lessons"] = regime_lessons
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve regime lessons: {e}")

            # Update regime tracking
            self._last_regimes[symbol] = current_regime

            # Inject cycle timing into brain_state
            cycle_duration_ms = (time.monotonic() - cycle_start) * 1000
            brain_state["cycle_duration_ms"] = cycle_duration_ms
            
            # Stream raw sensory data into DMN Tier-0 (Phase 16)
            self.memory.stream_raw_data({"symbol": symbol, "price": price_curr, "state": brain_state})
            
            # 2. Debate (Phase 14)
            try:
                debate_summary = await self.debate_chamber.debate(brain_state)
            except Exception as e:
                self.logger.error(f"Debate chamber failed (LLM offline?): {e}. Using empirical fallback.")
                
                vars_snap = brain_state.get("variables", {})
                rsi = vars_snap.get("RSI_14", 50)
                adx = vars_snap.get("ADX", 20)
                if rsi < 40 and adx > 25:
                    emp_stance = "bullish"
                    emp_synth = "LLM Offline. Empirical Logic Fallback: Macro oversold structural divergence detected."
                elif rsi > 60 and adx > 25:
                    emp_stance = "bearish"
                    emp_synth = "LLM Offline. Empirical Logic Fallback: Macro overbought structural collapse detected."
                else:
                    emp_stance = "neutral"
                    emp_synth = "LLM Engine Offline: Market structure neutral. No causal direction."
                
                class MockDebate:
                    summary = emp_synth
                    synthesis = emp_synth
                    dominant_stance = emp_stance
                    overall_conviction = 0.5
                    arguments = []
                debate_summary = MockDebate()
            
            # 3. Dual-Confirmation Gate (Phase 20)
            confirmation = self.dual_confirm.evaluate(brain_state, debate_summary)
            
            # 4. Generate High-Fidelity Dashboard Data
            await self._update_high_fidelity_dashboard(symbol, brain_state, debate_summary, confirmation)
            
            # ═══ EVOLUTION: Track this prediction for future verification ═══
            await self._track_prediction(symbol, brain_state, confirmation)

            # ═══ PATTERN LIBRARY: Detect and store patterns ═══
            await self._detect_and_store_patterns(symbol, brain_state, confirmation)

            # ═══ GOD MODE: Update cognitive cache for instant dashboard ═══
            await self._update_god_mode_cache(symbol, brain_state, confirmation)

            if not confirmation.confirmed:
                return  # Pattern and Logic disagree → ABORT
            
            # 4. Decide (Phase 14) — only reached if dual-confirm passes
            intent = self.decider.decide(brain_state, debate_summary)
            
            # 5. Act (Phase 15: Action Engine Handoff)
            if intent.is_actionable():
                command = self.command_receiver.process_intent(symbol=symbol, price=price_curr, intent=intent)
                if command and command.status == "validated":
                    self.action_orchestrator.execute_intent(command)
                
        except Exception as e:
            self.logger.error(f"Cognitive cycle failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════
    # EVOLUTION LOOP: Prediction Tracking & Verification
    # ═══════════════════════════════════════════════════════════════════════

    async def _track_prediction(self, symbol: str, brain_state: dict, confirmation: Any) -> None:
        """Save the current cycle's prediction for future verification."""
        price = brain_state.get("price", 0.0)
        if price <= 0:
            return
            
        # Get the latest PredictionPacket from the FusionNode
        # The orchestrator runs FusionNode.predict() inside observe_and_reflect
        fusion = self.orchestrator.fusion_node
        if fusion.prediction_count == 0:
            return
            
        # Build a prediction record from what we know
        # Direction comes from the dual-confirm gate
        direction = "neutral"
        if confirmation.confirmed:
            direction = "bullish" if "buy" in confirmation.final_direction else "bearish"
        elif confirmation.pattern_signal in ("buy", "sell"):
            direction = "bullish" if confirmation.pattern_signal == "buy" else "bearish"
        else:
            return  # No directional signal to verify
        
        from atomicx.fusion.prediction import PredictionAction
        from atomicx.fusion.regime import MarketRegime
        
        # Create a lightweight prediction packet for tracking
        try:
            regime_enum = MarketRegime(brain_state.get("regime", "unknown"))
        except ValueError:
            regime_enum = MarketRegime.UNKNOWN
            
        packet = PredictionPacket(
            symbol=symbol,
            timeframe="5m",  # Verification timeframe
            direction=direction,
            confidence=confirmation.score,
            action=PredictionAction.BET if confirmation.confirmed else PredictionAction.STAY_OUT,
            regime=regime_enum,
        )
        
        pending = PendingPrediction(
            packet=packet,
            entry_price=price,
            symbol=symbol,
            regime=brain_state.get("regime", "unknown"),
        )
        self.pending_predictions.append(pending)

        # ═══ FIX: Persist prediction to Redis + database ═══
        try:
            # Save to Redis (pending list with 15 min TTL)
            # Use model_dump_json() to get a JSON string directly
            packet_json = packet.model_dump_json()
            await self.persistence.save_pending_prediction(
                prediction_id=packet.prediction_id,
                symbol=symbol,
                direction=direction,
                entry_price=price,
                confidence=confirmation.score,
                regime=brain_state.get("regime", "unknown"),
                packet_json=packet_json,
            )

            # Save to database (for historical tracking)
            # Capture EXACT raw data for causality tracking
            raw_metadata = {
                "complete_variables": brain_state.get("variables", {}),  # ALL variables, not truncated
                "layer_states_raw": brain_state.get("senses", {}),  # Includes swarm raw result, strategic, narrative, pattern
                "all_agents_raw": brain_state.get("all_agents", []),  # Every atomic agent's signal
                "confirmation_details": {
                    "pattern_signal": confirmation.pattern_signal,
                    "pattern_confidence": confirmation.pattern_confidence,
                    "logic_signal": confirmation.logic_signal,
                    "final_direction": confirmation.final_direction,
                    "confirmed": confirmation.confirmed,
                    "score": confirmation.score,
                    "logic_reason": confirmation.logic_reason,
                },
                "regime_full": brain_state.get("regime", "unknown"),
                "trust_weights": brain_state.get("trust_weights", {}),
                "monologue": brain_state.get("monologue", {}).model_dump() if hasattr(brain_state.get("monologue", {}), "model_dump") else str(brain_state.get("monologue", {})),
            }

            await self.persistence.save_prediction_outcome(
                prediction_id=packet.prediction_id,
                symbol=symbol,
                timeframe="5m",
                predicted_direction=direction,
                confidence=confirmation.score,
                entry_price=price,
                verification_price=None,
                was_correct=None,
                actual_return=None,
                profit_return=None,
                regime=brain_state.get("regime", "unknown"),
                predicted_at=pending.created_at,
                verified_at=None,
                reasoning=packet.reasoning_summary,
                variable_snapshot=brain_state.get("variables", {}),
                metadata=raw_metadata,  # Store all raw data for causality viewer
            )
        except Exception as e:
            self.logger.warning(f"Failed to persist prediction {packet.prediction_id}: {e}")

        self.logger.info(
            f"[EVOLUTION] Tracked prediction: {symbol} {direction} @ ${price:,.2f} "
            f"(conf: {confirmation.score:.0%}, regime: {brain_state.get('regime', '?')})"
        )

    async def _verify_pending_predictions(self) -> None:
        """Check if any pending predictions have aged past the verification window.
        
        For each expired prediction:
        1. Compare predicted direction vs actual price movement
        2. Create a PredictionOutcome
        3. Feed it into SelfImprovementLoop (agent weights, lessons)
        4. Feed it into MemoryOrchestrator (PerformanceGenome, CausalRL)
        5. Persist to Qdrant
        """
        if not self.pending_predictions:
            return
            
        now = datetime.now(tz=timezone.utc)
        still_pending = []
        
        for pred in self.pending_predictions:
            age_seconds = (now - pred.created_at).total_seconds()
            
            if age_seconds < self.VERIFICATION_DELAY_SECONDS:
                still_pending.append(pred)
                continue
                
            # Time to verify!
            current_price = self._last_prices.get(pred.symbol, 0.0)
            if current_price <= 0:
                still_pending.append(pred)  # Can't verify without a price
                continue
            
            # Calculate actual return
            actual_return = (current_price - pred.entry_price) / pred.entry_price

            # FIXED: Pure directional correctness
            # If predicted UP and price went UP (any amount) = CORRECT
            # If predicted DOWN and price went DOWN (any amount) = CORRECT
            predicted_bullish = pred.packet.direction == "bullish"

            if predicted_bullish:
                # Bullish prediction: correct if price went up AT ALL
                was_correct = actual_return > 0
            else:
                # Bearish prediction: correct if price went down AT ALL
                was_correct = actual_return < 0

            # For bearish predictions, invert the return (since you'd be shorting)
            # This ensures profit is positive when bearish prediction is correct
            profit_return = actual_return if predicted_bullish else -actual_return

            # Log the verification result with clear details
            self.logger.success(
                f"[VERIFY] {pred.symbol}: Predicted {'UP' if predicted_bullish else 'DOWN'}, "
                f"Actual: {actual_return:+.4%} → {'✓ CORRECT' if was_correct else '✗ WRONG'} | "
                f"Entry: ${pred.entry_price:.2f} → Exit: ${current_price:.2f} | "
                f"Profit: {profit_return:+.2%}"
            )

            # Create the outcome
            from atomicx.evolution import PredictionOutcome
            outcome = PredictionOutcome(
                prediction_id=pred.packet.prediction_id,
                symbol=pred.symbol,
                predicted_direction=pred.packet.direction,
                predicted_confidence=pred.packet.confidence,
                actual_return=actual_return,
                profit_return=profit_return,
                was_correct=was_correct,
                entry_price=pred.entry_price,
                exit_price=current_price,
                duration_hours=age_seconds / 3600,
                regime_at_prediction=pred.regime,
            )
            
            # ═══ FEED THE LEARNING LOOP ═══

            # 1. SelfImprovementLoop: Update agent weights + extract lessons + persist to Qdrant
            if self.self_improvement:
                try:
                    adjustments = await self.self_improvement.process_outcome(outcome)

                    # Log systematic issues
                    if adjustments.get("systematic_issues"):
                        for issue in adjustments["systematic_issues"]:
                            self.logger.error(f"[EVOLUTION] SYSTEMATIC ISSUE: {issue}")

                    # Log agent changes
                    if adjustments.get("agent_adjustments"):
                        for change in adjustments["agent_adjustments"]:
                            self.logger.warning(f"[EVOLUTION] AGENT CHANGE: {change}")

                except Exception as e:
                    self.logger.error(f"[EVOLUTION] SelfImprovement failed: {e}")

            # ═══ FIX: Feed outcome to CausalRL for weight optimization ═══
            try:
                # Convert PredictionOutcome to dict format expected by CausalRL
                trade_outcome_dict = {
                    "profit": profit_return,
                    "slippage_bps": 0,  # No slippage tracking yet
                    "regime": pred.regime,
                    "was_correct": was_correct,
                    "actual_return": actual_return,
                }
                strategy_id = f"{pred.regime}_{pred.packet.direction}"
                self.memory.causal_rl.update_from_outcome(strategy_id, trade_outcome_dict)
                self.logger.debug(f"[CAUSAL_RL] Updated with outcome: {was_correct}, profit={profit_return:.4f}")
            except Exception as e:
                self.logger.error(f"[CAUSAL_RL] Update failed: {e}")
            
            # 2. PerformanceGenome: Feed the strategy DNA for evolution
            strategy_id = f"{pred.regime}_{pred.packet.direction}"
            try:
                self.memory.log_strategy_outcome(strategy_id, {
                    "profit": profit_return,
                    "slippage_bps": 0,
                    "regime": pred.regime,
                })
            except Exception as e:
                self.logger.error(f"[EVOLUTION] Genome feed failed: {e}")
            
            # 3. Store the verification itself in Qdrant for permanent history  
            try:
                from atomicx.memory.service import MemoryEntry, MemoryType
                await self.memory_service.store(MemoryEntry(
                    memory_type=MemoryType.EPISODIC,
                    content=(
                        f"PREDICTION VERIFIED: {pred.symbol} predicted {pred.packet.direction} "
                        f"@ ${pred.entry_price:,.2f}, actual price ${current_price:,.2f} "
                        f"({actual_return:+.2%}). Result: {'✓ CORRECT' if was_correct else '✗ INCORRECT'}. "
                        f"Regime: {pred.regime}. Confidence was {pred.packet.confidence:.0%}."
                    ),
                    symbol=pred.symbol,
                    importance=0.9 if not was_correct else 0.6,  # Failures are more valuable
                    metadata={
                        "prediction_id": pred.packet.prediction_id,
                        "outcome": "correct" if was_correct else "incorrect",
                        "actual_return": actual_return,
                        "regime": pred.regime,
                    }
                ))
            except Exception as e:
                self.logger.warning(f"[EVOLUTION] Failed to persist verification to Qdrant: {e}")
            
            # Save to local history for dashboard
            self.prediction_history.append({
                "id": pred.packet.prediction_id,
                "symbol": pred.symbol,
                "direction": pred.packet.direction,
                "confidence": pred.packet.confidence,
                "entry": pred.entry_price,
                "exit": current_price,
                "return": profit_return,  # Use profit_return for P&L tracking
                "actual_return": actual_return,  # Keep raw return for reference
                "correct": was_correct,
                "regime": pred.regime,
                "verified_at": now.isoformat(),
            })

            # ═══ FIX: Persist verified outcome to database and remove from Redis ═══
            try:
                # Retrieve existing metadata (has all the raw data)
                from sqlalchemy import text
                from atomicx.data.storage.database import get_session
                existing_metadata = {}
                try:
                    async with get_session() as session:
                        result = await session.execute(
                            text("SELECT metadata FROM prediction_outcomes WHERE prediction_id = :pred_id"),
                            {"pred_id": pred.packet.prediction_id}
                        )
                        row = result.fetchone()
                        if row and row[0]:
                            existing_metadata = row[0] if isinstance(row[0], dict) else {}
                except Exception:
                    pass

                # Merge with verification data
                merged_metadata = {
                    **existing_metadata,
                    "verification": {
                        "age_seconds": age_seconds,
                        "verified_at": now.isoformat(),
                        "exit_price": current_price,
                        "actual_return": actual_return,
                        "profit_return": profit_return,
                        "was_correct": was_correct,
                    }
                }

                # Update database with verification results
                await self.persistence.save_prediction_outcome(
                    prediction_id=pred.packet.prediction_id,
                    symbol=pred.symbol,
                    timeframe="5m",
                    predicted_direction=pred.packet.direction,
                    confidence=pred.packet.confidence,
                    entry_price=pred.entry_price,
                    verification_price=current_price,
                    was_correct=was_correct,
                    actual_return=actual_return,
                    profit_return=profit_return,
                    regime=pred.regime,
                    predicted_at=pred.created_at,
                    verified_at=now,
                    reasoning=pred.packet.reasoning_summary,
                    variable_snapshot=None,
                    metadata=merged_metadata,  # Preserve all raw data + verification
                )

                # Create decision audit with complete causality trail
                await self._create_decision_audit(
                    prediction_id=pred.packet.prediction_id,
                    symbol=pred.symbol,
                    direction=pred.packet.direction,
                    confidence=pred.packet.confidence,
                    was_correct=was_correct,
                    actual_return=actual_return,
                    predicted_at=pred.created_at,
                    verified_at=now,
                    raw_metadata=merged_metadata,
                )

                # Remove from Redis (no longer pending)
                await self.persistence.remove_pending_prediction(pred.packet.prediction_id)

                # ═══ NEW: INGEST TO WIKI (Karpathy LLM Wiki Pattern) ═══
                try:
                    await self.memory.wiki.ingest_prediction_outcome(
                        prediction_id=pred.packet.prediction_id,
                        symbol=pred.symbol,
                        pattern=merged_metadata.get("confirmation_details", {}).get("logic_reason"),
                        regime=pred.regime,
                        direction=pred.packet.direction,
                        confidence=pred.packet.confidence,
                        was_correct=was_correct,
                        actual_return=actual_return,
                        lessons=None if was_correct or abs(actual_return) < 0.05 else f"Prediction failed. Review why {pred.packet.direction} did not work in {pred.regime} regime."
                    )
                    self.logger.debug(f"[WIKI] Ingested outcome: {pred.symbol} {pred.packet.direction} → {'✓' if was_correct else '✗'}")
                except Exception as e:
                    self.logger.warning(f"[WIKI] Failed to ingest outcome: {e}")

            except Exception as e:
                self.logger.warning(f"Failed to persist verified outcome: {e}")

            emoji = "✓" if was_correct else "✗"
            self.logger.success(
                f"[EVOLUTION] {emoji} Prediction verified: {pred.symbol} "
                f"{pred.packet.direction} @ ${pred.entry_price:,.2f} → ${current_price:,.2f} "
                f"(P&L: {profit_return:+.2%}, price Δ: {actual_return:+.2%}) | Regime: {pred.regime} | "
                f"Win Rate: {self._get_win_rate():.1%} ({len(self.prediction_history)} total)"
            )

            # Update dashboard performance metrics
            self.dashboard.update_performance(
                win_rate=self._get_win_rate(),
                total=len(self.prediction_history),
                last_lesson=f"{emoji} {pred.symbol} {pred.packet.direction} ${pred.entry_price:,.0f}→${current_price:,.0f} ({profit_return:+.1%})"
            )
        
        self.pending_predictions = still_pending

    def _safe_create_task(self, coro, name: str) -> asyncio.Task:
        """Create a background task with automatic exception logging.

        FIX: Prevents silent failures in background tasks. All exceptions are logged.
        """
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: self._handle_task_exception(t, name))
        return task

    def _handle_task_exception(self, task: asyncio.Task, name: str) -> None:
        """Handle exceptions from background tasks."""
        try:
            task.result()  # Raises exception if task failed
        except asyncio.CancelledError:
            # Expected on graceful shutdown
            self.logger.debug(f"[BACKGROUND TASK] {name} cancelled (shutdown)")
        except Exception as e:
            # Unexpected failure - log with full traceback
            self.logger.error(
                f"[BACKGROUND TASK] {name} failed with exception: {e}",
                exc_info=True
            )

    def _get_win_rate(self) -> float:
        """Calculate current win rate from prediction history."""
        if not self.prediction_history or len(self.prediction_history) == 0:
            return 0.0
        wins = sum(1 for p in self.prediction_history if p.get("correct", False))
        # Double-check to prevent division by zero in async context
        if len(self.prediction_history) == 0:
            return 0.0
        return wins / len(self.prediction_history)

    def _get_dominant_regime(self) -> str:
        """Get most aggressive regime across all tracked symbols.

        FIX: Prevents regime flip-flop when tracking multiple symbols.
        Returns the strongest market state rather than last-processed symbol.
        """
        if not self._last_regimes:
            return 'TRANSITION'

        regimes = list(self._last_regimes.values())

        # Priority order: Most aggressive first
        if "BULL_BREAKOUT" in regimes:
            return "BULL_BREAKOUT"
        if "BEAR_BREAKOUT" in regimes:
            return "BEAR_BREAKOUT"
        if "BULLISH" in regimes:
            return "BULLISH"
        if "BEARISH" in regimes:
            return "BEARISH"

        return "TRANSITION"

    async def _update_god_mode_cache(self, symbol: str, brain_state: dict, confirmation: Any) -> None:
        """Update God Mode cognitive cache for instant dashboard (< 5ms).

        PERFORMANCE FIX:
        - Before: Dashboard computed on-demand (60,000ms blocking)
        - After: Brain updates cache in background, dashboard reads instantly

        This method is called after every cognitive cycle to keep the dashboard
        data fresh without blocking API requests.
        """
        try:
            from atomicx.dashboard.god_mode_cache import get_god_mode_cache
            from atomicx.fusion.prediction import PredictionPacket, PredictionAction
            from atomicx.fusion.regime import MarketRegime

            cache = get_god_mode_cache()

            # Build prediction packet from confirmation
            direction = "neutral"
            if confirmation.confirmed:
                direction = "bullish" if "buy" in confirmation.final_direction else "bearish"
            elif confirmation.pattern_signal in ("buy", "sell"):
                direction = "bullish" if confirmation.pattern_signal == "buy" else "bearish"

            try:
                regime_enum = MarketRegime(brain_state.get("regime", "unknown"))
            except ValueError:
                regime_enum = MarketRegime.UNKNOWN

            # Create prediction packet
            packet = PredictionPacket(
                symbol=symbol,
                timeframe="5m",
                direction=direction,
                confidence=confirmation.score,
                action=PredictionAction.BET if confirmation.confirmed else PredictionAction.STAY_OUT,
                regime=regime_enum,
            )

            # Extract swarm result with RICH METRICS
            swarm_result = None
            senses = brain_state.get("senses", {})
            swarm_raw = senses.get("swarm", {})
            if swarm_raw:
                agent_count = swarm_raw.get("agent_count", 500)
                bullish = int(swarm_raw.get("bullish_agents", 250))
                bearish = int(swarm_raw.get("bearish_agents", 250))
                neutral = agent_count - bullish - bearish

                # FIX: Calculate proper direction from agent counts
                # If 250/250 split, it's NEUTRAL not bullish!
                if bullish > bearish + 50:  # Meaningful bullish majority
                    direction = "bullish"
                    confidence = bullish / agent_count
                elif bearish > bullish + 50:  # Meaningful bearish majority
                    direction = "bearish"
                    confidence = bearish / agent_count
                else:  # Mixed or 50/50
                    direction = "neutral"
                    confidence = 0.5

                swarm_result = {
                    "name": "MARL Swarm",
                    "direction": direction,
                    "confidence": float(confidence),
                    "agent_count": agent_count,
                    "bullish_agents": bullish,
                    "bearish_agents": bearish,
                    "neutral_agents": max(0, neutral),  # Ensure non-negative

                    # Rich metrics for visualization
                    "convergence": float(swarm_raw.get("convergence", 0.85)),
                    "diversity": float(swarm_raw.get("diversity", 0.30)),
                    "stability": float(swarm_raw.get("stability", 0.80)),
                    "reward_signal": float(swarm_raw.get("reward_signal", 0.0)),
                    "top_strategy": swarm_raw.get("top_strategy", "N/A"),
                    "consensus_steps": int(swarm_raw.get("consensus_steps", 0)),

                    "data_source": "REAL"
                }

            # Extract narrative state
            narrative_state = None
            narrative_raw = senses.get("narrative", {})
            if narrative_raw:
                narrative_state = {
                    "social_momentum": narrative_raw.get("social_momentum", "neutral"),
                    "social_confidence": float(narrative_raw.get("social_confidence", 0.5)),
                    "sentiment": float(narrative_raw.get("sentiment", 0.5)),
                    "recent_signals": narrative_raw.get("recent_signals", 0),
                }

            # Extract causal links (if available)
            causal_links = brain_state.get("causal_links", [])

            # Get variables and augment with daily change for stocks
            variables = brain_state.get("variables", {})

            # For stocks, calculate the true daily change from 1d OHLCV data
            # (current implementation shows intraday change, but stocks need day-over-day)
            if "/" not in symbol:  # Stocks don't have "/" (e.g., META vs BTC/USDT)
                try:
                    # Fetch last 2 daily candles from database
                    from atomicx.data.storage.database import get_session_factory
                    from atomicx.data.storage.models import OHLCV
                    from sqlalchemy import select

                    session_factory = get_session_factory()
                    async with session_factory() as session:
                        stmt = (
                            select(OHLCV.close)
                            .where(OHLCV.symbol == symbol)
                            .where(OHLCV.timeframe == "1d")
                            .order_by(OHLCV.timestamp.desc())
                            .limit(2)
                        )
                        result = await session.execute(stmt)
                        closes = [row[0] for row in result.fetchall()]

                        if len(closes) >= 2:
                            today_close = closes[0]
                            yesterday_close = closes[1]
                            daily_change = ((today_close - yesterday_close) / yesterday_close) * 100
                            variables["DAILY_CHANGE_PERCENT"] = float(daily_change)
                            self.logger.info(f"[STOCK] Updated {symbol} daily change to {daily_change:.2f}% (today: ${today_close:.2f}, yesterday: ${yesterday_close:.2f})")
                        else:
                            self.logger.warning(f"[STOCK] Not enough 1d candles for {symbol} (found {len(closes)})")
                except Exception as e:
                    self.logger.warning(f"Could not fetch 1d daily change for {symbol}: {e}")

            # Update cache
            await cache.update(
                symbol=symbol,
                prediction=packet,
                variables=variables,
                swarm_result=swarm_result,
                narrative_state=narrative_state,
                causal_links=causal_links,
                all_agents=brain_state.get("all_agents", []),  # Pass all 46+ atomic agents
            )

        except Exception as e:
            self.logger.warning(f"[GOD-MODE-CACHE] Failed to update cache for {symbol}: {e}")

    def _sync_dashboard_stats(self) -> None:
        """Push evolution stats to the dashboard."""
        if self.self_improvement and hasattr(self.self_improvement, 'stats'):
            stats = self.self_improvement.stats
            self.dashboard.performance["win_rate"] = stats.win_rate
            self.dashboard.performance["total_predictions"] = stats.total_predictions
            self.dashboard.performance["intelligence_score"] = stats.edge_over_random
            
            # Find latest lesson
            if self.prediction_history:
                latest = self.prediction_history[-1]
                direction = latest.get('direction', 'unknown')
                return_pct = latest.get('return', 0.0)
                regime = latest.get('regime', 'unknown')
                symbol = latest.get('symbol', 'BTC/USDT')
                correct = latest.get('correct', False)
                self.dashboard.performance["last_lesson"] = (
                    f"{'✓' if correct else '✗'} {symbol} "
                    f"{direction} {return_pct:+.2%} "
                    f"(regime: {regime})"
                )
        elif self.prediction_history:
            # Fallback: compute from local history
            self.dashboard.performance["win_rate"] = self._get_win_rate()
            self.dashboard.performance["total_predictions"] = len(self.prediction_history)
            self.dashboard.performance["intelligence_score"] = self._get_win_rate() - 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # DASHBOARD DATA
    # ═══════════════════════════════════════════════════════════════════════

    async def _update_high_fidelity_dashboard(self, symbol: str, brain_state: dict, debate: Any, conf_gate: Any) -> None:
        """Populate the new columnar dashboard with rich data for any given symbol."""
        if not self.dashboard: return

        # Base price and variance (Dynamic)
        price_curr = brain_state.get("price", 0.0)
        if price_curr <= 0:
            return  # No price, skip dashboard update

        # Confidence logic (Dynamic)
        conf_val = int(conf_gate.score * 100)

        # Direction determined by dual-confirm (Dynamic)
        sig = "WATCH"
        if conf_gate.confirmed:
            sig = "LONG" if "buy" in conf_gate.final_direction else "SHORT"
        elif conf_gate.pattern_signal in ("buy", "sell"):
            sig = "WATCH"

        badge = "badge-long" if sig == "LONG" else ("badge-short" if sig == "SHORT" else "badge-watch")
        chart_color = "#00e5a0" if sig == "LONG" else ("#ff4455" if sig == "SHORT" else "#888")

        # Calculate real chart points and OHLCV data for candlestick rendering
        recent = brain_state.get("variables", {}).get("RECENT_PRICES", [])
        ohlcv_bars = []

        # Try to fetch recent OHLCV from database for proper candlestick chart
        try:
            from sqlalchemy import select
            from atomicx.data.storage.database import get_session_factory
            from atomicx.data.storage.models import OHLCV

            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    select(OHLCV)
                    .where(OHLCV.symbol == symbol)
                    .where(OHLCV.timeframe == "1m")
                    .order_by(OHLCV.timestamp.desc())
                    .limit(60)  # Last 60 candles for chart
                )
                rows = result.scalars().all()

                if rows:
                    ohlcv_bars = [
                        {
                            "timestamp": r.timestamp.isoformat(),
                            "open": float(r.open),
                            "high": float(r.high),
                            "low": float(r.low),
                            "close": float(r.close),
                            "volume": float(r.volume),
                        }
                        for r in reversed(rows)
                    ]
        except Exception as e:
            self.logger.debug(f"Failed to fetch OHLCV for chart: {e}")

        # Build simple line chart from prices
        if not recent:
            chart_points = "0,80 30,75 60,85 90,70 120,75 150,68 180,72 210,65 240,70 270,66 300,68"
            support = f"${price_curr * 0.95:,.2f}"
            resist = f"${price_curr * 1.05:,.2f}"
        else:
            min_p = min(recent)
            max_p = max(recent)
            rng = max_p - min_p if max_p > min_p else 1
            support = f"${min_p:,.2f}"
            resist = f"${max_p:,.2f}"
            pts = []
            for i, p in enumerate(recent):
                x = (i / max(1, len(recent)-1)) * 300
                y = 120 - ((p - min_p) / rng) * 95
                pts.append(f"{x:.1f},{y:.1f}")
            chart_points = " ".join(pts)

        # ═══ FIX: Generate REAL ML-based predictions from pattern library ═══
        preds = {}
        try:
            # Query pattern library for historical performance
            pattern_name = conf_gate.pattern_signal.upper() if conf_gate.pattern_signal else "NEUTRAL"
            regime = brain_state.get("regime", "unknown")

            pattern_history = await self.pattern_verifier.get_pattern_history(
                pattern_name=pattern_name,
                regime=regime,
                symbol=symbol,
                limit=50
            )

            # Calculate actual expected returns from pattern history
            if pattern_history and len(pattern_history) > 5:
                returns = [p["outcome_return"] for p in pattern_history if p.get("outcome_verified")]
                if returns:
                    avg_return = sum(returns) / len(returns)
                    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                else:
                    avg_return = 0.0
                    std_return = 0.02
            else:
                # Fallback to fusion node prediction if no pattern history
                fusion_pred = self.orchestrator.fusion_node
                if hasattr(fusion_pred, '_last_prediction') and fusion_pred._last_prediction:
                    avg_return = fusion_pred._last_prediction.get("expected_return", 0.0)
                    std_return = 0.02
                else:
                    avg_return = 0.0
                    std_return = 0.02

            # Generate predictions for different timeframes based on real ML data
            timeframe_mults = {"1h": 0.3, "1d": 1.0, "1w": 3.5}
            for tf, mult in timeframe_mults.items():
                expected_move = avg_return * mult
                target_price = price_curr * (1 + expected_move)

                preds[tf] = {
                    "price": f"${target_price:,.2f}",
                    "move": f"{abs(expected_move)*100:.2f}%",
                    "dir": "up" if expected_move > 0 else "down",
                    "conf": conf_val,
                    "low": f"${target_price * (1 - std_return * mult):,.2f}",
                    "mid": f"${target_price:,.2f}",
                    "high": f"${target_price * (1 + std_return * mult):,.2f}",
                }
        except Exception as e:
            self.logger.warning(f"Failed to generate ML predictions: {e}. Using fallback.")
            # Minimal fallback
            for tf in ["1h", "1d", "1w"]:
                preds[tf] = {
                    "price": f"${price_curr:,.2f}",
                    "move": "0.0%",
                    "dir": "up" if sig == "LONG" else "down",
                    "conf": conf_val,
                    "low": f"${price_curr * 0.98:,.2f}",
                    "mid": f"${price_curr:,.2f}",
                    "high": f"${price_curr * 1.02:,.2f}",
                }

        # Map real arguments to UI format
        agents_firing = []
        debate_args_for_reasoning = []
        for arg in getattr(debate, 'arguments', []):
            cls = "ar-long" if arg.stance == "bullish" else ("ar-short" if arg.stance == "bearish" else "ar-neutral")
            agents_firing.append({
                "name": f"Debate: {arg.agent_name}",
                "sig": arg.stance.upper(),
                "acc": f"{arg.conviction:.0%}",
                "cls": cls,
                "logic": getattr(arg, "reasoning", "No LLM monologue available.")
            })
            # Store for reasoning transparency
            debate_args_for_reasoning.append({
                "name": arg.agent_name,
                "stance": arg.stance,
                "conviction": arg.conviction
            })

        # Add the 62-layer hierarchy agents we harvested in observe_and_reflect
        for hr_agent in brain_state.get("all_agents", []):
            agents_firing.append(hr_agent)

        # Extract dynamic data for L7, L8, L9
        senses = brain_state.get("senses", {})
        narr_res = senses.get("narrative", {})
        swarm_res = senses.get("swarm", {})
        vars_snap = brain_state.get("variables", {})
        weights = brain_state.get("trust_weights", {})

        # ═══ FIX: Normalize weights to sum to 100% ═══
        raw_math = 0.3  # Technical baseline
        raw_narrative = weights.get("narrative", 0.1)
        raw_swarm = weights.get("swarm", 0.1)
        raw_causal = weights.get("causal", 0.4)

        total_weight = raw_math + raw_narrative + raw_swarm + raw_causal
        if total_weight > 0:
            math_w = int((raw_math / total_weight) * 100)
            sn_w = int(((raw_narrative + raw_swarm) / total_weight) * 100)
            causal_w = int((raw_causal / total_weight) * 100)

            # Ensure sum is exactly 100 (handle rounding)
            weight_sum = math_w + sn_w + causal_w
            if weight_sum != 100:
                math_w += (100 - weight_sum)  # Add rounding error to math
        else:
            math_w, sn_w, causal_w = 33, 33, 34

        # ═══ ADD: Database transparency stats ═══
        db_stats = {}
        try:
            from sqlalchemy import select, func
            from atomicx.data.storage.database import get_session_factory
            from atomicx.data.storage.models import OHLCV
            from atomicx.variables.models import ComputedVariable

            session_factory = get_session_factory()
            async with session_factory() as session:
                # Count OHLCV candles
                ohlcv_result = await session.execute(
                    select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
                )
                db_stats["ohlcv_count"] = ohlcv_result.scalar() or 0

                # Count computed variables
                var_result = await session.execute(
                    select(func.count(ComputedVariable.id)).where(ComputedVariable.symbol == symbol)
                )
                db_stats["var_count"] = var_result.scalar() or 0

                # Get latest timestamp
                latest_result = await session.execute(
                    select(func.max(OHLCV.timestamp)).where(OHLCV.symbol == symbol)
                )
                latest_ts = latest_result.scalar()
                db_stats["last_update"] = latest_ts.strftime("%Y-%m-%d %H:%M UTC") if latest_ts else "Never"
        except Exception as e:
            self.logger.warning(f"Failed to fetch DB stats: {e}")
            db_stats = {"ohlcv_count": 0, "var_count": 0, "last_update": "Error"}

        # Pattern library stats
        try:
            pattern_stats = await self.pattern_verifier.get_pattern_stats(symbol=symbol)
            db_stats["pattern_total"] = pattern_stats.get("total_detected", 0)
            db_stats["pattern_verified"] = pattern_stats.get("verified_count", 0)
            win_rate = pattern_stats.get("win_rate", 0.0)
            db_stats["pattern_win_rate"] = f"{win_rate:.1%}" if win_rate > 0 else "N/A"
        except Exception as e:
            self.logger.warning(f"Failed to fetch pattern stats: {e}")
            db_stats["pattern_total"] = 0
            db_stats["pattern_verified"] = 0
            db_stats["pattern_win_rate"] = "N/A"

        # Memory tier stats
        try:
            mem_stats = await self.memory_service.get_stats()
            db_stats["mem_tier1"] = mem_stats.get("tier1_count", 0)
            db_stats["mem_tier234"] = mem_stats.get("tier234_count", 0)
        except Exception as e:
            self.logger.warning(f"Failed to fetch memory stats: {e}")
            db_stats["mem_tier1"] = 0
            db_stats["mem_tier234"] = 0

        # ═══ ADD: Reasoning transparency ═══
        # Identify which variables triggered the signal
        triggering_vars = []
        for var_name, var_val in vars_snap.items():
            if var_name.startswith("RSI") and var_val < 30:
                triggering_vars.append(f"{var_name}={var_val:.1f} (oversold)")
            elif var_name.startswith("RSI") and var_val > 70:
                triggering_vars.append(f"{var_name}={var_val:.1f} (overbought)")
            elif var_name == "MACD_HISTOGRAM" and abs(var_val) > 0.5:
                triggering_vars.append(f"MACD_HISTOGRAM={var_val:.2f} (strong momentum)")
            elif var_name == "ADX" and var_val > 25:
                triggering_vars.append(f"ADX={var_val:.1f} (trending)")

        data_sources = [
            "Binance WebSocket (OHLCV)",
            "TimescaleDB (Historical)",
            "Variable Engine (46 Indicators)",
            "Pattern Library (Verified Outcomes)",
            "LLM Debate (Claude 3.5 Sonnet)"
        ]

        self.dashboard.update_symbol_data(symbol, {
            "signal": sig,
            "confidence": conf_val,
            "badge_class": badge,
            "chart_color": chart_color,
            "chart_points": chart_points,
            "ohlcv_bars": ohlcv_bars,  # For candlestick rendering
            "support": support,
            "resist": resist,
            "preds": preds,
            "agents": agents_firing,
            "reasoning": conf_gate.logic_reason,
            "regime": brain_state.get("regime", "Unknown"),
            "current_price": f"${price_curr:,.4f}",
            "cycle_time_ms": brain_state.get('cycle_duration_ms', 1200),  # FIX: Renamed from latency
            "math_passed": conf_gate.pattern_signal in ("buy", "sell") and conf_gate.pattern_confidence > 0.5,
            "causal_passed": conf_gate.logic_signal in ("bullish", "bearish"),
            "ob_imbalance": f"{vars_snap.get('OB_IMBALANCE', 0.0):.3f}",
            "spread": f"{vars_snap.get('SPREAD', 0.0001):.4f}",
            "narrative_sentiment": narr_res.get("direction", "NEUTRAL").upper(),
            "narrative_conf": f"{narr_res.get('confidence', 0.0):.1%}",
            "swarm_dir": swarm_res.get("direction", "WATCH").upper(),
            "swarm_strength": f"{swarm_res.get('confidence', 0.0):.1%}",
            "pattern_signal": conf_gate.pattern_signal.upper(),
            "weights": {
                "math": math_w,
                "swarm": sn_w,
                "causal": causal_w
            },
            # ═══ ADD: All variables for variables panel ═══
            "all_variables": vars_snap,
            # ═══ ADD: Database transparency ═══
            "db_stats": db_stats,
            # ═══ ADD: Reasoning transparency ═══
            "debate_args": debate_args_for_reasoning,
            "triggering_vars": triggering_vars,
            "data_sources": data_sources,
            "crystal_ball": self.orchestrator.get_crystal_ball_predictions(),
            # Evolution stats
            "pending_predictions": len(self.pending_predictions),
            "verified_predictions": len(self.prediction_history),
            "evolution_win_rate": f"{self._get_win_rate():.1%}",
        })

    async def _intelligence_scan(self) -> None:
        """Phase 17 + 21: Scan for breaking news and run impact prediction."""
        try:
            stories = await self.news_scanner.scan_cycle()

            for story in stories:
                if story.deep_dive_triggered:
                    research = await self.browser_agent.deep_dive(story)
                    self.knowledge_graph.ingest_research(research)

                    impact = await self.impact_predictor.predict_impact(story)

                    # Store news event for pattern learning
                    try:
                        # Check for matching patterns first
                        matching_patterns = await self.news_intelligence.get_matching_patterns(story)

                        predicted_impact = None
                        if matching_patterns:
                            # Use pattern with highest confidence
                            best_pattern = max(matching_patterns, key=lambda p: p["confidence"])
                            predicted_impact = {
                                "direction": "BULLISH" if best_pattern["expected_impact"] > 0 else "BEARISH",
                                "magnitude": abs(best_pattern["expected_impact"]),
                                "confidence": best_pattern["confidence"]
                            }
                            self.logger.info(
                                f"[PATTERN-MATCH] Found pattern: {best_pattern['pattern_type']} → "
                                f"{best_pattern['expected_impact']:+.1f}% (conf: {best_pattern['confidence']:.0%})"
                            )

                        # Store event with analysis
                        event_id = await self.news_intelligence.store_news_event(
                            news_item=story,
                            analysis=research,
                            predicted_impact=predicted_impact
                        )

                        # Schedule outcome tracking (1h, 4h, 24h after news)
                        self._safe_create_task(
                            self._track_news_outcome(event_id, story),
                            f"news_outcome_tracking_{event_id}"
                        )

                    except Exception as e:
                        self.logger.error(f"[NEWS-INTEL] Failed to store news event: {e}")

                    if impact["decision"]["action"] == "FAST_PATH_INTENT":
                        self.logger.warning(
                            f"[CRYSTAL BALL] FAST PATH: {impact['decision']['reason']}"
                        )

        except Exception as e:
            self.logger.error(f"Intelligence scan failed: {e}")

    async def _track_news_outcome(self, event_id: int, news_item: Any) -> None:
        """Track actual market outcomes after a news event (1h, 4h, 24h)."""
        try:
            # Get current price as baseline
            brain_state = await self.orchestrator.get_brain_state(self.symbols[0])
            price_before = brain_state.get("price", 0.0)

            if price_before <= 0:
                self.logger.warning(f"[NEWS-OUTCOME] Can't track outcome - no price available")
                return

            # Wait 1 hour
            await asyncio.sleep(3600)
            state_1h = await self.orchestrator.get_brain_state(self.symbols[0])
            price_1h = state_1h.get("price", price_before)

            # Wait 3 more hours (total 4h)
            await asyncio.sleep(3 * 3600)
            state_4h = await self.orchestrator.get_brain_state(self.symbols[0])
            price_4h = state_4h.get("price", price_before)

            # Wait 20 more hours (total 24h)
            await asyncio.sleep(20 * 3600)
            state_24h = await self.orchestrator.get_brain_state(self.symbols[0])
            price_24h = state_24h.get("price", price_before)

            # Calculate volume and volatility changes (simplified for now)
            volume_change = 1.0  # TODO: Track actual volume change
            volatility_change = 1.0  # TODO: Track actual volatility change

            # Track outcome
            await self.news_intelligence.track_outcome(
                event_id=event_id,
                price_before=price_before,
                price_1h=price_1h,
                price_4h=price_4h,
                price_24h=price_24h,
                volume_change=volume_change,
                volatility_change=volatility_change
            )

        except asyncio.CancelledError:
            # Task was cancelled (e.g., system shutdown)
            pass
        except Exception as e:
            self.logger.error(f"[NEWS-OUTCOME] Failed to track outcome for event {event_id}: {e}")

    def _run_titan_checks(self) -> None:
        """Phases 22-24: Run all Titan-Killer intelligence modules."""
        try:
            hidden_signals = self.kernel_engine.get_hidden_signals()
            if hidden_signals:
                self.logger.info(f"[KERNEL] {len(hidden_signals)} hidden alt-data signals active")

            trap = self.retail_flow.detect_liquidity_traps()
            if trap:
                self.logger.warning(
                    f"[CITADEL] LIQUIDITY TRAP: {trap.trap_type} | "
                    f"Action: {trap.recommended_action} | Confidence: {trap.confidence:.0%}"
                )

            expiry_event = self.expiry_sentinel.is_expiry_window()
            if expiry_event:
                alerts = self.expiry_sentinel.get_dashboard_alerts()
                for alert in alerts:
                    self.dashboard.current_position["expiry_alert"] = alert["message"]

        except Exception as e:
            self.logger.error(f"Titan-Killer check failed: {e}")

    def _log_narrative_health_report(self) -> None:
        """Log a comprehensive health report for the narrative intelligence system."""
        scanner_health = self.news_scanner.get_health_status()
        narrative_health = self.orchestrator.narrative.get_health_status()

        self.logger.info(
            f"[NARRATIVE HEALTH] Scanner: {scanner_health['healthy_sources']}/{scanner_health['total_sources']} sources online | "
            f"Success rate: {scanner_health['success_rate']:.1%} | "
            f"Narrative: {narrative_health['narrative_count']} narratives | "
            f"{narrative_health['total_signals_ingested']} signals ingested | "
            f"Last update: {narrative_health['seconds_since_ingest']:.0f}s ago"
        )

        # Log individual source status
        for source_name, source_health in scanner_health["sources"].items():
            status = "✓ ONLINE" if source_health["healthy"] else "✗ OFFLINE"
            failures = source_health["consecutive_failures"]
            self.logger.debug(
                f"[NARRATIVE] {source_name}: {status} "
                f"(failures: {failures})"
            )

    # ═══════════════════════════════════════════════════════════════════════
    # PATTERN LIBRARY: Detection & Verification
    # ═══════════════════════════════════════════════════════════════════════

    async def _detect_and_store_patterns(
        self,
        symbol: str,
        brain_state: dict,
        confirmation: Any
    ) -> None:
        """Detect patterns in current state and store them for verification."""
        try:
            variables = brain_state.get("variables", {})
            price = brain_state.get("price", 0.0)
            regime = brain_state.get("regime", "unknown")

            if not variables or price <= 0:
                return

            # Detect patterns
            patterns = await self.pattern_verifier.detect_and_store_patterns(
                symbol=symbol,
                timeframe="1m",  # Using 1m timeframe for pattern detection
                variables=variables,
                price=price,
                regime=regime,
            )

            if patterns:
                self.logger.debug(
                    f"[PATTERN] Detected {len(patterns)} patterns for {symbol}"
                )

        except Exception as e:
            self.logger.error(f"[PATTERN] Detection failed: {e}")

    async def _verify_pattern_outcomes(self) -> None:
        """Verify outcomes for patterns that have aged past verification window."""
        try:
            stats = await self.pattern_verifier.verify_pending_patterns()

            if stats["patterns_verified"] > 0:
                self.logger.info(
                    f"[PATTERN] Verified {stats['patterns_verified']} pattern outcomes"
                )

        except Exception as e:
            self.logger.error(f"[PATTERN] Verification failed: {e}")

    async def get_pattern_history(
        self,
        pattern_name: str,
        regime: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get historical pattern outcomes for agent analysis.

        This is the key function that strategic and swarm layers call
        to answer: "Last 50 times this pattern appeared, what happened?"

        Args:
            pattern_name: Pattern to query (e.g., "RSI_OVERSOLD_30")
            regime: Optional regime filter (e.g., "trending_volatile")
            symbol: Optional symbol filter
            limit: Max results (default 50)

        Returns:
            List of historical pattern occurrences with outcomes
        """
        try:
            return await self.pattern_verifier.get_pattern_history(
                pattern_name=pattern_name,
                regime=regime,
                symbol=symbol,
                limit=limit,
            )
        except Exception as e:
            self.logger.error(f"[PATTERN] History query failed: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════
    # CAUSAL INTELLIGENCE & LEARNING
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_causal_discovery(self) -> None:
        """Periodically run causal discovery to learn relationships from data.

        Discovers "which variables actually cause price moves" using:
        - NOTEARS (structure learning)
        - PC Algorithm (constraint-based)
        - Granger Causality (time-series)

        Updates FusionNode with learned causal weights.
        """
        try:
            for symbol in self.symbols:
                self.logger.info(f"[CAUSAL] Running discovery for {symbol}...")

                # Run discovery on 30 days of historical data
                await self.orchestrator.fusion_node.causal_engine.discover(
                    symbol=symbol,
                    timeframe="1h",
                    lookback_days=30
                )

                self.logger.success(
                    f"[CAUSAL] Discovery complete for {symbol}. "
                    f"Learned causal relationships will be used in next cycle."
                )
        except Exception as e:
            self.logger.error(f"[CAUSAL] Discovery failed: {e}")

    async def _adjust_causal_weights(self) -> None:
        """Apply CausalRL weight adjustments based on outcome profitability.

        Uses causal inference to estimate "what would have happened if we
        changed momentum_weight or risk_scaling?" and optimizes accordingly.
        """
        try:
            adjusted_weights = self.memory.causal_rl.periodic_weight_adjustment()

            if adjusted_weights:
                # Apply to fusion node's causal engine
                self.orchestrator.fusion_node.causal_engine.apply_weights(adjusted_weights)

                self.logger.success(
                    f"[CAUSAL_RL] Applied weight adjustments: "
                    f"{len(adjusted_weights)} variables updated"
                )
            else:
                self.logger.debug("[CAUSAL_RL] No adjustments needed this cycle")

        except Exception as e:
            self.logger.error(f"[CAUSAL_RL] Weight adjustment failed: {e}")

    async def _enact_evolution_proposals(self) -> None:
        """Enact high-confidence evolution proposals from the Evolver.

        Applies system improvements like:
        - Weight adjustments (permanently_decrease_swarm)
        - Agent spawning (add_tiebreaker_agent)
        - Strategy changes (shift_to_mean_reversion)
        """
        try:
            proposals = getattr(self.evolver, 'proposals', [])

            if not proposals:
                return

            enacted_count = 0
            for proposal in proposals:
                # Only enact high-confidence proposals
                if proposal.confidence < 0.7 or proposal.status != "pending":
                    continue

                success = await self._apply_proposal(proposal)
                if success:
                    proposal.status = "enacted"
                    enacted_count += 1
                    self.logger.success(
                        f"[EVOLVER] ✓ Enacted: {proposal.mutation_type} "
                        f"(confidence: {proposal.confidence:.2f})"
                    )
                else:
                    proposal.status = "failed"
                    self.logger.warning(
                        f"[EVOLVER] ✗ Failed to enact: {proposal.mutation_type}"
                    )

            if enacted_count > 0:
                self.logger.info(f"[EVOLVER] Enacted {enacted_count} proposals this cycle")

        except Exception as e:
            self.logger.error(f"[EVOLVER] Proposal enactment failed: {e}")

    async def _apply_proposal(self, proposal: Any) -> bool:
        """Apply a single evolution proposal to the system.

        Args:
            proposal: EvolutionProposal object with mutation_type and rationale

        Returns:
            True if successfully applied, False otherwise
        """
        try:
            mutation = proposal.mutation_type

            # Weight adjustments
            if mutation == "permanently_decrease_swarm":
                current = self.orchestrator.self_model.trust_weights.get("swarm", 1.0)
                new_weight = max(0.1, current * 0.8)  # Reduce by 20%, floor at 0.1
                self.orchestrator.self_model.trust_weights["swarm"] = new_weight
                self.logger.info(f"[EVOLVER] Swarm weight: {current:.2f} → {new_weight:.2f}")
                return True

            elif mutation == "permanently_increase_causal":
                current = self.orchestrator.self_model.trust_weights.get("causal", 1.0)
                new_weight = min(2.0, current * 1.2)  # Increase by 20%, cap at 2.0
                self.orchestrator.self_model.trust_weights["causal"] = new_weight
                self.logger.info(f"[EVOLVER] Causal weight: {current:.2f} → {new_weight:.2f}")
                return True

            elif mutation == "permanently_decrease_narrative":
                current = self.orchestrator.self_model.trust_weights.get("narrative", 1.0)
                new_weight = max(0.1, current * 0.8)
                self.orchestrator.self_model.trust_weights["narrative"] = new_weight
                self.logger.info(f"[EVOLVER] Narrative weight: {current:.2f} → {new_weight:.2f}")
                return True

            elif mutation == "add_tiebreaker_agent":
                self.logger.warning(
                    f"[EVOLVER] Agent spawning not yet implemented. "
                    f"Proposal logged for manual review."
                )
                return False

            elif mutation == "shift_to_mean_reversion":
                # Could adjust strategy parameters here
                self.logger.warning(
                    f"[EVOLVER] Strategy shifting not yet implemented. "
                    f"Proposal logged for manual review."
                )
                return False

            else:
                self.logger.warning(f"[EVOLVER] Unknown mutation type: {mutation}")
                return False

        except Exception as e:
            self.logger.error(f"[EVOLVER] Failed to apply {proposal.mutation_type}: {e}")
            return False

    async def _create_decision_audit(
        self,
        prediction_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        was_correct: bool,
        actual_return: float,
        predicted_at: datetime,
        verified_at: datetime,
        raw_metadata: dict,
    ) -> None:
        """Create a complete decision audit with all causality data.

        This captures EXACT raw data for the causality viewer:
        - Complete variable snapshot
        - Swarm agent positions and consensus
        - Layer outputs (strategic, narrative, pattern, swarm)
        - All agent signals
        - News content (if applicable)
        """
        try:
            import hashlib
            import json
            from sqlalchemy import text
            from atomicx.data.storage.database import get_session

            audit_id = hashlib.md5(f"{prediction_id}{predicted_at}".encode()).hexdigest()

            # Build causal chain from raw data
            causal_chain = []

            # Step 1: Variable state
            if raw_metadata.get("complete_variables"):
                var_count = len(raw_metadata["complete_variables"])
                causal_chain.append({
                    "step": 1,
                    "what": "Market State Analyzed",
                    "data": {
                        "variable_count": var_count,
                        "top_5_variables": dict(list(raw_metadata["complete_variables"].items())[:5])
                    }
                })

            # Step 2: Layer analysis
            layer_states = raw_metadata.get("layer_states_raw", {})
            if layer_states:
                causal_chain.append({
                    "step": 2,
                    "what": "Multi-Layer Analysis",
                    "data": {
                        "strategic": layer_states.get("strategic", {}),
                        "narrative": layer_states.get("narrative", {}),
                        "swarm": layer_states.get("swarm", {}),
                        "pattern": layer_states.get("pattern", {}),
                    }
                })

            # Step 3: Agent signals
            all_agents = raw_metadata.get("all_agents_raw", [])
            if all_agents:
                causal_chain.append({
                    "step": 3,
                    "what": f"Agent Hierarchy Consensus ({len(all_agents)} agents)",
                    "data": {
                        "agent_count": len(all_agents),
                        "sample_agents": all_agents[:10] if len(all_agents) > 10 else all_agents
                    }
                })

            # Step 4: Confirmation
            confirmation = raw_metadata.get("confirmation_details", {})
            if confirmation:
                causal_chain.append({
                    "step": 4,
                    "what": "Dual Confirmation",
                    "data": confirmation
                })

            # Step 5: Prediction made
            causal_chain.append({
                "step": 5,
                "what": "Prediction Made",
                "data": {
                    "symbol": symbol,
                    "direction": direction.upper(),
                    "confidence": confidence,
                    "regime": raw_metadata.get("regime_full", "unknown")
                }
            })

            # Step 6: Outcome
            causal_chain.append({
                "step": 6,
                "what": "Outcome Verified",
                "data": {
                    "actual_return": actual_return,
                    "was_correct": was_correct,
                    "verified_after": (verified_at - predicted_at).total_seconds(),
                }
            })

            # Build reasoning tree with weights
            reasoning_tree = {
                "root": f"Predicted {direction.upper()} for {symbol}",
                "branches": []
            }

            # Add layer contributions
            for layer_name, layer_data in layer_states.items():
                if isinstance(layer_data, dict):
                    reasoning_tree["branches"].append({
                        "factor": layer_name.title(),
                        "weight": layer_data.get("confidence", 0),
                        "direction": layer_data.get("direction", "neutral"),
                        "details": layer_data
                    })

            # Build thinking log
            thinking_log = [
                {"thought": "Gathered market variables", "result": f"{len(raw_metadata.get('complete_variables', {}))} variables captured"},
                {"thought": "Analyzed market layers", "result": f"{len(layer_states)} layers evaluated"},
                {"thought": f"Consensus: {direction.upper()}", "result": f"Confidence: {confidence:.0%}"},
            ]

            if all_agents:
                thinking_log.append({
                    "thought": "Agent hierarchy consulted",
                    "result": f"{len(all_agents)} agents provided signals"
                })

            thinking_log.append({
                "thought": "Dual confirmation applied",
                "result": f"Confirmed: {confirmation.get('confirmed', False)}"
            })

            # Build factors_analyzed from layer contributions
            factors_analyzed = []
            for layer_name, layer_data in layer_states.items():
                if isinstance(layer_data, dict):
                    layer_confidence = layer_data.get("confidence", 0)
                    layer_direction = layer_data.get("direction", "neutral")

                    # Calculate contribution (confidence weighted by direction)
                    contrib_value = layer_confidence if layer_direction == direction else -layer_confidence
                    contrib_str = f"{contrib_value:+.2f}" if contrib_value != 0 else "0"

                    factors_analyzed.append({
                        "factor": layer_name.title(),
                        "value": layer_direction,
                        "weight": round(layer_confidence, 3),
                        "contributed": contrib_str
                    })

            # Build variables_changed from complete_variables
            variables_changed = {}
            complete_vars = raw_metadata.get("complete_variables", {})
            if complete_vars:
                # Track key variables that influenced the decision
                key_vars = ["PRICE", "EMA_9", "RSI_14", "MACD_HISTOGRAM", "VOLUME", "VWAP"]
                for var_name in key_vars:
                    if var_name in complete_vars:
                        variables_changed[var_name] = {
                            "value": round(complete_vars[var_name], 4) if isinstance(complete_vars[var_name], (int, float)) else str(complete_vars[var_name]),
                            "impact_on_system": f"Contributed to {direction} signal"
                        }

            # Store in database
            async with get_session() as session:
                await session.execute(text("""
                    INSERT INTO decision_audits (
                        audit_id,
                        decision_type,
                        decision_timestamp,
                        decision_outcome,
                        predicted_outcome,
                        actual_outcome,
                        was_correct,
                        error_magnitude,
                        causal_chain,
                        reasoning_tree,
                        thinking_log,
                        factors_analyzed,
                        variables_changed,
                        created_at
                    ) VALUES (
                        :audit_id,
                        'prediction',
                        :predicted_at,
                        :direction,
                        :direction,
                        :actual_outcome,
                        :was_correct,
                        :error_mag,
                        :causal_chain,
                        :reasoning_tree,
                        :thinking_log,
                        :factors_analyzed,
                        :variables_changed,
                        :predicted_at
                    )
                    ON CONFLICT (audit_id) DO UPDATE SET
                        actual_outcome = EXCLUDED.actual_outcome,
                        was_correct = EXCLUDED.was_correct,
                        error_magnitude = EXCLUDED.error_magnitude,
                        factors_analyzed = EXCLUDED.factors_analyzed,
                        variables_changed = EXCLUDED.variables_changed
                """), {
                    "audit_id": audit_id,
                    "predicted_at": predicted_at,
                    "direction": direction.upper(),
                    "actual_outcome": "BULLISH" if actual_return > 0 else "BEARISH",
                    "was_correct": was_correct,
                    "error_mag": abs(actual_return),
                    "causal_chain": json.dumps(causal_chain),
                    "reasoning_tree": json.dumps(reasoning_tree),
                    "thinking_log": json.dumps(thinking_log),
                    "factors_analyzed": json.dumps(factors_analyzed),
                    "variables_changed": json.dumps(variables_changed)
                })
                await session.commit()

            self.logger.debug(f"[CAUSALITY] Created decision audit {audit_id[:8]} with complete raw data")

        except Exception as e:
            self.logger.error(f"[CAUSALITY] Failed to create decision audit: {e}")
