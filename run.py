"""AtomicX Main Execution Pipeline — Paper Trading Mode.

This script runs the entire 12-layer Causal Intelligence Engine
in a continuous loop for local paper trading.

Requirements:
- Docker containers running (TimescaleDB, Redis, Kafka, Qdrant)
- Historical backfill completed
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
from datetime import datetime, timezone

from loguru import logger
from dotenv import load_dotenv

# Load environment before any other imports that might use settings
load_dotenv()

# Verify critical environment
assert os.getenv("AWS_DEFAULT_REGION"), "AWS_DEFAULT_REGION not set in .env"

from atomicx.common.logging import setup_logging
from atomicx.config import get_settings
from atomicx.data.connectors.binance_ws import BinanceWebSocketConnector
from atomicx.data.connectors.ohlcv_saver import OHLCVSaver

from atomicx.evolution import SelfImprovementLoop, PredictionOutcome
from atomicx.evolution.engine import EvolutionEngine
from atomicx.fusion.engine import FusionNode
from atomicx.fusion.prediction import PredictionAction
from atomicx.guardrails import TradabilityGuardrails
from atomicx.memory.service import MemoryService
from atomicx.narrative import NarrativeTracker
from atomicx.strategic import StrategicActorLayer
from atomicx.swarm import SwarmSimulator
from atomicx.variables.engine import VariableComputeEngine
from atomicx.brain.orchestrator import MetaOrchestrator
from atomicx.brain.loop import CognitiveLoop

# Global shutdown event
shutdown_event = asyncio.Event()


def handle_shutdown(sig, frame):
    """Handle graceful shutdown."""
    logger.warning("Shutdown signal received. Stopping gracefully...")
    shutdown_event.set()


async def run_pipeline(symbols: list[str], timeframe: str = "1h") -> None:
    """Run the 12-layer intelligence pipeline."""
    setup_logging()

    # INSTITUTIONAL FIX: Initialize Apple Silicon optimizer
    from atomicx.common.hardware import get_optimizer
    optimizer = get_optimizer()
    sys_info = optimizer.get_system_info()
    if sys_info["is_apple_silicon"]:
        logger.success(
            f"[M3-OPTIMIZER] Apple Silicon optimizations enabled: "
            f"{sys_info['performance_cores']}P + {sys_info['efficiency_cores']}E cores, "
            f"{sys_info['memory_available_gb']:.1f}GB available"
        )

    # Register shutdown handlers (only works in main thread)
    try:
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    except ValueError:
        pass

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  AtomicX — Local Paper Trading Engine Starting   ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # 1. Initialize core engines
    settings = get_settings()

    # 3. Intelligence Layers (Cognitive Architecture)
    if settings.aws_access_key_id or settings.aws_bearer_token:
        logger.info(f"LLM enabled via AWS Bedrock (Proxy/Custom: {settings.bedrock_model_id})")
    
    # Wait for Database Readiness
    import socket
    db_port = 5432
    db_host = "127.0.0.1" # Standard for local docker postgres
    logger.info("Checking database connectivity...")
    db_available = False
    for attempt in range(15):
        try:
            with socket.create_connection((db_host, db_port), timeout=2.0):
                logger.success("Database connection established!")
                db_available = True
                break
        except Exception:
            logger.warning(f"Database not ready on {db_host}:{db_port}. Waiting 2s... (Attempt {attempt + 1}/15)")
            await asyncio.sleep(2)
    else:
        logger.error("Database connection timed out. System may operate in degraded fallback mode (CCXT live data only).")

    # ═══ AUTONOMOUS MULTI-ASSET: Auto-load from database ═══
    if db_available:
        logger.info("═══ AUTONOMOUS ASSET LOADING ═══")
        logger.info("Querying portfolio database for active assets...")
        try:
            from atomicx.data.storage.database import get_session
            from atomicx.data.storage.models import PortfolioAsset
            from atomicx.data.asset_manager import get_asset_manager
            from sqlalchemy import select

            async with get_session() as session:
                # Load all active and ready assets
                result = await session.execute(
                    select(PortfolioAsset).where(
                        PortfolioAsset.is_active == True,
                        PortfolioAsset.status == 'active'  # Only load fully initialized assets
                    )
                )
                active_assets = result.scalars().all()

                # Add active assets to tracking list
                for asset in active_assets:
                    if asset.symbol not in symbols:
                        symbols.append(asset.symbol)
                        logger.success(
                            f"[PORTFOLIO] ✓ Loaded {asset.symbol} "
                            f"({asset.asset_type}, {asset.data_source})"
                        )

                if active_assets:
                    logger.success(
                        f"[PORTFOLIO] {len(active_assets)} active assets ready for tracking"
                    )

                # Check for pending/initializing assets (need background init)
                pending_result = await session.execute(
                    select(PortfolioAsset).where(
                        PortfolioAsset.is_active == True,
                        PortfolioAsset.status.in_(['pending', 'initializing', 'error'])
                    )
                )
                pending_assets = pending_result.scalars().all()

                if pending_assets:
                    logger.warning(
                        f"[PORTFOLIO] Found {len(pending_assets)} assets needing initialization"
                    )

                    # Initialize them in background
                    asset_manager = get_asset_manager()
                    for asset in pending_assets:
                        logger.info(f"[PORTFOLIO] Initializing {asset.symbol} in background...")
                        await asset_manager.initialize_asset_background(
                            symbol=asset.symbol,
                            asset_type=asset.asset_type
                        )

                if not active_assets and not pending_assets:
                    logger.info(
                        "[PORTFOLIO] No assets in database. "
                        "Add assets through UI at http://localhost:8001/"
                    )

        except Exception as e:
            logger.warning(f"[PORTFOLIO] Failed to load assets from database: {e}")

    # Run data ingestion pipeline if DB is available
    if db_available:
        from atomicx.data.startup import DataStartupOrchestrator

        logger.info("\n" + "="*80)
        logger.info("RUNNING DATA INGESTION PIPELINE")
        logger.info("="*80)

        data_orchestrator = DataStartupOrchestrator(symbols=symbols)
        try:
            startup_results = await data_orchestrator.run_startup_pipeline(
                force_full_backfill=True,  # FIX: Download full 500 candles for indicator computation
                run_pattern_discovery=False,  # Skip expensive analysis on startup
            )

            if startup_results["status"] == "failed":
                logger.error("Data pipeline startup failed. Continuing with degraded mode...")
            elif startup_results["status"] == "partial":
                logger.warning("Data pipeline completed with warnings. Some features may be degraded.")
            else:
                logger.success("Data pipeline startup complete — all systems ready!")

        except Exception as e:
            logger.error(f"Data pipeline startup error: {e}. Continuing with degraded mode...")
        finally:
            await data_orchestrator.close()
    else:
        logger.warning("Skipping data ingestion pipeline (database unavailable)")

    # Sensory Organs
    var_engine = VariableComputeEngine()

    
    # Construct the memory service with user_id
    memory = MemoryService(user_id="atomicx_paper_trader")
    await memory.initialize()
    
    fusion_node = FusionNode(memory=memory)
    await fusion_node.initialize()
    
    strategic_layer = StrategicActorLayer()
    narrative_tracker = NarrativeTracker()
    swarm_sim = SwarmSimulator()
    guardrails = TradabilityGuardrails()
    self_improvement = SelfImprovementLoop(hierarchy=fusion_node.hierarchy, memory=memory)
    evolution = EvolutionEngine()

    logger.success("All 12 architecture layers initialized.")

    # 2. Connect to live data (Binance WebSocket)
    # FIX: OHLCV Saver - Persists real-time candles to TimescaleDB
    ohlcv_saver = OHLCVSaver(batch_size=10, flush_interval=5.0)
    await ohlcv_saver.start()
    logger.info("OHLCV Saver started (batch writes to TimescaleDB)")

    ws_client = BinanceWebSocketConnector(
        symbols=symbols,
        on_kline=ohlcv_saver.on_kline,  # FIX: Wire saver to WebSocket kline events
        on_cvd=ohlcv_saver.on_cvd,  # INSTITUTIONAL FIX: CVD persistence
        on_liquidation=ohlcv_saver.on_liquidation  # INSTITUTIONAL FIX: Liquidation persistence (Pain Map)
    )
    asyncio.create_task(ws_client.start())

    # FIX: Start Variable Engine background compute loop (stores to registry/database every 60s)
    asyncio.create_task(var_engine.start(compute_interval=60.0))
    logger.info("Variable Engine background compute loop started (60s interval)")

    logger.info(f"Subscribed to live streams for: {symbols}")

    # 3. Instantiate the Brain (AtomicX)
    orchestrator = MetaOrchestrator(
        var_engine=var_engine,
        fusion_node=fusion_node,
        strategic_layer=strategic_layer,
        narrative_tracker=narrative_tracker,
        swarm_sim=swarm_sim
    )
    
    # 8. Cognitive Loop (Observe -> Simulate -> Debate -> Decide -> Act -> Reflect)
    brain_loop = CognitiveLoop(
        orchestrator=orchestrator, 
        symbols=symbols,
        self_improvement=self_improvement,
        ws_client=ws_client
    )
    
    # 3.5 Start Dashboard Server (Phase 18)
    from atomicx.dashboard.app import create_dashboard_app
    from atomicx.brain.chatbot_agent import AgenticChatbot
    import uvicorn

    # Create agentic chatbot with tool use
    chatbot = AgenticChatbot(orchestrator)
    logger.info("[CHATBOT] Agentic chatbot initialized with dynamic tool use")

    # ═══ FIX: Link dynamic symbol addition from portfolio API to cognitive loop ═══
    brain_loop.dashboard.on_add_symbol = brain_loop.add_symbol

    # Connect portfolio API to cognitive loop
    from atomicx.dashboard.portfolio_api import set_add_symbol_callback
    set_add_symbol_callback(brain_loop.add_symbol)
    logger.success("[PORTFOLIO] Connected portfolio API to cognitive loop for dynamic asset tracking")

    dashboard_app = create_dashboard_app(
        state=brain_loop.dashboard,
        chat_handler=chatbot.ask,
        orchestrator=orchestrator
    )

    
    config = uvicorn.Config(dashboard_app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    
    # Run dashboard in background
    asyncio.create_task(server.serve())
    logger.success("Dashboard active at http://localhost:8001")

    # Configure shutdown task to stop brain loop
    async def shutdown_watcher():
        await shutdown_event.wait()
        brain_loop.stop()
        await server.shutdown()
        
    asyncio.create_task(shutdown_watcher())

    # 4. Start the Cognitive Loop (as background task)
    asyncio.create_task(brain_loop.start())
    logger.success("Cognitive loop started in background")

    # Keep server alive until shutdown signal
    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received")

    # Cleanup
    await ws_client.stop()
    await ohlcv_saver.stop()  # FIX: Flush remaining candles before shutdown
    logger.info("AtomicX pipeline shutdown gracefully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtomicX Local Paper Trading Pipeline")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC/USDT"],
        help="Trading pairs to monitor",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h",
        help="Primary analysis timeframe",
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(run_pipeline(args.symbols, args.timeframe))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
