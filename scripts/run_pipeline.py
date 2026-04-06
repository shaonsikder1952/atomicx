"""Entry point for the AtomicX data pipeline service."""

import asyncio
import signal

from loguru import logger

from atomicx.common.logging import setup_logging
from atomicx.data.pipeline import DataPipelineOrchestrator


async def main() -> None:
    """Run the data pipeline."""
    setup_logging()
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  AtomicX — Causal Intelligence Engine    ║")
    logger.info("║  Data Pipeline Service v0.1.0            ║")
    logger.info("╚══════════════════════════════════════════╝")

    pipeline = DataPipelineOrchestrator()

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(pipeline.stop()))

    try:
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
