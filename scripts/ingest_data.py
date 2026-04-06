#!/usr/bin/env python3
"""Standalone script for running data ingestion pipeline.

Usage:
    python scripts/ingest_data.py                       # Default: incremental sync for configured symbols
    python scripts/ingest_data.py --force-backfill      # Full re-download of all data
    python scripts/ingest_data.py --pattern-discovery   # Also run pattern discovery (expensive)
    python scripts/ingest_data.py --health-only         # Only check data health
    python scripts/ingest_data.py --symbols BTC/USDT ETH/USDT  # Specific symbols
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv()

from atomicx.data.startup import run_startup_with_args


if __name__ == "__main__":
    asyncio.run(run_startup_with_args())
