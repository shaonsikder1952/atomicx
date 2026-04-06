"""Alternative Data Sources for Edge.

Data sources that provide signals not available in standard OHLCV:
- Mempool monitoring (whale transactions)
- Funding rates (leverage signals)
- Options flow (institutional positioning)
- Stablecoin flows (capital movement)
"""

from __future__ import annotations

__all__ = [
    "MempoolMonitor",
    "FundingRateTracker",
    "OptionsFlowScanner",
    "StablecoinFlowAnalyzer",
]
