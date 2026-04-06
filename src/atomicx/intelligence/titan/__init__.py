"""Titan-Killer Intelligence Package.

Institutional-grade modules stolen from the world's best quant firms,
adapted for the "Small Fish" advantage.
"""

from atomicx.intelligence.titan.kernel_engine import KernelCorrelationEngine
from atomicx.intelligence.titan.retail_flow import RetailFlowAnticipator
from atomicx.intelligence.titan.expiry_sentinel import ExpiryDaySentinel

__all__ = ["KernelCorrelationEngine", "RetailFlowAnticipator", "ExpiryDaySentinel"]
