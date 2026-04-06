"""Sensory Cache — Singleton for real-time market microstructure data.

Bridges the high-frequency WebSocket streams (Depth, Trade) 
to the lower-frequency Variable Engine and Cognitive Loop.
"""

from __future__ import annotations

import time
from typing import Any
from loguru import logger


class SensoryCache:
    """In-memory cache for the latest market sensory data."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SensoryCache, cls).__new__(cls)
            cls._instance._data = {}
        return cls._instance

    def update(self, symbol: str, key: str, value: Any) -> None:
        """Update a specific sensory metric for a symbol."""
        if symbol not in self._data:
            self._data[symbol] = {}
        
        self._data[symbol][key] = {
            "value": value,
            "timestamp": time.time()
        }

    def get(self, symbol: str, key: str, default: Any = None) -> Any:
        """Get the latest cached value for a symbol/key."""
        node = self._data.get(symbol, {}).get(key)
        if node:
            # TTL check: If data is older than 60 seconds, it's stale
            if time.time() - node["timestamp"] < 60:
                return node["value"]
        return default

    def get_all(self, symbol: str) -> dict[str, Any]:
        """Get all fresh sensory data for a symbol."""
        symbol_data = self._data.get(symbol, {})
        fresh = {}
        now = time.time()
        for k, v in symbol_data.items():
            if now - v["timestamp"] < 60:
                fresh[k] = v["value"]
        return fresh

    def get_timestamp(self, symbol: str, key: str) -> Any:
        """Get the timestamp of when a specific key was last updated.

        FIX: Used by staleness gate to detect zombie data streams.

        Returns:
            datetime object (UTC timezone) of last update, or None if not found
        """
        from datetime import datetime, timezone
        node = self._data.get(symbol, {}).get(key)
        if node:
            return datetime.fromtimestamp(node["timestamp"], tz=timezone.utc)
        return None


def get_sensory_cache() -> SensoryCache:
    """Get the global SensoryCache instance."""
    return SensoryCache()
