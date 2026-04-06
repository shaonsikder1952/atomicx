"""DMN Memory Tiers.

Defines the 5 living tiers of the Dynamic Memory Nexus:
Tier 0: Instant Sensory Buffer
Tier 1: Short-Term Pattern Memory
Tier 2: Strategy Genome
Tier 3: Long-Term Causal Knowledge
Tier 4: Evolutionary Archive
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Base structure for an entry in any DMN tier."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    data: dict[str, Any]


class SensoryBufferTier0:
    """Tier 0: Holds exactly the last 60 seconds of raw market data."""
    def __init__(self) -> None:
        self.buffer: list[MemoryEntry] = []
        self.max_age_seconds = 60

    def add(self, data: dict[str, Any]) -> None:
        """Push raw sensory data into the instant buffer."""
        self.buffer.append(MemoryEntry(data=data))
        self.prune()

    def prune(self) -> None:
        """Discard data older than 60 seconds."""
        now = datetime.now(tz=timezone.utc)
        self.buffer = [
            e for e in self.buffer 
            if (now - e.timestamp).total_seconds() <= self.max_age_seconds
        ]


class ShortTermTier1:
    """Tier 1: Holds emerging patterns and current narratives."""
    def __init__(self) -> None:
        # In the future, this interfaces with Qdrant/Mem0
        self.active_patterns: dict[str, MemoryEntry] = {}

    def insert_pattern(self, pattern_type: str, details: dict[str, Any]) -> None:
        """Store a discovered pattern or narrative."""
        self.active_patterns[pattern_type] = MemoryEntry(data=details)


class StrategyGenomeTier2:
    """Tier 2: Log of every executed strategy and its actual R:R."""
    def __init__(self) -> None:
        self.genome_log: list[MemoryEntry] = []

    def record_performance(self, strategy_id: str, outcome_metrics: dict[str, Any]) -> None:
        """Log the true outcome of a strategy cycle."""
        self.genome_log.append(MemoryEntry(data={"strategy_id": strategy_id, "metrics": outcome_metrics}))


class CausalKnowledgeTier3:
    """Tier 3: Formally proven traps, regime rules, and deep abstractions."""
    def __init__(self) -> None:
        self.knowledge_graph: dict[str, MemoryEntry] = {}


class EvolutionaryArchiveTier4:
    """Tier 4: Cold storage of all strategies and historical outcomes."""
    def __init__(self) -> None:
        self.archive: list[MemoryEntry] = []
        
    def archive_strategy(self, state: dict[str, Any]) -> None:
        """Permanently store a strategy block to feed to Meta-Reflection."""
        self.archive.append(MemoryEntry(data=state))
