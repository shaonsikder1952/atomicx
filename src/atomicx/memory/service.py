"""Memory types and Mem0 adapter.

Defines the 4 memory types (episodic, semantic, causal, procedural)
and wraps Mem0 for self-improving memory with domain-specific metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from atomicx.config import get_settings

try:
    from mem0.embeddings.base import EmbeddingBase
    from mem0.llms.base import LLMBase
except ImportError:
    EmbeddingBase = object
    LLMBase = object



class MemoryType(str, Enum):
    """The 4 memory types in AtomicX."""

    EPISODIC = "episodic"  # Specific events with full variable snapshots
    SEMANTIC = "semantic"  # Distilled understandings and learned rules
    CAUSAL = "causal"  # Mechanism chains with reasoning traces
    PROCEDURAL = "procedural"  # Behavioral rules and action patterns


class MemoryEntry(BaseModel):
    """A memory entry to be stored."""

    memory_type: MemoryType
    content: str = Field(description="Natural language description of the memory")
    symbol: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Context
    variable_snapshot: dict[str, float] = Field(default_factory=dict)
    prediction_id: str | None = None
    outcome: str | None = None  # "correct", "incorrect", None

    # Mem0 metadata
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


import httpx
import json

class ProxyBedrockEmbedding(EmbeddingBase):
    """Custom embedder that uses the AtomicX Bedrock proxy."""
    def __init__(self, settings: Any):
        super().__init__()
        self.settings = settings
        self.url = f"{settings.aws_endpoint_url or f'https://bedrock-runtime.{settings.aws_region_name}.amazonaws.com'}/model/amazon.titan-embed-text-v1/invoke"
        self.headers = {
            "Authorization": f"Bearer {settings.aws_bearer_token}",
            "x-api-key": settings.aws_bearer_token,
            "Content-Type": "application/json",
        }

    def embed(self, text: str, *args, **kwargs):
        payload = {"inputText": text}
        with httpx.Client() as client:
            resp = client.post(self.url, json=payload, headers=self.headers, timeout=30.0)
            resp.raise_for_status()
            return resp.json().get("embedding")

class ProxyBedrockLLM(LLMBase):
    """Custom LLM that uses the AtomicX Bedrock proxy."""
    def __init__(self, settings: Any):
        from mem0.configs.llms.base import BaseLlmConfig
        config = BaseLlmConfig(model=settings.bedrock_model_id)
        super().__init__(config)
        self.settings = settings
        self.base_url = settings.aws_endpoint_url or f"https://bedrock-runtime.{settings.aws_region_name}.amazonaws.com"
        self.headers = {
            "Authorization": f"Bearer {settings.aws_bearer_token}",
            "x-api-key": settings.aws_bearer_token,
            "Content-Type": "application/json",
        }

    def generate_response(self, messages: list[dict[str, str]], **kwargs) -> str:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [m for m in messages if m["role"] != "system"],
            "system": next((m["content"] for m in messages if m["role"] == "system"), ""),
        }
        url = f"{self.base_url}/model/{self.settings.bedrock_model_id}/invoke"
        with httpx.Client() as client:
            resp = client.post(url, json=payload, headers=self.headers, timeout=30.0)
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            return text

class MemoryService:
    """Wraps Mem0 for self-improving memory.

    Falls back to Qdrant direct if Mem0 is unavailable.
    Provides unified API for all memory operations.
    """

    def __init__(self, user_id: str = "atomicx_system") -> None:
        self._user_id = user_id
        self._mem0_client = None
        self._local_store: list[MemoryEntry] = []  # Fallback
        self._initialized = False
        # ═══ FIX: Direct Qdrant client for stats queries ═══
        self._qdrant_client = None
        self._collection_name = "atomicx_memory"

    async def initialize(self) -> None:
        """Initialize Mem0 connection."""
        settings = get_settings()

        # ═══ FIX: Initialize direct Qdrant client for stats queries ═══
        try:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(url=settings.qdrant_url)
            logger.info(f"Direct Qdrant client initialized: {settings.qdrant_url}")
        except ImportError:
            logger.warning("qdrant-client not installed, stats will be estimated")
            self._qdrant_client = None
        except Exception as e:
            logger.warning(f"Qdrant client init failed: {e}")
            self._qdrant_client = None

        try:
            from mem0 import Memory

            # Base configuration with Qdrant
            # We provide dummy 'openai' configs to satisfy Mem0's internal validation
            # before we swap them for our custom ProxyBedrock providers.
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "url": settings.qdrant_url,
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": { "api_key": "dummy" }
                },
                "llm": {
                    "provider": "openai",
                    "config": { "api_key": "dummy" }
                },
                "custom_fact_extraction_prompt": (
                    "You are the Causal Intelligence Engine for AtomicX. "
                    "Your goal is to extract market regimes, causal links, whale activity, "
                    "and directional setups from the input text. "
                    "Identify specific entities (symbols like BTC/USDT), regimes (TRENDING_BULLISH), "
                    "and causal triggers (whale activity). "
                    "Extracted facts must be concise and objective."
                ),
                "version": "v1.1"
            }

            if settings.aws_bearer_token:
                # Use custom proxy-aware providers
                custom_embedder = ProxyBedrockEmbedding(settings)
                custom_llm = ProxyBedrockLLM(settings)

                # Initialize and swap
                self._mem0_client = Memory.from_config(config)
                self._mem0_client.embedding_model = custom_embedder
                self._mem0_client.llm = custom_llm

                logger.info(f"Mem0 initialized with Qdrant and ProxyBedrock")
            else:
                self._mem0_client = Memory.from_config(config)
                logger.info(f"Mem0 initialized with Qdrant (default providers)")

            self._initialized = True
        except ImportError:
            logger.warning("Mem0 not installed, using local fallback memory")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Mem0 init failed ({e}), using local fallback")
            self._initialized = True

    async def store(self, entry: MemoryEntry) -> str | None:
        """Store a memory entry.

        Returns the memory ID if successful.
        """
        if not self._initialized:
            await self.initialize()

        # Build content with context
        content = self._format_entry(entry)

        if self._mem0_client:
            try:
                result = self._mem0_client.add(
                    content,
                    user_id=self._user_id,
                    metadata={
                        "type": entry.memory_type.value,
                        "symbol": entry.symbol,
                        "timestamp": entry.timestamp.isoformat(),
                        "importance": entry.importance,
                        "prediction_id": entry.prediction_id,
                        "outcome": entry.outcome,
                        **entry.metadata,
                    },
                )
                
                # Mem0 v1.1+ returns {"results": [{"id": "...", ...}]}
                memory_id = None
                if isinstance(result, dict) and "results" in result:
                    results = result["results"]
                    if isinstance(results, list) and len(results) > 0:
                        memory_id = results[0].get("id")
                
                logger.debug(f"Stored {entry.memory_type.value} memory: {content[:80]}...")
                return memory_id
            except Exception as e:
                logger.error(f"Mem0 store failed: {e}")

        # Fallback: local store
        self._local_store.append(entry)
        return f"local_{len(self._local_store)}"

    async def store_prediction(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        reasoning: str,
        variable_snapshot: dict[str, float],
        prediction_id: str,
    ) -> None:
        """Auto-store a prediction as an episodic memory."""
        await self.store(MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content=(
                f"Prediction {prediction_id}: {symbol} {direction} "
                f"@ {confidence:.0%} confidence. Reasoning: {reasoning}"
            ),
            symbol=symbol,
            variable_snapshot=variable_snapshot,
            prediction_id=prediction_id,
            importance=confidence,
            tags=["prediction", direction, symbol],
        ))

    async def store_outcome(
        self,
        prediction_id: str,
        was_correct: bool,
        actual_return: float,
        lessons: str = "",
    ) -> None:
        """Auto-store a prediction outcome as semantic + causal memory."""
        outcome = "correct" if was_correct else "incorrect"

        # Semantic: distilled understanding
        await self.store(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content=(
                f"Prediction {prediction_id} was {outcome}. "
                f"Actual return: {actual_return:.2%}. {lessons}"
            ),
            prediction_id=prediction_id,
            outcome=outcome,
            importance=0.8 if not was_correct else 0.5,  # Failures are more important
            tags=["outcome", outcome],
        ))

        # Causal: what caused the outcome
        if lessons:
            await self.store(MemoryEntry(
                memory_type=MemoryType.CAUSAL,
                content=f"Causal analysis for {prediction_id}: {lessons}",
                prediction_id=prediction_id,
                importance=0.9,
                tags=["causal_analysis", outcome],
            ))

    async def retrieve(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a query.

        Uses Mem0's semantic search with optional type filtering.
        """
        if not self._initialized:
            await self.initialize()

        if self._mem0_client:
            try:
                results = self._mem0_client.search(
                    query, user_id=self._user_id, limit=limit
                )
                memories = results if isinstance(results, list) else results.get("results", [])

                # Normalize to include 'content' key (Mem0 used 'memory')
                for m in memories:
                    if "memory" in m and "content" not in m:
                        m["content"] = m["memory"]

                if memory_type:
                    memories = [
                        m for m in memories
                        if m.get("metadata", {}).get("type") == memory_type.value
                    ]

                return memories
            except Exception as e:
                logger.error(f"Mem0 retrieve failed: {e}")

        # Fallback: simple keyword search
        results = []
        query_lower = query.lower()
        for entry in reversed(self._local_store):
            if memory_type and entry.memory_type != memory_type:
                continue
            if query_lower in entry.content.lower():
                results.append({
                    "content": entry.content,
                    "metadata": {
                        "type": entry.memory_type.value,
                        "symbol": entry.symbol,
                        "timestamp": entry.timestamp.isoformat(),
                    },
                })
            if len(results) >= limit:
                break
        return results

    async def retrieve_similar_setups(
        self, variable_snapshot: dict[str, float], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find past predictions with similar variable conditions."""
        # Build a natural language query from the snapshot
        conditions = []
        for var_id, value in sorted(variable_snapshot.items()):
            # Only include numeric values in the query
            if isinstance(value, (int, float)):
                conditions.append(f"{var_id}={value:.2f}")
            else:
                conditions.append(f"{var_id}={value}")
        query = f"Market conditions: {', '.join(conditions[:10])}"
        return await self.retrieve(query, memory_type=MemoryType.EPISODIC, limit=limit)

    async def get_lessons_for_regime(self, regime: str, limit: int = 10) -> list[dict]:
        """Retrieve past lessons learned in a specific market regime."""
        return await self.retrieve(
            f"Market regime: {regime}", memory_type=MemoryType.SEMANTIC, limit=limit
        )

    def _format_entry(self, entry: MemoryEntry) -> str:
        """Format a memory entry for storage."""
        parts = [entry.content]
        if entry.symbol:
            parts.append(f"[Symbol: {entry.symbol}]")
        if entry.variable_snapshot:
            top_vars = list(entry.variable_snapshot.items())[:5]
            vars_str = ", ".join(f"{k}={v:.2f}" for k, v in top_vars)
            parts.append(f"[Vars: {vars_str}]")
        return " ".join(parts)

    @property
    def memory_count(self) -> int:
        """Total memories stored."""
        if self._mem0_client:
            try:
                all_mems = self._mem0_client.get_all(user_id=self._user_id)
                return len(all_mems) if isinstance(all_mems, list) else 0
            except Exception:
                pass
        return len(self._local_store)

    async def get_stats(self) -> dict[str, Any]:
        """Get memory tier statistics for dashboard.

        Returns:
            Dict with tier1_count (episodic), tier234_count (other tiers)
        """
        tier1_count = 0
        tier234_count = 0

        # ═══ FIX: Count from Qdrant if available ═══
        if self._qdrant_client:
            try:
                # Check if collection exists first
                collections = self._qdrant_client.get_collections()
                collection_names = [c.name for c in collections.collections]

                if self._collection_name in collection_names:
                    collection_info = self._qdrant_client.get_collection(self._collection_name)
                    total_points = collection_info.points_count if hasattr(collection_info, 'points_count') else 0

                    # Rough estimation: 70% episodic (Tier 1), 30% other tiers (Tier 2-4)
                    tier1_count = int(total_points * 0.7)
                    tier234_count = total_points - tier1_count

                    logger.debug(f"[MEMORY STATS] Qdrant: {total_points} total points")
                else:
                    logger.debug(f"[MEMORY STATS] Collection '{self._collection_name}' not yet created")
            except Exception as e:
                logger.warning(f"Failed to get Qdrant stats: {e}")

        # Fallback to local store if no Qdrant or collection empty
        if tier1_count == 0 and tier234_count == 0:
            local_count = len(self._local_store)
            tier1_count = int(local_count * 0.7)
            tier234_count = local_count - tier1_count

        return {
            "tier1_count": tier1_count,
            "tier234_count": tier234_count,
        }
