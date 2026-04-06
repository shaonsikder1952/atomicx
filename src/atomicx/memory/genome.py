"""Performance Genome.

A living scoring system that tracks every strategy's real-world performance
across multiple dimensions: expectancy, Sharpe, drawdown, regime-specific
accuracy, and edge decay rate. This is the "DNA" of the system's intelligence.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field

from atomicx.memory.tiers import StrategyGenomeTier2, EvolutionaryArchiveTier4


class StrategyGene(BaseModel):
    """A single strategy's living performance profile."""
    gene_id: str = Field(default_factory=lambda: f"gene-{uuid.uuid4().hex[:8]}")
    strategy_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Core metrics (updated after every trade)
    total_trades: int = 0
    winning_trades: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0

    # Derived scores (recomputed periodically)
    win_rate: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    edge_decay_rate: float = 0.0  # How fast this strategy loses effectiveness

    # Regime-specific tracking
    regime_scores: dict[str, float] = Field(default_factory=dict)
    best_regime: str = "unknown"
    worst_regime: str = "unknown"

    # Evolution metadata
    generation: int = 1
    parent_gene_id: str | None = None
    status: str = "active"  # active, degrading, retired

    # ═══ FIX: Strategy parameters for real mutations ═══
    parameters: dict[str, float] = Field(default_factory=lambda: {
        "confidence_threshold": 0.72,
        "stop_loss_atr_multiple": 1.5,
        "take_profit_atr_multiple": 2.5,
        "position_size_pct": 0.01,
        "regime_confidence_min": 0.6,
        "trend_strength_threshold": 0.5,
        "volume_threshold": 1.2,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    })

    def mutate(self, mutation_rate: float = 0.15) -> StrategyGene:
        """Create a mutated copy of this gene with parameter variance.

        Args:
            mutation_rate: Percentage variance for each parameter (default 15%)

        Returns:
            New StrategyGene with mutated parameters
        """
        import random

        mutant = StrategyGene(
            strategy_id=f"{self.strategy_id}_mutant",
            generation=self.generation + 1,
            parent_gene_id=self.gene_id,
        )

        # Copy and mutate each parameter
        for param_name, param_value in self.parameters.items():
            # Apply random variance: ±mutation_rate
            variance = random.uniform(-mutation_rate, mutation_rate)
            new_value = param_value * (1.0 + variance)

            # Apply parameter-specific bounds
            if "threshold" in param_name or "confidence" in param_name:
                # Thresholds/confidence: clamp to [0.0, 1.0]
                new_value = max(0.0, min(1.0, new_value))
            elif "atr_multiple" in param_name:
                # ATR multiples: reasonable range [0.5, 10.0]
                new_value = max(0.5, min(10.0, new_value))
            elif "position_size" in param_name:
                # Position size: max 5%
                new_value = max(0.001, min(0.05, new_value))
            elif "rsi" in param_name:
                # RSI levels: [0, 100]
                new_value = max(0, min(100, new_value))
            else:
                # Default: positive values only
                new_value = max(0.0, new_value)

            mutant.parameters[param_name] = round(new_value, 4)

        return mutant


class PerformanceGenome:
    """The living DNA registry of all strategies."""
    
    def __init__(self, tier2: StrategyGenomeTier2, tier4: EvolutionaryArchiveTier4) -> None:
        self.genes: dict[str, StrategyGene] = {}
        self.tier2 = tier2
        self.tier4 = tier4
        self.logger = logger.bind(module="memory.genome")
        
    def register_strategy(self, strategy_id: str, regime: str = "unknown") -> StrategyGene:
        """Create a new gene entry for a strategy."""
        gene = StrategyGene(strategy_id=strategy_id)
        gene.regime_scores[regime] = 0.0
        self.genes[strategy_id] = gene
        self.logger.info(f"[Genome] Registered new gene: {gene.gene_id} for strategy '{strategy_id}'")
        return gene
        
    def record_trade(self, strategy_id: str, profit: float, regime: str) -> None:
        """Update a strategy's gene with a new trade outcome.

        ═══ FIX: Now persists to database after each trade ═══
        """
        if strategy_id not in self.genes:
            self.register_strategy(strategy_id, regime)

        gene = self.genes[strategy_id]
        gene.total_trades += 1
        gene.total_profit += profit

        if profit > 0:
            gene.winning_trades += 1

        # Update running drawdown
        if profit < gene.max_drawdown:
            gene.max_drawdown = profit

        # Recalculate derived scores
        gene.win_rate = gene.winning_trades / gene.total_trades if gene.total_trades > 0 else 0
        gene.expectancy = gene.total_profit / gene.total_trades if gene.total_trades > 0 else 0

        # Update regime-specific score
        current_regime_score = gene.regime_scores.get(regime, 0.0)
        gene.regime_scores[regime] = current_regime_score + profit

        # Find best/worst regimes
        if gene.regime_scores:
            gene.best_regime = max(gene.regime_scores, key=gene.regime_scores.get)
            gene.worst_regime = min(gene.regime_scores, key=gene.regime_scores.get)

        # ═══ FIX: Persist gene to database ═══
        try:
            import asyncio
            asyncio.create_task(self.save_gene(gene))
        except Exception as e:
            logger.warning(f"Failed to persist gene {gene.gene_id}: {e}")
        
    def compute_edge_decay(self) -> list[StrategyGene]:
        """Identify strategies whose edge is decaying over time."""
        degrading = []
        for sid, gene in self.genes.items():
            if gene.total_trades < 10:
                continue  # Not enough data
                
            # Simple decay heuristic: if recent expectancy < historical expectancy
            recent_trades = self.tier2.genome_log[-5:] if len(self.tier2.genome_log) >= 5 else []
            recent_profits = [e.data.get("metrics", {}).get("profit", 0) for e in recent_trades]
            recent_avg = sum(recent_profits) / len(recent_profits) if recent_profits else 0
            
            if recent_avg < gene.expectancy * 0.5:  # Edge decayed by 50%+
                gene.edge_decay_rate = abs(gene.expectancy - recent_avg) / max(abs(gene.expectancy), 0.001)
                gene.status = "degrading"
                degrading.append(gene)
                self.logger.warning(f"[Genome] Edge decay detected: '{sid}' | Decay rate: {gene.edge_decay_rate:.2%}")
                
        return degrading
        
    def retire_strategy(self, strategy_id: str) -> None:
        """Archive a degraded strategy to Tier-4 and mark as retired."""
        if strategy_id in self.genes:
            gene = self.genes[strategy_id]
            gene.status = "retired"
            self.tier4.archive_strategy(gene.model_dump())
            self.logger.info(f"[Genome] Retired strategy '{strategy_id}' to Tier-4 Archive.")
            
    def get_genome_summary(self) -> dict[str, Any]:
        """Produce a summary report of the entire genome."""
        active = [g for g in self.genes.values() if g.status == "active"]
        degrading = [g for g in self.genes.values() if g.status == "degrading"]
        retired = [g for g in self.genes.values() if g.status == "retired"]

        return {
            "total_genes": len(self.genes),
            "active": len(active),
            "degrading": len(degrading),
            "retired": len(retired),
            "best_performer": max(self.genes.values(), key=lambda g: g.expectancy).strategy_id if self.genes else None,
            "worst_performer": min(self.genes.values(), key=lambda g: g.expectancy).strategy_id if self.genes else None,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE METHODS (Fix for genome state loss on restart)
    # ═══════════════════════════════════════════════════════════════════════════

    async def save_gene(self, gene: StrategyGene) -> None:
        """Persist a strategy gene to database.

        Called after every trade execution. Enables cross-session evolution.
        """
        from atomicx.data.storage.database import get_session_factory
        from atomicx.data.storage.models import StrategyGenome
        from sqlalchemy.dialects.postgresql import insert

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                stmt = insert(StrategyGenome).values(
                    strategy_id=gene.strategy_id,
                    gene_id=gene.gene_id,
                    total_trades=gene.total_trades,
                    winning_trades=gene.winning_trades,
                    total_profit=float(gene.total_profit),
                    max_drawdown=float(gene.max_drawdown),
                    win_rate=float(gene.win_rate),
                    expectancy=float(gene.expectancy),
                    sharpe_ratio=float(gene.sharpe_ratio),
                    edge_decay_rate=float(gene.edge_decay_rate),
                    regime_scores=gene.regime_scores,
                    best_regime=gene.best_regime,
                    worst_regime=gene.worst_regime,
                    parameters=gene.parameters,
                    generation=gene.generation,
                    parent_gene_id=gene.parent_gene_id,
                    status=gene.status,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['strategy_id'],
                    set_={
                        'total_trades': stmt.excluded.total_trades,
                        'winning_trades': stmt.excluded.winning_trades,
                        'total_profit': stmt.excluded.total_profit,
                        'max_drawdown': stmt.excluded.max_drawdown,
                        'win_rate': stmt.excluded.win_rate,
                        'expectancy': stmt.excluded.expectancy,
                        'sharpe_ratio': stmt.excluded.sharpe_ratio,
                        'edge_decay_rate': stmt.excluded.edge_decay_rate,
                        'regime_scores': stmt.excluded.regime_scores,
                        'best_regime': stmt.excluded.best_regime,
                        'worst_regime': stmt.excluded.worst_regime,
                        'parameters': stmt.excluded.parameters,
                        'status': stmt.excluded.status,
                        'updated_at': datetime.now(tz=timezone.utc),
                    }
                )
                await session.execute(stmt)
                await session.commit()
                logger.debug(f"[PERSISTENCE] Saved gene {gene.gene_id} ({gene.strategy_id}) to DB")
        except Exception as e:
            logger.error(f"[PERSISTENCE] Failed to save gene {gene.gene_id}: {e}")

    async def load_genes(self) -> int:
        """Load all strategy genes from database on startup.

        Returns number of genes loaded.
        """
        from atomicx.data.storage.database import get_session_factory
        from atomicx.data.storage.models import StrategyGenome as StrategyGenomeDB
        from sqlalchemy import select

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(select(StrategyGenomeDB))
                db_genes = result.scalars().all()

                count = 0
                for db_gene in db_genes:
                    gene = StrategyGene(
                        strategy_id=db_gene.strategy_id,
                        gene_id=db_gene.gene_id,
                        created_at=db_gene.created_at,
                    )
                    gene.total_trades = db_gene.total_trades
                    gene.winning_trades = db_gene.winning_trades
                    gene.total_profit = float(db_gene.total_profit)
                    gene.max_drawdown = float(db_gene.max_drawdown)
                    gene.win_rate = float(db_gene.win_rate)
                    gene.expectancy = float(db_gene.expectancy)
                    gene.sharpe_ratio = float(db_gene.sharpe_ratio)
                    gene.edge_decay_rate = float(db_gene.edge_decay_rate)
                    gene.regime_scores = db_gene.regime_scores
                    gene.best_regime = db_gene.best_regime
                    gene.worst_regime = db_gene.worst_regime
                    gene.parameters = db_gene.parameters
                    gene.generation = db_gene.generation
                    gene.parent_gene_id = db_gene.parent_gene_id
                    gene.status = db_gene.status

                    self.genes[gene.strategy_id] = gene
                    count += 1

                logger.info(f"[PERSISTENCE] Loaded {count} strategy genes from DB")
                return count
        except Exception as e:
            logger.warning(f"[PERSISTENCE] Failed to load genes: {e}")
            return 0
