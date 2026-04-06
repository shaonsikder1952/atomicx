"""Daily Evolution Cycle.

Bridges the Brain's EvolverAgent with the Memory's PerformanceGenome.
Runs daily (or on major events) to propose bigger architectural changes:
retiring harmful strategies, mutating successful ones, spawning new variants.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from atomicx.memory.genome import PerformanceGenome
from atomicx.memory.tiers import EvolutionaryArchiveTier4


class DailyEvolutionCycle:
    """Orchestrates the daily evolution pass across the genome."""
    
    def __init__(self, genome: PerformanceGenome, tier4: EvolutionaryArchiveTier4) -> None:
        self.genome = genome
        self.tier4 = tier4
        self.logger = logger.bind(module="memory.evolution")
        self.evolution_count = 0
        
    def run_evolution(self, memory_service: Any = None) -> dict[str, Any]:
        """Execute one complete evolution cycle.
        
        Args:
            memory_service: Optional MemoryService for persisting evolution to Qdrant.
        """
        self.evolution_count += 1
        self.logger.info(f"=== DAILY EVOLUTION CYCLE #{self.evolution_count} ===")
        
        results = {
            "cycle": self.evolution_count,
            "retired": [],
            "mutated": [],
            "spawned": [],
        }
        
        # Step 1: Detect edge decay across the genome
        degrading = self.genome.compute_edge_decay()
        
        # Step 2: Retire strategies that have decayed beyond recovery
        for gene in degrading:
            if gene.edge_decay_rate > 0.7:  # 70%+ edge loss → retire
                self.genome.retire_strategy(gene.strategy_id)
                results["retired"].append(gene.strategy_id)
                self.logger.warning(f"[Evolution] RETIRED '{gene.strategy_id}' — edge decayed {gene.edge_decay_rate:.0%}")
                
        # ═══ FIX: Dynamic "Top 20%" threshold instead of static 55% ═══
        # Calculate dynamic mutation threshold based on top performers
        active_genes = [g for g in self.genome.genes.values() if g.status == "active" and g.total_trades >= 20]

        if active_genes:
            # Sort by win rate descending
            sorted_genes = sorted(active_genes, key=lambda g: g.win_rate, reverse=True)

            # Calculate top 20% threshold (or top 3 if less than 15 genes)
            top_count = max(3, len(sorted_genes) // 5)  # 20% or minimum 3
            mutation_candidates = sorted_genes[:top_count]

            if mutation_candidates:
                mutation_threshold = mutation_candidates[-1].win_rate  # Lowest win rate in top 20%
                self.logger.info(
                    f"[Evolution] Dynamic mutation threshold: {mutation_threshold:.1%} "
                    f"(top {top_count}/{len(sorted_genes)} performers)"
                )
            else:
                mutation_threshold = 0.50  # Fallback to 50% if no candidates
                self.logger.warning("[Evolution] No mutation candidates found, using 50% fallback threshold")

            # Mutate strategies in top 20%
            for gene in mutation_candidates:
                sid = gene.strategy_id

                # Create mutant with actual parameter changes (15% variance)
                mutant_gene = gene.mutate(mutation_rate=0.15)
                mutant_id = f"{sid}_mutant_v{self.evolution_count}"
                mutant_gene.strategy_id = mutant_id

                # Register the mutant in genome
                self.genome.genes[mutant_id] = mutant_gene

                # Track parameter changes for logging
                param_diffs = {}
                for key in gene.parameters:
                    old_val = gene.parameters[key]
                    new_val = mutant_gene.parameters[key]
                    if old_val != new_val:
                        diff_pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                        param_diffs[key] = f"{old_val:.3f}→{new_val:.3f} ({diff_pct:+.1f}%)"

                results["mutated"].append({
                    "parent": sid,
                    "child": mutant_id,
                    "parameter_changes": param_diffs,
                    "parent_win_rate": gene.win_rate,
                })

                self.logger.success(
                    f"[Evolution] MUTATED '{sid}' (win_rate={gene.win_rate:.1%}) → '{mutant_id}' (Gen {mutant_gene.generation}). "
                    f"Changed parameters: {list(param_diffs.keys())}"
                )
        else:
            self.logger.warning("[Evolution] No active strategies with >=20 trades - skipping mutation")
                
        # Step 4: Spawn fresh strategies if genome is too thin
        if len([g for g in self.genome.genes.values() if g.status == "active"]) < 3:
            spawn_id = f"auto_spawn_{self.evolution_count}"
            self.genome.register_strategy(spawn_id)
            results["spawned"].append(spawn_id)
            self.logger.info(f"[Evolution] SPAWNED new strategy '{spawn_id}' — genome was too thin.")
            
        # Step 5: Log genome summary
        summary = self.genome.get_genome_summary()
        self.logger.info(f"[Evolution] Genome Summary: {summary}")
        
        # Step 6: Persist evolution results to Qdrant (permanent treasure)
        if memory_service:
            import asyncio
            try:
                from atomicx.memory.service import MemoryEntry, MemoryType
                loop = asyncio.get_event_loop()
                loop.create_task(memory_service.store(MemoryEntry(
                    memory_type=MemoryType.PROCEDURAL,
                    content=(
                        f"EVOLUTION CYCLE #{self.evolution_count}: "
                        f"Retired {len(results['retired'])} strategies, "
                        f"Mutated {len(results['mutated'])} strategies, "
                        f"Spawned {len(results['spawned'])} strategies. "
                        f"Genome: {summary}"
                    ),
                    importance=0.95,
                    tags=["evolution", f"cycle_{self.evolution_count}"],
                    metadata=results,
                )))
            except Exception as e:
                self.logger.warning(f"Failed to persist evolution to Qdrant: {e}")

        # Step 7: Ingest evolution cycle to Wiki (Karpathy LLM Wiki Pattern)
        if memory_service and hasattr(memory_service, 'wiki'):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(memory_service.wiki.ingest_evolution_cycle(
                    cycle_number=self.evolution_count,
                    retired=[{"name": s, "reason": "Edge decay >70%"} for s in results["retired"]],
                    mutated=[{"name": m["child"], "mutation": f"Mutated from {m['parent']}", "parameter_changes": m["parameter_changes"]} for m in results["mutated"]],
                    spawned=[{"name": s, "description": "Auto-spawned due to thin genome"} for s in results["spawned"]],
                    genome_summary=summary
                ))
                self.logger.debug(f"[WIKI] Ingested evolution cycle #{self.evolution_count}")
            except Exception as e:
                self.logger.warning(f"[WIKI] Failed to ingest evolution cycle: {e}")

        return results
