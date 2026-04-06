"""Evolver Agent for AtomicX.

Scans the system's past performance and internal monologues to suggest
radical self-modifications (Phase 14). Proposes architectural mutations
which the MaintainerAgent (Phase 16) can eventually ratify and enact.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from atomicx.brain.reflector import RecursiveReflector


class EvolutionProposal(BaseModel):
    """A proposed mutation to the system's architecture or reasoning."""
    component: str        # e.g., "debate_weights", "guardrail_thresholds"
    mutation_type: str    # "increase_trust", "decrease_trust", "spawn_agent"
    rationale: str
    confidence: float
    status: str = "pending" # pending, approved, rejected, enacted


class EvolverAgent:
    """Proposes architectural and weighted mutations based on reflection history."""

    def __init__(self, reflector: RecursiveReflector) -> None:
        self.reflector = reflector
        self.proposals: list[EvolutionProposal] = []
        self.logger = logger.bind(module="brain.evolver")

    async def analyze_and_propose(self) -> list[EvolutionProposal]:
        """Review recent monologues and propose system evolutions."""
        monologues = await self.reflector.get_recent_monologues(count=20)
        if len(monologues) < 5:
            self.logger.debug("Not enough history to propose evolutions.")
            return []
            
        new_proposals = []
        
        # Example Heuristic 1: If Swarm is consistently out of alignment with the regime
        low_alignment_count = sum(1 for m in monologues if m.regime_alignment < 0.7)
        if low_alignment_count >= 3:
            proposal = EvolutionProposal(
                component="trust_weights",
                mutation_type="permanently_decrease_swarm",
                rationale=f"{low_alignment_count} recent cycles showed poor regime alignment. Suggesting a hard cap on Swarm trust.",
                confidence=0.85
            )
            new_proposals.append(proposal)

        # Example Heuristic 2: If Causal Purist keeps voting against a strong narrative but we miss trades
        conflict_count = sum(1 for m in monologues if "conflict" in m.reasoning.lower())
        if conflict_count >= 4:
            proposal = EvolutionProposal(
                component="debate_chamber",
                mutation_type="add_tiebreaker_agent",
                rationale=f"High conflict rate ({conflict_count} recent cycles). The Debate Chamber needs a dedicated Tiebreaker Agent focusing purely on Order Book Imbalance.",
                confidence=0.9
            )
            new_proposals.append(proposal)

        for p in new_proposals:
            self.logger.warning(f"EVOLUTION PROPOSAL GENERATED: [{p.mutation_type.upper()}] - {p.rationale}")
            self.proposals.append(p)
            
        return new_proposals
