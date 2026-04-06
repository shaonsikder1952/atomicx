"""AtomicX (v4.0) — The Brain Architecture.

The Brain serves as the central consciousness above the v3.0 nervous system.
It consists of:
- Meta-Orchestrator: The top-level consciousness loop
- Recursive Reflector: Self-awareness and meta-reasoning
- Evolver (Phase 14+): Proposes and tests radical self-changes
- Maintainer (Phase 16+): Health and drift control
- Decider Core (Phase 14+): Translates fused signals into high-level intent
- Executor Fleet (Phase 15+): Spawns autonomous execution child agents
"""

from atomicx.brain.orchestrator import MetaOrchestrator
from atomicx.brain.reflector import RecursiveReflector
from atomicx.brain.debate import DebateChamber, DebateSummary, DebateArgument
from atomicx.brain.decider import DeciderCore, DecisionIntent
from atomicx.brain.evolver import EvolverAgent, EvolutionProposal

# CognitiveLoop is imported lazily to avoid circular import with execution module.
# Import it directly: `from atomicx.brain.loop import CognitiveLoop`


def __getattr__(name: str):
    if name == "CognitiveLoop":
        from atomicx.brain.loop import CognitiveLoop
        return CognitiveLoop
    raise AttributeError(f"module 'atomicx.brain' has no attribute {name!r}")


__all__ = [
    "MetaOrchestrator",
    "RecursiveReflector",
    "CognitiveLoop",
    "DebateChamber",
    "DebateSummary",
    "DebateArgument",
    "DeciderCore",
    "DecisionIntent",
    "EvolverAgent",
    "EvolutionProposal"
]

