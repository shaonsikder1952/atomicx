"""Evolution Engine — Variable and Agent Lifecycle (Phase 12).

The Evolution Engine:
1. Discovers new candidate variables from data patterns
2. Promotes high-performing variables/agents
3. Retires underperforming ones
4. Spawns new agent groups based on discovered causal chains
5. Maintains the system's long-term compounding intelligence
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class VariableCandidate(BaseModel):
    """A candidate variable discovered by the Evolution Engine."""
    variable_id: str
    name: str
    formula: str = ""
    source: str = ""  # How it was discovered
    estimated_edge: float = 0.0
    test_results: dict[str, float] = Field(default_factory=dict)
    promoted: bool = False
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class EvolutionAction(BaseModel):
    """A single evolutionary action taken."""
    action_type: str  # "promote", "demote", "spawn", "retire", "adjust"
    target: str  # variable/agent ID
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionEngine:
    """Long-term system evolution — discovery, promotion, retirement.

    Runs on a schedule (after every N predictions) to evolve the system.
    """

    PROMOTION_THRESHOLD = 0.03  # 3% edge to promote
    DEMOTION_THRESHOLD = 0.01  # <1% edge after 200+ predictions → demote
    MIN_SAMPLES = 200  # Minimum predictions before judging

    def __init__(self) -> None:
        self._candidates: list[VariableCandidate] = []
        self._actions: list[EvolutionAction] = []
        self._generation: int = 0

    def evolve(
        self,
        agent_performance: list[dict[str, Any]],
        variable_performance: list[dict[str, Any]],
    ) -> list[EvolutionAction]:
        """Run one evolution cycle.

        Args:
            agent_performance: List of {id, win_rate, predictions, edge, active}
            variable_performance: List of {id, weight, edge, predictions}

        Returns:
            List of evolution actions taken
        """
        self._generation += 1
        actions: list[EvolutionAction] = []

        logger.info(f"Evolution Engine — Generation {self._generation}")

        # 1. Promote high performers
        for agent in agent_performance:
            if (
                agent["predictions"] >= self.MIN_SAMPLES
                and agent["edge"] >= self.PROMOTION_THRESHOLD
                and agent["active"]
            ):
                action = EvolutionAction(
                    action_type="promote",
                    target=agent["id"],
                    reason=f"Edge {agent['edge']:.2%} > {self.PROMOTION_THRESHOLD:.2%} "
                           f"after {agent['predictions']} predictions",
                    metadata={"new_weight": min(agent.get("weight", 1.0) * 1.2, 2.0)},
                )
                actions.append(action)

        # 2. Demote underperformers
        for agent in agent_performance:
            if (
                agent["predictions"] >= self.MIN_SAMPLES
                and agent["edge"] < self.DEMOTION_THRESHOLD
                and agent["active"]
            ):
                action = EvolutionAction(
                    action_type="demote",
                    target=agent["id"],
                    reason=f"Edge {agent['edge']:.2%} < {self.DEMOTION_THRESHOLD:.2%} "
                           f"after {agent['predictions']} predictions",
                    metadata={"new_weight": max(agent.get("weight", 1.0) * 0.7, 0.1)},
                )
                actions.append(action)

        # 3. Retire dead agents
        for agent in agent_performance:
            if not agent["active"] and agent["predictions"] >= self.MIN_SAMPLES:
                actions.append(EvolutionAction(
                    action_type="retire",
                    target=agent["id"],
                    reason=f"Auto-pruned after {agent['predictions']} predictions",
                ))

        # 4. Spawn candidates for high-edge variable combinations
        top_vars = sorted(
            variable_performance,
            key=lambda v: v.get("edge", 0),
            reverse=True,
        )[:5]

        if len(top_vars) >= 2:
            # Suggest interaction variable
            v1, v2 = top_vars[0], top_vars[1]
            candidate = VariableCandidate(
                variable_id=f"{v1['id']}_x_{v2['id']}",
                name=f"Interaction: {v1['id']} × {v2['id']}",
                formula=f"{v1['id']} * {v2['id']}",
                source="evolution_engine",
                estimated_edge=(v1.get("edge", 0) + v2.get("edge", 0)) / 2,
            )
            self._candidates.append(candidate)
            actions.append(EvolutionAction(
                action_type="spawn",
                target=candidate.variable_id,
                reason=f"Interaction of top 2 performers (est. edge: {candidate.estimated_edge:.2%})",
                metadata={"formula": candidate.formula},
            ))

        # 5. Log cycle
        promotions = sum(1 for a in actions if a.action_type == "promote")
        demotions = sum(1 for a in actions if a.action_type == "demote")
        retirements = sum(1 for a in actions if a.action_type == "retire")
        spawns = sum(1 for a in actions if a.action_type == "spawn")

        logger.info(
            f"Evolution Gen {self._generation}: "
            f"{promotions} promoted, {demotions} demoted, "
            f"{retirements} retired, {spawns} spawned"
        )

        self._actions.extend(actions)
        return actions

    def get_candidate_variables(self) -> list[VariableCandidate]:
        """Get all discovered candidate variables."""
        return self._candidates

    def get_evolution_history(self) -> list[EvolutionAction]:
        """Get the complete evolution history."""
        return self._actions

    @property
    def generation(self) -> int:
        return self._generation
