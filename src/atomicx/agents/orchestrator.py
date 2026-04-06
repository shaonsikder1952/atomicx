"""Agent Hierarchy Orchestrator — builds and runs the full agent tree.

Constructs the complete hierarchy from the variable catalog:
  46 variables → 46 Atomic Agents → ~10 Group Leaders → ~3 Super Groups
  → 1 Verification Layer → 1 Common-Sense Layer → 1 Domain Leader → Fusion Node
"""

from __future__ import annotations

from loguru import logger

from atomicx.agents.base import AgentConfig
from atomicx.agents.hierarchy.atomic import AtomicAgent
from atomicx.agents.hierarchy.group_leader import GroupLeader
from atomicx.agents.hierarchy.upper import (
    CommonSenseLayer,
    DomainLeader,
    SuperGroup,
    VerificationLayer,
)
from atomicx.agents.signals import AgentSignal, SignalDirection
from atomicx.variables.catalog import get_default_variables


class AgentHierarchy:
    """Builds and manages the complete agent hierarchy.

    Automatically constructs the tree from the variable catalog,
    grouping variables by category into Group Leaders, then into
    Super Groups by domain.
    """

    def __init__(self) -> None:
        self.atomic_agents: dict[str, AtomicAgent] = {}
        self.group_leaders: dict[str, GroupLeader] = {}
        self.super_groups: dict[str, SuperGroup] = {}
        self.verification_layer: VerificationLayer | None = None
        self.common_sense_layer: CommonSenseLayer | None = None
        self.domain_leader: DomainLeader | None = None
        self._built = False

    def build(self) -> None:
        """Construct the full hierarchy from the variable catalog."""
        variables = get_default_variables()

        # Level 1: Create Atomic Agents
        for var in variables:
            agent = AtomicAgent(variable_id=var.id)
            self.atomic_agents[var.id] = agent

        logger.info(f"Created {len(self.atomic_agents)} atomic agents")

        # Level 2: Group by category → Group Leaders
        category_map: dict[str, list[AtomicAgent]] = {}
        for var in variables:
            cat = var.category
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(self.atomic_agents[var.id])

        for category, agents in category_map.items():
            gl = GroupLeader(
                group_id=category,
                name=f"Group Leader: {category.replace('_', ' ').title()}",
                children=agents,
            )
            self.group_leaders[category] = gl

        logger.info(f"Created {len(self.group_leaders)} group leaders: {list(self.group_leaders.keys())}")

        # Level 3: Group by domain → Super Groups
        domain_groups: dict[str, list[GroupLeader]] = {
            "technical": [],  # trend, momentum, volatility, volume
            "microstructure": [],  # microstructure, leverage
            "macro": [],  # market_overview, trend_strength, time_cycle, market_cycle
        }

        for category, gl in self.group_leaders.items():
            if category in ("trend", "momentum", "volatility", "volume"):
                domain_groups["technical"].append(gl)
            elif category in ("microstructure", "leverage"):
                domain_groups["microstructure"].append(gl)
            else:
                domain_groups["macro"].append(gl)

        for domain, leaders in domain_groups.items():
            if leaders:
                sg = SuperGroup(
                    super_group_id=domain,
                    name=f"Super Group: {domain.title()}",
                    group_leaders=leaders,
                )
                self.super_groups[domain] = sg

        logger.info(f"Created {len(self.super_groups)} super groups: {list(self.super_groups.keys())}")

        # Level 4: Verification Layer
        self.verification_layer = VerificationLayer(
            super_groups=list(self.super_groups.values())
        )

        # Level 5: Common-Sense Layer
        self.common_sense_layer = CommonSenseLayer()

        # Level 6: Domain Leader
        self.domain_leader = DomainLeader(
            domain="economic",
            verification_layer=self.verification_layer,
            common_sense_layer=self.common_sense_layer,
        )

        self._built = True
        total_agents = (
            len(self.atomic_agents)
            + len(self.group_leaders)
            + len(self.super_groups)
            + 3  # verification, common-sense, domain leader
        )
        logger.info(f"Hierarchy built: {total_agents} total agents across 6 levels")

    def get_all_agents(self) -> list:
        """Get all agents in the hierarchy for persistence operations.

        ═══ FIX: Added for agent performance loading on startup ═══
        """
        agents = []
        agents.extend(self.atomic_agents.values())
        agents.extend(self.group_leaders.values())
        agents.extend(self.super_groups.values())
        if self.verification_layer:
            agents.append(self.verification_layer)
        if self.common_sense_layer:
            agents.append(self.common_sense_layer)
        if self.domain_leader:
            agents.append(self.domain_leader)
        return agents

    async def evaluate(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal | None:
        """Run the full hierarchy and produce a final signal.

        Args:
            symbol: Trading pair
            timeframe: Analysis timeframe
            context: Market context including variable snapshot

        Returns:
            Final domain leader signal
        """
        if not self._built:
            self.build()

        if not self.domain_leader:
            return None

        signal = await self.domain_leader.evaluate(symbol, timeframe, context)
        return signal

    def get_agent_tree(self) -> dict:
        """Return the hierarchy structure for visualization."""
        return {
            "atomic_agents": len(self.atomic_agents),
            "group_leaders": {
                name: [a.agent_id for a in gl.children]
                for name, gl in self.group_leaders.items()
            },
            "super_groups": {
                name: [gl.agent_id for gl in sg.group_leaders]
                for name, sg in self.super_groups.items()
            },
            "verification": self.verification_layer.agent_id if self.verification_layer else None,
            "common_sense": self.common_sense_layer.agent_id if self.common_sense_layer else None,
            "domain_leader": self.domain_leader.agent_id if self.domain_leader else None,
        }

    def get_performance_summary(self) -> list[dict]:
        """Get performance summary for all agents."""
        results = []
        for agent_id, agent in self.atomic_agents.items():
            results.append({
                "id": agent_id,
                "type": "atomic",
                "win_rate": agent.win_rate,
                "predictions": agent.config.total_predictions,
                "edge": agent.config.performance_edge,
                "active": agent.is_active,
            })
        return sorted(results, key=lambda x: x["edge"], reverse=True)
