"""Knowledge Graph.

Stores people, their statements, and measured market reactions.
Enables queries like: "When Person X last spoke about Y, what happened to BTC?"
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field


class PersonNode(BaseModel):
    """A person tracked in the knowledge graph."""
    node_id: str = Field(default_factory=lambda: f"person-{uuid.uuid4().hex[:8]}")
    name: str
    role: str = ""
    significance_score: float = 1.0
    statements: list[dict[str, Any]] = Field(default_factory=list)
    added_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class KnowledgeGraph:
    """Living knowledge graph of people, events, and market reactions."""
    
    def __init__(self) -> None:
        self.people: dict[str, PersonNode] = {}
        self.logger = logger.bind(module="intelligence.graph")

    def add_person(self, name: str, role: str, statements: list[dict[str, Any]] | None = None) -> PersonNode:
        """Add or update a person in the knowledge graph."""
        if name in self.people:
            node = self.people[name]
            if statements:
                node.statements.extend(statements)
            self.logger.info(f"[GRAPH] Updated person '{name}' with {len(statements or [])} new statements")
            return node
            
        node = PersonNode(name=name, role=role, statements=statements or [])
        self.people[name] = node
        self.logger.success(f"[GRAPH] Added new person: '{name}' ({role})")
        return node
        
    def ingest_research(self, research: dict[str, Any]) -> None:
        """Ingest a full BrowserAgent research result into the graph."""
        for person_data in research.get("related_people", []):
            self.add_person(
                name=person_data["name"],
                role=person_data.get("role", ""),
                statements=person_data.get("past_statements", [])
            )
            
    def query_person_impact(self, name: str) -> dict[str, Any]:
        """Query the historical market impact of a person's statements."""
        if name not in self.people:
            return {"error": f"Person '{name}' not found in graph"}
            
        node = self.people[name]
        reactions = [s.get("market_reaction", "0%") for s in node.statements]
        
        # Parse reaction percentages
        parsed = []
        for r in reactions:
            try:
                parsed.append(float(r.replace("%", "").replace("+", "")))
            except (ValueError, AttributeError):
                pass
                
        avg_impact = sum(parsed) / len(parsed) if parsed else 0
        
        return {
            "name": name,
            "role": node.role,
            "total_statements": len(node.statements),
            "avg_market_impact": f"{avg_impact:+.1f}%",
            "significance": node.significance_score
        }
        
    def lower_significance(self, name: str, factor: float = 0.5) -> None:
        """Lower a person's significance score (used by Data Pruner)."""
        if name in self.people:
            self.people[name].significance_score *= factor
            self.logger.info(f"[GRAPH] Lowered significance for '{name}' to {self.people[name].significance_score:.2f}")
