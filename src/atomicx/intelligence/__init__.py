"""Autonomous Web Intelligence Package.

Enables the system to browse the web, discover breaking stories,
identify key people, and build a living knowledge graph.
"""

from atomicx.intelligence.scanner import NewsScanner
from atomicx.intelligence.browser_agent import BrowserAgent
from atomicx.intelligence.knowledge_graph import KnowledgeGraph

__all__ = ["NewsScanner", "BrowserAgent", "KnowledgeGraph"]
