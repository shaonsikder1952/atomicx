"""Trading Intelligence Wiki - Persistent Knowledge Base.

Implements Karpathy's LLM Wiki pattern for AtomicX.
Enables compounding knowledge across all 62 agents.

Core Operations:
- INGEST: Add new knowledge from prediction outcomes, evolution cycles, lessons
- QUERY: Search and synthesize from existing wiki pages
- LINT: Health-check for contradictions, staleness, orphans
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger


class TradingWiki:
    """Persistent markdown-based knowledge base for AtomicX intelligence."""

    def __init__(self, wiki_path: Path | str | None = None):
        """Initialize wiki with directory structure.

        Args:
            wiki_path: Path to wiki root directory. Defaults to project/wiki/
        """
        if wiki_path is None:
            # Default to project root / wiki
            project_root = Path(__file__).parent.parent.parent.parent
            wiki_path = project_root / "wiki"

        self.wiki_path = Path(wiki_path)
        self.logger = logger.bind(module="memory.wiki")

        # Ensure directory structure exists
        self._ensure_structure()

        self.logger.info(f"[WIKI] Initialized at {self.wiki_path}")

    def _ensure_structure(self) -> None:
        """Create wiki directory structure if it doesn't exist."""
        self.wiki_path.mkdir(exist_ok=True)

        subdirs = [
            "regimes",
            "symbols",
            "patterns",
            "strategies",
            "strategies/retired",
            "lessons",
            "variables",
        ]

        for subdir in subdirs:
            (self.wiki_path / subdir).mkdir(parents=True, exist_ok=True)

    def _append_to_log(
        self,
        operation: Literal["ingest", "query", "lint"],
        title: str,
        result: str,
        files: list[str],
        insight: str | None = None,
    ) -> None:
        """Append operation to chronological log.

        Args:
            operation: Type of operation
            title: Human-readable title
            result: Outcome summary
            files: List of pages created/modified
            insight: Key takeaway (optional)
        """
        log_path = self.wiki_path / "log.md"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        entry = f"\n## [{timestamp}] {operation} | {title}\n"
        entry += f"- **Result**: {result}\n"
        entry += f"- **Files**: {', '.join(files)}\n"
        if insight:
            entry += f"- **Insight**: {insight}\n"
        entry += "\n"

        with open(log_path, "a") as f:
            f.write(entry)

        self.logger.debug(f"[WIKI] Logged: {operation} | {title}")

    async def ingest_prediction_outcome(
        self,
        prediction_id: str,
        symbol: str,
        pattern: str | None,
        regime: str,
        direction: str,
        confidence: float,
        was_correct: bool,
        actual_return: float | None,
        lessons: str | None = None,
    ) -> None:
        """Ingest prediction outcome into wiki.

        Updates:
        - Symbol page with new performance data
        - Pattern page with win rate update
        - Regime page with regime-specific learnings
        - Lessons page if significant

        Args:
            prediction_id: Unique prediction identifier
            symbol: Trading pair
            pattern: Pattern that triggered prediction
            regime: Market regime at prediction time
            direction: Predicted direction
            confidence: Prediction confidence
            was_correct: Whether prediction was correct
            actual_return: Actual return achieved
            lessons: Optional lessons learned
        """
        self.logger.info(
            f"[WIKI-INGEST] Outcome: {symbol} {direction} @ {confidence:.1%} "
            f"→ {'✓ CORRECT' if was_correct else '✗ WRONG'} ({actual_return:+.2%})"
        )

        files_modified = []

        # 1. Update symbol page
        symbol_file = self._normalize_symbol_path(symbol)
        symbol_page = self.wiki_path / "symbols" / f"{symbol_file}.md"
        await self._update_symbol_page(
            symbol_page, symbol, pattern, regime, direction, confidence, was_correct, actual_return
        )
        files_modified.append(f"symbols/{symbol_file}.md")

        # 2. Update pattern page (if pattern exists)
        if pattern:
            pattern_file = self._normalize_pattern_name(pattern)
            pattern_page = self.wiki_path / "patterns" / f"{pattern_file}.md"
            await self._update_pattern_page(
                pattern_page, pattern, symbol, regime, was_correct, actual_return
            )
            files_modified.append(f"patterns/{pattern_file}.md")

        # 3. Update regime page
        regime_file = self._normalize_regime_name(regime)
        regime_page = self.wiki_path / "regimes" / f"{regime_file}.md"
        await self._update_regime_page(
            regime_page, regime, symbol, pattern, was_correct, actual_return
        )
        files_modified.append(f"regimes/{regime_file}.md")

        # 4. Create lesson page if significant
        if lessons or (not was_correct and abs(actual_return or 0) > 0.05):
            lesson_file = f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{prediction_id[:8]}"
            lesson_page = self.wiki_path / "lessons" / f"{lesson_file}.md"
            await self._create_lesson_page(
                lesson_page,
                prediction_id,
                symbol,
                pattern,
                regime,
                direction,
                confidence,
                was_correct,
                actual_return,
                lessons,
            )
            files_modified.append(f"lessons/{lesson_file}.md")

        # 5. Append to log
        result_emoji = "✓" if was_correct else "✗"
        self._append_to_log(
            operation="ingest",
            title=f"{symbol} {direction} prediction {result_emoji}",
            result=f"{direction} @ {confidence:.1%} → {actual_return:+.2%} ({'correct' if was_correct else 'wrong'})",
            files=files_modified,
            insight=lessons if lessons else None,
        )

    async def ingest_evolution_cycle(
        self,
        cycle_number: int,
        retired: list[dict],
        mutated: list[dict],
        spawned: list[dict],
        genome_summary: dict,
    ) -> None:
        """Ingest evolution cycle results into wiki.

        Documents strategy mutations, retirements, and spawns.

        Args:
            cycle_number: Evolution cycle counter
            retired: List of retired strategies
            mutated: List of mutated strategies
            spawned: List of newly spawned strategies
            genome_summary: Current genome state
        """
        self.logger.info(
            f"[WIKI-INGEST] Evolution cycle #{cycle_number}: "
            f"{len(retired)} retired, {len(mutated)} mutated, {len(spawned)} spawned"
        )

        files_modified = []

        # Create evolution cycle page
        cycle_file = f"evolution_cycle_{cycle_number:04d}"
        cycle_page = self.wiki_path / "lessons" / f"{cycle_file}.md"

        content = f"# Evolution Cycle #{cycle_number}\n\n"
        content += f"*Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC*\n\n"
        content += f"---\n\n"

        if retired:
            content += f"## Retired Strategies ({len(retired)})\n\n"
            for strategy in retired:
                content += f"- **{strategy['name']}**: {strategy.get('reason', 'N/A')}\n"
            content += "\n"

        if mutated:
            content += f"## Mutated Strategies ({len(mutated)})\n\n"
            for strategy in mutated:
                content += f"- **{strategy['name']}**: {strategy.get('mutation', 'N/A')}\n"
            content += "\n"

        if spawned:
            content += f"## Spawned Strategies ({len(spawned)})\n\n"
            for strategy in spawned:
                content += f"- **{strategy['name']}**: {strategy.get('description', 'N/A')}\n"
            content += "\n"

        content += f"## Genome Summary\n\n"
        content += f"```json\n{genome_summary}\n```\n"

        cycle_page.write_text(content)
        files_modified.append(f"lessons/{cycle_file}.md")

        # Append to log
        self._append_to_log(
            operation="ingest",
            title=f"Evolution Cycle #{cycle_number}",
            result=f"{len(retired)} retired, {len(mutated)} mutated, {len(spawned)} spawned",
            files=files_modified,
            insight=f"System continues to evolve. Cycle #{cycle_number} complete.",
        )

    async def search_relevant(
        self,
        query: str,
        symbol: str | None = None,
        regime: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search wiki for relevant pages.

        Args:
            query: Search query
            symbol: Filter by symbol (optional)
            regime: Filter by regime (optional)
            limit: Maximum pages to return

        Returns:
            Concatenated content of relevant pages
        """
        self.logger.debug(f"[WIKI-QUERY] Searching: {query}")

        # Build search paths based on filters
        search_paths = []

        if symbol:
            symbol_file = self._normalize_symbol_path(symbol)
            symbol_path = self.wiki_path / "symbols" / f"{symbol_file}.md"
            if symbol_path.exists():
                search_paths.append(symbol_path)

        if regime:
            regime_file = self._normalize_regime_name(regime)
            regime_path = self.wiki_path / "regimes" / f"{regime_file}.md"
            if regime_path.exists():
                search_paths.append(regime_path)

        # Add general search across lessons and patterns
        for subdir in ["lessons", "patterns", "variables"]:
            dir_path = self.wiki_path / subdir
            if dir_path.exists():
                search_paths.extend(list(dir_path.glob("*.md")))

        # Simple keyword search (can be enhanced with embeddings later)
        query_lower = query.lower()
        relevant_pages = []

        for page_path in search_paths[:limit * 3]:  # Check more than limit
            if page_path.exists():
                content = page_path.read_text()
                if query_lower in content.lower():
                    relevant_pages.append((page_path, content))

            if len(relevant_pages) >= limit:
                break

        # Concatenate results
        if not relevant_pages:
            return "No relevant wiki pages found."

        result = f"# Wiki Search Results ({len(relevant_pages)} pages)\n\n"
        for page_path, content in relevant_pages:
            relative_path = page_path.relative_to(self.wiki_path)
            result += f"## Source: `{relative_path}`\n\n{content}\n\n---\n\n"

        return result

    async def _update_symbol_page(
        self,
        page_path: Path,
        symbol: str,
        pattern: str | None,
        regime: str,
        direction: str,
        confidence: float,
        was_correct: bool,
        actual_return: float | None,
    ) -> None:
        """Update or create symbol page with new outcome."""
        if page_path.exists():
            content = page_path.read_text()
        else:
            content = f"# {symbol}\n\n*Trading Intelligence for {symbol}*\n\n---\n\n"
            content += f"## Performance History\n\n"

        # Append new outcome
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        outcome_line = (
            f"- **[{timestamp}]** {regime} regime: {direction} @ {confidence:.1%} "
            f"→ {'✓' if was_correct else '✗'} ({actual_return:+.2%})"
        )
        if pattern:
            outcome_line += f" | Pattern: {pattern}"
        outcome_line += "\n"

        # Find Performance History section and append
        if "## Performance History" in content:
            content = content.replace(
                "## Performance History\n\n", f"## Performance History\n\n{outcome_line}"
            )
        else:
            content += f"\n## Performance History\n\n{outcome_line}"

        page_path.write_text(content)

    async def _update_pattern_page(
        self,
        page_path: Path,
        pattern: str,
        symbol: str,
        regime: str,
        was_correct: bool,
        actual_return: float | None,
    ) -> None:
        """Update or create pattern page with new performance data."""
        if page_path.exists():
            content = page_path.read_text()
        else:
            content = f"# Pattern: {pattern}\n\n*Trading pattern definition and performance*\n\n---\n\n"
            content += f"## Recent Outcomes\n\n"

        # Append outcome
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        outcome_line = (
            f"- **[{timestamp}]** {symbol} in {regime}: "
            f"{'✓ WIN' if was_correct else '✗ LOSS'} ({actual_return:+.2%})\n"
        )

        if "## Recent Outcomes" in content:
            content = content.replace(
                "## Recent Outcomes\n\n", f"## Recent Outcomes\n\n{outcome_line}"
            )
        else:
            content += f"\n## Recent Outcomes\n\n{outcome_line}"

        page_path.write_text(content)

    async def _update_regime_page(
        self,
        page_path: Path,
        regime: str,
        symbol: str,
        pattern: str | None,
        was_correct: bool,
        actual_return: float | None,
    ) -> None:
        """Update or create regime page with performance in this regime."""
        if page_path.exists():
            content = page_path.read_text()
        else:
            content = f"# Regime: {regime}\n\n*Market state characteristics and optimal strategies*\n\n---\n\n"
            content += f"## Recent Performance\n\n"

        # Append outcome
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        outcome_line = f"- **[{timestamp}]** {symbol}: {'✓' if was_correct else '✗'} ({actual_return:+.2%})"
        if pattern:
            outcome_line += f" | {pattern}"
        outcome_line += "\n"

        if "## Recent Performance" in content:
            content = content.replace(
                "## Recent Performance\n\n", f"## Recent Performance\n\n{outcome_line}"
            )
        else:
            content += f"\n## Recent Performance\n\n{outcome_line}"

        page_path.write_text(content)

    async def _create_lesson_page(
        self,
        page_path: Path,
        prediction_id: str,
        symbol: str,
        pattern: str | None,
        regime: str,
        direction: str,
        confidence: float,
        was_correct: bool,
        actual_return: float | None,
        lessons: str | None,
    ) -> None:
        """Create a new lesson page for significant events."""
        content = f"# Lesson: {symbol} {direction} {'WIN' if was_correct else 'LOSS'}\n\n"
        content += f"*Prediction ID: {prediction_id}*\n"
        content += f"*Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC*\n\n"
        content += f"---\n\n"

        content += f"## Prediction Details\n\n"
        content += f"- **Symbol**: {symbol}\n"
        content += f"- **Pattern**: {pattern or 'N/A'}\n"
        content += f"- **Regime**: {regime}\n"
        content += f"- **Direction**: {direction}\n"
        content += f"- **Confidence**: {confidence:.1%}\n"
        content += f"- **Was Correct**: {'Yes ✓' if was_correct else 'No ✗'}\n"
        content += f"- **Actual Return**: {actual_return:+.2%}\n\n"

        if lessons:
            content += f"## Lessons Learned\n\n{lessons}\n\n"
        else:
            content += f"## Auto-Generated Insight\n\n"
            if not was_correct:
                content += f"This prediction failed. Review why {pattern or 'this pattern'} "
                content += f"did not work in {regime} regime for {symbol}.\n\n"

        content += f"## Related Pages\n\n"
        content += f"- [[symbols/{self._normalize_symbol_path(symbol)}]]\n"
        if pattern:
            content += f"- [[patterns/{self._normalize_pattern_name(pattern)}]]\n"
        content += f"- [[regimes/{self._normalize_regime_name(regime)}]]\n"

        page_path.write_text(content)

    def _normalize_symbol_path(self, symbol: str) -> str:
        """Convert symbol to valid filename (e.g., BTC/USDT → BTC_USDT)."""
        return symbol.replace("/", "_").replace(":", "_")

    def _normalize_pattern_name(self, pattern: str) -> str:
        """Convert pattern name to valid filename."""
        # Remove special characters, replace spaces with underscores
        normalized = re.sub(r"[^a-zA-Z0-9_\s]", "", pattern)
        normalized = normalized.replace(" ", "_").lower()
        return normalized

    def _normalize_regime_name(self, regime: str) -> str:
        """Convert regime name to valid filename."""
        return regime.replace(" ", "_").lower()

    async def get_variable_insights(
        self, variable_names: list[str], regime: str | None = None
    ) -> str:
        """Retrieve accumulated insights about specific variables.

        Args:
            variable_names: List of variable names to query
            regime: Optional regime filter

        Returns:
            Concatenated insights from variable pages
        """
        insights = []

        for var_name in variable_names:
            var_file = self._normalize_pattern_name(var_name)
            var_path = self.wiki_path / "variables" / f"{var_file}.md"

            if var_path.exists():
                content = var_path.read_text()
                insights.append(f"## {var_name}\n\n{content}")

        if not insights:
            return "No variable insights found in wiki yet."

        return "\n\n---\n\n".join(insights)


# Global singleton instance
_wiki_instance: TradingWiki | None = None


def get_trading_wiki() -> TradingWiki:
    """Get the global trading wiki instance."""
    global _wiki_instance
    if _wiki_instance is None:
        _wiki_instance = TradingWiki()
    return _wiki_instance
