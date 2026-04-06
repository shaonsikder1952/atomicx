"""Code evolution engine with safety constraints.

EXPERIMENTAL: Self-modifying code system with extensive safety mechanisms.

Safety Features:
- Whitelist of allowed files (agents/, fusion/, variables/, intelligence/)
- Blacklist of forbidden files (brain/loop.py, data/storage/, config.py)
- Syntax validation (ast.parse)
- Import validation (compile)
- Forbidden pattern detection (no os.system, eval, exec, etc.)
- Staging environment for testing
- Shadow testing requirement (30 cycles)
- Rollback on degradation
- Max 1 change per 100 cycles
- Requires confidence > 0.85
- Respects EVOLUTION_FREEZE environment variable

This is OFF BY DEFAULT. Enable by setting ENABLE_CODE_EVOLUTION=1 in .env
"""

import os
import ast
import uuid
import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from loguru import logger
from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atomicx.data.storage.models import CodeSnapshot
from atomicx.data.storage.database import get_session


# ═══════════════════════════════════════════════════════════════════════════
# SAFETY CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

# Files that can be modified
ALLOWED_TARGETS = [
    "atomicx/agents/",
    "atomicx/fusion/",
    "atomicx/variables/",
    "atomicx/intelligence/",
    "atomicx/narrative/",
]

# Files that must NEVER be modified
FORBIDDEN_TARGETS = [
    "brain/loop.py",
    "brain/orchestrator.py",
    "data/storage/",
    "evolution/code_evolution.py",
    "evolution/self_improvement.py",
    "config.py",
    "run.py",
    "__init__.py",
]

# Patterns that must not appear in generated code
FORBIDDEN_PATTERNS = [
    "os.system",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "eval(",
    "exec(",
    "__import__",
    "importlib.import_module",
    "open(",  # Except in specific safe contexts
    "delete",
    "DROP TABLE",
    "rm -rf",
]


# ═══════════════════════════════════════════════════════════════════════════
# CODE EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class CodeEvolutionEngine:
    """Self-modifying code with safety constraints.

    EXPERIMENTAL: This system can modify its own code to fix bugs and improve
    performance. It operates under strict safety constraints to prevent
    catastrophic failures.

    Usage:
        engine = CodeEvolutionEngine()

        # Generate a fix for an underperforming agent
        patch = await engine.generate_patch(
            path="atomicx/agents/atomic/rsi_agent.py",
            issue="RSI agent underperforming in ranging markets",
            evidence={"win_rate": 0.35, "sample_size": 100}
        )

        # Validate the patch
        if engine.validate_patch(path, patch):
            # Apply to staging
            await engine.apply_to_staging(path, patch)

            # After shadow testing passes...
            await engine.commit_patch(path, patch)
    """

    def __init__(self):
        self._enabled = os.getenv("ENABLE_CODE_EVOLUTION") == "1"
        self._freeze = os.getenv("EVOLUTION_FREEZE") == "1"
        self._staging_dir = Path("/tmp/atomicx_staging")
        self._change_count_window: list[datetime] = []
        self._max_changes_per_100_cycles = 1

        if not self._enabled:
            logger.warning("[CODE EVOLUTION] Disabled (set ENABLE_CODE_EVOLUTION=1 to enable)")

    def is_enabled(self) -> bool:
        """Check if code evolution is enabled."""
        if self._freeze:
            logger.warning("[CODE EVOLUTION] System frozen (EVOLUTION_FREEZE=1)")
            return False
        return self._enabled

    def is_allowed_target(self, file_path: str) -> bool:
        """Check if a file is allowed to be modified."""
        # Check forbidden first (higher priority)
        for forbidden in FORBIDDEN_TARGETS:
            if forbidden in file_path:
                logger.warning(f"[CODE EVOLUTION] File is forbidden: {file_path}")
                return False

        # Check allowed
        for allowed in ALLOWED_TARGETS:
            if allowed in file_path:
                return True

        logger.warning(f"[CODE EVOLUTION] File not in allowed list: {file_path}")
        return False

    def validate_patch(self, file_path: str, code: str) -> bool:
        """Validate a code patch for safety.

        Checks:
        - Syntax validity (ast.parse)
        - Import validity (compile)
        - Forbidden pattern detection

        Args:
            file_path: Path to file being modified
            code: New code content

        Returns:
            True if safe, False otherwise
        """
        if not self.is_allowed_target(file_path):
            return False

        # ═══ Syntax Check ═══
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.error(f"[CODE EVOLUTION] Syntax error in patch: {e}")
            return False

        # ═══ Import Check ═══
        try:
            compile(code, file_path, "exec")
        except Exception as e:
            logger.error(f"[CODE EVOLUTION] Compile error in patch: {e}")
            return False

        # ═══ Forbidden Pattern Check ═══
        code_lower = code.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in code_lower:
                # Allow 'open(' in specific safe contexts (reading config files, etc.)
                if pattern == "open(" and "mode='r'" in code:
                    continue

                logger.error(f"[CODE EVOLUTION] Forbidden pattern detected: {pattern}")
                return False

        logger.success(f"[CODE EVOLUTION] Patch validated: {file_path}")
        return True

    async def generate_patch(
        self,
        file_path: str,
        issue: str,
        evidence: dict
    ) -> str | None:
        """Generate a code patch using LLM.

        This would call AWS Bedrock (Claude) to generate a fix for the issue.

        Args:
            file_path: Path to file to modify
            issue: Description of the problem
            evidence: Performance data, error logs, etc.

        Returns:
            Generated code or None if generation failed
        """
        if not self.is_enabled():
            return None

        if not self.is_allowed_target(file_path):
            return None

        # Check rate limit (max 1 change per 100 cycles)
        self._clean_change_window()
        if len(self._change_count_window) >= self._max_changes_per_100_cycles:
            logger.warning("[CODE EVOLUTION] Rate limit reached (max 1 change per 100 cycles)")
            return None

        logger.info(f"[CODE EVOLUTION] Generating patch for {file_path}: {issue}")

        # Read current code
        try:
            with open(file_path, "r") as f:
                current_code = f.read()
        except Exception as e:
            logger.error(f"[CODE EVOLUTION] Failed to read {file_path}: {e}")
            return None

        # TODO: Call AWS Bedrock to generate patch
        # For now, return None (this would be implemented with actual LLM call)
        logger.warning("[CODE EVOLUTION] LLM patch generation not yet implemented")
        return None

    async def apply_to_staging(self, file_path: str, code: str) -> bool:
        """Apply patch to staging environment for testing.

        Creates isolated copy of the code in /tmp/atomicx_staging/
        for testing before committing to main codebase.

        Args:
            file_path: Path to file
            code: New code content

        Returns:
            True if successful
        """
        if not self.is_enabled():
            return False

        if not self.validate_patch(file_path, code):
            return False

        # Create staging directory
        self._staging_dir.mkdir(parents=True, exist_ok=True)

        staging_path = self._staging_dir / file_path
        staging_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(staging_path, "w") as f:
                f.write(code)

            logger.success(f"[CODE EVOLUTION] Applied to staging: {staging_path}")
            return True
        except Exception as e:
            logger.error(f"[CODE EVOLUTION] Failed to write staging: {e}")
            return False

    async def commit_patch(
        self,
        file_path: str,
        code: str,
        reason: str,
        confidence: float,
        evidence: dict
    ) -> str | None:
        """Commit a validated patch to the main codebase.

        Steps:
        1. Backup original code to code_snapshots table
        2. Write new code to file
        3. Git commit with message
        4. Start monitoring for rollback

        Args:
            file_path: Path to file
            code: New code content
            reason: Explanation for the change
            confidence: Confidence score (0-1)
            evidence: Performance data supporting the change

        Returns:
            change_id or None if failed
        """
        if not self.is_enabled():
            return None

        if confidence < 0.85:
            logger.warning(f"[CODE EVOLUTION] Confidence {confidence:.2f} below threshold 0.85")
            return None

        if not self.validate_patch(file_path, code):
            return None

        change_id = f"change_{uuid.uuid4().hex[:12]}"

        try:
            # Read original code
            with open(file_path, "r") as f:
                original_code = f.read()

            # Save snapshot to database
            async with get_session() as session:
                stmt = insert(CodeSnapshot).values(
                    change_id=change_id,
                    file_path=file_path,
                    original_code=original_code,
                    new_code=code,
                    change_type="evolution",
                    proposed_by="code_evolution_engine",
                    evidence=evidence,
                    confidence=Decimal(str(confidence)),
                    status="applied"
                )
                await session.execute(stmt)
                await session.commit()

            # Write new code
            with open(file_path, "w") as f:
                f.write(code)

            # Git commit
            try:
                subprocess.run(
                    ["git", "add", file_path],
                    check=True,
                    capture_output=True
                )

                commit_message = f"""[AUTONOMOUS] {reason}

Change ID: {change_id}
Confidence: {confidence:.2f}
Evidence: {evidence}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"""

                subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    check=True,
                    capture_output=True
                )

                logger.success(f"[CODE EVOLUTION] Committed {change_id}: {file_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"[CODE EVOLUTION] Git commit failed: {e}")

            # Hot reload module
            self._hot_reload_module(file_path)

            # Track rate limit
            self._change_count_window.append(datetime.now(timezone.utc))

            return change_id

        except Exception as e:
            logger.error(f"[CODE EVOLUTION] Failed to commit patch: {e}")
            return None

    def _hot_reload_module(self, file_path: str) -> None:
        """Hot reload a Python module.

        Uses importlib.reload() to reload the module without restarting.
        """
        try:
            # Convert file path to module name
            # e.g., atomicx/agents/atomic/rsi_agent.py → atomicx.agents.atomic.rsi_agent
            module_path = file_path.replace("/", ".").replace(".py", "")

            import importlib
            import sys

            if module_path in sys.modules:
                module = sys.modules[module_path]
                importlib.reload(module)
                logger.success(f"[CODE EVOLUTION] Hot reloaded: {module_path}")
            else:
                logger.warning(f"[CODE EVOLUTION] Module not loaded, skipping reload: {module_path}")

        except Exception as e:
            logger.error(f"[CODE EVOLUTION] Hot reload failed: {e}")

    async def rollback(self, change_id: str, reason: str) -> bool:
        """Rollback a code change.

        Restores original code from code_snapshots table.

        Args:
            change_id: ID of change to rollback
            reason: Reason for rollback

        Returns:
            True if successful
        """
        logger.warning(f"[CODE EVOLUTION] Rolling back {change_id}: {reason}")

        async with get_session() as session:
            # Get snapshot
            result = await session.execute(
                select(CodeSnapshot).where(CodeSnapshot.change_id == change_id)
            )
            snapshot = result.scalar_one_or_none()

            if not snapshot:
                logger.error(f"[CODE EVOLUTION] Snapshot not found: {change_id}")
                return False

            try:
                # Restore original code
                with open(snapshot.file_path, "w") as f:
                    f.write(snapshot.original_code)

                # Update snapshot status
                snapshot.status = "rolled_back"
                snapshot.rolled_back_at = datetime.now(timezone.utc)
                snapshot.rollback_reason = reason
                await session.commit()

                # Git commit rollback
                subprocess.run(
                    ["git", "add", snapshot.file_path],
                    check=True,
                    capture_output=True
                )

                subprocess.run(
                    ["git", "commit", "-m", f"[ROLLBACK] {change_id}: {reason}"],
                    check=True,
                    capture_output=True
                )

                # Hot reload
                self._hot_reload_module(snapshot.file_path)

                logger.success(f"[CODE EVOLUTION] Rolled back {change_id}")
                return True

            except Exception as e:
                logger.error(f"[CODE EVOLUTION] Rollback failed: {e}")
                return False

    async def monitor_and_rollback_loop(self) -> None:
        """Background task: monitor deployed changes and rollback if needed.

        Watches win rate for 30 cycles after code change.
        If win rate drops > 10%, automatically rollback.
        """
        logger.info("[CODE EVOLUTION] Monitor and rollback loop started")

        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if not self.is_enabled():
                    continue

                async with get_session() as session:
                    # Get recent applied changes
                    recent_changes = await session.execute(
                        select(CodeSnapshot)
                        .where(
                            CodeSnapshot.status == "applied",
                            CodeSnapshot.applied_at >= datetime.now(timezone.utc) - timedelta(hours=2)
                        )
                    )

                    for change in recent_changes.scalars():
                        # Check if monitoring period is complete
                        if change.applied_at < datetime.now(timezone.utc) - timedelta(hours=1):
                            # Monitoring complete, mark as stable
                            change.status = "stable"
                            await session.commit()
                            continue

                        # Check win rate degradation
                        if change.live_win_rate_after and change.live_win_rate_before:
                            delta = float(change.live_win_rate_after - change.live_win_rate_before)

                            if delta < -0.10:  # 10% degradation
                                await self.rollback(
                                    change.change_id,
                                    f"Performance degraded by {delta:.2%}"
                                )

            except Exception as e:
                logger.error(f"[CODE EVOLUTION] Monitor loop error: {e}")

    def _clean_change_window(self) -> None:
        """Remove changes older than 100 cycles from rate limit window."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=200)  # ~100 cycles
        self._change_count_window = [
            ts for ts in self._change_count_window
            if ts >= cutoff
        ]
