"""Meta-learning engine - learning from the learning process.

Analyzes patterns in the evolution system itself to improve:
- Which types of proposals historically work best
- Calibration (predicted improvement vs actual improvement)
- Evolution parameter optimization (when to diagnose, how long to A/B test)
- Overconfidence detection and correction

This is "learning about learning" - the evolution system evolving itself.
"""

import uuid
from typing import Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from collections import defaultdict
from loguru import logger
from sqlalchemy import select, insert, func, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atomicx.data.storage.models import (
    EvolutionProposal, ABTestResult, DiagnosisLog,
    MetaLearningLog, PredictionOutcome
)
from atomicx.data.storage.database import get_session


# ═══════════════════════════════════════════════════════════════════════════
# META-LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MetaLearningEngine:
    """Learn from learning — improve the improvement process.

    Analyzes historical evolution data to discover:
    - Which proposal types improve performance most
    - Whether we're overconfident or underconfident in predictions
    - Optimal evolution parameters (diagnosis frequency, A/B test duration)
    - Patterns in successful vs failed experiments

    This enables the evolution system to continuously improve its own
    decision-making process.
    """

    def __init__(self):
        self._insights_cache: dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # PROPOSAL PATTERN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    async def analyze_proposal_patterns(self) -> dict[str, dict]:
        """Analyze which proposal types historically improved performance.

        Queries:
        - evolution_proposals joined with ab_test_results
        - Group by action_type
        - Calculate success rate, avg improvement, confidence calibration

        Returns:
            Dictionary of action_type → metrics
        """
        logger.info("[META-LEARNING] Analyzing proposal patterns...")

        patterns = {}

        async with get_session() as session:
            # Get all proposals with test results
            proposals_query = await session.execute(
                select(EvolutionProposal, ABTestResult)
                .join(
                    ABTestResult,
                    EvolutionProposal.proposal_id == ABTestResult.proposal_id
                )
                .where(EvolutionProposal.status.in_(["approved", "rejected", "deployed"]))
            )

            # Group by action type
            action_stats: dict[str, dict] = defaultdict(lambda: {
                "total": 0,
                "approved": 0,
                "deltas": [],
                "confidences": [],
                "expected_improvements": [],
                "actual_improvements": []
            })

            for proposal, test_result in proposals_query:
                action_type = proposal.action_type
                stats = action_stats[action_type]

                stats["total"] += 1
                if test_result.decision == "promote":
                    stats["approved"] += 1

                stats["deltas"].append(float(test_result.delta))
                stats["confidences"].append(float(proposal.confidence))

                if proposal.expected_improvement:
                    stats["expected_improvements"].append(float(proposal.expected_improvement))
                    stats["actual_improvements"].append(float(test_result.delta))

            # Calculate metrics for each action type
            for action_type, stats in action_stats.items():
                if stats["total"] == 0:
                    continue

                success_rate = stats["approved"] / stats["total"]
                avg_delta = sum(stats["deltas"]) / len(stats["deltas"]) if stats["deltas"] else 0.0
                avg_confidence = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.5

                # Calculate calibration (how accurate are our confidence scores?)
                if stats["expected_improvements"] and stats["actual_improvements"]:
                    calibration_error = sum(
                        abs(exp - act)
                        for exp, act in zip(stats["expected_improvements"], stats["actual_improvements"])
                    ) / len(stats["expected_improvements"])
                else:
                    calibration_error = None

                patterns[action_type] = {
                    "total_proposals": stats["total"],
                    "success_rate": success_rate,
                    "avg_improvement": avg_delta,
                    "avg_confidence": avg_confidence,
                    "calibration_error": calibration_error,
                    "recommendation": "pursue" if success_rate > 0.60 and avg_delta > 0.03 else "reduce"
                }

                logger.info(
                    f"[META-LEARNING] {action_type}: "
                    f"success={success_rate:.2%}, "
                    f"avg_delta={avg_delta:+.2%}, "
                    f"calibration_error={calibration_error:.3f if calibration_error else 'N/A'}"
                )

            # Save insight
            await self._save_insight(
                insight_type="proposal_patterns",
                content=f"Analyzed {sum(s['total'] for s in action_stats.values())} proposals across {len(patterns)} action types",
                evidence=patterns,
                confidence=0.85
            )

        logger.success(f"[META-LEARNING] Proposal pattern analysis complete: {len(patterns)} action types")
        return patterns

    # ═══════════════════════════════════════════════════════════════════════
    # OVERCONFIDENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    async def analyze_overconfidence(self) -> dict[str, float]:
        """Analyze prediction confidence vs actual win rate.

        Checks if the system is:
        - Overconfident (high confidence but low win rate)
        - Underconfident (low confidence but high win rate)
        - Well-calibrated (confidence matches win rate)

        Returns:
            Calibration metrics by confidence bucket
        """
        logger.info("[META-LEARNING] Analyzing prediction confidence calibration...")

        calibration = {}

        async with get_session() as session:
            # Get predictions with outcomes
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)

            outcomes_query = await session.execute(
                select(PredictionOutcome)
                .where(
                    and_(
                        PredictionOutcome.verified_at.isnot(None),
                        PredictionOutcome.verified_at >= cutoff_time
                    )
                )
            )

            # Group by confidence buckets
            confidence_buckets = {
                "0.4-0.5": {"predictions": [], "actual_wr": []},
                "0.5-0.6": {"predictions": [], "actual_wr": []},
                "0.6-0.7": {"predictions": [], "actual_wr": []},
                "0.7-0.8": {"predictions": [], "actual_wr": []},
                "0.8-0.9": {"predictions": [], "actual_wr": []},
                "0.9-1.0": {"predictions": [], "actual_wr": []},
            }

            for outcome in outcomes_query.scalars():
                confidence = float(outcome.confidence) if outcome.confidence else 0.5
                was_correct = outcome.was_correct or False

                # Find bucket
                bucket = None
                if 0.4 <= confidence < 0.5:
                    bucket = "0.4-0.5"
                elif 0.5 <= confidence < 0.6:
                    bucket = "0.5-0.6"
                elif 0.6 <= confidence < 0.7:
                    bucket = "0.6-0.7"
                elif 0.7 <= confidence < 0.8:
                    bucket = "0.7-0.8"
                elif 0.8 <= confidence < 0.9:
                    bucket = "0.8-0.9"
                elif 0.9 <= confidence <= 1.0:
                    bucket = "0.9-1.0"

                if bucket:
                    confidence_buckets[bucket]["predictions"].append(confidence)
                    confidence_buckets[bucket]["actual_wr"].append(1.0 if was_correct else 0.0)

            # Calculate calibration for each bucket
            for bucket, data in confidence_buckets.items():
                if not data["predictions"]:
                    continue

                avg_confidence = sum(data["predictions"]) / len(data["predictions"])
                actual_wr = sum(data["actual_wr"]) / len(data["actual_wr"])
                calibration_error = abs(avg_confidence - actual_wr)

                calibration[bucket] = {
                    "sample_size": len(data["predictions"]),
                    "avg_confidence": avg_confidence,
                    "actual_win_rate": actual_wr,
                    "calibration_error": calibration_error,
                    "status": "overconfident" if avg_confidence > actual_wr + 0.05 else (
                        "underconfident" if avg_confidence < actual_wr - 0.05 else "calibrated"
                    )
                }

                logger.info(
                    f"[META-LEARNING] Confidence {bucket}: "
                    f"predicted={avg_confidence:.2%}, "
                    f"actual={actual_wr:.2%}, "
                    f"error={calibration_error:.2%}, "
                    f"n={len(data['predictions'])}"
                )

            # Save insight
            await self._save_insight(
                insight_type="confidence_calibration",
                content=f"Analyzed confidence calibration across {sum(len(d['predictions']) for d in confidence_buckets.values())} predictions",
                evidence=calibration,
                confidence=0.80
            )

        logger.success(f"[META-LEARNING] Confidence calibration analysis complete")
        return calibration

    # ═══════════════════════════════════════════════════════════════════════
    # EVOLUTION PARAMETER OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════

    async def optimize_evolution_parameters(self) -> dict[str, Any]:
        """Optimize evolution system parameters.

        Analyzes:
        - Diagnosis frequency (is 50 cycles optimal?)
        - A/B test duration (is 20 cycles enough?)
        - Rollback thresholds (is -10% too aggressive?)

        Based on historical success rates of deployments.

        Returns:
            Recommended parameter adjustments
        """
        logger.info("[META-LEARNING] Optimizing evolution parameters...")

        recommendations = {}

        async with get_session() as session:
            # ═══ Analyze A/B Test Duration ═══
            ab_tests_query = await session.execute(
                select(ABTestResult)
                .order_by(ABTestResult.decided_at.desc())
                .limit(100)
            )

            test_durations = []
            test_outcomes = []

            for test in ab_tests_query.scalars():
                test_durations.append(test.cycles_tested)
                test_outcomes.append(1 if test.decision == "promote" else 0)

            if test_durations:
                avg_duration = sum(test_durations) / len(test_durations)
                promotion_rate = sum(test_outcomes) / len(test_outcomes) if test_outcomes else 0.0

                # If promotion rate is very low, we might be testing too briefly
                if promotion_rate < 0.20:
                    recommendations["shadow_test_cycles"] = {
                        "current": avg_duration,
                        "recommended": avg_duration * 1.5,
                        "reason": f"Low promotion rate ({promotion_rate:.2%}) suggests insufficient test duration"
                    }
                # If promotion rate is very high, we might be testing too long
                elif promotion_rate > 0.60:
                    recommendations["shadow_test_cycles"] = {
                        "current": avg_duration,
                        "recommended": max(10, avg_duration * 0.75),
                        "reason": f"High promotion rate ({promotion_rate:.2%}) suggests we can reduce test duration"
                    }

            # ═══ Analyze Diagnosis Frequency ═══
            diagnosis_query = await session.execute(
                select(DiagnosisLog)
                .order_by(DiagnosisLog.diagnosed_at.desc())
                .limit(50)
            )

            diagnosis_intervals = []
            prev_time = None

            for diagnosis in diagnosis_query.scalars():
                if prev_time:
                    interval = (prev_time - diagnosis.diagnosed_at).total_seconds() / 60
                    diagnosis_intervals.append(interval)
                prev_time = diagnosis.diagnosed_at

            if diagnosis_intervals:
                avg_interval_minutes = sum(diagnosis_intervals) / len(diagnosis_intervals)
                avg_interval_cycles = avg_interval_minutes / 2  # ~2 min per cycle

                # Check if we're diagnosing too frequently (no time to see results)
                if avg_interval_cycles < 30:
                    recommendations["diagnosis_interval_cycles"] = {
                        "current": avg_interval_cycles,
                        "recommended": 50,
                        "reason": "Diagnosing too frequently - allow more time for changes to show effect"
                    }

            # Save insight
            await self._save_insight(
                insight_type="parameter_optimization",
                content=f"Analyzed {len(test_durations)} A/B tests and {len(diagnosis_intervals)} diagnoses",
                evidence=recommendations,
                confidence=0.75
            )

        logger.success(
            f"[META-LEARNING] Parameter optimization complete: "
            f"{len(recommendations)} recommendations"
        )
        return recommendations

    # ═══════════════════════════════════════════════════════════════════════
    # REGIME-SPECIFIC LEARNING PATTERNS
    # ═══════════════════════════════════════════════════════════════════════

    async def analyze_regime_learning_patterns(self) -> dict[str, dict]:
        """Analyze how well the system learns in different market regimes.

        Some regimes might be:
        - Easy to learn (quick adaptation, high win rates)
        - Hard to learn (slow adaptation, unstable performance)
        - Volatile (performance swings widely)

        Returns:
            Learning patterns by regime
        """
        logger.info("[META-LEARNING] Analyzing regime-specific learning patterns...")

        regime_patterns = {}

        async with get_session() as session:
            # Get diagnosis history
            diagnoses = await session.execute(
                select(DiagnosisLog)
                .order_by(DiagnosisLog.diagnosed_at.desc())
                .limit(50)
            )

            regime_win_rates: dict[str, list[float]] = defaultdict(list)

            for diagnosis in diagnoses.scalars():
                if not diagnosis.win_rate_by_regime:
                    continue

                for regime, win_rate in diagnosis.win_rate_by_regime.items():
                    regime_win_rates[regime].append(win_rate)

            # Calculate metrics for each regime
            for regime, win_rates in regime_win_rates.items():
                if len(win_rates) < 5:
                    continue

                avg_wr = sum(win_rates) / len(win_rates)
                volatility = (
                    sum((wr - avg_wr) ** 2 for wr in win_rates) / len(win_rates)
                ) ** 0.5

                # Check for improvement trend (are we learning?)
                recent_avg = sum(win_rates[:5]) / 5 if len(win_rates) >= 5 else avg_wr
                older_avg = sum(win_rates[-5:]) / 5 if len(win_rates) >= 10 else avg_wr
                improvement_trend = recent_avg - older_avg

                regime_patterns[regime] = {
                    "avg_win_rate": avg_wr,
                    "volatility": volatility,
                    "improvement_trend": improvement_trend,
                    "sample_size": len(win_rates),
                    "learning_status": (
                        "improving" if improvement_trend > 0.03 else
                        "declining" if improvement_trend < -0.03 else
                        "stable"
                    ),
                    "difficulty": (
                        "hard" if volatility > 0.15 or avg_wr < 0.45 else
                        "medium" if volatility > 0.10 or avg_wr < 0.55 else
                        "easy"
                    )
                }

                logger.info(
                    f"[META-LEARNING] Regime {regime}: "
                    f"wr={avg_wr:.2%}, "
                    f"volatility={volatility:.3f}, "
                    f"trend={improvement_trend:+.2%}, "
                    f"difficulty={regime_patterns[regime]['difficulty']}"
                )

            # Save insight
            await self._save_insight(
                insight_type="regime_learning_patterns",
                content=f"Analyzed learning patterns across {len(regime_patterns)} regimes",
                evidence=regime_patterns,
                confidence=0.80
            )

        logger.success(
            f"[META-LEARNING] Regime learning analysis complete: "
            f"{len(regime_patterns)} regimes analyzed"
        )
        return regime_patterns

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════

    async def _save_insight(
        self,
        insight_type: str,
        content: str,
        evidence: dict,
        confidence: float
    ) -> None:
        """Save a meta-learning insight to database."""
        async with get_session() as session:
            stmt = insert(MetaLearningLog).values(
                insight_type=insight_type,
                content=content,
                evidence=evidence,
                confidence=Decimal(str(confidence))
            )
            await session.execute(stmt)
            await session.commit()

            logger.info(f"[META-LEARNING] Saved insight: {insight_type}")

    async def get_recent_insights(self, limit: int = 10) -> list[dict]:
        """Get recent meta-learning insights for dashboard.

        Returns:
            List of insights with type, content, evidence, confidence
        """
        insights = []

        async with get_session() as session:
            result = await session.execute(
                select(MetaLearningLog)
                .order_by(MetaLearningLog.created_at.desc())
                .limit(limit)
            )

            for log in result.scalars():
                insights.append({
                    "insight_type": log.insight_type,
                    "content": log.content,
                    "evidence": log.evidence,
                    "confidence": float(log.confidence),
                    "created_at": log.created_at.isoformat()
                })

        return insights
