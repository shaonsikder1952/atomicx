"""Formal Logic Engine for Rule-Based Reasoning.

Prolog-like rule engine for formal logical reasoning.
Catches edge cases that neural networks might miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from loguru import logger


@dataclass
class LogicRule:
    """Formal logic rule."""
    name: str
    conditions: List[tuple]  # [(var, operator, value), ...]
    conclusion: str
    confidence: float


class LogicEngine:
    """Formal logic rule engine.

    Usage:
        engine = LogicEngine()

        # Define rules
        engine.add_rule('''
            IF price_change < -0.03 AND cvd > 1000 AND ob_imbalance < -0.2
            THEN signal = "bear_trap"
            CONFIDENCE = 0.8
        ''')

        # Evaluate rules
        result = engine.evaluate(variables)

        if result.conclusion == "bear_trap":
            logger.warning("Bear trap detected by logic engine")
    """

    def __init__(self):
        self.rules: List[LogicRule] = []
        logger.info("[LOGIC] Initialized formal logic engine")

    def add_rule(self, rule_string: str):
        """Add rule from string format.

        Format:
            IF condition1 AND condition2 AND ...
            THEN conclusion
            CONFIDENCE = value
        """
        try:
            # Parse rule (simplified parser)
            lines = rule_string.strip().split("\n")

            # Extract IF conditions
            if_line = [l for l in lines if l.strip().startswith("IF")][0]
            conditions_str = if_line.replace("IF", "").strip()

            # Parse conditions
            conditions = []
            for cond_str in conditions_str.split(" AND "):
                cond_str = cond_str.strip()

                # Parse: var operator value
                if "<" in cond_str:
                    parts = cond_str.split("<")
                    conditions.append((parts[0].strip(), "<", float(parts[1].strip())))
                elif ">" in cond_str:
                    parts = cond_str.split(">")
                    conditions.append((parts[0].strip(), ">", float(parts[1].strip())))
                elif "=" in cond_str and "=" not in cond_str.replace("==", ""):
                    parts = cond_str.split("==")
                    conditions.append((parts[0].strip(), "==", parts[1].strip().strip("'")))

            # Extract THEN conclusion
            then_line = [l for l in lines if l.strip().startswith("THEN")][0]
            conclusion = then_line.replace("THEN", "").strip()

            # Extract CONFIDENCE
            conf_line = [l for l in lines if "CONFIDENCE" in l][0]
            confidence = float(conf_line.split("=")[1].strip())

            # Create rule
            rule = LogicRule(
                name=f"rule_{len(self.rules)}",
                conditions=conditions,
                conclusion=conclusion,
                confidence=confidence,
            )

            self.rules.append(rule)
            logger.debug(f"[LOGIC] Added rule: {conclusion} (conf={confidence})")

        except Exception as e:
            logger.error(f"[LOGIC] Failed to parse rule: {e}")

    def evaluate(self, variables: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Evaluate all rules against variables.

        Args:
            variables: Variable values

        Returns:
            Result dict or None if no rules match
        """
        for rule in self.rules:
            # Check if all conditions are satisfied
            all_satisfied = True

            for var_name, operator, value in rule.conditions:
                var_value = variables.get(var_name, 0.0)

                if operator == "<":
                    if not (var_value < value):
                        all_satisfied = False
                        break
                elif operator == ">":
                    if not (var_value > value):
                        all_satisfied = False
                        break
                elif operator == "==":
                    if not (var_value == value):
                        all_satisfied = False
                        break

            # If all conditions met, return conclusion
            if all_satisfied:
                logger.info(f"[LOGIC] Rule triggered: {rule.conclusion}")
                return {
                    "rule_name": rule.name,
                    "conclusion": rule.conclusion,
                    "confidence": rule.confidence,
                    "conditions_met": rule.conditions,
                }

        # No rules matched
        return None

    def get_rules_summary(self) -> List[str]:
        """Get summary of all rules."""
        return [f"{r.name}: {r.conclusion} (conf={r.confidence})" for r in self.rules]
