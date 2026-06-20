"""Deterministic rule engine for underwriting evaluations.

This module provides the RuleEngine class which loads rules from JSON files
and evaluates them against application data using safe expression evaluation.
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RuleEngine:
    """Loads and evaluates deterministic underwriting rules.

    The engine reads rule definitions from JSON files and evaluates each
    rule's condition against an application object. Matched rules are used
    to build a structured assessment including risk tier, flags, and
    recommendations.
    """

    def __init__(self, rules_dir: str = "./rules/death") -> None:
        """Initialise the rule engine.

        Args:
            rules_dir: Default directory path containing rule JSON files.
        """
        self.rules_dir = rules_dir

    def load_rules(self, path: str) -> Dict[str, Any]:
        """Load deterministic rules from a JSON file.

        Args:
            path: Filesystem path to the rules JSON file.

        Returns:
            Dict with a "rules" key containing a list of rule objects.

        Raises:
            FileNotFoundError: If the rules file does not exist.
            json.JSONDecodeError: If the rules file is not valid JSON.
        """
        logger.info(f"Loading rules from {path}")
        with open(path, "r") as f:
            return json.load(f)

    def evaluate_rules(
        self, application: Any, rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate all deterministic rules against the application.

        Each rule has a 'condition' field that is evaluated as a Python
        expression against the application object. Rules that match are
        collected and returned.

        The evaluation uses a restricted globals dict (no __builtins__) to
        prevent arbitrary code execution. The application is exposed via the
        'applicant' variable inside the condition expression.

        Args:
            application: An Application Pydantic model instance or any object
                with attributes accessible by the rule conditions.
            rules: Dict with a "rules" key containing a list of rule dicts.

        Returns:
            List of matched rule dictionaries.
        """
        # Safe builtins for eval — minimal set to prevent arbitrary code execution.
        _SAFE_BUILTINS: Dict[str, Any] = {
            "getattr": getattr,
            "len": len,
            "max": max,
            "min": min,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "filter": filter,
            "map": map,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "abs": abs,
            "round": round,
            "True": True,
            "False": False,
            "None": None,
        }

        matched: List[Dict[str, Any]] = []
        for rule in rules.get("rules", []):
            condition = rule.get("condition", "False")
            try:
                if eval(
                    condition,
                    {"__builtins__": _SAFE_BUILTINS},
                    {"applicant": application},
                ):
                    matched.append(rule)
                    logger.debug(
                        f"Rule {rule.get('rule_id', '?')} matched: "
                        f"{rule.get('description', '')}"
                    )
            except Exception as e:
                logger.error(
                    f"Error evaluating rule {rule.get('rule_id', '?')}: {e}"
                )
        return matched

    def build_assessment(
        self,
        application: Any,
        matched_rules: List[Dict[str, Any]],
        agent_name: str,
    ) -> Dict[str, Any]:
        """Build an assessment dict from matched deterministic rules.

        The highest-severity matched rule determines the risk tier.
        If no rules match, the assessment defaults to 'standard'.

        Args:
            application: An Application Pydantic model instance.
            matched_rules: List of rules that matched the application.
            agent_name: Name of the agent that produced these rules.

        Returns:
            Dict representing the assessment with keys matching
            AgentAssessment model fields.
        """
        if not matched_rules:
            return {
                "agent_name": agent_name,
                "risk_tier": "standard",
                "flags": [],
                "recommendation": "No risk factors identified. Standard terms.",
                "loading_range": [1.0, 1.0],
                "additional_evidence_required": [],
                "confidence_score": 1.0,
                "reasoning_summary": "All deterministic rules passed.",
                "apra_references": [],
                "llm_used": False,
            }

        # Priority: critical > high > moderate > low > none
        highest = max(
            matched_rules,
            key=lambda r: self._get_severity_rank(r.get("severity", "none")),
        )

        flags = [
            {
                "rule_id": r["rule_id"],
                "severity": r.get("severity", "unknown"),
                "description": r.get("description", ""),
            }
            for r in matched_rules
        ]

        loading_range = highest.get("loading_range", [1.0, 1.0])

        # Collect all additional evidence from matched rules (deduplicated)
        evidence: List[str] = []
        for r in matched_rules:
            for e in r.get("additional_evidence", []):
                if e not in evidence:
                    evidence.append(e)

        apra_refs = list(
            {r.get("apra_ref", "") for r in matched_rules if r.get("apra_ref")}
        )

        return {
            "agent_name": agent_name,
            "risk_tier": self._determine_risk_tier(highest),
            "flags": flags,
            "recommendation": highest.get("recommendation", "standard"),
            "loading_range": loading_range,
            "additional_evidence_required": evidence,
            "confidence_score": 1.0,
            "reasoning_summary": (
                f"Matched {len(matched_rules)} deterministic rule(s). "
                f"Highest severity: {highest.get('severity')} "
                f"({highest['rule_id']})."
            ),
            "apra_references": apra_refs,
            "llm_used": False,
        }

    @staticmethod
    def _determine_risk_tier(highest_rule: Dict[str, Any]) -> str:
        """Map rule recommendation to standardised risk tier.

        Args:
            highest_rule: The matched rule with the highest severity.

        Returns:
            One of: "standard", "loading", "decline", "refer"
        """
        rec = highest_rule.get("recommendation", "standard")
        if rec in ("standard", "standard_or_loading"):
            return "standard"
        if "loading" in rec:
            return "loading"
        if rec in ("decline", "manual_underwriting_or_decline"):
            return "decline"
        if rec in ("refer", "manual_underwriting"):
            return "refer"
        return "standard"

    @staticmethod
    def _get_severity_rank(severity: str) -> int:
        """Return a numeric rank for a severity level.

        Higher values indicate more severe conditions.

        Args:
            severity: Severity string (e.g. "critical", "high", "moderate",
                "low", "none").

        Returns:
            Integer rank.
        """
        severity_order: Dict[str, int] = {
            "critical": 5,
            "high": 4,
            "moderate": 3,
            "low": 2,
            "none": 1,
        }
        return severity_order.get(severity, 0)
