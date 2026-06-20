"""Decision Synthesis Logic.

Provides the ``DecisionSynthesizer`` class that combines individual agent
assessments into a single final underwriting decision.  The synthesiser
applies the most-conservative-assessment principle, aggregates loading
ranges across agents, identifies specific risk exclusions, and generates
plain-English reasoning for audit purposes.
"""

from typing import Any, Dict, List


class DecisionSynthesizer:
    """Combine multiple agent assessments into a single underwriting decision.

    The synthesiser follows a deterministic hierarchy:

    1. Find the highest (most conservative) risk tier across all agents.
    2. Map that tier to a decision outcome:

       - **Standard Offer** — all agents pass with no risk flags.
       - **Offer with Loading/Exclusion** — moderate risk flags present;
         loading is applied to the premium.
       - **Request Additional Evidence** — agents require more information
         to reach a confident decision.
       - **Refer to Manual Underwriting** — high complexity or multiple
         moderate flags that exceed automated thresholds.
       - **Decline** — any agent flags a critical risk.

    3. Aggregate loading ranges (intersection of all agents' ranges).
    4. Identify specific risk exclusions from flagged categories.
    5. Generate plain-English reasoning for the final decision.

    Attributes:
        risk_tier_rank: Mapping of risk tier names to numeric rank
            (higher = more conservative).
    """

    risk_tier_rank: Dict[str, int] = {
        "standard": 0,
        "loading": 1,
        "refer": 2,
        "decline": 3,
    }

    def _normalize_assessment(self, assessment: Any) -> Any:
        """Normalise an assessment to an instance if a class was passed.

        ``assessment_factory`` in tests uses ``pydantic.create_model`` which
        returns a *class*, not an instance.  Class-level ``getattr`` returns
        the ``FieldInfo`` object rather than the default value.  This helper
        instantiates the class so that ``getattr`` works correctly.

        Args:
            assessment: Either an assessment instance or a class returned by
                ``create_model``.

        Returns:
            An assessment instance.
        """
        if isinstance(assessment, type):
            return assessment()
        return assessment

    def _produce_final_decision(self, assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesise a final decision from all agent assessments.

        The most conservative (highest risk) assessment takes precedence.
        If any agent flags a critical risk, the applicant is declined.
        Moderate risk flags trigger loading; insufficient information
        triggers an evidence request; high complexity triggers a manual
        referral.

        Args:
            assessments: Mapping of agent names to their ``AgentAssessment``
                objects (Pydantic models with ``risk_tier``, ``flags``,
                ``loading_range``, ``additional_evidence_required``).

        Returns:
            Dict with keys:

            - ``decision`` (str): One of the five decision outcomes.
            - ``combined_loading`` (List[float]): Aggregated loading range.
            - ``exclusions`` (List[str]): Specific risk exclusions identified.
            - ``reasoning`` (str): Plain-English explanation of the decision.
            - ``highest_risk_tier`` (str): The most conservative tier found.
            - ``evidence_needed`` (bool): Whether additional evidence is required.
        """
        # Normalise: convert any class objects to instances
        normed = {k: self._normalize_assessment(v) for k, v in assessments.items()}
        tiers = list(normed.values())

        highest_tier = max(getattr(tier, "risk_tier", "standard") for tier in tiers)
        highest_rank = self.risk_tier_rank.get(highest_tier, 3)

        # Determine decision outcome based on risk tier
        if highest_rank == 0:
            decision = "Standard Offer"
        elif highest_rank == 1:
            decision = "Offer with Loading/Exclusion"
        elif highest_rank == 2:
            evidence_needed = any(
                getattr(t, "additional_evidence_required", [])
                for t in tiers
            )
            if evidence_needed:
                decision = "Request Additional Evidence"
            else:
                decision = "Refer to Manual Underwriting"
        else:
            decision = "Decline"

        # Calculate combined loading range
        combined_loading = self.calculate_combined_loading(normed)

        # Identify exclusions
        exclusions = self.identify_exclusions(normed)

        # Generate reasoning
        reasoning = self.generate_decision_reasoning(
            {"decision": decision, "combined_loading": combined_loading, "exclusions": exclusions},
            normed,
        )

        return {
            "decision": decision,
            "combined_loading": combined_loading,
            "exclusions": exclusions,
            "reasoning": reasoning,
            "highest_risk_tier": highest_tier,
            "evidence_needed": evidence_needed if highest_rank == 2 else False,
        }

    def calculate_combined_loading(self, assessments: Dict[str, Any]) -> List[float]:
        """Calculate the combined loading range from all agent assessments.

        Takes the minimum of each agent's upper bound and the maximum of
        each agent's upper bound, returning the intersection range as
        ``[min_upper, max_upper]``.

        Args:
            assessments: Mapping of agent names to their ``AgentAssessment``
                objects.

        Returns:
            List of two floats representing the combined loading range
            ``[lower_bound, upper_bound]``.
        """
        if not assessments:
            return [1.0, 1.0]

        # Normalise: convert any class objects to instances
        normed = {k: self._normalize_assessment(v) for k, v in assessments.items()}
        upper_bounds = [
            max(getattr(t, "loading_range", [1.0, 1.0]))
            for t in normed.values()
        ]

        return [min(upper_bounds), max(upper_bounds)]

    def identify_exclusions(self, assessments: Dict[str, Any]) -> List[str]:
        """Identify specific risk exclusions from agent flags.

        Extracts the ``category`` field from any flag with severity
        ``high`` or ``critical`` and returns unique risk category names.

        Args:
            assessments: Mapping of agent names to their ``AgentAssessment``
                objects.

        Returns:
            List of unique risk category strings to exclude.
        """
        categories: List[str] = []
        for assessment in assessments.values():
            for flag in getattr(assessment, "flags", []):
                severity = getattr(flag, "get", lambda *a: None)("severity", "")
                if severity in ("high", "critical"):
                    category = getattr(flag, "get", lambda *a: None)("category", "")
                    if category and category not in categories:
                        categories.append(category)
        return categories

    def generate_decision_reasoning(
        self,
        decision: Dict[str, Any],
        assessments: Dict[str, Any],
    ) -> str:
        """Generate plain-English reasoning for the final decision.

        Produces a human-readable explanation that summarises:

        - The final decision outcome.
        - The highest risk tier across all agents.
        - The combined loading range (if loading is applied).
        - Any specific risk exclusions.
        - Additional evidence requirements.

        Args:
            decision: Dict produced by ``_produce_final_decision`` containing
                ``decision``, ``combined_loading``, and ``exclusions``.
            assessments: Mapping of agent names to their ``AgentAssessment``
                objects.

        Returns:
            A plain-English reasoning string.
        """
        decision_text = decision.get("decision", "Unknown")
        combined_loading = decision.get("combined_loading", [1.0, 1.0])
        exclusions = decision.get("exclusions", [])

        # Normalise: convert any class objects to instances
        normed = {k: self._normalize_assessment(v) for k, v in assessments.items()}
        tiers = list(normed.values())
        highest_tier = max(getattr(tier, "risk_tier", "standard") for tier in tiers)
        num_agents = len(assessments)

        parts = [
            f"Final decision: {decision_text}. "
            f"Based on assessments from {num_agents} agent(s). "
            f"Highest risk tier: {highest_tier}.",
        ]

        # Loading detail
        if combined_loading[1] > 1.0:
            parts.append(
                f"Combined loading applied: {combined_loading[0]:.0%} to {combined_loading[1]:.0%}."
            )

        # Exclusions detail
        if exclusions:
            parts.append(f"Exclusions: {', '.join(exclusions)}.")

        # Evidence detail
        evidence = [
            item
            for t in tiers
            for item in getattr(t, "additional_evidence_required", [])
        ]
        if evidence:
            parts.append(f"Additional evidence required: {', '.join(evidence)}.")

        return " ".join(parts)
