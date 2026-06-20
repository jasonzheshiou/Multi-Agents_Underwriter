"""Tests for the DecisionSynthesizer class."""

import pytest
from pydantic import Field, create_model


@pytest.fixture()
def assessment_factory():
    """Return a factory function that creates assessment-like Pydantic models.

    Example::

        assess = assessment_factory(
            risk_tier="standard",
            flags=[{"category": "heart_disease", "severity": "high"}],
        )
    """
    def _factory(
        risk_tier: str = "standard",
        flags: list | None = None,
        loading_range: list | None = None,
        additional_evidence_required: list | None = None,
        recommendation: str = "Approve",
        confidence_score: float = 0.95,
        reasoning_summary: str = "Test reasoning",
    ):
        return create_model(
            "AgentAssessment",
            agent_name=(str, Field(default="test_agent")),
            risk_tier=(str, Field(default=risk_tier)),
            flags=(list, Field(default=flags or [])),
            recommendation=(str, Field(default=recommendation)),
            loading_range=(list, Field(default=loading_range or [1.0, 1.0])),
            additional_evidence_required=(
                list, Field(default=additional_evidence_required or [])
            ),
            confidence_score=(float, Field(default=confidence_score)),
            reasoning_summary=(str, Field(default=reasoning_summary)),
            apra_references=(list, Field(default=[])),
            llm_used=(bool, Field(default=False)),
        )

    return _factory


# ------------------------------------------------------------------
# Helper: determine expected decision from risk tiers
# ------------------------------------------------------------------

def _expected_decision(tiers: list[str], has_evidence: bool) -> str:
    """Return the expected decision outcome given a list of risk tiers."""
    tier_rank = {"standard": 0, "loading": 1, "refer": 2, "decline": 3}
    highest_rank = max(tier_rank.get(t, 0) for t in tiers)

    if highest_rank == 0:
        return "Standard Offer"
    if highest_rank == 1:
        return "Offer with Loading/Exclusion"
    if highest_rank == 2:
        return "Request Additional Evidence" if has_evidence else "Refer to Manual Underwriting"
    return "Decline"


# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------

class TestDecisionSynthesizer:
    """Test suite for DecisionSynthesizer."""

    def test_instantiation(self):
        """Test that DecisionSynthesizer can be instantiated."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assert synthesizer is not None
        assert synthesizer.risk_tier_rank == {
            "standard": 0,
            "loading": 1,
            "refer": 2,
            "decline": 3,
        }

    def test_standard_offer_when_all_agents_pass(self, assessment_factory):
        """Standard Offer when all agents pass with standard risk tier."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(risk_tier="standard"),
            "Financial Agent": assessment_factory(risk_tier="standard"),
            "Compliance Agent": assessment_factory(risk_tier="standard"),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert result["decision"] == "Standard Offer"
        assert result["highest_risk_tier"] == "standard"
        assert result["evidence_needed"] is False

    def test_loading_applied_when_moderate_risk_flags_present(self, assessment_factory):
        """Loading applied when moderate risk flags are present."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.05, 1.10],
                flags=[
                    {
                        "rule_id": "MED-001",
                        "severity": "moderate",
                        "category": "smoker_status",
                        "description": "Former smoker",
                    },
                ],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.10, 1.20],
                flags=[
                    {
                        "rule_id": "FIN-002",
                        "severity": "moderate",
                        "category": "income_multiple",
                        "description": "High sum-insured-to-income ratio",
                    },
                ],
            ),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert result["decision"] == "Offer with Loading/Exclusion"
        assert result["highest_risk_tier"] == "loading"
        assert result["combined_loading"] == [1.1, 1.2]
        assert "loading" in result["reasoning"].lower()

    def test_exclusions_identified_for_specific_risks(self, assessment_factory):
        """Exclusions identified when high/critical risk categories are flagged."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.10, 1.20],
                flags=[
                    {
                        "rule_id": "MED-010",
                        "severity": "high",
                        "category": "heart_disease",
                        "description": "Pre-existing heart condition",
                    },
                ],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.05, 1.10],
                flags=[
                    {
                        "rule_id": "FIN-005",
                        "severity": "high",
                        "category": "cancer",
                        "description": "Cancer history flagged by financial review",
                    },
                ],
            ),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert result["decision"] == "Offer with Loading/Exclusion"
        assert "heart_disease" in result["exclusions"]
        assert "cancer" in result["exclusions"]
        assert "exclusion" in result["reasoning"].lower()

    def test_additional_evidence_requested_when_insufficient_info(self, assessment_factory):
        """Request Additional Evidence when agents need more information."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.05, 1.10],
                flags=[],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="refer",
                loading_range=[1.10, 1.20],
                additional_evidence_required=["Income verification letter"],
                flags=[
                    {
                        "rule_id": "FIN-003",
                        "severity": "moderate",
                        "category": "income_verification",
                        "description": "Income verification required",
                    },
                ],
            ),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert result["decision"] == "Request Additional Evidence"
        assert result["highest_risk_tier"] == "refer"
        assert result["evidence_needed"] is True
        assert "Income verification letter" in result["reasoning"]

    def test_manual_referral_for_high_complexity(self, assessment_factory):
        """Refer to Manual Underwriting for high complexity (no evidence needed)."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.10, 1.20],
                flags=[
                    {
                        "rule_id": "MED-015",
                        "severity": "moderate",
                        "category": "complex_medical",
                        "description": "Complex medical profile",
                    },
                ],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="refer",
                loading_range=[1.15, 1.25],
                flags=[
                    {
                        "rule_id": "FIN-008",
                        "severity": "moderate",
                        "category": "complex_financial",
                        "description": "Complex financial arrangement",
                    },
                ],
            ),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert result["decision"] == "Refer to Manual Underwriting"
        assert result["highest_risk_tier"] == "refer"
        assert result["evidence_needed"] is False
        assert "manual" in result["reasoning"].lower()

    def test_decline_with_plain_english_reasoning(self, assessment_factory):
        """Decline when critical risk is flagged, with plain-English reasoning."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="decline",
                loading_range=[1.0, 1.0],
                flags=[
                    {
                        "rule_id": "MED-099",
                        "severity": "critical",
                        "category": "terminal_illness",
                        "description": "Terminal illness diagnosed within past 12 months",
                    },
                ],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="decline",
                loading_range=[1.0, 1.0],
                flags=[
                    {
                        "rule_id": "FIN-099",
                        "severity": "critical",
                        "category": "fraud_suspected",
                        "description": "Fraud suspected in application",
                    },
                ],
            ),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert result["decision"] == "Decline"
        assert result["highest_risk_tier"] == "decline"
        # Reasoning should be plain English, not just a tier name
        reasoning = result["reasoning"]
        assert isinstance(reasoning, str)
        assert len(reasoning) > 20
        assert "decline" in reasoning.lower()

    def test_combined_loading_calculation_correct(self, assessment_factory):
        """Combined loading range is calculated correctly from multiple agents."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.05, 1.10],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.10, 1.20],
            ),
        }

        combined = synthesizer.calculate_combined_loading(assessments)

        # [min of upper bounds, max of upper bounds]
        assert combined == [1.1, 1.2]

    def test_excluded_risk_categories_identified(self, assessment_factory):
        """Specific risk categories are identified from high/critical flags."""
        from underwriting.agents.decision_synthesis import DecisionSynthesizer

        synthesizer = DecisionSynthesizer()
        assessments = {
            "Medical Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.10, 1.20],
                flags=[
                    {
                        "rule_id": "MED-010",
                        "severity": "high",
                        "category": "heart_disease",
                        "description": "Heart disease",
                    },
                ],
            ),
            "Financial Agent": assessment_factory(
                risk_tier="loading",
                loading_range=[1.05, 1.10],
                flags=[
                    {
                        "rule_id": "FIN-005",
                        "severity": "high",
                        "category": "cancer",
                        "description": "Cancer history",
                    },
                ],
            ),
        }

        result = synthesizer._produce_final_decision(assessments)

        assert "heart_disease" in result["exclusions"]
        assert "cancer" in result["exclusions"]
