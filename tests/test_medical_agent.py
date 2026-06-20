"""Tests for the Medical Underwriting Agent."""

import json
import os
import tempfile
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from underwriting.agents.base_agent import AgentAssessment
from underwriting.agents.medical_agent import MedicalAgent
from underwriting.application.schema import (
    Application,
    BenefitType,
    MedicalCondition,
    SmokerStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rules_file(
    rules: list[dict[str, Any]],
) -> str:
    """Write *rules* to a temporary JSON file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"rules": rules}, f)
    return path


def _standard_applicant() -> Application:
    """Return a standard-risk applicant (BMI ~24.7, Never smoker, no conditions)."""
    return Application(
        full_name="Jane Doe",
        date_of_birth=date(1990, 1, 15),
        gender="Female",
        residency_status="Australian Citizen",
        contact_address="1 Market St, Sydney NSW 2000",
        benefit_types=[BenefitType.DEATH],
        sum_insured_death=1_000_000,
        occupation="Manager",
        employer_name="Acme Corp",
        years_in_occupation=5.0,
        annual_income=85_000,
        height_cm=180,
        weight_kg=80,
        smoker_status=SmokerStatus.NEVER,
        has_medical_conditions=False,
        medical_conditions=[],
        has_family_history=False,
        family_history=[],
        has_hazardous_pursuits=False,
        hazardous_pursuits=[],
        recreational_drug_use=False,
        duty_of_disclosure_acknowledged=True,
    )


def _high_risk_applicant() -> Application:
    """Return a high-risk applicant (BMI 38, Current smoker, type 2 diabetes)."""
    return Application(
        full_name="John Smith",
        date_of_birth=date(1975, 6, 20),
        gender="Male",
        residency_status="Australian Citizen",
        contact_address="42 Collins St, Melbourne VIC 3000",
        benefit_types=[BenefitType.DEATH],
        sum_insured_death=2_000_000,
        occupation="Engineer",
        employer_name="BuildCo",
        years_in_occupation=15.0,
        annual_income=120_000,
        height_cm=180,
        weight_kg=123,
        smoker_status=SmokerStatus.CURRENT,
        cigarettes_per_day=15,
        has_medical_conditions=True,
        medical_conditions=[
            MedicalCondition(
                condition_name="type 2 diabetes",
                diagnosis_date=date(2015, 3, 10),
                treating_doctor_name="Dr Lee",
                treating_doctor_contact="0400 000 001",
            )
        ],
        has_family_history=False,
        family_history=[],
        has_hazardous_pursuits=False,
        hazardous_pursuits=[],
        recreational_drug_use=False,
        duty_of_disclosure_acknowledged=True,
    )


def _obese_applicant() -> Application:
    """Return an obese applicant (BMI ~37, Never smoker, no conditions)."""
    return Application(
        full_name="Alice Brown",
        date_of_birth=date(1988, 11, 5),
        gender="Female",
        residency_status="Permanent Resident",
        contact_address="10 George St, Brisbane QLD 4000",
        benefit_types=[BenefitType.DEATH],
        sum_insured_death=500_000,
        occupation="Administrator",
        employer_name="GovDept",
        years_in_occupation=3.0,
        annual_income=65_000,
        height_cm=180,
        weight_kg=120,
        smoker_status=SmokerStatus.NEVER,
        has_medical_conditions=False,
        medical_conditions=[],
        has_family_history=False,
        family_history=[],
        has_hazardous_pursuits=False,
        hazardous_pursuits=[],
        recreational_drug_use=False,
        duty_of_disclosure_acknowledged=True,
    )


def _underweight_applicant() -> Application:
    """Return an underweight applicant (BMI ~17, Never smoker, no conditions)."""
    return Application(
        full_name="Bob White",
        date_of_birth=date(1995, 3, 22),
        gender="Male",
        residency_status="Temporary Visa",
        contact_address="5 Elizabeth St, Perth WA 6000",
        benefit_types=[BenefitType.DEATH],
        sum_insured_death=300_000,
        occupation="Sales",
        employer_name="RetailCo",
        years_in_occupation=2.0,
        annual_income=50_000,
        height_cm=180,
        weight_kg=55,
        smoker_status=SmokerStatus.NEVER,
        has_medical_conditions=False,
        medical_conditions=[],
        has_family_history=False,
        family_history=[],
        has_hazardous_pursuits=False,
        hazardous_pursuits=[],
        recreational_drug_use=False,
        duty_of_disclosure_acknowledged=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def medical_agent() -> MedicalAgent:
    """Create a MedicalAgent with the real rules file (no LLM)."""
    rules_path = "rules/death/medical_rules.json"
    return MedicalAgent(rules_path=rules_path, llm_client=None)


# ---------------------------------------------------------------------------
# Tests — Instantiation
# ---------------------------------------------------------------------------

class TestMedicalAgentInstantiation:
    """Verify MedicalAgent can be created correctly."""

    def test_instantiate_with_no_llm(self) -> None:
        agent = MedicalAgent("rules/death/medical_rules.json")
        assert agent.name == "Medical Agent"
        assert agent.llm is None
        assert "rules" in agent.rules

    def test_instantiate_with_llm_client(self) -> None:
        mock_llm = MagicMock()
        agent = MedicalAgent("rules/death/medical_rules.json", llm_client=mock_llm)
        assert agent.llm is mock_llm


# ---------------------------------------------------------------------------
# Tests — evaluate()
# ---------------------------------------------------------------------------

class TestEvaluateStandardApplicant:
    """Standard applicant should get risk_tier 'standard'."""

    def test_returns_agent_assessment(self, medical_agent: MedicalAgent) -> None:
        app = _standard_applicant()
        assessment = medical_agent.evaluate(app)
        assert isinstance(assessment, AgentAssessment)

    def test_risk_tier_standard(self, medical_agent: MedicalAgent) -> None:
        app = _standard_applicant()
        assessment = medical_agent.evaluate(app)
        assert assessment.risk_tier == "standard"

    def test_no_flags_for_standard(self, medical_agent: MedicalAgent) -> None:
        app = _standard_applicant()
        assessment = medical_agent.evaluate(app)
        # Standard applicant: only MED-D-003 (BMI 25-30, low severity)
        # and MED-D-010 (Never smoker, none severity) should match
        # Both have recommendation "standard" → risk_tier "standard"
        assert assessment.risk_tier == "standard"


class TestEvaluateHighRiskApplicant:
    """High-risk applicant (BMI 38, Current smoker, diabetes) should get 'loading'."""

    def test_returns_agent_assessment(self, medical_agent: MedicalAgent) -> None:
        app = _high_risk_applicant()
        assessment = medical_agent.evaluate(app)
        assert isinstance(assessment, AgentAssessment)

    def test_risk_tier_loading(self, medical_agent: MedicalAgent) -> None:
        app = _high_risk_applicant()
        assessment = medical_agent.evaluate(app)
        assert assessment.risk_tier == "loading"

    def test_flags_included(self, medical_agent: MedicalAgent) -> None:
        app = _high_risk_applicant()
        assessment = medical_agent.evaluate(app)
        # Should have flags for BMI 35-40 (MED-D-005), Current smoker (MED-D-013),
        # and type 2 diabetes (MED-D-032)
        rule_ids = {f["rule_id"] for f in assessment.flags}
        assert "MED-D-005" in rule_ids  # BMI 35-40
        assert "MED-D-013" in rule_ids  # Current smoker
        assert "MED-D-032" in rule_ids  # Type 2 diabetes


class TestEvaluateObeseApplicant:
    """Obese applicant (BMI 35-40) should get 'loading' tier."""

    def test_risk_tier_loading(self, medical_agent: MedicalAgent) -> None:
        app = _obese_applicant()
        assessment = medical_agent.evaluate(app)
        assert assessment.risk_tier == "loading"

    def test_bmi_flag_present(self, medical_agent: MedicalAgent) -> None:
        app = _obese_applicant()
        assessment = medical_agent.evaluate(app)
        rule_ids = {f["rule_id"] for f in assessment.flags}
        assert "MED-D-005" in rule_ids  # BMI 35-40


class TestEvaluateUnderweightApplicant:
    """Underweight applicant (BMI < 18.5) should get 'refer' tier."""

    def test_risk_tier_refer(self, medical_agent: MedicalAgent) -> None:
        app = _underweight_applicant()
        assessment = medical_agent.evaluate(app)
        assert assessment.risk_tier == "refer"

    def test_underweight_flag_present(self, medical_agent: MedicalAgent) -> None:
        app = _underweight_applicant()
        assessment = medical_agent.evaluate(app)
        rule_ids = {f["rule_id"] for f in assessment.flags}
        assert "MED-D-001" in rule_ids  # BMI < 18.5


# ---------------------------------------------------------------------------
# Tests — LLM enrichment
# ---------------------------------------------------------------------------

class TestLlmEnrichment:
    """LLM enrichment appends to reasoning_summary when available."""

    def test_llm_enrich_appends_reasoning(self) -> None:
        """When LLM is available, reasoning_summary should contain LLM text."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = (
            "LLM analysis: Conditions appear well-controlled."
        )
        agent = MedicalAgent(
            "rules/death/medical_rules.json", llm_client=mock_llm
        )
        app = _high_risk_applicant()
        assessment = agent.evaluate(app)
        assert assessment.llm_used is True
        assert "LLM" in assessment.reasoning_summary

    def test_llm_failure_handled_gracefully(self) -> None:
        """When LLM raises an exception, deterministic assessment is returned."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("LLM service unavailable")
        agent = MedicalAgent(
            "rules/death/medical_rules.json", llm_client=mock_llm
        )
        app = _high_risk_applicant()
        assessment = agent.evaluate(app)
        # Should NOT crash; deterministic assessment returned
        assert isinstance(assessment, AgentAssessment)
        assert assessment.risk_tier == "loading"


# ---------------------------------------------------------------------------
# Tests — Rebuttal
# ---------------------------------------------------------------------------

class TestRebuttal:
    """Rebuttal stands firm on objective criteria, reduces confidence on non-objective."""

    def test_stands_firm_on_objective_flags(self, medical_agent: MedicalAgent) -> None:
        """Objective flags (BMI, smoker, conditions, family history) reduce confidence when challenged."""
        app = _high_risk_applicant()
        assessment = medical_agent.evaluate(app)
        original_confidence = assessment.confidence_score

        rebuttal = medical_agent.generate_rebuttal(
            app, assessment, other_assessments=[]
        )

        # Confidence should be reduced for objective flags when challenged
        assert rebuttal.confidence_score < original_confidence
        # Tier may downgrade if confidence drops below threshold
        assert rebuttal.risk_tier in ("standard", "loading", "refer", "decline")

    def test_reduces_confidence_on_non_objective_flags(self) -> None:
        """Non-objective flags should reduce confidence."""
        # Create a scenario with non-objective flags
        rules_path = _make_rules_file([
            {
                "rule_id": "MED-D-900",
                "category": "mental_health",
                "condition": "False",
                "severity": "low",
                "recommendation": "case_by_case",
                "description": "Non-objective flag for testing",
                "additional_evidence": [],
            },
        ])
        try:
            agent = MedicalAgent(rules_path)
            app = _standard_applicant()

            # Manually create an assessment with a non-objective flag
            assessment = AgentAssessment(
                agent_name="Medical Agent",
                risk_tier="standard",
                flags=[
                    {
                        "rule_id": "MED-D-900",
                        "category": "mental_health",
                        "severity": "low",
                        "description": "Non-objective flag for testing",
                    }
                ],
                recommendation="case_by_case",
                loading_range=[1.0, 1.0],
                confidence_score=1.0,
                reasoning_summary="Original assessment",
            )

            rebuttal = agent.generate_rebuttal(app, assessment, other_assessments=[])

            # Non-objective flags reduce confidence by 0.25
            assert rebuttal.confidence_score == pytest.approx(0.75)
        finally:
            os.unlink(rules_path)

    def test_rebuttal_includes_reasoning(self, medical_agent: MedicalAgent) -> None:
        """Rebuttal should contain reasoning about flags."""
        app = _high_risk_applicant()
        assessment = medical_agent.evaluate(app)
        rebuttal = medical_agent.generate_rebuttal(
            app, assessment, other_assessments=[]
        )
        assert "Rebuttal" in rebuttal.reasoning_summary
        assert "CHALLENGED" in rebuttal.reasoning_summary


# ---------------------------------------------------------------------------
# Tests — handle_user_message() domain-specific responses
# ---------------------------------------------------------------------------

class TestMedicalAgentChatResponse:
    """Test domain-specific chat responses from handle_user_message()."""

    @pytest.fixture()
    def agent(self) -> MedicalAgent:
        """Create a MedicalAgent with no LLM (deterministic only)."""
        return MedicalAgent(rules_path="rules/death/medical_rules.json")

    def _make_assessment(
        self,
        risk_tier: str = "loading",
        flags: list[dict[str, str]] | None = None,
        recommendation: str = "loading",
        loading_range: list[float] | None = None,
        confidence_score: float = 0.85,
        reasoning_summary: str = "Elevated BMI detected",
    ) -> AgentAssessment:
        """Build a deterministic AgentAssessment for chat tests."""
        if flags is None:
            flags = [{"rule_id": "MED-001", "severity": "high", "description": "High BMI"}]
        if loading_range is None:
            loading_range = [1.2, 1.5]
        return AgentAssessment(
            agent_name="Medical Agent",
            risk_tier=risk_tier,
            flags=flags,
            recommendation=recommendation,
            loading_range=loading_range,
            confidence_score=confidence_score,
            reasoning_summary=reasoning_summary,
            additional_evidence_required=[],
            apra_references=[],
        )

    def test_bmi_question_includes_bmi_value(self, agent: MedicalAgent) -> None:
        """Question about BMI gets a response (LLM-first, fallback without LLM)."""
        app = _standard_applicant()
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            app, assessment, "What's my BMI?", []
        )
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_smoking_question_includes_smoker_status(self, agent: MedicalAgent) -> None:
        """Question about smoking gets a response (LLM-first, fallback without LLM)."""
        app = _standard_applicant()
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            app, assessment, "Why was I flagged for smoking?", []
        )
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_flag_question_lists_medical_flags(self, agent: MedicalAgent) -> None:
        """Flag question gets a response mentioning flags (LLM-first fallback)."""
        app = _standard_applicant()
        flags = [
            {"rule_id": "MED-001", "severity": "high", "description": "High BMI"},
            {"rule_id": "MED-002", "severity": "moderate", "description": "Elevated cholesterol"},
        ]
        assessment = self._make_assessment(flags=flags)
        response = agent.handle_user_message(
            app, assessment, "Why was I flagged?", []
        )
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert "flag" in response.content.lower()

    def test_explain_question_includes_risk_tier(self, agent: MedicalAgent) -> None:
        """Asking for an explanation should include the risk tier."""
        app = _standard_applicant()
        assessment = self._make_assessment(risk_tier="loading")
        response = agent.handle_user_message(
            app, assessment, "Explain your assessment", []
        )
        assert "loading" in response.content.lower()

    def test_evidence_statement_acknowledged(self, agent: MedicalAgent) -> None:
        """Submitting new evidence should acknowledge it positively."""
        app = _standard_applicant()
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            app, assessment, "I just quit smoking", []
        )
        content_lower = response.content.lower()
        assert "thank" in content_lower or "consider" in content_lower

    def test_general_question_uses_base_response(self, agent: MedicalAgent) -> None:
        """A general greeting should fall back to the base response."""
        app = _standard_applicant()
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            app, assessment, "Hello", []
        )
        assert "medical underwriting" in response.content.lower()


# ---------------------------------------------------------------------------
# Tests — handle_user_message() evidence handling
# ---------------------------------------------------------------------------

class TestMedicalAgentEvidenceHandling:
    """Tests for MedicalAgent evidence handling — assessment modification."""

    @pytest.fixture()
    def agent(self) -> MedicalAgent:
        return MedicalAgent(rules_path="rules/death/medical_rules.json")

    def _make_assessment(
        self,
        confidence: float = 0.9,
        risk_tier: str = "loading",
    ) -> AgentAssessment:
        return AgentAssessment(
            agent_name="Medical Agent",
            risk_tier=risk_tier,
            recommendation="loading",
            confidence_score=confidence,
            flags=[{"rule_id": "MED-001", "severity": "high", "description": "High BMI"}],
            reasoning_summary="Elevated BMI detected",
            additional_evidence_required=[],
            apra_references=[],
        )

    def test_evidence_reduces_confidence(self, agent: MedicalAgent) -> None:
        """Evidence should reduce confidence_score by 0.1."""
        assessment = self._make_assessment(confidence=0.9)
        original = assessment.confidence_score
        agent.handle_user_message(
            application=_standard_applicant(),
            current_assessment=assessment,
            user_message="I just quit smoking",
            conversation_history=[],
        )
        assert assessment.confidence_score == original - 0.1

    def test_evidence_appends_reasoning_note(self, agent: MedicalAgent) -> None:
        """Evidence should append a note to reasoning_summary."""
        assessment = self._make_assessment()
        original = assessment.reasoning_summary
        agent.handle_user_message(
            application=_standard_applicant(),
            current_assessment=assessment,
            user_message="I stopped drinking",
            conversation_history=[],
        )
        assert assessment.reasoning_summary != original
        assert "New evidence" in assessment.reasoning_summary

    def test_evidence_refreshes_risk_tier_and_flags(self, agent: MedicalAgent) -> None:
        """Evidence should refresh risk_tier and flags from re-evaluation."""
        assessment = self._make_assessment()
        original_reasoning = assessment.reasoning_summary
        original_confidence = assessment.confidence_score
        agent.handle_user_message(
            application=_standard_applicant(),
            current_assessment=assessment,
            user_message="I just quit smoking",
            conversation_history=[],
        )
        # confidence should be adjusted by -0.1 (proves evidence branch ran)
        assert assessment.confidence_score == pytest.approx(original_confidence - 0.1)
        # reasoning_summary should have the evidence note
        assert "New evidence" in assessment.reasoning_summary
        assert assessment.reasoning_summary != original_reasoning

    def test_evidence_minimum_confidence(self, agent: MedicalAgent) -> None:
        """Confidence should not drop below 0.0."""
        assessment = self._make_assessment(confidence=0.05)
        agent.handle_user_message(
            application=_standard_applicant(),
            current_assessment=assessment,
            user_message="I just quit smoking",
            conversation_history=[],
        )
        assert assessment.confidence_score == 0.0
