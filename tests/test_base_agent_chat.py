"""Tests for BaseAgent handle_user_message and _build_deterministic_chat_response."""

import json
import os
import tempfile
from typing import Any, List

import pytest

from unittest.mock import MagicMock

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.debate.chat_models import ChatMessage
from underwriting.llm.llm_client import FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConcreteAgent(BaseAgent):
    """A concrete subclass of BaseAgent for testing chat methods."""

    def evaluate(self, application: Any) -> AgentAssessment:
        matched = self.evaluate_rules(application)
        return self.build_deterministic_assessment(application, matched)

    def generate_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> AgentAssessment:
        return my_assessment

    def handle_user_message(
        self,
        application: Any,
        current_assessment: AgentAssessment,
        user_message: str,
        conversation_history: List[ChatMessage],
    ) -> ChatMessage:
        return self._build_deterministic_chat_response(
            user_message, current_assessment, "test underwriting", application
        )


class _ConcreteAgentWithRules(BaseAgent):
    """A concrete subclass that properly initializes base class with rules.

    Used for testing evidence re-evaluation where evaluate() must
    return a fresh assessment based on the application's attributes.
    """

    def __init__(self, name: str, rules_path: str, llm_client=None):
        super().__init__(name, rules_path, llm_client)

    def evaluate(self, application: Any) -> AgentAssessment:
        matched = self.evaluate_rules(application)
        return self.build_deterministic_assessment(application, matched)

    def generate_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> AgentAssessment:
        return my_assessment

    def handle_user_message(
        self,
        application: Any,
        current_assessment: AgentAssessment,
        user_message: str,
        conversation_history: List[ChatMessage],
    ) -> ChatMessage:
        return self._build_deterministic_chat_response(
            user_message, current_assessment, "test underwriting", application
        )


def _write_rules_file(rules_data: dict) -> str:
    """Write rules dict to a temporary JSON file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(rules_data, f)
    return path


# ---------------------------------------------------------------------------
# Abstract method existence
# ---------------------------------------------------------------------------


class TestHandleUserMessageAbstract:
    """Verify handle_user_message is an abstract method on BaseAgent."""

    def test_handle_user_message_is_abstract(self):
        """Cannot instantiate BaseAgent without implementing handle_user_message."""
        rules_path = _write_rules_file({"rules": []})
        try:
            with pytest.raises(TypeError):
                BaseAgent(name="Test", rules_path=rules_path)
        finally:
            os.unlink(rules_path)

    def test_concrete_agent_can_call_handle_user_message(self):
        """A concrete agent that implements handle_user_message works."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
            )

            class FakeApp:
                pass

            result = agent.handle_user_message(
                application=FakeApp(),
                current_assessment=assessment,
                user_message="What is my risk tier?",
                conversation_history=[],
            )

            assert isinstance(result, ChatMessage)
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# _build_deterministic_chat_response
# ---------------------------------------------------------------------------


class TestBuildDeterministicChatResponse:
    """Tests for the _build_deterministic_chat_response helper method."""

    def test_returns_valid_chat_message(self):
        """The helper returns a valid ChatMessage instance."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="loading",
                recommendation="ok",
            )

            result = agent._build_deterministic_chat_response(
                user_message="Test input",
                current_assessment=assessment,
                domain_description="medical underwriting",
            )

            assert isinstance(result, ChatMessage)
            assert result.sender == "TestAgent"
            assert result.message_type == "text"
            assert result.content != ""
            assert result.reasoning != ""
        finally:
            os.unlink(rules_path)

    def test_response_contains_agent_name(self):
        """Response content includes the agent's name."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="Medical Agent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="Medical Agent",
                risk_tier="standard",
                recommendation="ok",
            )

            result = agent._build_deterministic_chat_response(
                user_message="Test",
                current_assessment=assessment,
                domain_description="medical underwriting",
            )

            assert result.sender == "Medical Agent"
        finally:
            os.unlink(rules_path)

    def test_response_contains_tier(self):
        """Response content includes the current risk tier in uppercase."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="decline",
                recommendation="decline",
            )

            result = agent._build_deterministic_chat_response(
                user_message="Test",
                current_assessment=assessment,
                domain_description="financial underwriting",
            )

            assert "DECLINE" in result.content
        finally:
            os.unlink(rules_path)

    def test_response_contains_flag_count(self):
        """Response content includes the number of flags."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="loading",
                recommendation="loading",
                flags=[
                    {"rule_id": "R1", "severity": "high", "description": "Flag 1"},
                    {"rule_id": "R2", "severity": "moderate", "description": "Flag 2"},
                    {"rule_id": "R3", "severity": "low", "description": "Flag 3"},
                ],
            )

            result = agent._build_deterministic_chat_response(
                user_message="Test",
                current_assessment=assessment,
                domain_description="compliance review",
            )

            assert "3 flag(s)" in result.content
        finally:
            os.unlink(rules_path)

    def test_zero_flags(self):
        """Response correctly shows 0 flags when none exist."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
                flags=[],
            )

            result = agent._build_deterministic_chat_response(
                user_message="Test",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert "0 flag(s)" in result.content
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# Evidence handling — assessment modification
# ---------------------------------------------------------------------------


class TestEvidenceHandling:
    """Tests for evidence intent — assessment should be modified in-place."""

    def test_evidence_intent_reduces_confidence(self):
        """Evidence intent: if re-evaluation produces same tier/flags, reduce confidence by 0.1.
        
        When tier or flags change, confidence is preserved (evidence triggered
        meaningful reassessment). When nothing changes, confidence drops to
        reflect stale-application re-evaluation.
        """
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="standard",
                confidence_score=0.9,
                flags=[],
            )
            original_confidence = assessment.confidence_score

            agent._build_deterministic_chat_response(
                user_message="I just quit smoking",
                current_assessment=assessment,
                domain_description="medical underwriting",
            )

            # Empty rules → fresh assessment is "standard" (same tier, same flags)
            # → confidence reduced by 0.1
            assert assessment.confidence_score == original_confidence - 0.1
        finally:
            os.unlink(rules_path)

    def test_evidence_intent_appends_reasoning_note(self):
        """Evidence intent should append an evidence-related note to reasoning_summary.
        
        When re-evaluation produces the same result (tier/flags unchanged),
        the note reflects that evidence was noted but did not change the outcome.
        """
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
                confidence_score=1.0,
                flags=[],
                reasoning_summary="Initial assessment",
            )
            original_reasoning = assessment.reasoning_summary

            agent._build_deterministic_chat_response(
                user_message="I stopped drinking alcohol",
                current_assessment=assessment,
                domain_description="medical underwriting",
            )

            assert assessment.reasoning_summary != original_reasoning
            assert "Evidence noted" in assessment.reasoning_summary
        finally:
            os.unlink(rules_path)

    def test_evidence_intent_refreshes_risk_tier_and_flags(self):
        """Evidence intent SHOULD refresh risk_tier and flags from re-evaluation.

        When evidence is provided, the agent re-evaluates the application
        via self.evaluate() and updates the current assessment with the
        fresh risk_tier, flags, recommendation, loading_range, and
        additional_evidence_required from the re-evaluation.
        """
        rules_data = {
            "rules": [
                {
                    "rule_id": "R1",
                    "condition": "applicant.bmi < 25",
                    "severity": "none",
                    "recommendation": "standard",
                    "description": "Healthy BMI",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                },
                {
                    "rule_id": "R2",
                    "condition": "applicant.bmi >= 25",
                    "severity": "high",
                    "recommendation": "loading_applied",
                    "description": "Overweight",
                    "loading_range": [1.25, 1.75],
                    "additional_evidence": ["gp_report"],
                },
            ],
        }
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgentWithRules(name="TestAgent", rules_path=rules_path)

            # Application with BMI 22 — rules will evaluate to "standard"
            app = type("App", (), {"bmi": 22})()

            # Initial assessment has "loading" tier — should be refreshed
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="loading",
                recommendation="loading",
                confidence_score=0.85,
                flags=[{"rule_id": "R1", "severity": "high", "description": "Flag"}],
                loading_range=[1.5, 2.0],
                additional_evidence_required=["old_evidence"],
            )

            agent._build_deterministic_chat_response(
                user_message="I have new medical report",
                current_assessment=assessment,
                domain_description="test domain",
                application=app,
            )

            # risk_tier should be refreshed from re-evaluation (BMI 22 -> standard)
            assert assessment.risk_tier == "standard"
            # flags should be refreshed from re-evaluation (R1 matched, severity=none)
            assert len(assessment.flags) == 1
            assert assessment.flags[0]["rule_id"] == "R1"
            # recommendation should be refreshed
            assert assessment.recommendation == "standard"
            # confidence preserved when tier changed (loading→standard means
            # evidence triggered meaningful reassessment)
            assert assessment.confidence_score == pytest.approx(0.85)
            # reasoning_summary should note the tier change from evidence
            assert "Re-evaluated with user evidence" in assessment.reasoning_summary
            assert "loading to standard" in assessment.reasoning_summary
        finally:
            os.unlink(rules_path)

    def test_evidence_intent_minimum_confidence(self):
        """Confidence should not drop below 0.0."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
                confidence_score=0.05,
                flags=[],
            )

            agent._build_deterministic_chat_response(
                user_message="I just quit smoking",
                current_assessment=assessment,
                domain_description="medical underwriting",
            )

            assert assessment.confidence_score == 0.0
        finally:
            os.unlink(rules_path)

    def test_evidence_intent_refreshes_flags(self):
        """Evidence intent should refresh flags from the re-evaluated assessment."""
        rules_data = {
            "rules": [
                {
                    "rule_id": "R1",
                    "condition": "applicant.bmi < 18.5",
                    "severity": "moderate",
                    "recommendation": "manual_underwriting",
                    "description": "Underweight",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                },
                {
                    "rule_id": "R2",
                    "condition": "applicant.bmi >= 30",
                    "severity": "moderate",
                    "recommendation": "loading_applied",
                    "description": "Obese Class I",
                    "loading_range": [1.25, 1.75],
                    "additional_evidence": ["gp_report"],
                },
            ],
        }
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgentWithRules(name="TestAgent", rules_path=rules_path)

            # Application with BMI 33 — rules will match R2 (obese)
            app = type("App", (), {"bmi": 33})()

            # Initial assessment has standard tier and empty flags
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="standard",
                confidence_score=0.95,
                flags=[],
                loading_range=[1.0, 1.0],
            )

            agent._build_deterministic_chat_response(
                user_message="I have new medical report",
                current_assessment=assessment,
                domain_description="test domain",
                application=app,
            )

            # risk_tier should be refreshed from re-evaluation
            assert assessment.risk_tier == "loading"
            # flags should be refreshed from re-evaluation (R2 has "Obese Class I")
            assert len(assessment.flags) == 1
            assert assessment.flags[0]["rule_id"] == "R2"
            assert assessment.flags[0]["description"] == "Obese Class I"
            # recommendation should be refreshed
            assert assessment.recommendation == "loading_applied"
            # loading_range should be refreshed
            assert assessment.loading_range == [1.25, 1.75]
            # confidence preserved when tier changed (standard→loading means
            # evidence triggered meaningful reassessment)
            assert assessment.confidence_score == pytest.approx(0.95)
            # additional_evidence_required should be refreshed
            assert "gp_report" in assessment.additional_evidence_required
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# LLM-enriched chat response
# ---------------------------------------------------------------------------


class TestBuildLLMChatResponse:
    """Tests for the _build_llm_chat_response helper method.

    Verifies that when an LLM is available, the agent calls the LLM,
    parses its JSON response, and updates the assessment accordingly.
    Falls back to deterministic when LLM returns FALLBACK_MESSAGE,
    returns malformed JSON, or is None.
    """

    def test_llm_returns_response_text(self):
        """LLM returns plain text → ChatMessage.content contains that text."""
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = (
                "Specialist report noted. Risk tier adjusted to standard."
            )
            mock_llm.is_available.return_value = True

            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=mock_llm
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="loading",
                recommendation="loading",
            )

            result = agent._build_llm_chat_response(
                user_message="What is my risk tier?",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert isinstance(result, ChatMessage)
            assert "Specialist report noted" in result.content
            mock_llm.chat.assert_called_once()
        finally:
            os.unlink(rules_path)

    def test_llm_updates_risk_tier(self):
        """LLM returns JSON with risk_tier_update → assessment.risk_tier changes."""
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = json.dumps(
                {"response_text": "Done", "risk_tier_update": "standard"}
            )
            mock_llm.is_available.return_value = True

            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=mock_llm
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="loading",
                recommendation="loading",
            )

            agent._build_llm_chat_response(
                user_message="Reassess my risk",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert assessment.risk_tier == "standard"
        finally:
            os.unlink(rules_path)

    def test_llm_updates_confidence(self):
        """LLM returns JSON with confidence_update → assessment.confidence_score changes."""
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = json.dumps(
                {"response_text": "ok", "confidence_update": 0.75}
            )
            mock_llm.is_available.return_value = True

            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=mock_llm
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
                confidence_score=0.5,
            )

            agent._build_llm_chat_response(
                user_message="Reassess confidence",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert assessment.confidence_score == 0.75
        finally:
            os.unlink(rules_path)

    def test_llm_updates_reasoning(self):
        """LLM returns JSON with reasoning → reasoning_summary appends it."""
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = json.dumps(
                {
                    "response_text": "ok",
                    "reasoning": "Specialist cleared cardiovascular concern",
                }
            )
            mock_llm.is_available.return_value = True

            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=mock_llm
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
                reasoning_summary="Initial assessment",
            )

            agent._build_llm_chat_response(
                user_message="Any specialist input?",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert "Specialist cleared cardiovascular concern" in assessment.reasoning_summary
        finally:
            os.unlink(rules_path)

    def test_llm_unavailable_fallback(self):
        """LLM returns FALLBACK_MESSAGE → deterministic response (no FALLBACK text)."""
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = FALLBACK_MESSAGE
            mock_llm.is_available.return_value = True

            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=mock_llm
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="loading",
                recommendation="loading",
            )

            result = agent._build_llm_chat_response(
                user_message="What is my risk tier?",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert isinstance(result, ChatMessage)
            assert FALLBACK_MESSAGE not in result.content
        finally:
            os.unlink(rules_path)

    def test_llm_malformed_json_fallback(self):
        """LLM returns malformed JSON → deterministic response (no crash)."""
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = "not valid json {{{"
            mock_llm.is_available.return_value = True

            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=mock_llm
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
            )

            result = agent._build_llm_chat_response(
                user_message="What is my risk tier?",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert isinstance(result, ChatMessage)
            assert result.content != ""
        finally:
            os.unlink(rules_path)

    def test_no_llm_fallback(self):
        """LLM is None → deterministic response."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(
                name="TestAgent", rules_path=rules_path, llm_client=None
            )
            assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
            )

            result = agent._build_llm_chat_response(
                user_message="What is my risk tier?",
                current_assessment=assessment,
                domain_description="test domain",
            )

            assert isinstance(result, ChatMessage)
            assert result.sender == "TestAgent"
            assert result.content != ""
        finally:
            os.unlink(rules_path)
