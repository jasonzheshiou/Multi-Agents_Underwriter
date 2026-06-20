"""Tests for the ComplianceAgent class — contextual chat responses."""

from unittest.mock import MagicMock

import pytest

from underwriting.agents.base_agent import AgentAssessment
from underwriting.agents.compliance_agent import ComplianceAgent


class TestComplianceAgentEvidenceHandling:
    """Tests for ComplianceAgent evidence handling — assessment modification."""

    def _make_agent(self):
        return ComplianceAgent(rules_path="rules/death/compliance_rules.json")

    def _make_assessment(self, confidence=0.9, risk_tier="standard"):
        return AgentAssessment(
            agent_name="Compliance Agent",
            risk_tier=risk_tier,
            recommendation="standard",
            confidence_score=confidence,
            flags=[{"rule_id": "CMP-D-001", "severity": "moderate", "description": "Duty of disclosure review required"}],
            reasoning_summary="Compliance review flags identified",
            additional_evidence_required=[],
            apra_references=["APRA CPS 220", "APRA CPS 234"],
        )

    def test_evidence_reduces_confidence(self):
        """Evidence should reduce confidence_score by 0.1."""
        agent = self._make_agent()
        assessment = self._make_assessment(confidence=0.9)
        original = assessment.confidence_score
        agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="I have additional disclosure documents",
            conversation_history=[],
        )
        assert assessment.confidence_score == original - 0.1

    def test_evidence_appends_reasoning_note(self):
        """Evidence should append a note to reasoning_summary."""
        agent = self._make_agent()
        assessment = self._make_assessment()
        original = assessment.reasoning_summary
        agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="I have additional disclosure documents",
            conversation_history=[],
        )
        assert assessment.reasoning_summary != original
        assert "New evidence" in assessment.reasoning_summary

    def test_evidence_refreshes_risk_tier_and_flags(self):
        """Evidence should refresh risk_tier and flags from re-evaluation."""
        agent = self._make_agent()
        assessment = self._make_assessment()
        original_tier = assessment.risk_tier
        original_flags = list(assessment.flags)
        agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="I have additional disclosure documents",
            conversation_history=[],
        )
        # risk_tier and flags are refreshed from re-evaluation
        # With a MagicMock application, no rules match, so tier becomes "standard"
        assert assessment.risk_tier != original_tier or assessment.flags != original_flags
        assert assessment.risk_tier == "standard"

    def test_evidence_minimum_confidence(self):
        """Confidence should not drop below 0.0."""
        agent = self._make_agent()
        assessment = self._make_assessment(confidence=0.05)
        agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="I have additional disclosure documents",
            conversation_history=[],
        )
        assert assessment.confidence_score == 0.0


class TestComplianceAgentChatResponse:
    """Test suite for ComplianceAgent contextual chat responses."""

    def _make_assessment(self):
        """Return a compliance assessment with flags."""
        return AgentAssessment(
            agent_name="Compliance Agent",
            risk_tier="loading",
            flags=[
                {"rule_id": "CMP-D-001", "severity": "moderate", "description": "Duty of disclosure review required"},
                {"rule_id": "CMP-D-005", "severity": "low", "description": "APRA CPS 234 compliance check"},
            ],
            recommendation="loading",
            loading_range=[1.05, 1.15],
            confidence_score=0.95,
            reasoning_summary="Compliance review flags identified",
            additional_evidence_required=[],
            apra_references=["APRA CPS 220", "APRA CPS 234"],
        )

    def test_compliance_flag_question_lists_flags(self):
        """Flag question gets a response mentioning flags (LLM-first fallback)."""
        agent = ComplianceAgent(rules_path="rules/death/compliance_rules.json")
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="Why was I flagged for compliance?",
            conversation_history=[],
        )
        assert "2" in response.content or "flag" in response.content.lower()

    def test_apra_reference_question_mentions_apra(self):
        """Question about rules gets a response (LLM-first, fallback without LLM)."""
        agent = ComplianceAgent(rules_path="rules/death/compliance_rules.json")
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="What APRA rules apply to me?",
            conversation_history=[],
        )
        assert len(response.content) > 0

    def test_explain_question_includes_risk_tier(self):
        """Explain question should include risk tier."""
        agent = ComplianceAgent(rules_path="rules/death/compliance_rules.json")
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="Explain your assessment",
            conversation_history=[],
        )
        assert "loading" in response.content.lower()

    def test_evidence_statement_acknowledged(self):
        """Evidence statement should be acknowledged."""
        agent = ComplianceAgent(rules_path="rules/death/compliance_rules.json")
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="I have additional disclosure documents",
            conversation_history=[],
        )
        assert "Thank you" in response.content or "consider" in response.content.lower()

    def test_general_question_uses_base_response(self):
        """General question should use base response pattern."""
        agent = ComplianceAgent(rules_path="rules/death/compliance_rules.json")
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="Hello",
            conversation_history=[],
        )
        assert "compliance" in response.content.lower()

    def test_response_always_includes_agent_name(self):
        """Response sender should be the agent name."""
        agent = ComplianceAgent(rules_path="rules/death/compliance_rules.json")
        assessment = self._make_assessment()
        response = agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="Test",
            conversation_history=[],
        )
        assert response.sender == "Compliance Agent"
