"""Tests for the base agent class."""

import json
import os
import tempfile
from typing import Any, List
from unittest import mock

import pytest
from pydantic import ValidationError

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.debate.chat_models import ChatMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConcreteAgent(BaseAgent):
    """A concrete subclass of BaseAgent for testing purposes."""

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
            user_message, current_assessment, "test underwriting"
        )


def _write_rules_file(rules_data: dict) -> str:
    """Write rules dict to a temporary JSON file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(rules_data, f)
    return path


# ---------------------------------------------------------------------------
# AgentAssessment
# ---------------------------------------------------------------------------


class TestAgentAssessment:
    """Tests for the AgentAssessment Pydantic model."""

    def test_create_minimal(self):
        assessment = AgentAssessment(agent_name="Test", risk_tier="standard", recommendation="ok")
        assert assessment.agent_name == "Test"
        assert assessment.risk_tier == "standard"
        assert assessment.flags == []
        assert assessment.loading_range == [1.0, 1.0]
        assert assessment.additional_evidence_required == []
        assert assessment.confidence_score == 1.0
        assert assessment.reasoning_summary == ""
        assert assessment.apra_references == []
        assert assessment.llm_used is False
        assert assessment.timestamp != ""

    def test_create_full(self):
        assessment = AgentAssessment(
            agent_name="Test",
            risk_tier="loading",
            flags=[{"rule_id": "R1", "severity": "high", "description": "Test flag"}],
            recommendation="Loading",
            loading_range=[1.25, 1.5],
            additional_evidence_required=["Medical report"],
            confidence_score=0.8,
            reasoning_summary="Some reasoning",
            apra_references=["APRA-CFI-2023-001"],
            llm_used=True,
        )
        assert assessment.agent_name == "Test"
        assert assessment.risk_tier == "loading"
        assert len(assessment.flags) == 1
        assert assessment.loading_range == [1.25, 1.5]
        assert assessment.additional_evidence_required == ["Medical report"]
        assert assessment.confidence_score == 0.8
        assert assessment.reasoning_summary == "Some reasoning"
        assert assessment.apra_references == ["APRA-CFI-2023-001"]
        assert assessment.llm_used is True

    def test_confidence_score_validation_below(self):
        with pytest.raises(ValidationError):
            AgentAssessment(
                agent_name="Test",
                risk_tier="standard",
                recommendation="ok",
                confidence_score=-0.1,
            )

    def test_confidence_score_validation_above(self):
        with pytest.raises(ValidationError):
            AgentAssessment(
                agent_name="Test",
                risk_tier="standard",
                recommendation="ok",
                confidence_score=1.1,
            )

    def test_confidence_score_boundary_zero(self):
        assessment = AgentAssessment(
            agent_name="Test",
            risk_tier="standard",
            recommendation="ok",
            confidence_score=0.0,
        )
        assert assessment.confidence_score == 0.0

    def test_confidence_score_boundary_one(self):
        assessment = AgentAssessment(
            agent_name="Test",
            risk_tier="standard",
            recommendation="ok",
            confidence_score=1.0,
        )
        assert assessment.confidence_score == 1.0

    def test_timestamp_is_iso_format(self):
        assessment = AgentAssessment(agent_name="Test", risk_tier="standard", recommendation="ok")
        # Should be a valid ISO format string
        from datetime import datetime
        datetime.fromisoformat(assessment.timestamp)


# ---------------------------------------------------------------------------
# BaseAgent — Initialisation
# ---------------------------------------------------------------------------


class TestBaseAgentInit:
    """Tests for BaseAgent initialisation."""

    def test_init_loads_rules(self):
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            assert agent.name == "TestAgent"
            assert agent.rules == {"rules": []}
            assert agent.llm is None
        finally:
            os.unlink(rules_path)

    def test_init_with_llm_client(self):
        rules_path = _write_rules_file({"rules": []})
        try:
            mock_client = mock.MagicMock()
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path, llm_client=mock_client)
            assert agent.llm is mock_client
        finally:
            os.unlink(rules_path)

    def test_init_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _ConcreteAgent(name="TestAgent", rules_path="/nonexistent/path/rules.json")

    def test_init_invalid_json(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("{invalid json}")
            with pytest.raises(json.JSONDecodeError):
                _ConcreteAgent(name="TestAgent", rules_path=path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# load_rules
# ---------------------------------------------------------------------------


class TestLoadRules:
    """Tests for the load_rules method."""

    def test_load_valid_rules(self):
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "low", "description": "Desc1"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            loaded = agent.load_rules(rules_path)
            assert loaded == rules_data
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# evaluate_rules
# ---------------------------------------------------------------------------


class TestEvaluateRules:
    """Tests for the evaluate_rules method."""

    def test_no_matched_rules(self):
        rules_path = _write_rules_file({"rules": [
            {"rule_id": "R1",
             "condition": "False",
             "severity": "low",
             "description": "Never matches"},
        ]})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            # A simple mock object with no attributes
            class FakeApp:
                pass
            result = agent.evaluate_rules(FakeApp())
            assert result == []
        finally:
            os.unlink(rules_path)

    def test_all_rules_match(self):
        rules_data = {"rules": [
            {"rule_id": "R1",
             "condition": "True",
             "severity": "low",
             "description": "Always matches"},
            {"rule_id": "R2",
             "condition": "True",
             "severity": "high",
             "description": "Also matches"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            result = agent.evaluate_rules(FakeApp())
            assert len(result) == 2
            assert result[0]["rule_id"] == "R1"
            assert result[1]["rule_id"] == "R2"
        finally:
            os.unlink(rules_path)

    def test_condition_accessing_applicant(self):
        """Test that conditions can access applicant attributes."""
        rules_data = {"rules": [
            {"rule_id": "R1",
             "condition": "applicant.age > 60",
             "severity": "moderate",
             "description": "Old applicant"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)

            class FakeApp:
                age = 30

            result = agent.evaluate_rules(FakeApp())
            assert result == []
        finally:
            os.unlink(rules_path)

    def test_rule_with_missing_condition_defaults_to_false(self):
        """Rules without a condition field should not match."""
        rules_data = {"rules": [
            {"rule_id": "R1", "severity": "low", "description": "No condition"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            result = agent.evaluate_rules(FakeApp())
            assert result == []
        finally:
            os.unlink(rules_path)

    def test_empty_rules_list(self):
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            result = agent.evaluate_rules(FakeApp())
            assert result == []
        finally:
            os.unlink(rules_path)

    def test_missing_rules_key_treated_as_empty(self):
        """If the 'rules' key is missing, treat as empty list."""
        rules_path = _write_rules_file({"other": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            result = agent.evaluate_rules(FakeApp())
            assert result == []
        finally:
            os.unlink(rules_path)

    def test_eval_safety_no_builtins(self):
        """Ensure eval cannot access dangerous builtins like open, eval, exec."""
        rules_data = {"rules": [
            {"rule_id": "R1",
             "condition": "open",
             "severity": "low",
             "description": "Should not match — open is blocked"},
            {"rule_id": "R2",
             "condition": "eval",
             "severity": "low",
             "description": "Should not match — eval is blocked"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            result = agent.evaluate_rules(FakeApp())
            # Should not match because dangerous builtins are blocked
            assert result == []
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# build_deterministic_assessment
# ---------------------------------------------------------------------------


class TestBuildDeterministicAssessment:
    """Tests for the build_deterministic_assessment method."""

    def test_no_matched_rules(self):
        class FakeApp:
            pass
        agent = _ConcreteAgent(name="TestAgent", rules_path=_write_rules_file({"rules": []}))
        assessment = agent.build_deterministic_assessment(FakeApp(), [])

        assert assessment.agent_name == "TestAgent"
        assert assessment.risk_tier == "standard"
        assert assessment.recommendation == "No risk factors identified. Standard terms."
        assert assessment.reasoning_summary == "All deterministic rules passed."
        assert assessment.confidence_score == 1.0
        assert assessment.llm_used is False
        assert assessment.flags == []

    def test_single_matched_rule(self):
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "high", "description": "High risk",
             "recommendation": "loading", "loading_range": [1.25, 1.5]},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            assert assessment.agent_name == "TestAgent"
            assert assessment.risk_tier == "loading"
            assert len(assessment.flags) == 1
            assert assessment.flags[0]["rule_id"] == "R1"
            assert assessment.flags[0]["severity"] == "high"
            assert assessment.loading_range == [1.25, 1.5]
            assert assessment.confidence_score == 1.0
            assert assessment.llm_used is False
        finally:
            os.unlink(rules_path)

    def test_multiple_ruleshighest_severity_wins(self):
        """The highest severity rule determines the risk tier."""
        rules_data = {"rules": [
            {"rule_id": "R1",
             "condition": "True",
             "severity": "low",
             "description": "Low",
             "recommendation": "standard_or_loading",
             "loading_range": [1.0, 1.1]},
            {"rule_id": "R2",
             "condition": "True",
             "severity": "critical",
             "description": "Critical",
             "recommendation": "decline",
             "loading_range": [1.0, 1.0]},
            {"rule_id": "R3",
             "condition": "True",
             "severity": "moderate",
             "description": "Moderate",
             "recommendation": "refer", "loading_range": [1.1, 1.2]},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            assert assessment.risk_tier == "decline"
            assert assessment.recommendation == "decline"
            assert len(assessment.flags) == 3
        finally:
            os.unlink(rules_path)

    def test_all_flags_collected(self):
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "low", "description": "Flag1"},
            {"rule_id": "R2", "condition": "True", "severity": "high", "description": "Flag2"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            assert len(assessment.flags) == 2
            flag_ids = {f["rule_id"] for f in assessment.flags}
            assert flag_ids == {"R1", "R2"}
        finally:
            os.unlink(rules_path)

    def test_additional_evidence_deduplication(self):
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "low", "description": "R1",
             "additional_evidence": ["Medical report", "Income proof"]},
            {"rule_id": "R2", "condition": "True", "severity": "moderate", "description": "R2",
             "additional_evidence": ["Medical report", "Employment history"]},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            assert assessment.additional_evidence_required == [
                "Medical report", "Income proof",
                "Employment history",
            ]
        finally:
            os.unlink(rules_path)

    def test_apra_references_deduplication(self):
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "low", "description": "R1",
             "apra_ref": "APRA-CFI-2023-001"},
            {"rule_id": "R2", "condition": "True", "severity": "moderate", "description": "R2",
             "apra_ref": "APRA-CFI-2023-001"},
            {"rule_id": "R3", "condition": "True", "severity": "high", "description": "R3",
             "apra_ref": "APRA-CFI-2023-002"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            # Deduplicated
            assert len(assessment.apra_references) == 2
            assert "APRA-CFI-2023-001" in assessment.apra_references
            assert "APRA-CFI-2023-002" in assessment.apra_references
        finally:
            os.unlink(rules_path)

    def test_default_loading_range(self):
        """If a rule has no loading_range, default to [1.0, 1.0]."""
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "low", "description": "R1",
             "recommendation": "standard"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            assert assessment.loading_range == [1.0, 1.0]
        finally:
            os.unlink(rules_path)

    def test_reasoning_summary_format(self):
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "high", "description": "R1"},
            {"rule_id": "R2", "condition": "True", "severity": "moderate", "description": "R2"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            class FakeApp:
                pass
            assessment = agent.build_deterministic_assessment(FakeApp(), rules_data["rules"])

            assert "2 deterministic rule(s)" in assessment.reasoning_summary
            assert "high" in assessment.reasoning_summary
            assert "R1" in assessment.reasoning_summary
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# _determine_risk_tier
# ---------------------------------------------------------------------------


class TestDetermineRiskTier:
    """Tests for the _determine_risk_tier static method."""

    @pytest.mark.parametrize("recommendation,expected", [
        ("standard", "standard"),
        ("standard_or_loading", "standard"),
        ("loading", "loading"),
        ("standard_loading", "loading"),
        ("decline", "decline"),
        ("manual_underwriting_or_decline", "decline"),
        ("refer", "refer"),
        ("manual_underwriting", "refer"),
    ])
    def test_mapping(self, recommendation, expected):
        highest_rule = {"recommendation": recommendation}
        result = BaseAgent._determine_risk_tier(highest_rule)
        assert result == expected

    def test_default_returns_standard(self):
        """If no recommendation key, default to 'standard'."""
        highest_rule = {"severity": "low"}
        result = BaseAgent._determine_risk_tier(highest_rule)
        assert result == "standard"


# ---------------------------------------------------------------------------
# Abstract class enforcement
# ---------------------------------------------------------------------------


class TestAbstractClass:
    """Tests ensuring BaseAgent cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseAgent(name="Test", rules_path=_write_rules_file({"rules": []}))

    def test_concrete_subclass_eval_works(self):
        """Concrete subclass can call evaluate."""
        rules_data = {"rules": [
            {"rule_id": "R1", "condition": "True", "severity": "low", "description": "R1"},
        ]}
        rules_path = _write_rules_file(rules_data)
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)

            class FakeApp:
                pass

            assessment = agent.evaluate(FakeApp())
            assert isinstance(assessment, AgentAssessment)
            assert assessment.agent_name == "TestAgent"
        finally:
            os.unlink(rules_path)

    def test_generate_rebuttal_returns_assessment(self):
        """Concrete subclass generate_rebuttal works."""
        rules_path = _write_rules_file({"rules": []})
        try:
            agent = _ConcreteAgent(name="TestAgent", rules_path=rules_path)
            my_assessment = AgentAssessment(
                agent_name="TestAgent",
                risk_tier="standard",
                recommendation="ok",
            )
            result = agent.generate_rebuttal(None, my_assessment, [])
            assert result is my_assessment
        finally:
            os.unlink(rules_path)


# ---------------------------------------------------------------------------
# _analyze_question_intent
# ---------------------------------------------------------------------------


class TestAnalyzeQuestionIntent:
    """Tests for the _analyze_question_intent method."""

    def _make_agent(self):
        return _ConcreteAgent(name="TestAgent", rules_path=_write_rules_file({"rules": []}))

    def test_flag_intent_when_question_about_flags(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("Why was I flagged?")
        assert result == "flag"

    def test_explain_intent_when_question_about_concepts(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("Explain your assessment")
        assert result == "explain"

    def test_evidence_intent_when_providing_info(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("I just quit smoking")
        assert result == "evidence"

    def test_general_intent_for_other_questions(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("Hello")
        assert result == "general"

    def test_flag_intent_case_insensitive(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("WHY was I FLAGGED?")
        assert result == "flag"

    def test_explain_intent_with_what_is_query(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("What is my BMI?")
        assert result == "explain"

    def test_evidence_intent_with_i_have(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("I have a new medical report")
        assert result == "evidence"

    def test_general_intent_for_empty_message(self):
        agent = self._make_agent()
        result = agent._analyze_question_intent("")
        assert result == "general"


# ---------------------------------------------------------------------------
# _build_deterministic_chat_response — contextual / intent-aware
# ---------------------------------------------------------------------------


class TestBuildDeterministicChatResponseContextual:
    """Tests for the enhanced _build_deterministic_chat_response method."""

    def _make_assessment(self):
        return AgentAssessment(
            agent_name="Test Agent",
            risk_tier="loading",
            flags=[{"rule_id": "TEST-001", "severity": "high", "description": "Test flag"}],
            recommendation="loading",
            loading_range=[1.2, 1.5],
            confidence_score=0.85,
            reasoning_summary="Test reasoning",
            additional_evidence_required=[],
            apra_references=[],
        )

    def _make_agent(self):
        return _ConcreteAgent(name="TestAgent", rules_path=_write_rules_file({"rules": []}))

    def test_flag_response_mentions_flagged_items(self):
        agent = self._make_agent()
        assessment = self._make_assessment()
        msg = agent._build_deterministic_chat_response(
            "Why was I flagged?", assessment, "test underwriting"
        )
        assert "flag" in msg.content.lower()

    def test_explain_response_mentions_assessment(self):
        agent = self._make_agent()
        assessment = self._make_assessment()
        msg = agent._build_deterministic_chat_response(
            "Explain your assessment", assessment, "test underwriting"
        )
        assert "assessment" in msg.content.lower()

    def test_evidence_response_acknowledges_input(self):
        agent = self._make_agent()
        assessment = self._make_assessment()
        msg = agent._build_deterministic_chat_response(
            "I just quit smoking", assessment, "test underwriting"
        )
        assert "thank you" in msg.content.lower()

    def test_response_always_includes_risk_tier(self):
        """Every response must reference the risk tier (uppercase) somewhere in the
        content. The pre-fix test asserted the literal "loading" was preserved,
        but evidence intent now triggers ``self.evaluate()`` and may legitimately
        change the tier. The contract is: a tier is *mentioned* — not that the
        pre-evaluation tier is preserved.
        """
        agent = self._make_agent()
        assessment = self._make_assessment()
        for text in [
            "Why was I flagged?",
            "Explain your assessment",
            "Hello",
        ]:
            msg = agent._build_deterministic_chat_response(
                text, assessment, "test underwriting"
            )
            # Pre-evaluation tier is preserved for non-evidence intents.
            assert "LOADING" in msg.content

    def test_evidence_response_references_post_eval_tier(self):
        """For evidence intent, the response must reference the post-evaluation
        tier (which may differ from the pre-evaluation tier). The pre-fix
        contract — "the pre-eval tier is preserved" — is gone. The new contract
        is: the response reflects the fresh evaluate() result.
        """
        agent = self._make_agent()
        assessment = self._make_assessment()
        # _ConcreteAgent.evaluate() with empty rules returns "standard".
        msg = agent._build_deterministic_chat_response(
            "I just quit smoking", assessment, "test underwriting"
        )
        # The fresh evaluation tier is referenced (uppercase).
        assert "STANDARD" in msg.content
