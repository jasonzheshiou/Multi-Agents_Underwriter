"""Tests for MedicalAgent._analyze_question_intent keyword classification.

These tests verify Bug D fix: the keyword overlap where "smoker" triggered
"flag" intent instead of "evidence", and evidence keywords are checked first.
"""

import pytest

from underwriting.agents.medical_agent import MedicalAgent


@pytest.fixture
def agent():
    """Create a MedicalAgent instance for intent testing."""
    return MedicalAgent(rules_path="rules/death/medical_rules.json")


# ---------------------------------------------------------------------------
# Evidence Intent Tests (user-provided information)
# ---------------------------------------------------------------------------


class TestEvidenceIntent:
    """Messages that should be classified as 'evidence' intent."""

    def test_quit_smoking(self, agent):
        """'I just quit smoking' → 'evidence' (was incorrectly 'flag' before fix)."""
        assert agent._analyze_question_intent("I just quit smoking") == "evidence"

    def test_stopped_drinking(self, agent):
        """'I stopped drinking' → 'evidence'."""
        assert agent._analyze_question_intent("I stopped drinking") == "evidence"

    def test_i_am_smoker(self, agent):
        """'I am a smoker' → 'evidence' (was incorrectly 'flag' before fix)."""
        assert agent._analyze_question_intent("I am a smoker") == "evidence"

    def test_i_have_condition(self, agent):
        """'I have a medical condition' → 'evidence'."""
        assert agent._analyze_question_intent("I have a medical condition") == "evidence"

    def test_i_quit_months_ago(self, agent):
        """'I've quit for 6 months' → 'evidence'."""
        assert agent._analyze_question_intent("I've quit for 6 months") == "evidence"

    def test_i_no_longer_smoke(self, agent):
        """'I no longer smoke' → 'evidence'."""
        assert agent._analyze_question_intent("I no longer smoke") == "evidence"

    def test_evidence_keyword_direct(self, agent):
        """'Here is my evidence' → 'evidence'."""
        assert agent._analyze_question_intent("Here is my evidence") == "evidence"

    def test_note_keyword(self, agent):
        """'Please note my BMI has changed' → 'evidence'."""
        assert agent._analyze_question_intent("Please note my BMI has changed") == "evidence"

    def test_new_report(self, agent):
        """'I have a new report from my doctor' → 'evidence'."""
        assert agent._analyze_question_intent("I have a new report from my doctor") == "evidence"

    def test_diagnosed(self, agent):
        """'I was diagnosed last year' → 'evidence'."""
        assert agent._analyze_question_intent("I was diagnosed last year") == "evidence"

    def test_was_told(self, agent):
        """'I was told my results are normal' → 'evidence'."""
        assert agent._analyze_question_intent("I was told my results are normal") == "evidence"


# ---------------------------------------------------------------------------
# Flag Intent Tests (questions about flags/declines/rejections)
# ---------------------------------------------------------------------------


class TestFlagIntent:
    """Messages that should be classified as 'flag' intent."""

    def test_why_flagged(self, agent):
        """'Why was I flagged?' → 'flag'."""
        assert agent._analyze_question_intent("Why was I flagged?") == "flag"

    def test_why_flag(self, agent):
        """'Why did I get a flag?' → 'flag'."""
        assert agent._analyze_question_intent("Why did I get a flag?") == "flag"

    def test_reason_for_decline(self, agent):
        """'What is the reason for decline?' → 'flag'."""
        assert agent._analyze_question_intent("What is the reason for decline?") == "flag"

    def test_why_rejected(self, agent):
        """'Why was my application rejected?' → 'flag'."""
        assert agent._analyze_question_intent("Why was my application rejected?") == "flag"

    def test_flagged_with_context(self, agent):
        """'I was flagged for BMI' → 'flag'."""
        assert agent._analyze_question_intent("I was flagged for BMI") == "flag"


# ---------------------------------------------------------------------------
# Explain Intent Tests (requests for explanation)
# ---------------------------------------------------------------------------


class TestExplainIntent:
    """Messages that should be classified as 'explain' intent."""

    def test_explain_assessment(self, agent):
        """'Explain your assessment' → 'explain'."""
        assert agent._analyze_question_intent("Explain your assessment") == "explain"

    def test_how_did_you_determine(self, agent):
        """'How did you determine my risk?' → 'explain'."""
        assert agent._analyze_question_intent("How did you determine my risk?") == "explain"

    def test_what_does_this_mean(self, agent):
        """'What does this tier mean?' → 'explain'."""
        assert agent._analyze_question_intent("What does this tier mean?") == "explain"

    def test_criteria_question(self, agent):
        """'What criteria was used?' → 'explain'."""
        assert agent._analyze_question_intent("What criteria was used?") == "explain"

    def test_meaning_of_flag(self, agent):
        """'What is the meaning of this flag?' → 'flag' ('flag' matches before 'meaning')."""
        assert agent._analyze_question_intent("What is the meaning of this flag?") == "flag"


# ---------------------------------------------------------------------------
# General Intent Tests (greetings, small talk)
# ---------------------------------------------------------------------------


class TestGeneralIntent:
    """Messages that should be classified as 'general' intent."""

    def test_hello(self, agent):
        """'Hello' → 'general'."""
        assert agent._analyze_question_intent("Hello") == "general"

    def test_thanks(self, agent):
        """'Thank you' → 'general'."""
        assert agent._analyze_question_intent("Thank you") == "general"

    def test_ok(self, agent):
        """'OK' → 'general'."""
        assert agent._analyze_question_intent("OK") == "general"

    def test_empty_message(self, agent):
        """Empty message → 'general'."""
        assert agent._analyze_question_intent("") == "general"

    def test_random_text(self, agent):
        """'Random text with no keywords' → 'general'."""
        assert agent._analyze_question_intent("Random text with no keywords") == "general"
