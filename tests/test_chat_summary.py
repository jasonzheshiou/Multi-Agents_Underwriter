"""Tests for ``generate_debate_summary()`` — HTML summary of agent assessments."""

from unittest.mock import patch

import pytest

from typing import Dict

from underwriting.debate.chat_models import Conversation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STANDARD_ASSESSMENT = {
    "risk_tier": "standard",
    "flags": [],
    "recommendation": "Standard Offer",
    "loading_range": [1.0, 1.0],
    "confidence_score": 0.95,
    "reasoning_summary": "All indicators within normal range.",
    "additional_evidence_required": [],
    "apra_references": [],
}


_LOADING_ASSESSMENT = {
    "risk_tier": "loading",
    "flags": [
        {
            "rule_id": "MED-001",
            "severity": "high",
            "description": "High BMI (31.2)",
        },
    ],
    "recommendation": "Offer with Loading",
    "loading_range": [1.2, 1.5],
    "confidence_score": 0.85,
    "reasoning_summary": "Elevated BMI detected.",
    "additional_evidence_required": [],
    "apra_references": [],
}


_DECLINE_ASSESSMENT = {
    "risk_tier": "decline",
    "flags": [
        {
            "rule_id": "MED-010",
            "severity": "critical",
            "description": "Uncontrolled Type 1 diabetes",
        },
    ],
    "recommendation": "Decline",
    "loading_range": [0.0, 0.0],
    "confidence_score": 0.90,
    "reasoning_summary": "Critical medical risk identified.",
    "additional_evidence_required": ["Recent HbA1c results"],
    "apra_references": ["APRA CPS 220"],
}


def _make_conversation(
    agent_assessments: Dict[str, dict] | None = None,
    application_id: str = "app-001",
    applicant_name: str = "Jane Doe",
) -> Conversation:
    """Build a minimal Conversation with optional agent_assessments."""
    conv = Conversation(
        application_id=application_id,
        applicant_name=applicant_name,
    )
    if agent_assessments:
        conv.agent_assessments = agent_assessments
    return conv


# ---------------------------------------------------------------------------
# 1. Empty assessments
# ---------------------------------------------------------------------------


class TestEmptyAssessments:
    """Tests for conversations with no agent assessments."""

    def test_empty_assessments_returns_empty_html(self):
        """A conversation with no agent_assessments should return an empty string."""
        from underwriting.debate.chat_summary import generate_debate_summary

        conv = _make_conversation(agent_assessments={})
        result = generate_debate_summary(conv)
        assert result == ""


# ---------------------------------------------------------------------------
# 2. Consensus — all standard
# ---------------------------------------------------------------------------


class TestConsensusIndicator:
    """Tests for consensus detection when all agents agree."""

    def test_all_standard_tiers_shows_consensus(self):
        """All agents at 'standard' tier should render a consensus indicator."""
        from underwriting.debate.chat_summary import generate_debate_summary

        assessments = {
            "Medical Agent": _STANDARD_ASSESSMENT,
            "Financial Agent": _STANDARD_ASSESSMENT,
            "Compliance Agent": _STANDARD_ASSESSMENT,
        }
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        assert result != ""
        # Consensus indicator text or badge should be present
        assert "Consensus" in result or "consensus" in result.lower()
        # All agents should be mentioned
        assert "Medical Agent" in result
        assert "Financial Agent" in result
        assert "Compliance Agent" in result


# ---------------------------------------------------------------------------
# 3. Mixed tiers — debate warning
# ---------------------------------------------------------------------------


class TestDebateWarning:
    """Tests for debate warning when agents disagree."""

    def test_mixed_tiers_shows_debate_warning(self):
        """Agents at different tiers should render a debate warning."""
        from underwriting.debate.chat_summary import generate_debate_summary

        assessments = {
            "Medical Agent": _STANDARD_ASSESSMENT,
            "Financial Agent": _LOADING_ASSESSMENT,
            "Compliance Agent": _STANDARD_ASSESSMENT,
        }
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        assert result != ""
        # Debate warning should be present
        assert "Debate" in result or "debate" in result.lower()
        # Warning indicator should be present
        assert "Warning" in result or "warning" in result.lower()


# ---------------------------------------------------------------------------
# 4. Flag sorting by severity
# ---------------------------------------------------------------------------


class TestFlagSorting:
    """Tests for flag severity ordering."""

    def test_flags_sorted_by_severity(self):
        """Flags should be sorted: critical > high > moderate > low."""
        from underwriting.debate.chat_summary import generate_debate_summary

        multi_flag_assessment = {
            "risk_tier": "loading",
            "flags": [
                {"rule_id": "MED-004", "severity": "low", "description": "Mild smoker"},
                {"rule_id": "MED-001", "severity": "high", "description": "High BMI"},
                {"rule_id": "MED-007", "severity": "moderate", "description": "Family history"},
                {"rule_id": "MED-010", "severity": "critical", "description": "Uncontrolled diabetes"},
            ],
            "recommendation": "Refer",
            "loading_range": [1.0, 1.0],
            "confidence_score": 0.80,
            "reasoning_summary": "Multiple risk factors.",
            "additional_evidence_required": [],
            "apra_references": [],
        }

        assessments = {"Medical Agent": multi_flag_assessment}
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        # All flag descriptions should be present
        assert "Uncontrolled diabetes" in result
        assert "High BMI" in result
        assert "Family history" in result
        assert "Mild smoker" in result

        # Verify order: critical first, then high, moderate, low
        critical_pos = result.index("Uncontrolled diabetes")
        high_pos = result.index("High BMI")
        moderate_pos = result.index("Family history")
        low_pos = result.index("Mild smoker")

        assert critical_pos < high_pos < moderate_pos < low_pos


# ---------------------------------------------------------------------------
# 5. Top 5 flags only
# ---------------------------------------------------------------------------


class TestTop5Flags:
    """Tests for the top-5 flag limit."""

    def test_top_5_flags_only(self):
        """More than 5 flags should show only the top 5 by severity."""
        from underwriting.debate.chat_summary import generate_debate_summary

        many_flags_assessment = {
            "risk_tier": "loading",
            "flags": [
                {"rule_id": "MED-001", "severity": "critical", "description": "Flag 1 critical"},
                {"rule_id": "MED-002", "severity": "critical", "description": "Flag 2 critical"},
                {"rule_id": "MED-003", "severity": "high", "description": "Flag 3 high"},
                {"rule_id": "MED-004", "severity": "high", "description": "Flag 4 high"},
                {"rule_id": "MED-005", "severity": "moderate", "description": "Flag 5 moderate"},
                {"rule_id": "MED-006", "severity": "moderate", "description": "Flag 6 moderate"},
                {"rule_id": "MED-007", "severity": "low", "description": "Flag 7 low"},
                {"rule_id": "MED-008", "severity": "low", "description": "Flag 8 low"},
            ],
            "recommendation": "Refer",
            "loading_range": [1.0, 1.0],
            "confidence_score": 0.75,
            "reasoning_summary": "Many flags.",
            "additional_evidence_required": [],
            "apra_references": [],
        }

        assessments = {"Medical Agent": many_flags_assessment}
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        # Top 5 should be present
        assert "Flag 1 critical" in result
        assert "Flag 2 critical" in result
        assert "Flag 3 high" in result
        assert "Flag 4 high" in result
        assert "Flag 5 moderate" in result

        # Flags beyond 5 should NOT appear
        assert "Flag 6 moderate" not in result
        assert "Flag 7 low" not in result
        assert "Flag 8 low" not in result


# ---------------------------------------------------------------------------
# 6. Color coding — standard = green
# ---------------------------------------------------------------------------


class TestColorCodingStandard:
    """Tests for green color coding on standard tier."""

    def test_color_coding_standard_green(self):
        """'standard' tier should render with green color or class."""
        from underwriting.debate.chat_summary import generate_debate_summary

        assessments = {
            "Medical Agent": _STANDARD_ASSESSMENT,
        }
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        assert result != ""
        # Should contain green color indicator (CSS class or color value)
        assert "green" in result.lower() or "#28a745" in result or "standard-offer" in result


# ---------------------------------------------------------------------------
# 7. Color coding — decline = red
# ---------------------------------------------------------------------------


class TestColorCodingDecline:
    """Tests for red color coding on decline tier."""

    def test_color_coding_decline_red(self):
        """'decline' tier should render with red color or class."""
        from underwriting.debate.chat_summary import generate_debate_summary

        assessments = {
            "Medical Agent": _DECLINE_ASSESSMENT,
        }
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        assert result != ""
        # Should contain red color indicator (CSS class or color value)
        assert "red" in result.lower() or "#dc3545" in result or "decline" in result


# ---------------------------------------------------------------------------
# 8. Plain language explanation
# ---------------------------------------------------------------------------


class TestPlainLanguageExplanation:
    """Tests for plain-language summary text."""

    def test_plain_language_explanation_included(self):
        """Summary should include a plain-language explanation section."""
        from underwriting.debate.chat_summary import generate_debate_summary

        assessments = {
            "Medical Agent": _LOADING_ASSESSMENT,
        }
        conv = _make_conversation(agent_assessments=assessments)
        result = generate_debate_summary(conv)

        assert result != ""
        # Should contain a plain-language explanation heading or text
        assert "Summary" in result or "summary" in result.lower()
        # Should include the agent's reasoning summary
        assert "Elevated BMI detected" in result or "High BMI" in result
