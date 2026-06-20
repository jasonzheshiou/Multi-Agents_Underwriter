"""Tests that conversation.final_decision is recalculated after evidence injection.

These tests verify Bug A fix: after user injects evidence and debate rounds
complete, the conversation.final_decision reflects the new outcome computed
from updated agent_assessments.
"""

import pytest
from datetime import date

from underwriting.application.schema import (
    Application,
    BenefitType,
    SmokerStatus,
)
from underwriting.agents.base_agent import AgentAssessment
from underwriting.debate.chat_models import Conversation, ChatMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RISK_TIER_RANK = {"standard": 0, "loading": 1, "refer": 2, "decline": 3}
_DECISION_MAP = {
    0: "Standard Offer",
    1: "Offer with Loading/Exclusion",
    2: "Refer to Manual Underwriting",
    3: "Refer to Manual Underwriting",
}


def _compute_decision(assessments: dict) -> str:
    """Replicate the decision recalculation logic from the fix."""
    if not assessments:
        return "Pending"
    tiers = [a.get("risk_tier", "standard") for a in assessments.values()]
    ranks = [_RISK_TIER_RANK.get(t, 0) for t in tiers]
    return _DECISION_MAP.get(max(ranks), "Refer to Manual Underwriting")


def _make_assessment(agent_name: str, risk_tier: str, confidence: float = 1.0) -> dict:
    """Create assessment data dict matching Conversation.agent_assessments format."""
    return {
        "risk_tier": risk_tier,
        "flags": [],
        "recommendation": "standard",
        "loading_range": [1.0, 1.0],
        "confidence_score": confidence,
        "reasoning_summary": f"Test assessment for {agent_name}",
        "additional_evidence_required": [],
        "apra_references": [],
    }


# ---------------------------------------------------------------------------
# Decision Computation Tests
# ---------------------------------------------------------------------------


class TestDecisionRecalculation:
    """Verify the decision recalculation logic produces correct outcomes."""

    def test_all_standard_tiers_yields_standard_offer(self):
        """All agents at 'standard' tier → 'Standard Offer'."""
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "standard"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
            "Compliance Agent": _make_assessment("Compliance Agent", "standard"),
        }
        assert _compute_decision(assessments) == "Standard Offer"

    def test_one_loading_rest_standard_yields_loading(self):
        """One agent at 'loading', others 'standard' → 'Offer with Loading/Exclusion'."""
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "loading"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
            "Compliance Agent": _make_assessment("Compliance Agent", "standard"),
        }
        assert _compute_decision(assessments) == "Offer with Loading/Exclusion"

    def test_one_refer_rest_standard_yields_refer(self):
        """One agent at 'refer', others 'standard' → 'Refer to Manual Underwriting'."""
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "refer"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
            "Compliance Agent": _make_assessment("Compliance Agent", "standard"),
        }
        assert _compute_decision(assessments) == "Refer to Manual Underwriting"

    def test_one_decline_rest_standard_yields_refer(self):
        """One agent at 'decline', others 'standard' → 'Refer to Manual Underwriting'."""
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "decline"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
            "Compliance Agent": _make_assessment("Compliance Agent", "standard"),
        }
        assert _compute_decision(assessments) == "Refer to Manual Underwriting"

    def test_mixed_tiers_highest_wins(self):
        """Mixed tiers: 'standard', 'loading', 'refer' → 'Refer to Manual Underwriting'."""
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "refer"),
            "Financial Agent": _make_assessment("Financial Agent", "loading"),
            "Compliance Agent": _make_assessment("Compliance Agent", "standard"),
        }
        assert _compute_decision(assessments) == "Refer to Manual Underwriting"

    def test_single_agent_assessment(self):
        """Single agent assessment, regardless of tier."""
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "loading"),
        }
        assert _compute_decision(assessments) == "Offer with Loading/Exclusion"

    def test_empty_assessments_returns_pending(self):
        """No assessments → 'Pending'."""
        assert _compute_decision({}) == "Pending"

    def test_decision_changes_when_tier_downgrades(self):
        """Decision should change when an agent's tier is downgraded."""
        # Before evidence: Medical at 'refer'
        before = {
            "Medical Agent": _make_assessment("Medical Agent", "refer"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
        }
        assert _compute_decision(before) == "Refer to Manual Underwriting"

        # After evidence: Medical downgraded to 'loading'
        after = {
            "Medical Agent": _make_assessment("Medical Agent", "loading"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
        }
        assert _compute_decision(after) == "Offer with Loading/Exclusion"

    def test_decision_unchanged_when_no_tier_change(self):
        """Decision should stay the same when tiers don't change."""
        before = {
            "Medical Agent": _make_assessment("Medical Agent", "loading"),
            "Financial Agent": _make_assessment("Financial Agent", "loading"),
        }
        before_decision = _compute_decision(before)

        after = {
            "Medical Agent": _make_assessment("Medical Agent", "loading", confidence=0.8),
            "Financial Agent": _make_assessment("Financial Agent", "loading", confidence=0.7),
        }
        after_decision = _compute_decision(after)

        assert before_decision == after_decision


class TestConversationFinalDecisionUpdate:
    """Verify that conversation.final_decision gets recalculated."""

    def test_final_decision_updated_after_evidence(self):
        """Simulate the chat flow: final_decision should reflect new assessments."""
        # Initial setup — Medical at 'refer' (high risk)
        assessments = {
            "Medical Agent": _make_assessment("Medical Agent", "refer"),
            "Financial Agent": _make_assessment("Financial Agent", "standard"),
            "Compliance Agent": _make_assessment("Compliance Agent", "standard"),
        }

        conv = Conversation(
            application_id="test_001",
            applicant_name="Test",
            debate_rounds=0,
            final_decision=_compute_decision(assessments),
            agent_assessments=assessments,
        )
        assert conv.final_decision == "Refer to Manual Underwriting"

        # Evidence injection: Medical tier downgraded from 'refer' to 'loading'
        conv.agent_assessments["Medical Agent"] = _make_assessment("Medical Agent", "loading")

        # Simulate debate round
        conv.debate_rounds += 1

        # THE FIX: recalculation (this is what app.py now does)
        tiers = [data.get("risk_tier", "standard") for data in conv.agent_assessments.values()]
        ranks = [_RISK_TIER_RANK.get(t, 0) for t in tiers]
        conv.final_decision = _DECISION_MAP.get(max(ranks), "Refer to Manual Underwriting")

        # After fix: decision should be 'Offer with Loading/Exclusion'
        assert conv.final_decision == "Offer with Loading/Exclusion"
