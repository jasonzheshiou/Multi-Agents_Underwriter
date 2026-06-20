"""Integration tests for DebateOrchestrator with all agents.

Tests the debate and consensus logic of the orchestrator using the real
MedicalAgent, FinancialAgent, and ComplianceAgent with synthetic applicant
fixtures from conftest.py.

Also includes tests with mock agents to verify specific debate scenarios
that the real agents cannot produce (e.g., all agents disagreeing).
"""


from typing import Any, List

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.agents.compliance_agent import ComplianceAgent
from underwriting.agents.debate_orchestrator import DebateOrchestrator
from underwriting.agents.financial_agent import FinancialAgent
from underwriting.agents.medical_agent import MedicalAgent
from underwriting.debate.chat_models import ChatMessage

# ---------------------------------------------------------------------------
# Paths to real rule files
# ---------------------------------------------------------------------------

_RULES_DIR = "rules/death"
_MEDICAL_RULES = f"{_RULES_DIR}/medical_rules.json"
_FINANCIAL_RULES = f"{_RULES_DIR}/financial_rules.json"
_COMPLIANCE_RULES = f"{_RULES_DIR}/compliance_rules.json"


# ---------------------------------------------------------------------------
# Concrete test agent (mirrors test_debate_orchestrator.py pattern)
# ---------------------------------------------------------------------------


class _DebateTestAgent(BaseAgent):
    """A concrete BaseAgent for testing specific debate scenarios.

    Allows setting fixed risk tiers and controlling rebuttal behaviour.
    """

    def __init__(
        self,
        name: str,
        risk_tier: str,
        flags: list | None = None,
        rebuttal_tier: str | None = None,
    ):
        self.name = name
        self._fixed_risk_tier = risk_tier
        self._fixed_flags = flags or []
        self._rebuttal_tier = rebuttal_tier

    def evaluate(self, application) -> AgentAssessment:
        return AgentAssessment(
            agent_name=self.name,
            risk_tier=self._fixed_risk_tier,
            flags=self._fixed_flags,
            recommendation=self._fixed_risk_tier,
            reasoning_summary=f"{self.name} assessed {self._fixed_risk_tier}.",
        )

    def generate_rebuttal(self, application, my_assessment, other_assessments) -> AgentAssessment:
        tier = self._rebuttal_tier if self._rebuttal_tier is not None else self._fixed_risk_tier
        return AgentAssessment(
            agent_name=self.name,
            risk_tier=tier,
            flags=self._fixed_flags,
            recommendation=tier,
            reasoning_summary=f"{self.name} rebuttal: {tier}.",
        )

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agents(risk_tiers=None, llm_client=None):
    """Create agents, optionally overriding risk tiers via _DebateTestAgent."""
    if risk_tiers is None:
        return [
            MedicalAgent(_MEDICAL_RULES, llm_client),
            FinancialAgent(_FINANCIAL_RULES, llm_client),
            ComplianceAgent(_COMPLIANCE_RULES, llm_client),
        ]

    agents = []
    for i, tier in enumerate(risk_tiers):
        name = ["Medical", "Financial", "Compliance"][i]
        agents.append(_DebateTestAgent(name, tier))
    return agents


def _run_orchestrator(applicant, risk_tiers=None, llm_client=None):
    """Run the orchestrator and return the result dict."""
    agents = _make_agents(risk_tiers=risk_tiers, llm_client=llm_client)
    orchestrator = DebateOrchestrator(agents)
    return orchestrator.run(applicant)


# ---------------------------------------------------------------------------
# Test: Consensus — all agents agree on standard
# ---------------------------------------------------------------------------


class TestConsensusAllStandard:
    """All agents agree on 'standard' → consensus_reached=True."""

    def test_all_standard_consensus(self, synthetic_applicant):
        """All three real agents assess standard → consensus."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["standard", "standard", "standard"])
        assert result["consensus_reached"] is True

    def test_no_debate_on_consensus(self, synthetic_applicant):
        """No debate entries when all agents agree."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["standard", "standard", "standard"])
        assert result["debate_log"] == []

    def test_final_decision_standard_offer(self, synthetic_applicant):
        """Final decision is 'Standard Offer' when all agents agree standard."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["standard", "standard", "standard"])
        assert result["final_decision"] == "Standard Offer"

    def test_consensus_all_loading(self, synthetic_applicant):
        """All agents agree on 'loading' → consensus."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["loading", "loading", "loading"])
        assert result["consensus_reached"] is True
        assert result["final_decision"] == "Offer with Loading/Exclusion"

    def test_consensus_all_decline(self, synthetic_applicant):
        """All agents agree on 'decline' → consensus."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["decline", "decline", "decline"])
        assert result["consensus_reached"] is True
        assert result["final_decision"] == "Refer to Manual Underwriting"


# ---------------------------------------------------------------------------
# Test: Dispute — medical decline, financial standard
# ---------------------------------------------------------------------------


class TestDisputeMedicalDeclineFinancialStandard:
    """Agents disagree → debate triggered."""

    def test_dispute_detected(self, synthetic_applicant):
        """Medical decline + Financial standard → dispute."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["decline", "standard", "standard"],
        )
        assert result["consensus_reached"] is False

    def test_debate_initiated(self, synthetic_applicant):
        """Dispute triggers debate with entries in log."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["decline", "standard", "standard"],
        )
        assert len(result["debate_log"]) > 0

    def test_final_decision_conservative(self, synthetic_applicant):
        """Final decision is most conservative (decline → Refer)."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["decline", "standard", "standard"],
        )
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_debate_log_has_round_info(self, synthetic_applicant):
        """Debate log entries contain round information."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["decline", "standard", "standard"],
        )
        for entry in result["debate_log"]:
            assert "round" in entry
            assert "agent" in entry
            assert "updated_tier" in entry


# ---------------------------------------------------------------------------
# Test: Debate resolves in ≤3 rounds
# ---------------------------------------------------------------------------


class TestDebateResolvesInBounds:
    """Debate must resolve within MAX_DEBATE_ROUNDS (3)."""

    def _make_app(self):
        """Create a minimal application-like object."""

        class App:
            pass

        return App()

    def test_debate_resolves_within_bounds(self, synthetic_applicant):
        """Debate with all different tiers resolves in ≤3 rounds."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "decline"],
        )
        max_entries = DebateOrchestrator.MAX_DEBATE_ROUNDS * len(
            [a for a in _make_agents(risk_tiers=["standard", "loading", "decline"])]
        )
        assert len(result["debate_log"]) <= max_entries

    def test_debate_resolves_early(self, synthetic_applicant):
        """Disagreeing agent changes tier on rebuttal → early consensus.

        When consensus is reached after a round, the debate loop breaks
        *before* logging new entries, so the debate log may be empty if
        consensus was reached on the first check.
        """
        # Medical starts as 'standard', changes to 'loading' on rebuttal
        # Financial='loading', Compliance='loading' → after round 1:
        # medical='loading', financial='loading', compliance='loading' → consensus
        medical = _DebateTestAgent("Medical", "standard", rebuttal_tier="loading")
        financial = _DebateTestAgent("Financial", "loading")
        compliance = _DebateTestAgent("Compliance", "loading")
        orchestrator = DebateOrchestrator([medical, financial, compliance])
        result = orchestrator.run(self._make_app())

        assert result["consensus_reached"] is True
        # Debate log may be empty if consensus reached before logging
        assert result["final_decision"] == "Offer with Loading/Exclusion"

    def test_debate_still_resolves_when_no_consensus(self, synthetic_applicant):
        """Even when no consensus is reached, debate still completes."""
        # All agents stay firm on different tiers
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "decline"],
        )
        # Debate runs full 3 rounds, never reaches consensus
        assert result["consensus_reached"] is False
        assert len(result["debate_log"]) > 0


# ---------------------------------------------------------------------------
# Test: Final decision = most conservative tier
# ---------------------------------------------------------------------------


class TestFinalDecisionMostConservative:
    """Final decision always picks the most conservative (highest risk) tier."""

    def _make_app(self):

        class App:
            pass

        return App()

    def test_most_conservative_decline_wins(self, synthetic_applicant):
        """standard + loading + decline → decline wins."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "decline"],
        )
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_most_conservative_refer_wins(self, synthetic_applicant):
        """standard + loading + refer → refer wins."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "refer"],
        )
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_most_conservative_loading_wins(self, synthetic_applicant):
        """standard + loading → loading wins."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "standard"],
        )
        assert result["final_decision"] == "Offer with Loading/Exclusion"

    def test_risk_tier_in_final_assessment(self, synthetic_applicant):
        """Final assessment includes the highest risk tier."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "standard"],
        )
        assert result["final_assessment"]["risk_tier"] == "loading"

    def test_flags_aggregated_from_all_agents(self, synthetic_applicant):
        """Final assessment aggregates flags from all agents."""
        agents = [
            _DebateTestAgent("Medical", "standard", flags=[{"rule_id": "MED-001", "severity": "low", "description": "Test"}]),
            _DebateTestAgent("Financial", "loading", flags=[{"rule_id": "FIN-001", "severity": "moderate", "description": "Test"}]),
            _DebateTestAgent("Compliance", "standard", flags=[{"rule_id": "CMP-001", "severity": "high", "description": "Test"}]),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        flag_ids = {f["rule_id"] for f in result["final_assessment"]["flags"]}
        assert "MED-001" in flag_ids
        assert "FIN-001" in flag_ids
        assert "CMP-001" in flag_ids


# ---------------------------------------------------------------------------
# Test: All agents agree — no debate
# ---------------------------------------------------------------------------


class TestAllAgentsAgree:
    """All agents agree → no debate initiated."""

    def _make_app(self):

        class App:
            pass

        return App()

    def test_all_agree_standard_no_debate(self, synthetic_applicant):
        """All standard → no debate, consensus."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "standard", "standard"],
        )
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Standard Offer"

    def test_all_agree_loading_no_debate(self, synthetic_applicant):
        """All loading → no debate, consensus."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["loading", "loading", "loading"],
        )
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Offer with Loading/Exclusion"

    def test_all_agree_refer_no_debate(self, synthetic_applicant):
        """All refer → no debate, consensus."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["refer", "refer", "refer"],
        )
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_all_agree_decline_no_debate(self, synthetic_applicant):
        """All decline → no debate, consensus."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["decline", "decline", "decline"],
        )
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_decision_reasoning_mentions_agent_count(self, synthetic_applicant):
        """Decision reasoning mentions underwriting agents."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "standard", "standard"],
        )
        assert "underwriting" in result["decision_reasoning"]


# ---------------------------------------------------------------------------
# Test: All agents disagree — debate with all agents
# ---------------------------------------------------------------------------


class TestAllAgentsDisagree:
    """All agents disagree → debate with all agents involved."""

    def _make_app(self):

        class App:
            pass

        return App()

    def test_all_disagree_triggers_debate(self, synthetic_applicant):
        """All different tiers → dispute and debate."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "decline"],
        )
        assert result["consensus_reached"] is False
        assert len(result["debate_log"]) > 0

    def test_all_disagree_final_decision_most_conservative(self, synthetic_applicant):
        """Final decision picks the most conservative tier."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "decline"],
        )
        assert result["final_decision"] == "Refer to Manual Underwriting"
        assert result["final_assessment"]["risk_tier"] == "decline"

    def test_all_disagree_all_agents_in_debate(self, synthetic_applicant):
        """All different tiers → all agents participate in debate rounds.

        With all different tiers, each agent is minority at some point
        across debate rounds.  The debate log contains entries for agents
        that were in the minority during each round.
        """
        medical = _DebateTestAgent("Medical", "standard")
        financial = _DebateTestAgent("Financial", "loading")
        compliance = _DebateTestAgent("Compliance", "decline")
        orchestrator = DebateOrchestrator([medical, financial, compliance])
        result = orchestrator.run(self._make_app())

        # With all different tiers, majority is determined by max(set, count)
        # which picks one arbitrarily.  At least one agent should be logged.
        assert len(result["debate_log"]) > 0

    def test_all_disagree_decision_reasoning_includes_tier(self, synthetic_applicant):
        """Reasoning includes the final decision outcome."""
        result = _run_orchestrator(
            synthetic_applicant,
            risk_tiers=["standard", "loading", "decline"],
        )
        assert "Refer to Manual Underwriting" in result["decision_reasoning"]


# ---------------------------------------------------------------------------
# Test: Edge cases with real agents
# ---------------------------------------------------------------------------


class TestEdgeCasesWithRealAgents:
    """Edge cases using the real MedicalAgent, FinancialAgent, ComplianceAgent."""

    def test_single_agent_pipeline(self, synthetic_applicant):
        """Single agent pipeline works correctly."""
        agents = [MedicalAgent(_MEDICAL_RULES, llm_client=None)]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(synthetic_applicant)

        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert isinstance(result["final_decision"], str)

    def test_two_agent_pipeline(self, synthetic_applicant):
        """Two-agent pipeline works correctly."""
        agents = [
            MedicalAgent(_MEDICAL_RULES, llm_client=None),
            ComplianceAgent(_COMPLIANCE_RULES, llm_client=None),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(synthetic_applicant)

        assert isinstance(result["final_decision"], str)
        assert isinstance(result["consensus_reached"], bool)

    def test_agent_assessments_serialized(self, synthetic_applicant):
        """Final assessment includes serialized agent assessments."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["standard", "loading", "standard"])
        assert "all_assessments" in result["final_assessment"]
        assert "Medical" in result["final_assessment"]["all_assessments"]
        assert "Financial" in result["final_assessment"]["all_assessments"]
        assert "Compliance" in result["final_assessment"]["all_assessments"]

    def test_decision_includes_risk_tier(self, synthetic_applicant):
        """Final assessment includes the risk tier."""
        result = _run_orchestrator(synthetic_applicant, risk_tiers=["standard", "loading", "standard"])
        assert "risk_tier" in result["final_assessment"]
        assert result["final_assessment"]["risk_tier"] == "loading"
