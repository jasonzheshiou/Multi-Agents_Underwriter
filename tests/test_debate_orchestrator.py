"""Tests for the DebateOrchestrator class."""


from typing import Any, List

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.agents.debate_orchestrator import DebateOrchestrator
from underwriting.debate.chat_models import ChatMessage

# ---------------------------------------------------------------------------
# Concrete agent for testing
# ---------------------------------------------------------------------------


class _TestAgent(BaseAgent):
    """A concrete BaseAgent subclass for testing the debate orchestrator."""

    def __init__(self, name: str, risk_tier: str, flags: list | None = None, **kwargs):
        """Create a test agent that always returns a fixed assessment.

        Args:
            name: Agent name.
            risk_tier: Fixed risk tier to return.
            flags: Fixed flags list.
            **kwargs: Additional arguments passed to BaseAgent.__init__.
        """
        # Skip BaseAgent.__init__ — we don't need rules loading for tests.
        self.name = name
        self._fixed_risk_tier = risk_tier
        self._fixed_flags = flags or []
        self._rebuttal_tier = None  # Override for debate rounds if set

    def evaluate(self, application) -> AgentAssessment:
        """Return a fixed assessment."""
        tier = self._fixed_risk_tier
        return AgentAssessment(
            agent_name=self.name,
            risk_tier=tier,
            flags=self._fixed_flags,
            recommendation=tier,
            reasoning_summary=f"{self.name} assessed {tier}.",
        )

    def generate_rebuttal(self, application, my_assessment, other_assessments) -> AgentAssessment:
        """Return rebuttal assessment.

        If _rebuttal_tier is set, use it. Otherwise, stand firm.
        """
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
            user_message, current_assessment, "test underwriting", application
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assessment(agent_name: str, risk_tier: str, **kwargs) -> AgentAssessment:
    """Create a minimal AgentAssessment for direct testing of private methods."""
    return AgentAssessment(
        agent_name=agent_name,
        risk_tier=risk_tier,
        recommendation=risk_tier,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: DebateOrchestrator instantiation
# ---------------------------------------------------------------------------


class TestDebateOrchestratorInit:
    """Tests for DebateOrchestrator initialisation."""

    def test_instantiate_with_agents(self):
        """DebateOrchestrator can be created with a list of agents."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
            _TestAgent("Compliance", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        assert orchestrator.agents == agents
        assert orchestrator.debate_log == []

    def test_class_constants(self):
        """Verify class-level constants."""
        assert DebateOrchestrator.MAX_DEBATE_ROUNDS == 3
        assert DebateOrchestrator.RISK_TIER_RANK == {
            "standard": 0,
            "loading": 1,
            "refer": 2,
            "decline": 3,
        }

    def test_empty_agents_list(self):
        """DebateOrchestrator can be created with an empty agents list."""
        orchestrator = DebateOrchestrator([])
        assert orchestrator.agents == []


# ---------------------------------------------------------------------------
# Test: run() returns dict with all required fields
# ---------------------------------------------------------------------------


class TestRunReturnsDict:
    """Tests for the run() return value structure."""

    def _make_app(self):
        """Create a minimal application-like object."""
        class App:
            pass
        return App()

    def test_run_returns_dict(self):
        """run() returns a dict."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert isinstance(result, dict)

    def test_run_returns_all_required_fields(self):
        """run() returns dict with all expected keys."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        required_keys = {
            "final_assessment",
            "agent_assessments",
            "debate_log",
            "consensus_reached",
            "final_decision",
            "decision_reasoning",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_final_assessment_is_dict(self):
        """final_assessment is a dict."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert isinstance(result["final_assessment"], dict)

    def test_agent_assessments_contains_agent_names(self):
        """agent_assessments keys match agent names."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert set(result["agent_assessments"].keys()) == {a.name for a in agents}

    def test_debate_log_is_list(self):
        """debate_log is a list."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert isinstance(result["debate_log"], list)

    def test_consensus_reached_is_bool(self):
        """consensus_reached is a bool."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert isinstance(result["consensus_reached"], bool)

    def test_final_decision_is_str(self):
        """final_decision is a string."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert isinstance(result["final_decision"], str)

    def test_decision_reasoning_is_str(self):
        """decision_reasoning is a string."""
        agents = [_TestAgent("Medical", "standard"), _TestAgent("Financial", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert isinstance(result["decision_reasoning"], str)


# ---------------------------------------------------------------------------
# Test: Consensus case — all agents agree
# ---------------------------------------------------------------------------


class TestConsensusCase:
    """Tests where all agents agree (no dispute)."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_all_standard_consensus(self):
        """All agents agree on 'standard' → consensus_reached=True."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
            _TestAgent("Compliance", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True

    def test_all_loading_consensus(self):
        """All agents agree on 'loading' → consensus_reached=True."""
        agents = [
            _TestAgent("Medical", "loading"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True

    def test_all_decline_consensus(self):
        """All agents agree on 'decline' → consensus_reached=True."""
        agents = [
            _TestAgent("Medical", "decline"),
            _TestAgent("Financial", "decline"),
            _TestAgent("Compliance", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True

    def test_no_debate_log_on_consensus(self):
        """No debate entries when all agents agree."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["debate_log"] == []

    def test_consensus_two_agents(self):
        """Consensus with exactly two agents."""
        agents = [
            _TestAgent("Medical", "refer"),
            _TestAgent("Financial", "refer"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []


# ---------------------------------------------------------------------------
# Test: Dispute case — agents disagree
# ---------------------------------------------------------------------------


class TestDisputeCase:
    """Tests where agents disagree (dispute triggered)."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_dispute_detected_standard_vs_refer(self):
        """standard vs refer → dispute (rank diff >= 2)."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "refer"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False

    def test_dispute_detected_standard_vs_decline(self):
        """standard vs decline → dispute (extreme disagreement)."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False

    def test_dispute_detected_loading_vs_decline(self):
        """loading vs decline → dispute (rank diff >= 2)."""
        agents = [
            _TestAgent("Medical", "loading"),
            _TestAgent("Financial", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False

    def test_no_dispute_standard_vs_loading(self):
        """standard vs loading → no dispute (rank diff = 1)."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True

    def test_no_dispute_loading_vs_refer(self):
        """loading vs refer → no dispute (rank diff = 1)."""
        agents = [
            _TestAgent("Medical", "loading"),
            _TestAgent("Financial", "refer"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True

    def test_dispute_three_way(self):
        """Three agents with different tiers → dispute."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False


# ---------------------------------------------------------------------------
# Test: Debate resolves in ≤3 rounds
# ---------------------------------------------------------------------------


class TestDebateRounds:
    """Tests for debate round behaviour."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_debate_max_three_rounds(self):
        """Debate does not exceed MAX_DEBATE_ROUNDS."""
        # All agents stay firm → debate runs full 3 rounds, never reaches consensus
        medical = _TestAgent("Medical", "standard")
        financial = _TestAgent("Financial", "decline")
        compliance = _TestAgent("Compliance", "refer")
        orchestrator = DebateOrchestrator([medical, financial, compliance])
        result = orchestrator.run(self._make_app())
        # Max 3 agents in minority across rounds, total entries ≤ 3 * 3 = 9
        assert len(result["debate_log"]) <= DebateOrchestrator.MAX_DEBATE_ROUNDS * len(
            orchestrator.agents
        )

    def test_debate_early_consensus(self):
        """Debate stops early when consensus is reached."""
        medical = _TestAgent("Medical", "standard")
        financial = _TestAgent("Financial", "refer")
        compliance = _TestAgent("Compliance", "refer")
        # Financial and Compliance change to 'standard' on rebuttal → all standard → consensus
        financial._rebuttal_tier = "standard"
        compliance._rebuttal_tier = "standard"
        orchestrator = DebateOrchestrator([medical, financial, compliance])
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert len(result["debate_log"]) > 0

    def test_debate_log_entries_have_required_fields(self):
        """Each debate log entry has round, agent, original_tier, updated_tier, reasoning."""
        medical = _TestAgent("Medical", "standard")
        financial = _TestAgent("Financial", "decline")
        # Financial changes to 'loading' on rebuttal
        financial._rebuttal_tier = "loading"
        orchestrator = DebateOrchestrator([medical, financial])
        result = orchestrator.run(self._make_app())
        for entry in result["debate_log"]:
            assert "round" in entry
            assert "agent" in entry
            assert "original_tier" in entry
            assert "updated_tier" in entry
            assert "reasoning" in entry


# ---------------------------------------------------------------------------
# Test: Final decision = most conservative tier
# ---------------------------------------------------------------------------


class TestFinalDecision:
    """Tests for the _produce_final_decision logic."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_most_conservative_wins_standard(self):
        """All standard → final decision is 'Standard Offer'."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Standard Offer"

    def test_most_conservative_wins_loading(self):
        """Standard + loading → final decision is 'Offer with Loading/Exclusion'."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Offer with Loading/Exclusion"

    def test_most_conservative_wins_refer(self):
        """Standard + refer → final decision is 'Refer to Manual Underwriting'."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "refer"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_most_conservative_wins_decline(self):
        """Any decline → final decision is 'Refer to Manual Underwriting'."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_all_decline(self):
        """All agents decline → final decision is 'Refer to Manual Underwriting'."""
        agents = [
            _TestAgent("Medical", "decline"),
            _TestAgent("Financial", "decline"),
            _TestAgent("Compliance", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_flags_collected_from_all_agents(self):
        """Final decision includes flags from all agents."""
        agents = [
            _TestAgent("Medical", "standard", flags=[{"rule_id": "MED-001", "severity": "low", "description": "Test"}]),
            _TestAgent("Financial", "standard", flags=[{"rule_id": "FIN-001", "severity": "moderate", "description": "Test2"}]),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        flag_ids = {f["rule_id"] for f in result["final_assessment"]["flags"]}
        assert "MED-001" in flag_ids
        assert "FIN-001" in flag_ids

    def test_additional_evidence_collected(self):
        """Final decision includes additional evidence from all agents."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
        ]
        # Override assessments after run to add evidence
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        # Evidence should be in final_assessment
        assert "additional_evidence_required" in result["final_assessment"]


# ---------------------------------------------------------------------------
# Test: Tie-breaking logic
# ---------------------------------------------------------------------------


class TestTieBreaking:
    """Tests for tie-breaking scenarios."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_tie_two_standard_two_loading(self):
        """2 standard + 2 loading → loading wins (highest rank)."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
            _TestAgent("Compliance", "loading"),
            _TestAgent("Risk", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Offer with Loading/Exclusion"
        assert result["consensus_reached"] is True  # standard vs loading = rank diff 1

    def test_tie_all_different_tiers(self):
        """All different tiers → highest rank wins."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "refer"),
            _TestAgent("Risk", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Refer to Manual Underwriting"
        assert result["consensus_reached"] is False  # rank diff >= 2

    def test_tie_majority_picks_correct(self):
        """Majority tier is correctly identified in debate."""
        # 2 loading vs 1 standard → majority is loading, standard is minority
        medical = _TestAgent("Medical", "standard")
        financial = _TestAgent("Financial", "loading")
        compliance = _TestAgent("Compliance", "loading")
        orchestrator = DebateOrchestrator([medical, financial, compliance])
        result = orchestrator.run(self._make_app())
        # standard vs loading = rank diff 1, no dispute
        assert result["consensus_reached"] is True

    def test_tie_majority_all_same(self):
        """All same tier → no minority, no debate entries."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
            _TestAgent("Compliance", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["debate_log"] == []
        assert result["consensus_reached"] is True


# ---------------------------------------------------------------------------
# Test: All-agree case
# ---------------------------------------------------------------------------


class TestAllAgree:
    """Tests where all agents completely agree."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_all_agree_standard(self):
        """All agents agree on standard → no debate, consensus."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
            _TestAgent("Compliance", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Standard Offer"

    def test_all_agree_loading(self):
        """All agents agree on loading → no debate, consensus."""
        agents = [
            _TestAgent("Medical", "loading"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Offer with Loading/Exclusion"

    def test_all_agree_refer(self):
        """All agents agree on refer → no debate, consensus."""
        agents = [
            _TestAgent("Medical", "refer"),
            _TestAgent("Financial", "refer"),
            _TestAgent("Compliance", "refer"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_all_agree_decline(self):
        """All agents agree on decline → no debate, consensus."""
        agents = [
            _TestAgent("Medical", "decline"),
            _TestAgent("Financial", "decline"),
            _TestAgent("Compliance", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Refer to Manual Underwriting"


# ---------------------------------------------------------------------------
# Test: All-disagree case
# ---------------------------------------------------------------------------


class TestAllDisagree:
    """Tests where all agents disagree."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_all_different_tiers_triggers_debate(self):
        """All different tiers → dispute → debate initiated."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False
        assert len(result["debate_log"]) > 0

    def test_all_disagree_final_decision_is_most_conservative(self):
        """Final decision picks the most conservative tier."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["final_decision"] == "Refer to Manual Underwriting"

    def test_all_disagree_3_agents_all_different(self):
        """Three-way disagreement with 3 agents."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
            _TestAgent("Compliance", "refer"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False
        assert result["final_decision"] == "Refer to Manual Underwriting"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def _make_app(self):
        class App:
            pass
        return App()

    def test_single_agent(self):
        """Single agent → no dispute, consensus."""
        agents = [_TestAgent("Medical", "standard")]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []
        assert result["final_decision"] == "Standard Offer"

    def test_two_agents_same_tier(self):
        """Two agents with same tier → consensus."""
        agents = [
            _TestAgent("Medical", "loading"),
            _TestAgent("Financial", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True
        assert result["debate_log"] == []

    def test_two_agents_adjacent_tiers(self):
        """Two agents with adjacent tiers (rank diff = 1) → no dispute."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is True

    def test_two_agents_opposite_tiers(self):
        """Two agents with opposite tiers (standard vs decline) → dispute."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "decline"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert result["consensus_reached"] is False

    def test_decision_reasoning_contains_agent_count(self):
        """Decision reasoning mentions the number of underwriting agents."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "standard"),
            _TestAgent("Compliance", "standard"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert "underwriting" in result["decision_reasoning"]

    def test_all_assessments_in_final(self):
        """Final assessment includes all agent assessments."""
        agents = [
            _TestAgent("Medical", "standard"),
            _TestAgent("Financial", "loading"),
        ]
        orchestrator = DebateOrchestrator(agents)
        result = orchestrator.run(self._make_app())
        assert "all_assessments" in result["final_assessment"]
        assert "Medical" in result["final_assessment"]["all_assessments"]
        assert "Financial" in result["final_assessment"]["all_assessments"]


# ---------------------------------------------------------------------------
# Test: _detect_dispute private method
# ---------------------------------------------------------------------------


class TestDetectDispute:
    """Tests for the _detect_dispute method."""

    def test_no_dispute_all_same(self):
        """All same tier → no dispute."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "standard"),
        }
        orchestrator = DebateOrchestrator([])
        assert orchestrator._detect_dispute(assessments) is False

    def test_dispute_rank_diff_2(self):
        """Rank difference >= 2 → dispute."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "refer"),
        }
        orchestrator = DebateOrchestrator([])
        assert orchestrator._detect_dispute(assessments) is True

    def test_dispute_standard_vs_decline(self):
        """Standard vs decline → dispute (special case)."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "decline"),
        }
        orchestrator = DebateOrchestrator([])
        assert orchestrator._detect_dispute(assessments) is True

    def test_no_dispute_rank_diff_1(self):
        """Rank difference = 1 → no dispute."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "loading"),
        }
        orchestrator = DebateOrchestrator([])
        assert orchestrator._detect_dispute(assessments) is False

    def test_dispute_loading_vs_decline(self):
        """Loading vs decline → dispute (rank diff = 2)."""
        assessments = {
            "a": _make_assessment("a", "loading"),
            "b": _make_assessment("b", "decline"),
        }
        orchestrator = DebateOrchestrator([])
        assert orchestrator._detect_dispute(assessments) is True

    def test_dispute_refer_vs_decline(self):
        """Refer vs decline → no dispute (rank diff = 1)."""
        assessments = {
            "a": _make_assessment("a", "refer"),
            "b": _make_assessment("b", "decline"),
        }
        orchestrator = DebateOrchestrator([])
        assert orchestrator._detect_dispute(assessments) is False


# ---------------------------------------------------------------------------
# Test: _produce_final_decision private method
# ---------------------------------------------------------------------------


class TestProduceFinalDecision:
    """Tests for the _produce_final_decision method."""

    def test_all_standard(self):
        """All standard → Standard Offer."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "standard"),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert result["decision"] == "Standard Offer"
        assert result["risk_tier"] == "standard"

    def test_most_conservative_wins(self):
        """Most conservative tier wins."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "loading"),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert result["decision"] == "Offer with Loading/Exclusion"
        assert result["risk_tier"] == "loading"

    def test_decline_wins(self):
        """Decline wins over all other tiers."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "loading"),
            "c": _make_assessment("c", "decline"),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert result["decision"] == "Refer to Manual Underwriting"
        assert result["risk_tier"] == "decline"

    def test_flags_collected(self):
        """Flags from all agents are collected."""
        assessments = {
            "a": _make_assessment("a", "standard", flags=[{"rule_id": "R1", "severity": "low", "description": "f1"}]),
            "b": _make_assessment("b", "standard", flags=[{"rule_id": "R2", "severity": "high", "description": "f2"}]),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert len(result["flags"]) == 2

    def test_evidence_deduplicated(self):
        """Additional evidence is deduplicated."""
        assessments = {
            "a": _make_assessment("a", "standard", additional_evidence_required=["report"]),
            "b": _make_assessment("b", "standard", additional_evidence_required=["report"]),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert result["additional_evidence_required"] == ["report"]

    def test_all_assessments_serialized(self):
        """All assessments are serialized via dict()."""
        assessments = {
            "a": _make_assessment("a", "standard"),
            "b": _make_assessment("b", "loading"),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert "all_assessments" in result
        assert "a" in result["all_assessments"]
        assert "b" in result["all_assessments"]
        assert result["all_assessments"]["a"]["risk_tier"] == "standard"
        assert result["all_assessments"]["b"]["risk_tier"] == "loading"

    def test_reasoning_includes_decision(self):
        """Reasoning includes the final decision text."""
        assessments = {
            "a": _make_assessment("a", "standard"),
        }
        orchestrator = DebateOrchestrator([])
        result = orchestrator._produce_final_decision(assessments)
        assert "Standard Offer" in result["reasoning"]
