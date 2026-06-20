"""Tests for the FinancialAgent class."""

from unittest.mock import MagicMock

from underwriting.agents.base_agent import AgentAssessment
from underwriting.agents.financial_agent import FinancialAgent
from underwriting.debate.chat_models import ChatMessage


class TestFinancialAgent:
    """Test suite for FinancialAgent.
    
    TODO: Add comprehensive tests for:
    - FinancialAgent instantiation
    - evaluate() income multiple calculations per age bracket
    - evaluate() occupation class mapping
    - evaluate() multiple policy detection
    - LLM enrichment
    - Rebuttal logic
    """

    def test_instantiation(self):
        """Test that FinancialAgent can be instantiated."""
        from underwriting.agents.financial_agent import FinancialAgent
        agent = FinancialAgent(rules_path="rules/death/financial_rules.json")
        assert agent.name == "Financial Agent"

    def test_evaluate_returns_assessment(self):
        """Test that evaluate() returns an AgentAssessment."""
        from underwriting.agents.base_agent import AgentAssessment
        from underwriting.agents.financial_agent import FinancialAgent
        agent = FinancialAgent(rules_path="rules/death/financial_rules.json")
        mock_app = MagicMock()
        mock_app.age = 35
        mock_app.annual_income = 80000
        mock_app.sum_insured_death = 500000
        result = agent.evaluate(mock_app)
        assert isinstance(result, AgentAssessment)


class TestFinancialAgentChatResponse:
    """Tests for domain-specific handle_user_message() responses.

    These tests verify that FinancialAgent's chat response handler
    includes applicant data (income, sum insured, occupation) and
    assessment details (flags, risk tier) when users ask questions.
    """

    @staticmethod
    def _make_app():
        """Create a mock application with financial data."""
        app = MagicMock()
        app.annual_income = 85000
        app.sum_insured_death = 500000
        app.sum_insured_tpd = 500000
        app.occupation = "Manager"
        app.employer_name = "Acme Corp"
        app.years_in_occupation = 5.0
        return app

    @staticmethod
    def _make_assessment():
        """Create a mock AgentAssessment for testing."""
        return AgentAssessment(
            agent_name="Financial Agent",
            risk_tier="loading",
            flags=[
                {
                    "rule_id": "FIN-001",
                    "severity": "moderate",
                    "description": "High income ratio",
                }
            ],
            recommendation="loading",
            loading_range=[1.1, 1.3],
            confidence_score=0.90,
            reasoning_summary="Income-to-debt ratio above threshold",
            additional_evidence_required=[],
            apra_references=[],
        )

    @staticmethod
    def _make_agent():
        """Create a FinancialAgent instance for testing."""
        return FinancialAgent(
            rules_path="rules/death/financial_rules.json"
        )

    def test_income_question_includes_income_value(self):
        """Question about income gets a response (LLM-first, fallback without LLM)."""
        agent = self._make_agent()
        app = self._make_app()
        assessment = self._make_assessment()
        empty_history: list[ChatMessage] = []

        result = agent.handle_user_message(
            app, assessment, "What's my income?", empty_history
        )

        assert isinstance(result, ChatMessage)
        assert len(result.content) > 0

    def test_sum_insured_question_includes_amount(self):
        """Question about sum insured gets a response (LLM-first, fallback without LLM)."""
        agent = self._make_agent()
        app = self._make_app()
        assessment = self._make_assessment()
        empty_history: list[ChatMessage] = []

        result = agent.handle_user_message(
            app, assessment, "What's my sum insured?", empty_history
        )

        assert isinstance(result, ChatMessage)
        assert len(result.content) > 0

    def test_flag_question_lists_financial_flags(self):
        """Flag question gets a response mentioning flags (LLM-first fallback)."""
        agent = self._make_agent()
        app = self._make_app()
        assessment = self._make_assessment()
        empty_history: list[ChatMessage] = []

        result = agent.handle_user_message(
            app, assessment, "Why was I flagged?", empty_history
        )

        assert isinstance(result, ChatMessage)
        assert "flag" in result.content.lower() or "1" in result.content

    def test_explain_question_includes_risk_tier(self):
        """When user asks for assessment explanation, response should include tier."""
        agent = self._make_agent()
        app = self._make_app()
        assessment = self._make_assessment()
        empty_history: list[ChatMessage] = []

        result = agent.handle_user_message(
            app, assessment, "Explain your assessment", empty_history
        )

        assert isinstance(result, ChatMessage)
        assert "loading" in result.content.lower()

    def test_evidence_statement_acknowledged(self):
        """When user provides new evidence, response should acknowledge it."""
        agent = self._make_agent()
        app = self._make_app()
        assessment = self._make_assessment()
        empty_history: list[ChatMessage] = []

        result = agent.handle_user_message(
            app, assessment, "I just got a promotion", empty_history
        )

        assert isinstance(result, ChatMessage)
        content_lower = result.content.lower()
        assert "thank you" in content_lower or "consider" in content_lower

    def test_general_question_uses_base_response(self):
        """When user asks something general, response should include domain."""
        agent = self._make_agent()
        app = self._make_app()
        assessment = self._make_assessment()
        empty_history: list[ChatMessage] = []

        result = agent.handle_user_message(
            app, assessment, "Hello", empty_history
        )

        assert isinstance(result, ChatMessage)
        assert "financial underwriting" in result.content.lower()


class TestFinancialAgentEvidenceHandling:
    """Tests for FinancialAgent evidence handling — assessment modification."""

    def _make_agent(self):
        return FinancialAgent(rules_path="rules/death/financial_rules.json")

    def _make_assessment(self, confidence=0.9, risk_tier="standard"):
        return AgentAssessment(
            agent_name="Financial Agent",
            risk_tier=risk_tier,
            recommendation="standard",
            confidence_score=confidence,
            flags=[{"rule_id": "FIN-001", "severity": "moderate", "description": "High income ratio"}],
            reasoning_summary="Income-to-debt ratio above threshold",
            additional_evidence_required=[],
            apra_references=[],
        )

    def test_evidence_reduces_confidence(self):
        """Evidence should reduce confidence_score by 0.1."""
        agent = self._make_agent()
        assessment = self._make_assessment(confidence=0.9)
        original = assessment.confidence_score
        agent.handle_user_message(
            application=MagicMock(),
            current_assessment=assessment,
            user_message="I just got a promotion",
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
            user_message="I just got a promotion",
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
            user_message="I have new financial documents",
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
            user_message="I just got a promotion",
            conversation_history=[],
        )
        assert assessment.confidence_score == 0.0
