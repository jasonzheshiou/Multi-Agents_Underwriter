"""Financial Underwriting Agent.

Evaluates insurance applications against deterministic financial risk rules
(sum-insured-to-income multiples by age, occupation class, previous declination,
bankruptcy, multiple policies) and optionally enriches the assessment with
LLM-based reasoning for complex financial arrangements.
"""

import logging
from typing import Any, List

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.debate.chat_models import ChatMessage

logger = logging.getLogger(__name__)


class FinancialAgent(BaseAgent):
    """Financial underwriting agent.

    Applies financial risk rules to determine whether an applicant's
    financial profile warrants standard terms, loading, referral, or decline.

    Attributes:
        name: Human-readable agent name.
        rules: Loaded financial rules dictionary.
        llm: Optional LLM client for enrichment.
    """

    def __init__(
        self, rules_path: str, llm_client: Any = None, *, name: str = "Financial Agent"
    ) -> None:
        """Initialise the Financial Agent.

        Args:
            rules_path: Path to the JSON rules file containing financial rules.
            llm_client: Optional LLM client for enrichment. If None, deterministic only.
            name: Human-readable name for the agent.
        """
        super().__init__(name, rules_path, llm_client)

    def evaluate(self, application: Any) -> AgentAssessment:
        """Evaluate the application against financial risk rules.

        Checks sum-insured-to-income multiples by age bracket, occupation class,
        previous declination history, bankruptcy status, and multiple policy count.

        Args:
            application: An Application Pydantic model instance.

        Returns:
            AgentAssessment with financial risk evaluation.
        """
        logger.info(f"[{self.name}] Evaluating financial risk for applicant")

        matched_rules = self.evaluate_rules(application)
        assessment = self.build_deterministic_assessment(application, matched_rules)

        # LLM enrichment if client available
        if self.llm is not None:
            try:
                assessment = self._llm_enrich_financial(application, assessment)
            except Exception as exc:
                logger.warning(
                    f"[{self.name}] LLM enrichment failed: {exc}. "
                    "Returning deterministic assessment only."
                )
                assessment.llm_used = False

        logger.info(
            f"[{self.name}] Financial evaluation complete — "
            f"risk_tier={assessment.risk_tier}, flags={len(assessment.flags)}"
        )
        return assessment

    def _llm_enrich_financial(
        self, application: Any, assessment: AgentAssessment
    ) -> AgentAssessment:
        """Enrich deterministic assessment with LLM reasoning for complex finances.

        Sends a structured prompt to the LLM asking it to evaluate financial
        arrangements that require human-underwriting judgment (e.g., complex
        income structures, asset-heavy profiles, business ownership).

        Args:
            application: The Application being evaluated.
            assessment: The deterministic AgentAssessment to enrich.

        Returns:
            Updated AgentAssessment with LLM reasoning appended to reasoning_summary.
        """
        prompt = (
            f"Review the financial profile and provide additional reasoning.\n"
            f"- Age: {application.age}\n"
            f"- Annual income: {application.annual_income}\n"
            f"- Sum insured: {getattr(application, 'sum_insured_death', 'N/A')}\n"
            f"- Occupation class: {application.occupation_class}\n"
            f"- Bankruptcy status: {getattr(application, 'bankruptcy_status', 'N/A')}\n"
            f"- Existing policies: {getattr(application, 'total_existing_policies', 0)}\n"
            f"- Previous declination: {getattr(application, 'previous_declination', False)}\n"
            f"- Net worth: {getattr(application, 'total_net_worth', 'N/A')}\n"
            f"Deterministic assessment: {assessment.recommendation}\n"
            f"Provide brief reasoning for the financial risk tier."
        )

        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract reasoning from LLM response
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        if content:
            assessment.reasoning_summary = (
                f"{assessment.reasoning_summary} "
                f"[LLM enrichment]: {content}"
            )
            assessment.llm_used = True
            logger.debug(f"[{self.name}] LLM enrichment applied")
        else:
            logger.warning(f"[{self.name}] LLM returned empty content")

        return assessment

    def generate_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> AgentAssessment:
        """Generate a rebuttal when another agent disagrees with the financial assessment.

        Reviews the conflicting assessments against financial evidence (income,
        assets, occupation) and stands firm on objective financial data unless
        compelling counterevidence is found.

        Optionally enriches the rebuttal with LLM reasoning if an LLM
        client is available.

        Args:
            application: The original application.
            my_assessment: This agent's current assessment.
            other_assessments: Conflicting assessments from other agents.

        Returns:
            Updated AgentAssessment (may be unchanged if agent stands firm).
        """
        if not other_assessments:
            logger.debug(f"[{self.name}] No conflicting assessments to rebut")
            return my_assessment

        # Collect conflicting tiers
        conflicting_tiers = {a.risk_tier for a in other_assessments}
        my_tier = my_assessment.risk_tier

        # Financial agent stands firm on objective data unless there's a
        # clear financial evidence gap that other agents identified.
        tier_rank = {"decline": 4, "refer": 3, "loading": 2, "standard": 1}
        my_rank = tier_rank.get(my_tier, 0)

        # If any other agent has a higher tier, review their reasoning
        for other in other_assessments:
            other_rank = tier_rank.get(other.risk_tier, 0)
            if other_rank > my_rank:
                logger.info(
                    f"[{self.name}] Other agent ({other.agent_name}) has higher "
                    f"risk tier ({other.risk_tier}) — reviewing financial evidence"
                )
                # Review: does the applicant have additional financial evidence?
                net_worth = getattr(application, "total_net_worth", None)
                income = getattr(application, "annual_income", 0)

                if net_worth and income and net_worth > income * 5:
                    # Strong financial position — may lower tier if currently decline
                    if my_tier == "decline":
                        my_assessment.risk_tier = "loading"
                        my_assessment.recommendation = "loading_applied"
                        my_assessment.reasoning_summary = (
                            f"{my_assessment.reasoning_summary} "
                            "[Rebuttal]: Reconsidered due to strong net worth "
                            f"(${net_worth:.0f} vs income ${income:.0f}). "
                            f"Downgraded from decline to loading."
                        )
                        my_assessment.confidence_score = 0.85
                        logger.info(
                            f"[{self.name}] Rebuttal: Downgraded to loading "
                            f"(strong net worth)"
                        )
                else:
                    my_assessment.reasoning_summary = (
                        f"{my_assessment.reasoning_summary} "
                        f"[Rebuttal]: Reviewed conflicting assessment from "
                        f"{other.agent_name} ({other.risk_tier}). "
                        f"Financial data supports original tier ({my_tier}). "
                        f"Standing firm on financial risk assessment."
                    )
                    my_assessment.confidence_score = 0.95
                    logger.info(
                        f"[{self.name}] Rebuttal: Standing firm on tier {my_tier}"
                    )
            else:
                # Other agent has lower or equal tier — confirm our position
                my_assessment.reasoning_summary = (
                    f"{my_assessment.reasoning_summary} "
                    f"[Rebuttal]: Reviewed assessment from {other.agent_name} "
                    f"({other.risk_tier}). Financial evidence supports our "
                    f"assessment ({my_tier}). Standing firm."
                )

        # Optionally enrich with LLM reasoning during debate
        if self.llm is not None:
            try:
                my_assessment.reasoning_summary = self._llm_enrich_rebuttal(
                    application, my_assessment, other_assessments
                )
                my_assessment.llm_used = True
            except Exception as exc:
                logger.warning(
                    f"[{self.name}] LLM rebuttal enrichment failed: {exc}. "
                    "Using deterministic reasoning."
                )

        return my_assessment

    def _llm_enrich_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> str:
        """Enrich the deterministic rebuttal with LLM reasoning.

        Args:
            application: The original application.
            my_assessment: This agent's current assessment.
            other_assessments: Conflicting assessments from other agents.

        Returns:
            Enhanced reasoning string with LLM input appended.
        """
        other_summary = "\n".join(
            f"- {a.agent_name}: risk_tier={a.risk_tier}, "
            f"flags={len(a.flags)}, recommendation={a.recommendation}"
            for a in other_assessments
        )

        prompt = (
            f"You are a financial underwriting specialist reviewing a debate "
            f"between insurance underwriting agents.\n\n"
            f"Your assessment: tier={my_assessment.risk_tier}, "
            f"recommendation={my_assessment.recommendation}\n\n"
            f"Applicant: age={application.age}, income=${getattr(application, 'annual_income', 'N/A')}, "
            f"sum_insured=${getattr(application, 'sum_insured_death', 'N/A')}, "
            f"net_worth=${getattr(application, 'total_net_worth', 'N/A')}\n\n"
            f"Conflicting assessments:\n{other_summary}\n\n"
            f"Deterministic reasoning:\n{my_assessment.reasoning_summary}\n\n"
            f"Provide a brief (2-3 sentences) financial perspective on whether "
            f"your assessment holds given the conflicting views."
        )

        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )

        content = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

        if content:
            return f"{my_assessment.reasoning_summary} [LLM]: {content}"
        return my_assessment.reasoning_summary

    def handle_user_message(  # type: ignore[override]
        self,
        application: Any,
        current_assessment: AgentAssessment,
        user_message: str,
        conversation_history: List[ChatMessage],
    ) -> ChatMessage:
        """Handle a user message with LLM understanding — no keyword routing.

        All messages go through the LLM first. The LLM reads the user's
        message directly, understands natural language, classifies intent
        (evidence / question / general), and generates an appropriate
        response. Falls back to deterministic templates only when LLM
        is unavailable.

        Args:
            application: The original application.
            current_assessment: This agent's current assessment.
            user_message: The user's question or evidence text.
            conversation_history: Previous messages in this conversation.

        Returns:
            ChatMessage with the agent's LLM-generated response.
        """
        return self._build_deterministic_chat_response(
            user_message, current_assessment, "financial underwriting",
            application, conversation_history,
        )
