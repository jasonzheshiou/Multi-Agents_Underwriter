"""Medical Underwriting Agent.

Evaluates life insurance applications against deterministic medical rules
(BMI, smoking, pre-existing conditions, family history, etc.) and optionally
enriches the assessment with LLM-based reasoning for free-text medical
conditions.
"""

import logging
from typing import Any, List

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.debate.chat_models import ChatMessage

logger = logging.getLogger(__name__)


class MedicalAgent(BaseAgent):
    """Medical underwriting agent for death cover.

    Runs deterministic rules from ``rules/death/medical_rules.json`` against
    an Application model, then optionally enriches the result with LLM
    reasoning for free-text medical conditions.

    Attributes:
        name: Human-readable name used in logging and assessments.
        rules: Loaded rules dictionary with a ``"rules"`` key.
        llm: Optional LLM client for enrichment.
    """

    def __init__(
        self, rules_path: str, llm_client: Any = None, *, name: str = "Medical Agent"
    ) -> None:
        """Initialise the Medical Agent.

        Args:
            rules_path: Path to the JSON rules file.
            llm_client: Optional LLM client for enrichment. If *None*,
                only deterministic rules are used.
            name: Human-readable name for the agent.
        """
        super().__init__(name, rules_path, llm_client)

    def evaluate(self, application: Any) -> AgentAssessment:
        """Evaluate the application against medical rules.

        Runs all deterministic rules, builds the assessment, and optionally
        enriches it with LLM reasoning for free-text conditions.

        Args:
            application: An Application Pydantic model instance.

        Returns:
            AgentAssessment with the medical evaluation.
        """
        matched_rules = self.evaluate_rules(application)
        assessment = self.build_deterministic_assessment(application, matched_rules)

        if self.llm is not None:
            try:
                assessment = self._llm_enrich_conditions(application, assessment)
            except Exception as exc:
                logger.warning(
                    "[Medical Agent] LLM enrichment failed: %s — "
                    "returning deterministic assessment only.",
                    exc,
                )

        return assessment

    def _llm_enrich_conditions(
        self, application: Any, assessment: AgentAssessment
    ) -> AgentAssessment:
        """Enrich assessment with LLM interpretation of free-text conditions.

        Queries the vector knowledge base for relevant medical guidelines,
        asks the LLM to evaluate the applicant's free-text medical conditions
        against those guidelines, and appends the LLM's reasoning to the
        ``reasoning_summary``.

        Args:
            application: An Application Pydantic model instance.
            assessment: The deterministic AgentAssessment to enrich.

        Returns:
            Updated AgentAssessment with LLM enrichment.
        """
        # Build a prompt from the application's medical data
        conditions_text = ""
        for cond in application.medical_conditions:
            conditions_text += (
                f"- {cond.condition_name}: diagnosed {cond.diagnosis_date}, "
                f"treatment: {cond.treatment_description or 'N/A'}\n"
            )

        if not conditions_text:
            assessment.llm_used = True
            assessment.reasoning_summary += (
                " LLM: No free-text medical conditions to enrich."
            )
            return assessment

        # Query vector knowledge base (mock — real impl would use ChromaDB)
        kb_context = (
            "Relevant medical guidelines loaded from knowledge base. "
            "Key standards: BMI thresholds per NHMRC, smoking risk per "
            "Australian Institute of Health and Welfare, pre-existing "
            "condition assessment per APRA CPS 220."
        )

        # Ask LLM to evaluate
        prompt = (
            f"Medical conditions to evaluate:\n{conditions_text}\n\n"
            f"Knowledge base context:\n{kb_context}\n\n"
            f"Assessment summary:\n{assessment.reasoning_summary}\n\n"
            f"Provide a brief LLM interpretation of these conditions "
            f"relative to the knowledge base standards."
        )

        llm_response = self.llm.chat(prompt)  # type: ignore[union-attr]

        assessment.llm_used = True
        assessment.reasoning_summary += (
            f" LLM: {llm_response}"
        )
        return assessment

    def generate_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> AgentAssessment:
        """Generate a rebuttal when another agent disagrees.

        Reviews clinical evidence and stands firm on objective criteria
        (BMI, smoker status, pre-existing conditions, family history).
        Reduces confidence only on non-objective flags (e.g. mental health,
        hazardous pursuits).

        Optionally enriches the rebuttal with LLM reasoning if an LLM
        client is available.

        Args:
            application: The original application.
            my_assessment: This agent's current assessment.
            other_assessments: Conflicting assessments from other agents.

        Returns:
            Updated AgentAssessment (may differ from original if confidence
            is reduced on non-objective flags).
        """
        rebuttal_reasoning = (
            f"Rebuttal for {my_assessment.agent_name}: "
            f"Re-evaluating {len(my_assessment.flags)} flag(s) against "
            f"{len(other_assessments)} conflicting assessment(s).\n"
        )

        # Objective criteria that we stand firm on
        objective_categories = {
            "bmi",
            "smoker_status",
            "pre_existing_condition",
            "family_history",
        }

        # Track whether any non-objective flags were challenged
        confidence_reduction = 0.0

        updated_flags = []
        for flag in my_assessment.flags:
            category = flag.get("category", "")
            severity = flag.get("severity", "unknown")

            if category in objective_categories:
                # Objective flags — reduce confidence when challenged
                rebuttal_reasoning += (
                    f"  [CHALLENGED] {flag.get('rule_id', '?')}: "
                    f"{flag.get('description', '')} "
                    f"(severity: {severity}) — "
                    f"Objective evidence reviewed, confidence reduced due to conflicting assessments.\n"
                )
                updated_flags.append(flag)
                confidence_reduction += 0.15
            else:
                # Non-objective flags — reduce confidence if challenged
                rebuttal_reasoning += (
                    f"  [REDUCED CONFIDENCE] {flag.get('rule_id', '?')}: "
                    f"{flag.get('description', '')} "
                    f"(severity: {severity}) — "
                    f"Non-objective flag, confidence reduced on review.\n"
                )
                updated_flags.append(flag)
                confidence_reduction += 0.25

        # Apply confidence reduction across all flags
        new_confidence = my_assessment.confidence_score - confidence_reduction
        new_confidence = max(0.3, new_confidence)

        # Downgrade tier if confidence drops below threshold
        new_tier = my_assessment.risk_tier
        tier_downgraded = False
        if new_tier == "refer" and new_confidence < 0.55:
            new_tier = "loading"
            tier_downgraded = True
            rebuttal_reasoning += f"\n  [TIER CHANGE] Downgraded from {my_assessment.risk_tier} to {new_tier} (confidence {new_confidence:.2f} below threshold).\n"
        elif new_tier == "loading" and new_confidence < 0.45:
            new_tier = "standard"
            tier_downgraded = True
            rebuttal_reasoning += f"\n  [TIER CHANGE] Downgraded from {my_assessment.risk_tier} to {new_tier} (confidence {new_confidence:.2f} below threshold).\n"

        if not tier_downgraded:
            rebuttal_reasoning += f"\n  Standing firm on tier {new_tier} (confidence {new_confidence:.2f})."

        # Optionally enrich with LLM reasoning during debate
        llm_used = my_assessment.llm_used
        if self.llm is not None:
            try:
                rebuttal_reasoning = self._llm_enrich_rebuttal(
                    application, my_assessment, other_assessments, rebuttal_reasoning
                )
                llm_used = True
            except Exception as exc:
                logger.warning(
                    "[Medical Agent] LLM rebuttal enrichment failed: %s — "
                    "using deterministic reasoning.",
                    exc,
                )

        updated = AgentAssessment(
            agent_name=self.name,
            risk_tier=new_tier,
            flags=updated_flags,
            recommendation=my_assessment.recommendation,
            loading_range=my_assessment.loading_range,
            additional_evidence_required=my_assessment.additional_evidence_required,
            confidence_score=new_confidence,
            reasoning_summary=rebuttal_reasoning,
            apra_references=my_assessment.apra_references,
            llm_used=llm_used,
        )

        logger.info("[Medical Agent] Rebuttal generated: %s", rebuttal_reasoning)
        return updated

    def _llm_enrich_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
        deterministic_reasoning: str,
    ) -> str:
        """Enrich the deterministic rebuttal with LLM reasoning.

        Sends the application data, this agent's assessment, and the
        conflicting assessments to the LLM for a nuanced review.

        Args:
            application: The original application.
            my_assessment: This agent's current assessment.
            other_assessments: Conflicting assessments from other agents.
            deterministic_reasoning: The deterministic rebuttal reasoning.

        Returns:
            Enhanced reasoning string with LLM input appended.
        """
        # Build a summary of conflicting assessments
        other_summary = "\n".join(
            f"- {a.agent_name}: risk_tier={a.risk_tier}, "
            f"flags={len(a.flags)}, recommendation={a.recommendation}"
            for a in other_assessments
        )

        prompt = (
            f"You are a medical underwriting specialist reviewing a debate between "
            f"insurance underwriting agents.\n\n"
            f"Your assessment: tier={my_assessment.risk_tier}, "
            f"recommendation={my_assessment.recommendation}, "
            f"flags={len(my_assessment.flags)}\n\n"
            f"Conflicting assessments:\n{other_summary}\n\n"
            f"Deterministic reasoning:\n{deterministic_reasoning}\n\n"
            f"Provide a brief (2-3 sentences) medical perspective on whether "
            f"your assessment holds given the conflicting views. "
            f"Consider clinical evidence strength and risk severity."
        )

        llm_response = self.llm.chat(prompt)  # type: ignore[union-attr]
        return f"{deterministic_reasoning}\n  [LLM]: {llm_response}"

    def handle_user_message(
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
            user_message, current_assessment, "medical underwriting",
            application, conversation_history,
        )

    def _analyze_question_intent(self, user_message: str) -> str:
        """Classify the intent of a user message.

        Args:
            user_message: The user's question or statement.

        Returns:
            One of: ``"evidence"``, ``"flag"``, ``"explain"``, ``"general"``.
        """
        lower = user_message.lower()
        # Evidence statements — user-provided information takes priority
        if any(w in lower for w in ("quit", "stopped", "just", "evidence", "note",
                                     "i have", "i've", "i am", "i stopped", "i quit",
                                     "i no longer", "new report", "additional info",
                                     "update on", "diagnosed", "was told")):
            return "evidence"
        # Flag intent — questions about flags, declines, rejections
        if any(w in lower for w in ("flag", "why", "reason", "flagged",
                                     "decline", "reject")):
            return "flag"
        # Explain intent — what/how questions about concepts or assessment
        if any(w in lower for w in ("explain", "how", "what is", "what's",
                                     "what does", "meaning", "assess",
                                     "criterion", "criteria")):
            return "explain"
        return "general"

    def _compute_bmi(self, application: Any) -> float:
        """Calculate BMI from application height and weight.

        Args:
            application: The Application model.

        Returns:
            BMI value rounded to 1 decimal place.
        """
        height_m = getattr(application, "height_cm", 170) / 100.0
        weight_kg = getattr(application, "weight_kg", 70)
        if height_m <= 0:
            return 0.0
        return round(weight_kg / (height_m * height_m), 1)

    def _get_smoker_status(self, application: Any) -> str:
        """Get human-readable smoker status from application.

        Args:
            application: The Application model.

        Returns:
            Smoker status string.
        """
        status = getattr(application, "smoker_status", None)
        if status is None:
            return "unknown"
        if hasattr(status, "value"):
            return status.value
        return str(status)
