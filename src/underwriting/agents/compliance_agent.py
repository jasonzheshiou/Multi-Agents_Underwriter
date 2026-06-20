"""Compliance Underwriting Agent — regulatory compliance checks.

The ComplianceAgent evaluates insurance applications against a set of
deterministic compliance rules (duty of disclosure, anti-discrimination,
data privacy, vulnerable customer handling, etc.) and optionally enriches
the deterministic assessment with LLM-based cross-referencing against
the organisation's vector store of regulatory guidance.
"""

import logging
from typing import Any, Dict, List

from underwriting.agents.base_agent import AgentAssessment, BaseAgent
from underwriting.debate.chat_models import ChatMessage

logger = logging.getLogger(__name__)


class ComplianceAgent(BaseAgent):
    """Monitor regulatory compliance — observer only, NOT a decision-maker.

    The Compliance Agent evaluates applications against deterministic
    compliance rules covering disclosure, documentation, anti-discrimination,
    data privacy, vulnerable customer handling, AI governance, and other
    Australian regulatory requirements.

    **IMPORTANT**: The Compliance Agent is an **observer/informer**.
    It spots potential compliance gaps but does NOT:
    - Influence the final underwriting decision
    - Trigger or participate in underwriting disputes
    - Escalate its risk tier to override underwriting assessments

    Its output is informational — compliance observations are displayed
    separately from the underwriting recommendation.

    Attributes:
        name: Human-readable agent name (default: "Compliance Agent").
        rules: Dict with a "rules" key containing rule objects loaded from JSON.
        llm: Optional LLM client for enrichment.
    """

    def __init__(
        self, rules_path: str, llm_client: Any = None, *, name: str = "Compliance Agent"
    ) -> None:
        """Initialise the Compliance Agent.

        Args:
            rules_path: Path to the JSON compliance rules file.
            llm_client: Optional LLM client for enrichment. If None,
                deterministic evaluation only.
            name: Human-readable name for the agent.
        """
        super().__init__(name, rules_path, llm_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, application: Any) -> AgentAssessment:
        """Evaluate the application for regulatory compliance.

        Checks the following rule categories:
        - Duty of disclosure (CMP-D-001)
        - Decision documentation (CMP-D-002)
        - Decision timeframe (CMP-D-003)
        - Plain language requirement (CMP-D-004)
        - Mental health assessment (CMP-D-005)
        - Risk management framework (CMP-D-006)
        - Anti-discrimination (CMP-D-007)
        - Vulnerable customer handling (CMP-D-010)
        - Data privacy (CMP-D-020)

        Args:
            application: An Application Pydantic model instance or
                compatible object with attributes matching rule conditions.

        Returns:
            AgentAssessment with compliance flags, risk tier, and
            regulatory references.
        """
        logger.info(f"[{self.name}] Evaluating compliance for application")

        matched_rules = self.evaluate_rules(application)
        assessment = self.build_deterministic_assessment(application, matched_rules)

        # LLM enrichment if a client is available
        if self.llm is not None:
            assessment = self._llm_enrich_compliance(application, assessment)

        return assessment

    def generate_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> AgentAssessment:
        """Review compliance observations when other agents disagree.

        The Compliance Agent is an **observer only** — it does NOT escalate
        its risk tier or attempt to influence the underwriting decision.
        During debate, it reviews other agents' assessments for any
        compliance-relevant flags it may have missed, but it only adds
        them as informational observations.

        Args:
            application: The original application being evaluated.
            my_assessment: This agent's current compliance assessment.
            other_assessments: Assessments from underwriting agents.

        Returns:
            Updated AgentAssessment with new compliance observations
            but an UNCHANGED risk tier (observer only).
        """
        logger.info(
            f"[{self.name}] Reviewing debate — "
            f"my flags={len(my_assessment.flags)}, "
            f"other agents={len(other_assessments)}"
        )

        # Collect any compliance-relevant flags from underwriting agents
        # that we may have missed — compliance-only categories.
        compliance_categories = {
            "disclosure", "documentation", "process", "communication",
            "assessment", "risk_management", "fairness",
            "consumer_protection", "governance", "privacy",
            "operational_resilience", "ai_governance", "ai_ethics",
            "data_security",
        }
        additional_flags: List[Dict[str, str]] = []
        for oa in other_assessments:
            for flag in oa.flags:
                category = flag.get("category", "")
                if category in compliance_categories:
                    if not any(
                        f["rule_id"] == flag["rule_id"]
                        for f in additional_flags
                    ):
                        additional_flags.append(flag)

        llm_used = my_assessment.llm_used

        if additional_flags:
            existing_flag_ids = {f["rule_id"] for f in my_assessment.flags}
            new_flags = [
                f for f in additional_flags
                if f["rule_id"] not in existing_flag_ids
            ]
            if new_flags:
                updated_flags = my_assessment.flags + new_flags
                reasoning = (
                    f"Rebuttal review: noted {len(new_flags)} "
                    f"additional compliance-relevant observation(s) "
                    f"from underwriting agent assessments: "
                    + ", ".join(f["rule_id"] for f in new_flags)
                    + ". (Informational only — does not affect underwriting decision.)"
                )

                if self.llm is not None:
                    try:
                        reasoning = self._llm_enrich_rebuttal(
                            application, my_assessment, other_assessments, reasoning
                        )
                        llm_used = True
                    except Exception as exc:
                        logger.warning(
                            f"[{self.name}] LLM rebuttal enrichment failed: {exc}."
                        )

                # Return with additional flags but UNCHANGED risk tier
                # (Compliance is an observer, not a decision-maker)
                return AgentAssessment(
                    agent_name=self.name,
                    risk_tier=my_assessment.risk_tier,  # UNCHANGED
                    flags=updated_flags,
                    recommendation=my_assessment.recommendation,  # UNCHANGED
                    loading_range=my_assessment.loading_range,
                    additional_evidence_required=my_assessment.additional_evidence_required,
                    confidence_score=my_assessment.confidence_score,
                    reasoning_summary=reasoning,
                    apra_references=my_assessment.apra_references,
                    llm_used=llm_used,
                )

        # No new compliance observations — stand firm.
        logger.info(f"[{self.name}] No new compliance observations — standing firm")

        if self.llm is not None:
            try:
                my_assessment.reasoning_summary = self._llm_enrich_rebuttal(
                    application, my_assessment, other_assessments,
                    my_assessment.reasoning_summary,
                )
                llm_used = True
            except Exception as exc:
                logger.warning(
                    f"[{self.name}] LLM rebuttal enrichment failed: {exc}."
                )

        my_assessment.llm_used = llm_used
        return my_assessment

    def _llm_enrich_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
        deterministic_reasoning: str,
    ) -> str:
        """Enrich the deterministic rebuttal with LLM reasoning.

        Args:
            application: The original application.
            my_assessment: This agent's current assessment.
            other_assessments: Conflicting assessments from other agents.
            deterministic_reasoning: The deterministic rebuttal reasoning.

        Returns:
            Enhanced reasoning string with LLM input appended.
        """
        other_summary = "\n".join(
            f"- {a.agent_name}: risk_tier={a.risk_tier}, "
            f"flags={len(a.flags)}, recommendation={a.recommendation}"
            for a in other_assessments
        )

        prompt = (
            f"You are a regulatory compliance specialist reviewing a debate "
            f"between insurance underwriting agents.\n\n"
            f"Your assessment: tier={my_assessment.risk_tier}, "
            f"recommendation={my_assessment.recommendation}\n\n"
            f"Conflicting assessments:\n{other_summary}\n\n"
            f"Deterministic reasoning:\n{deterministic_reasoning}\n\n"
            f"Provide a brief (2-3 sentences) compliance perspective on whether "
            f"your assessment holds given the conflicting views. "
            f"Reference relevant regulatory principles."
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
            return f"{deterministic_reasoning} [LLM]: {content}"
        return deterministic_reasoning

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
            user_message, current_assessment, "compliance review",
            application, conversation_history,
        )

    def _analyze_question_intent(self, user_message: str) -> str:
        """Classify the user's message into an intent category.

        Extends the base class with compliance-domain keywords
        (APRA, regulatory, compliance, rule).

        Args:
            user_message: The user's input text.

        Returns:
            One of: "flag", "explain", "evidence", "general"
        """
        msg = user_message.lower().strip()

        # Flag keywords take priority over all else
        flag_keywords = ["why", "flagged", "flag", "decline", "reject"]
        if any(kw in msg for kw in flag_keywords):
            return "flag"

        # Compliance-specific explain keywords
        compliance_explain = [
            "apra", "regulatory", "compliance", "rule id",
            "rule_id", "which rules", "applies to me",
        ]
        if any(kw in msg for kw in compliance_explain):
            return "explain"

        return super()._analyze_question_intent(user_message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_enrich_compliance(
        self,
        application: Any,
        assessment: AgentAssessment,
    ) -> AgentAssessment:
        """Cross-reference the assessment with the vector store via LLM.

        Sends the application summary and current assessment to the LLM
        for cross-referencing against regulatory guidance stored in the
        vector store. Gracefully handles LLM failures by returning the
        original deterministic assessment.

        Args:
            application: The application being evaluated.
            assessment: The current deterministic AgentAssessment.

        Returns:
            Updated AgentAssessment enriched with LLM findings, or the
            original assessment if enrichment fails.
        """
        try:
            # Build a concise summary for the LLM
            summary = {
                "risk_tier": assessment.risk_tier,
                "flags": assessment.flags,
                "reasoning_summary": assessment.reasoning_summary,
            }

            response = self.llm.chat_completion(
                messages=[
                    {
                        "role": "system",
                            "content": (
                                "You are a compliance analyst reviewing an "
                                "underwriting assessment for regulatory gaps. "
                                "Focus ONLY on compliance/regulatory issues — "
                                "do NOT assess or modify the underwriting "
                                "risk tier or medical/financial flags. "
                                "Return findings as JSON with keys: "
                                "flags (compliance only), "
                                "additional_evidence_required, "
                                "reasoning_summary."
                            ),
                    },
                    {"role": "user", "content": str(summary)},
                ],
            )

            # Parse the LLM response
            if isinstance(response, dict):
                content = response.get("choices", [{}])[0].get(
                    "message", {}
                ).get("content", "{}")
                import json as _json
                enriched = _json.loads(content)
            else:
                logger.warning(
                    f"[{self.name}] LLM returned non-dict response, "
                    "skipping enrichment."
                )
                return assessment

            # Apply enrichment if the LLM found new compliance flags
            if enriched.get("flags"):
                existing_ids = {f["rule_id"] for f in assessment.flags}
                new_flags = [
                    f for f in enriched["flags"]
                    if f.get("rule_id") not in existing_ids
                    and f.get("rule_id", "").startswith("CMP-")
                ]
                assessment.flags.extend(new_flags)

            if enriched.get("additional_evidence_required"):
                for ev in enriched["additional_evidence_required"]:
                    if ev not in assessment.additional_evidence_required:
                        assessment.additional_evidence_required.append(ev)

            if enriched.get("reasoning_summary"):
                assessment.reasoning_summary += (
                    " [LLM: " + enriched["reasoning_summary"] + "]"
                )

            assessment.llm_used = True
            assessment.confidence_score = min(
                assessment.confidence_score,
                enriched.get("confidence_score", assessment.confidence_score),
            )

            logger.info(
                f"[{self.name}] LLM enrichment complete — "
                f"{len(assessment.flags)} total flags"
            )

        except Exception as exc:
            logger.warning(
                f"[{self.name}] LLM enrichment failed: {exc}. "
                "Returning deterministic assessment."
            )

        return assessment
