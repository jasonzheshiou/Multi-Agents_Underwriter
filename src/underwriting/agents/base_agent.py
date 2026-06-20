"""Abstract base class for all underwriting agents."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..debate.chat_models import ChatMessage
from ..llm.llm_client import FALLBACK_MESSAGE

# Configure structured logging
logger = logging.getLogger(__name__)


class AgentAssessment(BaseModel):
    """Standardised output from any underwriting agent.

    This model enforces a consistent structure so that the debate orchestrator
    can compare assessments across agents regardless of their domain.
    """

    agent_name: str
    risk_tier: str  # "standard", "loading", "decline", "refer"
    flags: List[Dict[str, str]] = Field(default_factory=list)
    # Each flag: {"rule_id": "...", "severity": "...", "description": "..."}
    recommendation: str
    loading_range: List[float] = Field(default_factory=lambda: [1.0, 1.0])
    additional_evidence_required: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning_summary: str = ""
    apra_references: List[str] = Field(default_factory=list)
    llm_used: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BaseAgent(ABC):
    """Abstract base class for all underwriting agents.

    All agents follow the same pattern:
    1. Load deterministic rules from JSON.
    2. Evaluate the application against those rules.
    3. Optionally enrich with LLM reasoning.
    4. Return a standardised AgentAssessment.

    This design ensures every agent is independently testable and replaceable.
    """

    def __init__(self, name: str, rules_path: str, llm_client=None):
        """Initialise the agent.

        Args:
            name: Human-readable agent name (e.g., "Medical Agent").
            rules_path: Path to the JSON rules file.
            llm_client: Optional LLM client for enrichment. If None, deterministic only.
        """
        self.name = name
        self.rules = self.load_rules(rules_path)
        self.llm = llm_client

    def load_rules(self, path: str) -> Dict[str, Any]:
        """Load deterministic rules from a JSON file.

        Args:
            path: Filesystem path to the rules JSON file.

        Returns:
            Dict with a "rules" key containing a list of rule objects.

        Raises:
            FileNotFoundError: If the rules file does not exist.
            json.JSONDecodeError: If the rules file is not valid JSON.
        """
        logger.info(f"[{self.name}] Loading rules from {path}")
        with open(path, "r") as f:
            return json.load(f)

    def evaluate_rules(self, application: Any) -> List[Dict[str, Any]]:
        """Evaluate all deterministic rules against the application.

        Each rule has a 'condition' field that is evaluated as a Python expression
        against the application object. Rules that match are collected and returned.

        Args:
            application: An Application Pydantic model instance.

        Returns:
            List of matched rule dictionaries.
        """
        # Safe builtins for eval — minimal set to prevent arbitrary code execution.
        _SAFE_BUILTINS: Dict[str, Any] = {
            "getattr": getattr,
            "len": len,
            "max": max,
            "min": min,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "filter": filter,
            "map": map,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "abs": abs,
            "round": round,
            "True": True,
            "False": False,
            "None": None,
        }

        matched = []
        for rule in self.rules.get("rules", []):
            condition = rule.get("condition", "False")
            try:
                # Safely evaluate condition against the application object.
                # The condition string uses 'applicant' as the variable name.
                if eval(condition, {"__builtins__": _SAFE_BUILTINS}, {"applicant": application}):
                    matched.append(rule)
                    logger.debug(
                        f"[{self.name}] Rule {rule['rule_id']} matched: "
                        f"{rule.get('description', '')}"
                    )
            except Exception as e:
                logger.error(
                    f"[{self.name}] Error evaluating rule {rule.get('rule_id', '?')}: {e}"
                )
        return matched

    def build_deterministic_assessment(
        self, application: Any, matched_rules: List[Dict[str, Any]]
    ) -> AgentAssessment:
        """Build an AgentAssessment from matched deterministic rules.

        The highest-severity matched rule determines the risk tier.
        If no rules match, the assessment defaults to 'standard'.

        Args:
            application: An Application Pydantic model instance.
            matched_rules: List of rules that matched the application.

        Returns:
            AgentAssessment with deterministic evaluation.
        """
        if not matched_rules:
            return AgentAssessment(
                agent_name=self.name,
                risk_tier="standard",
                recommendation="No risk factors identified. Standard terms.",
                reasoning_summary="All deterministic rules passed.",
            )

        # Priority: critical > high > moderate > low > none
        severity_order = {"critical": 5, "high": 4, "moderate": 3, "low": 2, "none": 1}
        highest = max(matched_rules, key=lambda r: severity_order.get(r.get("severity", "none"), 0))

        flags = [
            {
                "rule_id": r["rule_id"],
                "severity": r.get("severity", "unknown"),
                "description": r.get("description", ""),
                "category": r.get("category", ""),
            }
            for r in matched_rules
        ]

        loading_range = highest.get("loading_range", [1.0, 1.0])

        # Collect all additional evidence from matched rules (deduplicated)
        evidence = []
        for r in matched_rules:
            for e in r.get("additional_evidence", []):
                if e not in evidence:
                    evidence.append(e)

        apra_refs = list({r.get("apra_ref", "") for r in matched_rules if r.get("apra_ref")})

        return AgentAssessment(
            agent_name=self.name,
            risk_tier=self._determine_risk_tier(highest),
            flags=flags,
            recommendation=highest.get("recommendation", "standard"),
            loading_range=loading_range,
            additional_evidence_required=evidence,
            confidence_score=1.0,
            reasoning_summary=(
                f"Matched {len(matched_rules)} deterministic rule(s). "
                f"Highest severity: {highest.get('severity')} "
                f"({highest['rule_id']})."
            ),
            apra_references=apra_refs,
            llm_used=False,
        )

    def _analyze_question_intent(self, user_message: str) -> str:
        """Classify the user's message into an intent category.

        Intents:
            - "flag":   Questions about why a flag/decision was raised.
            - "explain": Requests for explanation of concepts or assessment.
            - "evidence": User providing new evidence or information.
            - "general": Everything else (greetings, small talk, etc.).

        Args:
            user_message: The user's input text.

        Returns:
            One of: "flag", "explain", "evidence", "general"
        """
        msg = user_message.lower().strip()

        if not msg:
            return "general"

        # Flag intent — questions about flags, declines, rejections
        flag_keywords = ["why", "flagged", "flag", "decline", "reject"]
        if any(kw in msg for kw in flag_keywords):
            return "flag"

        # Explain intent — what/how/why questions about concepts or assessment
        explain_keywords = ["explain", "what is", "what's", "how does", "how is",
                            "meaning", "assess", "criterion", "criteria"]
        if any(kw in msg for kw in explain_keywords):
            return "explain"

        # Evidence intent — user providing information (starts with "I ")
        evidence_keywords = ["i have", "i just", "i've", "i quit", "i no longer",
                             "i stopped", "i was diagnosed", "i was told",
                             "new report", "additional info", "update on"]
        if any(kw in msg for kw in evidence_keywords):
            return "evidence"

        return "general"

    def _build_chat_prompt(
        self,
        user_message: str,
        current_assessment: AgentAssessment,
        domain_description: str,
        application: Any = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> str:
        """Build an LLM prompt for chat-based agent interaction.

        Includes application context, current assessment, conversation history,
        and user message in a structured prompt with JSON response format instructions.

        Args:
            user_message: The user's input text.
            current_assessment: The agent's current assessment.
            domain_description: Description of the agent's domain.
            application: The Application model (optional).
            conversation_history: Previous messages (optional).

        Returns:
            Formatted prompt string for the LLM.
        """
        conversation_history = conversation_history or []

        lines = [
            f"You are a {domain_description} specialist for an Australian life insurance company.",
            "",
            "Provide your response in valid JSON format only.",
            "",
            "### Applicant Information",
        ]

        if application is not None:
            lines.append(f"- Age: {getattr(application, 'age', 'N/A')}")
            lines.append(f"- Smoker status: {getattr(application, 'smoker_status', 'N/A')}")
            lines.append(f"- BMI: {getattr(application, 'bmi', 'N/A')}")
            lines.append(f"- Annual income: {getattr(application, 'annual_income', 'N/A')}")
            lines.append(f"- Occupation: {getattr(application, 'occupation', 'N/A')}")
            lines.append(f"- Sum insured: {getattr(application, 'sum_insured_death', 'N/A')}")
            conditions = getattr(application, 'medical_conditions', None)
            if conditions:
                for cond in conditions:
                    name = getattr(cond, 'condition_name', 'Unknown')
                    lines.append(f"- Medical condition: {name}")
        else:
            lines.append("- No application data available")

        lines.extend([
            "",
            "### Current Assessment",
            f"- Risk tier: {current_assessment.risk_tier}",
            f"- Recommendation: {current_assessment.recommendation}",
            f"- Confidence score: {current_assessment.confidence_score}",
            f"- Loading range: {current_assessment.loading_range}",
            f"- Flags: {json.dumps(current_assessment.flags)}",
            f"- Reasoning: {current_assessment.reasoning_summary}",
            "",
        ])

        if conversation_history:
            lines.extend(["### Conversation History"])
            for msg in conversation_history[-5:]:
                if msg.sender == "user":
                    lines.append(f"User: {msg.content}")
                else:
                    lines.append(f"{msg.sender}: {msg.content}")
            lines.append("")

        lines.extend([
            "### User Message",
            user_message,
            "",
            "### Instructions",
            "Read the user's message carefully and understand their intent. Respond in valid JSON format only.",
            "",
            "### Response Format",
            """{
  "response_text": "Your natural language response to the user (required)",
  "risk_tier_update": "standard|loading|refer|decline" (only include if evidence changes your assessment),
  "recommendation_update": "Updated recommendation text" (only include if recommendation changes),
  "confidence_update": 0.0-1.0 (only include if evidence changes your confidence),
  "flags_to_remove": ["RULE-ID"] (only include if evidence contradicts specific flags),
  "flags_to_add": [{"rule_id":"X","severity":"low|moderate|high|critical","description":"..."}] (only include if new flags needed),
  "reasoning": "Updated reasoning summary" (only include if assessment changes)
}

Guidelines:
- If the user provides new information or evidence (e.g., specialist report, test results,
  doctor confirmation, updated health status, changed circumstances): acknowledge the
  specific evidence they mentioned, re-evaluate their assessment in light of it, and
  include appropriate assessment updates (risk_tier_update, recommendation_update,
  flags_to_remove, etc.).
- If the user asks a question: answer it clearly and professionally using the applicant
  data and current assessment as context. Do NOT change the assessment.
- If the user makes general conversation or small talk: respond naturally and
  professionally. Do NOT change the assessment.
- Always include "response_text" in your JSON response.
- Only include optional fields when the assessment actually changes.
- Be specific: reference the user's exact words and the relevant flags or rules
  in your response.""",
        ])

        return "\n".join(lines)

    def _build_llm_chat_response(
        self,
        user_message: str,
        current_assessment: AgentAssessment,
        domain_description: str,
        application: Any = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> ChatMessage:
        """Build a chat response using LLM, with deterministic fallback.

        Calls the LLM to generate an intelligent response, parses the JSON
        output, and updates the assessment accordingly. Falls back to
        deterministic behavior when LLM is unavailable or returns invalid data.

        Args:
            user_message: The user's input text.
            current_assessment: The agent's current assessment (modified in-place).
            domain_description: Description of the agent's domain.
            application: The Application model (optional).
            conversation_history: Previous messages (optional).

        Returns:
            ChatMessage with the response.
        """
        # If no LLM available, fall through to deterministic
        if self.llm is None or not self.llm.is_available():
            return self._build_deterministic_chat_response(
                user_message, current_assessment, domain_description,
                application, conversation_history,
            )

        # Build prompt and call LLM
        prompt = self._build_chat_prompt(
            user_message, current_assessment, domain_description,
            application, conversation_history,
        )

        response = self.llm.chat(prompt)

        # Check for FALLBACK_MESSAGE or empty/invalid LLM output
        if response == FALLBACK_MESSAGE or not response or not response.strip():
            return self._build_deterministic_chat_response(
                user_message, current_assessment, domain_description,
                application, conversation_history,
            )

        # Try to parse JSON
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON — use raw text as response, but fall back
            # to deterministic if the raw response is empty
            if not response or not response.strip():
                return self._build_deterministic_chat_response(
                    user_message, current_assessment, domain_description,
                    application, conversation_history,
                )
            return ChatMessage(
                sender=self.name,
                content=response,
                message_type="text",
                reasoning=response,
            )

        # Valid JSON — extract response_text (REQUIRED)
        response_text = data.get("response_text")
        if not response_text:
            return self._build_deterministic_chat_response(
                user_message, current_assessment, domain_description,
                application, conversation_history,
            )

        # Apply optional updates to current_assessment
        if "risk_tier_update" in data:
            current_assessment.risk_tier = data["risk_tier_update"]
        if "recommendation_update" in data:
            current_assessment.recommendation = data["recommendation_update"]
        if "confidence_update" in data:
            current_assessment.confidence_score = data["confidence_update"]
        if "flags_to_remove" in data:
            remove_ids = set(data["flags_to_remove"])
            current_assessment.flags = [
                f for f in current_assessment.flags
                if f.get("rule_id") not in remove_ids
            ]
        if "flags_to_add" in data:
            current_assessment.flags.extend(data["flags_to_add"])
        if "reasoning" in data:
            current_assessment.reasoning_summary += f" {data['reasoning']}"
        current_assessment.llm_used = True

        return ChatMessage(
            sender=self.name,
            content=response_text,
            message_type="text",
            reasoning=response_text,
        )

    def _build_deterministic_chat_response(
        self,
        user_message: str,
        current_assessment: AgentAssessment,
        domain_description: str,
        application: Any = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> ChatMessage:
        """Build a deterministic chat response based on domain templates.

        This method classifies the user's intent and generates an
        intent-specific response. It provides professional-sounding
        responses based on the agent's current assessment.

        Args:
            user_message: The user's input text.
            current_assessment: The agent's current assessment.
            domain_description: Description of the agent's domain (e.g. "medical underwriting").
            application: The Application model (required for evidence intent
                re-evaluation). Optional for other intents.

        Returns:
            ChatMessage with the response.
        """
        # Try LLM-powered chat response if available
        if self.llm is not None and self.llm.is_available() and application is not None:
            try:
                return self._build_llm_chat_response(
                    user_message, current_assessment, domain_description,
                    application, conversation_history,
                )
            except Exception:
                logger.warning(
                    "LLM chat response failed for [%s], falling back to deterministic",
                    self.name,
                )

        tier = current_assessment.risk_tier
        flag_count = len(current_assessment.flags)
        intent = self._analyze_question_intent(user_message)

        if intent == "flag":
            response = (
                f"Thank you for your question. I understand you'd like to know "
                f"about the flags on your assessment. You currently have "
                f"{flag_count} flag(s) identified during my {domain_description} review. "
                f"Your risk tier is {tier.upper()}. Each flag corresponds to a "
                f"specific underwriting criterion that was triggered. "
                f"I can provide more detail on any individual flag if you'd like."
            )
        elif intent == "explain":
            response = (
                f"Thank you for asking. Based on my {domain_description} assessment, "
                f"your risk tier is {tier.upper()}. This assessment considers "
                f"multiple factors including the {flag_count} flag(s) identified. "
                f"The assessment is based on deterministic underwriting rules "
                f"that evaluate your application against established criteria. "
                f"I'm happy to explain any specific aspect of the assessment."
            )
        elif intent == "evidence":
            # Re-evaluate the application to get a fresh assessment,
            # then update current_assessment fields from the fresh result.
            # Evidence re-evaluation: retain the original risk tier/flags if
            # the deterministic re-run produces identical results (stale app),
            # but reduce confidence and note evidence was considered.
            previous_tier = current_assessment.risk_tier
            previous_flag_count = len(current_assessment.flags)
            original_confidence = current_assessment.confidence_score

            fresh_assessment = self.evaluate(application)

            current_assessment.risk_tier = fresh_assessment.risk_tier
            current_assessment.flags = fresh_assessment.flags
            current_assessment.recommendation = fresh_assessment.recommendation
            current_assessment.loading_range = fresh_assessment.loading_range
            current_assessment.additional_evidence_required = (
                fresh_assessment.additional_evidence_required
            )
            current_assessment.apra_references = fresh_assessment.apra_references

            # Compute evidence-based confidence adjustment
            tier_changed = previous_tier != fresh_assessment.risk_tier
            flags_changed = previous_flag_count != len(fresh_assessment.flags)
            if tier_changed or flags_changed:
                # Evidence triggered meaningful reassessment — confidence stays high
                new_confidence = original_confidence
                evidence_note = (
                    f" [Re-evaluated with user evidence — "
                    f"risk tier changed from {previous_tier} to {fresh_assessment.risk_tier}]"
                    if tier_changed else
                    f" [Re-evaluated with user evidence — "
                    f"flag count changed from {previous_flag_count} to {len(fresh_assessment.flags)}]"
                )
            else:
                # Stale app produces same result — confidence reduced pending full review
                new_confidence = max(0.0, original_confidence - 0.1)
                evidence_note = (
                    f" [Evidence noted — deterministic re-evaluation produced same result. "
                    f"Confidence adjusted to {new_confidence:.2f}. "
                    f"Full re-evaluation requires application data update.]"
                )

            current_assessment.confidence_score = new_confidence
            current_assessment.reasoning_summary += evidence_note

            if tier_changed:
                response = (
                    f"Thank you for providing this evidence. Following re-evaluation, "
                    f"your risk tier has been updated from {previous_tier.upper()} to "
                    f"{fresh_assessment.risk_tier.upper()}. "
                    f"You now have {len(fresh_assessment.flags)} flag(s) in my "
                    f"{domain_description} review. Let me know if you have further questions."
                )
            else:
                response = (
                    f"Thank you for providing this information. I have conducted a "
                    f"deterministic re-evaluation of your application. "
                    f"Your risk tier remains {current_assessment.risk_tier.upper()} with "
                    f"{len(fresh_assessment.flags)} flag(s). "
                    f"Confidence score: {new_confidence:.2f}. "
                    f"To trigger a full re-evaluation with updated application data, "
                    f"please re-run the pipeline with corrected information."
                )
        else:
            response = (
                f"As a {domain_description} specialist, I have reviewed your input. "
                f"Based on my current assessment, the risk tier is {tier.upper()} "
                f"with {flag_count} flag(s) identified. "
                f"I will consider your input in the context of the full application profile."
            )

        return ChatMessage(
            sender=self.name,
            content=response,
            message_type="text",
            reasoning=response,
        )

    @staticmethod
    def _determine_risk_tier(highest_rule: Dict[str, Any]) -> str:
        """Map rule recommendation to standardised risk tier.

        Args:
            highest_rule: The matched rule with the highest severity.

        Returns:
            One of: "standard", "loading", "decline", "refer"
        """
        rec = highest_rule.get("recommendation", "standard")
        if rec in ["standard", "standard_or_loading"]:
            return "standard"
        if "loading" in rec:
            return "loading"
        if rec in ["decline", "manual_underwriting_or_decline"]:
            return "decline"
        if rec in ["refer", "manual_underwriting"]:
            return "refer"
        return "standard"

    @abstractmethod
    def evaluate(self, application: Any) -> AgentAssessment:
        """Evaluate the application and return an assessment.

        Subclasses must implement this method. The typical pattern is:
        1. Call self.evaluate_rules(application) for deterministic check.
        2. Call self.build_deterministic_assessment(application, matched).
        3. Optionally enrich with LLM (self.llm_enrich).
        4. Return the final AgentAssessment.

        Args:
            application: An Application Pydantic model instance.

        Returns:
            AgentAssessment with the agent's full evaluation.
        """

    @abstractmethod
    def generate_rebuttal(
        self,
        application: Any,
        my_assessment: AgentAssessment,
        other_assessments: List[AgentAssessment],
    ) -> AgentAssessment:
        """Generate a rebuttal when another agent disagrees.

        Called by the debate orchestrator. The agent reviews its own assessment
        and the conflicting assessments, then issues an updated position.

        Args:
            application: The original application.
            my_assessment: This agent's current assessment.
            other_assessments: Conflicting assessments from other agents.

        Returns:
            Updated AgentAssessment (may be unchanged if agent stands firm).
        """

    @abstractmethod
    def handle_user_message(
        self,
        application: Any,
        current_assessment: AgentAssessment,
        user_message: str,
        conversation_history: List[ChatMessage],
    ) -> ChatMessage:
        """Handle a user-injected question or evidence.

        The agent reviews the user's input, considers it against its domain rules,
        and responds with professional underwriting language.

        Args:
            application: The original application.
            current_assessment: This agent's current assessment.
            user_message: The user's question or evidence text.
            conversation_history: Previous messages in this conversation.

        Returns:
            ChatMessage with the agent's response.
        """
