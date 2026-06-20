"""Multi-agent debate orchestrator."""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..debate.chat_models import ChatMessage, Conversation
from ..debate.persistence import ConversationStore
from .base_agent import AgentAssessment, BaseAgent

logger = logging.getLogger(__name__)


class DebateOrchestrator:
    """Coordinates the multi-agent underwriting decision process.

    Flow:
    1. Run all agents in parallel on the application.
    2. Compare agent assessments for consensus.
    3. If disagreement detected, run structured debate (max 3 rounds).
    4. Produce a final reconciled assessment.
    5. Log everything to the audit trail.
    """

    MAX_DEBATE_ROUNDS = 3
    RISK_TIER_RANK = {
        "standard": 0,
        "loading": 1,
        "refer": 2,
        "decline": 3,
    }

    # Compliance Agent is an observer/informer — its risk tier does NOT
    # affect the final underwriting decision or trigger debates.
    COMPLIANCE_AGENT_NAME = "Compliance Agent"

    def __init__(
        self,
        agents: List[BaseAgent],
        chat_store: Optional[ConversationStore] = None,
    ):
        """Initialise with the list of agents to coordinate.

        Args:
            agents: List of BaseAgent instances (Medical, Financial, Compliance).
            chat_store: Optional conversation store for chat persistence.
        """
        self.agents = agents
        self.debate_log: List[Dict] = []
        self.chat_store = chat_store

    def run(self, application: Any) -> Dict:
        """Execute the full multi-agent underwriting process.

        Args:
            application: An Application Pydantic model instance.

        Returns:
            Dict containing: final_assessment, agent_assessments, debate_log,
            consensus_reached, final_decision, decision_reasoning.
        """
        # Phase 1: Parallel agent evaluation
        assessments: Dict[str, AgentAssessment] = {}
        for agent in self.agents:
            logger.info(f"Running {agent.name}...")
            assessments[agent.name] = agent.evaluate(application)

        # Phase 2: Detect disagreements
        dispute = self._detect_dispute(assessments)

        # Phase 3: Debate if needed
        if dispute:
            logger.info("Disagreement detected. Initiating debate...")
            assessments = self._run_debate(application, assessments, dispute)

        # Phase 4: Produce final decision
        final = self._produce_final_decision(assessments)

        return {
            "final_assessment": final,
            "agent_assessments": assessments,
            "debate_log": self.debate_log,
            "consensus_reached": not dispute or self._check_post_debate_consensus(assessments),
            "final_decision": final.get("decision"),
            "decision_reasoning": final.get("reasoning", ""),
        }

    def _detect_dispute(self, assessments: Dict[str, AgentAssessment]) -> bool:
        """Check whether underwriting agents disagree on risk tier.

        Excludes the Compliance Agent — it is an observer that spots
        compliance gaps but does not drive underwriting disputes.

        A dispute exists if any two underwriting agents have different
        risk tiers that are more than one rank apart, or if one says
        'decline' while another says 'standard'.

        Returns:
            True if a dispute exists, False if consensus.
        """
        underwriting = {
            name: a for name, a in assessments.items()
            if name != self.COMPLIANCE_AGENT_NAME
        }
        if len(underwriting) < 2:
            return False
        tiers = [a.risk_tier for a in underwriting.values()]
        ranks = [self.RISK_TIER_RANK.get(t, 0) for t in tiers]

        if max(ranks) - min(ranks) >= 2:
            return True
        if "decline" in tiers and "standard" in tiers:
            return True
        return False

    def _run_debate(
        self,
        application: Any,
        assessments: Dict[str, AgentAssessment],
        initial_dispute: bool,
    ) -> Dict[str, AgentAssessment]:
        """Execute structured debate rounds.

        In each round, ALL agents review the current state of assessments
        and may revise their position. This ensures every agent participates
        in the debate, not just those in the minority.

        Args:
            application: The original application.
            assessments: Current agent assessments.
            initial_dispute: Whether a dispute existed before debate.

        Returns:
            Updated assessments after debate.
        """
        for round_num in range(1, self.MAX_DEBATE_ROUNDS + 1):
            logger.info(f"Debate round {round_num}/{self.MAX_DEBATE_ROUNDS}")

            # All agents review the full picture and may revise
            for agent in self.agents:
                other_assessments = [
                    a for name, a in assessments.items() if name != agent.name
                ]
                updated = agent.generate_rebuttal(
                    application,
                    assessments[agent.name],
                    other_assessments,
                )
                assessments[agent.name] = updated
                self.debate_log.append({
                    "round": round_num,
                    "agent": agent.name,
                    "original_tier": [a.risk_tier for a in assessments.values()],
                    "updated_tier": updated.risk_tier,
                    "reasoning": updated.reasoning_summary,
                })

            # Check if consensus reached after this round
            if not self._detect_dispute(assessments):
                logger.info(f"Consensus reached after round {round_num}")
                break

        return assessments

    def _check_post_debate_consensus(self, assessments: Dict[str, AgentAssessment]) -> bool:
        """Check if all agents agree after debate."""
        return not self._detect_dispute(assessments)

    def _produce_final_decision(self, assessments: Dict[str, AgentAssessment]) -> Dict:
        """Synthesise a final underwriting decision from agent assessments.

        The most conservative (highest risk) underwriting assessment
        takes precedence. The Compliance Agent is excluded — it acts
        as a compliance-gap spotter, not an underwriting decision-maker.

        If any underwriting agent says 'decline', the final is at
        least 'refer'.
        """
        # Exclude Compliance Agent from underwriting decision
        underwriting = {
            name: a for name, a in assessments.items()
            if name != self.COMPLIANCE_AGENT_NAME
        }
        if not underwriting:
            underwriting = assessments  # fallback if only Compliance exists

        tiers = [a.risk_tier for a in underwriting.values()]
        ranks = [self.RISK_TIER_RANK.get(t, 0) for t in tiers]
        highest_rank = max(ranks) if ranks else 0

        decision_map = {0: "Standard Offer", 1: "Offer with Loading/Exclusion",
                        2: "Refer to Manual Underwriting", 3: "Refer to Manual Underwriting"}

        final_decision = decision_map.get(highest_rank, "Refer to Manual Underwriting")

        # Collect flags from underwriting agents (risk-relevant)
        all_flags = []
        all_evidence = []
        for a in underwriting.values():
            all_flags.extend(a.flags)
            all_evidence.extend(a.additional_evidence_required)

        # Also collect compliance flags separately for informational display
        compliance = assessments.get(self.COMPLIANCE_AGENT_NAME)
        compliance_flags = compliance.flags if compliance else []
        compliance_gaps = [
            f.get("description", f.get("rule_id", "?"))
            for f in compliance_flags
        ]

        reasoning = (
            f"Final underwriting decision: {final_decision}. "
            f"Based on assessments from {len(underwriting)} underwriting agents."
        )
        if compliance_gaps:
            reasoning += (
                f" Compliance noted {len(compliance_flags)} potential "
                f"process gap(s): {', '.join(compliance_gaps[:3])}"
                f"{'...' if len(compliance_gaps) > 3 else ''}."
            )

        return {
            "decision": final_decision,
            "risk_tier": max(tiers, key=lambda t: self.RISK_TIER_RANK.get(t, 0)) if tiers else "standard",
            "reasoning": reasoning,
            "flags": all_flags,
            "additional_evidence_required": list(set(all_evidence)),
            "compliance_flags": compliance_flags,
            "compliance_gaps": compliance_gaps,
            "all_assessments": {name: a.model_dump() for name, a in assessments.items()},
        }

    def run_with_chat(
        self,
        application: Any,
        applicant_name: str = "Unknown",
        chat_store: Optional[ConversationStore] = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline and create a chat conversation.

        Runs the existing debate flow, then wraps the results into a
        Conversation with ChatMessage objects for the chat UI.

        Args:
            application: The Application model.
            applicant_name: Name for the conversation.
            chat_store: Optional conversation store (uses self.chat_store if None).

        Returns:
            Dict with same format as run(), plus 'conversation' key.
        """
        store = chat_store or self.chat_store

        # Run existing debate flow
        results = self.run(application)

        # Create conversation
        app_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        conversation = Conversation(
            application_id=app_id,
            applicant_name=applicant_name,
            debate_rounds=self._count_debate_rounds(),
            final_decision=results.get("final_decision", ""),
            agents_participating=list(results.get("agent_assessments", {}).keys()),
        )

        # Wrap debate log entries into ChatMessages
        for entry in results.get("debate_log", []):
            msg = ChatMessage(
                sender=entry.get("agent", "Unknown"),
                content=entry.get("reasoning", ""),
                message_type="text",
                risk_tier_update=entry.get("updated_tier"),
                reasoning=entry.get("reasoning", ""),
            )
            conversation.add_message(msg)

        # Save if store provided
        if store:
            store.save(conversation)

        results["conversation"] = conversation
        results["conversation_id"] = app_id
        return results

    def _count_debate_rounds(self) -> int:
        """Count the number of debate rounds in the log."""
        if not self.debate_log:
            return 0
        return max(entry.get("round", 0) for entry in self.debate_log)

    def inject_user_message(
        self,
        application: Any,
        user_message: str,
        conversation_id: str,
        chat_store: Optional[ConversationStore] = None,
    ) -> Dict[str, Any]:
        """Inject a user message and get agent responses.

        Loads the conversation, appends the user message, then calls
        handle_user_message() on each agent to get their responses.

        Args:
            application: The Application model.
            user_message: The user's question or evidence.
            conversation_id: The conversation to update.
            chat_store: Optional store (uses self.chat_store if None).

        Returns:
            Dict with updated conversation and new agent assessments.
        """
        store = chat_store or self.chat_store

        if store is None:
            raise ValueError("ConversationStore required for user message injection")

        # Load conversation
        conversation = store.load(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Append user message
        user_msg = ChatMessage(
            sender="user",
            content=user_message,
            message_type="question",
            is_user_input=True,
        )
        conversation.add_message(user_msg)

        # Get current assessments from conversation or re-evaluate
        assessments = {}
        for agent in self.agents:
            # Try to get current assessment from conversation
            current_assessment = None
            for msg in reversed(conversation.messages):
                if msg.sender == agent.name:
                    # Assessment will be re-evaluated from the application
                    break

            # If no assessment found, re-evaluate
            if current_assessment is None:
                current_assessment = agent.evaluate(application)

            # Call handle_user_message
            response = agent.handle_user_message(
                application=application,
                current_assessment=current_assessment,
                user_message=user_message,
                conversation_history=conversation.messages[:-1],  # Exclude user msg
            )

            # Append agent response
            conversation.add_message(response)

            # Track assessment
            assessments[agent.name] = current_assessment

        # Save updated conversation
        store.save(conversation)

        return {
            "conversation": conversation,
            "agent_assessments": assessments,
        }

    def get_conversation(self, application_id: str) -> Optional[Conversation]:
        """Get a conversation from the store.

        Args:
            application_id: The conversation ID.

        Returns:
            The Conversation object, or None.
        """
        if self.chat_store is None:
            return None
        return self.chat_store.load(application_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations.

        Returns:
            List of summary dicts.
        """
        if self.chat_store is None:
            return []
        return self.chat_store.list_applications()

    def delete_conversation(self, application_id: str) -> bool:
        """Delete a conversation.

        Args:
            application_id: The conversation ID.

        Returns:
            True if deleted, False if not found.
        """
        if self.chat_store is None:
            return False
        return self.chat_store.delete(application_id)
