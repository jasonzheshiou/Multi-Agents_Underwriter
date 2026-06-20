"""Pydantic v2 models for the interactive debate log feature."""

import uuid
from datetime import UTC, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the debate log conversation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    sender: str
    content: str
    message_type: str = Field(default="text")
    risk_tier_update: Optional[str] = None
    reasoning: str = ""
    is_user_input: bool = False

    def model_post_init(self, __context) -> None:  # noqa: ANN001, D102
        """Validate message_type after initialization."""
        valid_types = {"text", "evidence", "question", "system"}
        if self.message_type not in valid_types:
            raise ValueError(
                f"message_type must be one of {valid_types}, got '{self.message_type}'"
            )


class Conversation(BaseModel):
    """A complete debate conversation for a single application."""

    application_id: str
    applicant_name: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    messages: List[ChatMessage] = Field(default_factory=list)
    status: str = Field(default="active")
    debate_rounds: int = 0
    final_decision: str = ""
    agents_participating: List[str] = Field(default_factory=list)
    agent_assessments: dict[str, dict[str, object]] = Field(default_factory=dict)
    applicant_data: dict[str, object] = Field(default_factory=dict)
    user_evidence_applied: bool = Field(default=False)
    evidence_re_evaluated: bool = Field(default=False)
    decision_summary: str = Field(default="")

    def add_message(self, message: ChatMessage) -> None:
        """Append a message and update the conversation timestamp.

        Args:
            message: The ChatMessage to append.
        """
        self.messages.append(message)
        self.updated_at = datetime.now(UTC).isoformat()

        # Track participating agents (non-user, non-system senders)
        sender = message.sender
        if sender not in ("user", "system") and sender not in self.agents_participating:
            self.agents_participating.append(sender)

    def add_system_message(self, content: str) -> ChatMessage:
        """Create and append a system message.

        Args:
            content: The system message text.

        Returns:
            The created ChatMessage instance.
        """
        message = ChatMessage(
            sender="system",
            content=content,
            message_type="system",
        )
        self.add_message(message)
        return message

    def to_summary(self) -> dict[str, object]:
        """Return a summary dict for sidebar display.

        Returns:
            Dict with keys: application_id, applicant_name, status,
            debate_rounds, final_decision, message_count, created_at,
            updated_at, agents_participating.
        """
        return {
            "application_id": self.application_id,
            "applicant_name": self.applicant_name,
            "status": self.status,
            "debate_rounds": self.debate_rounds,
            "final_decision": self.final_decision,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agents_participating": self.agents_participating,
            "user_evidence_applied": self.user_evidence_applied,
            "evidence_re_evaluated": self.evidence_re_evaluated,
        }
