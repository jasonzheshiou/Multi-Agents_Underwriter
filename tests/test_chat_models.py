"""Tests for chat data models (ChatMessage and Conversation)."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from underwriting.debate.chat_models import ChatMessage, Conversation

# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessageCreation:
    """Tests for ChatMessage creation and validation."""

    def test_create_minimal(self):
        msg = ChatMessage(sender="Medical Agent", content="Standard risk.")
        assert msg.sender == "Medical Agent"
        assert msg.content == "Standard risk."
        assert msg.message_type == "text"
        assert msg.risk_tier_update is None
        assert msg.reasoning == ""
        assert msg.is_user_input is False
        assert len(msg.id) == 8
        assert msg.timestamp != ""

    def test_create_full(self):
        msg = ChatMessage(
            sender="Financial Agent",
            content="Income-to-debt ratio acceptable.",
            message_type="evidence",
            risk_tier_update="loading",
            reasoning="Debt-to-income ratio is 35%, below threshold of 40%.",
            is_user_input=False,
        )
        assert msg.sender == "Financial Agent"
        assert msg.message_type == "evidence"
        assert msg.risk_tier_update == "loading"
        assert msg.reasoning == "Debt-to-income ratio is 35%, below threshold of 40%."
        assert msg.is_user_input is False

    def test_user_input_message(self):
        msg = ChatMessage(
            sender="user",
            content="What is my risk tier?",
            is_user_input=True,
            message_type="question",
        )
        assert msg.is_user_input is True
        assert msg.sender == "user"
        assert msg.message_type == "question"

    def test_system_message(self):
        msg = ChatMessage(
            sender="system",
            content="Debate round 2 started.",
            message_type="system",
        )
        assert msg.sender == "system"
        assert msg.message_type == "system"
        assert msg.is_user_input is False

    def test_id_is_8_char_hex(self):
        msg = ChatMessage(sender="Test", content="Hi")
        # Should be exactly 8 hex characters
        assert len(msg.id) == 8
        int(msg.id, 16)  # Should not raise

    def test_id_is_unique_per_message(self):
        msg1 = ChatMessage(sender="Test", content="First")
        msg2 = ChatMessage(sender="Test", content="Second")
        assert msg1.id != msg2.id

    def test_timestamp_is_iso_format(self):
        msg = ChatMessage(sender="Test", content="Hi")
        datetime.fromisoformat(msg.timestamp)

    def test_valid_message_types(self):
        for msg_type in ("text", "evidence", "question", "system"):
            msg = ChatMessage(sender="Test", content="Hi", message_type=msg_type)
            assert msg.message_type == msg_type

    def test_invalid_message_type_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(sender="Test", content="Hi", message_type="invalid")

    def test_sender_required(self):
        with pytest.raises(ValidationError):
            ChatMessage(content="Hi")

    def test_content_required(self):
        with pytest.raises(ValidationError):
            ChatMessage(sender="Test")


class TestChatMessageJSON:
    """Tests for ChatMessage JSON serialization."""

    def test_to_dict(self):
        msg = ChatMessage(
            sender="Medical Agent",
            content="BMI is 24.5.",
            message_type="evidence",
            risk_tier_update="standard",
        )
        data = msg.model_dump()
        assert data["sender"] == "Medical Agent"
        assert data["message_type"] == "evidence"
        assert data["risk_tier_update"] == "standard"
        assert data["is_user_input"] is False

    def test_json_roundtrip(self):
        msg = ChatMessage(
            sender="Medical Agent",
            content="BMI is 24.5.",
            message_type="evidence",
            risk_tier_update="standard",
            reasoning="BMI in normal range.",
        )
        json_str = msg.model_dump_json()
        restored = ChatMessage.model_validate_json(json_str)
        assert restored.sender == msg.sender
        assert restored.content == msg.content
        assert restored.message_type == msg.message_type
        assert restored.risk_tier_update == msg.risk_tier_update
        assert restored.reasoning == msg.reasoning

    def test_json_serializable_with_none_fields(self):
        msg = ChatMessage(sender="Test", content="Hi")
        data = json.loads(msg.model_dump_json())
        assert data["risk_tier_update"] is None


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class TestConversationCreation:
    """Tests for Conversation creation with default values."""

    def test_create_minimal(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        assert conv.application_id == "abc123"
        assert conv.applicant_name == "Jane Doe"
        assert conv.messages == []
        assert conv.status == "active"
        assert conv.debate_rounds == 0
        assert conv.final_decision == ""
        assert conv.agents_participating == []
        assert conv.created_at != ""
        assert conv.updated_at != ""

    def test_timestamps_are_iso_format(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        datetime.fromisoformat(conv.created_at)
        datetime.fromisoformat(conv.updated_at)

    def test_created_before_updated(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        # created_at should be <= updated_at (same factory call)
        assert conv.created_at <= conv.updated_at


class TestConversationAddMessage:
    """Tests for add_message method."""

    def test_add_single_message(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = ChatMessage(sender="Medical Agent", content="Standard risk.")
        conv.add_message(msg)
        assert len(conv.messages) == 1
        assert conv.messages[0] is msg

    def test_add_multiple_messages(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg1 = ChatMessage(sender="Medical Agent", content="Msg 1")
        msg2 = ChatMessage(sender="Financial Agent", content="Msg 2")
        conv.add_message(msg1)
        conv.add_message(msg2)
        assert len(conv.messages) == 2
        assert conv.messages[0] is msg1
        assert conv.messages[1] is msg2

    def test_updated_at_changes_on_add(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        original_updated = conv.updated_at
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)
        msg = ChatMessage(sender="Medical Agent", content="New message")
        conv.add_message(msg)
        assert conv.updated_at > original_updated

    def test_agent_tracked_on_add(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = ChatMessage(sender="Medical Agent", content="Standard risk.")
        conv.add_message(msg)
        assert "Medical Agent" in conv.agents_participating

    def test_user_sender_not_tracked(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = ChatMessage(sender="user", content="Question?", is_user_input=True)
        conv.add_message(msg)
        assert "user" not in conv.agents_participating

    def test_system_sender_not_tracked(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = ChatMessage(sender="system", content="Round started.")
        conv.add_message(msg)
        assert "system" not in conv.agents_participating

    def test_agent_not_duplicated(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg1 = ChatMessage(sender="Medical Agent", content="First")
        msg2 = ChatMessage(sender="Medical Agent", content="Second")
        conv.add_message(msg1)
        conv.add_message(msg2)
        assert conv.agents_participating == ["Medical Agent"]


class TestConversationAddSystemMessage:
    """Tests for add_system_message method."""

    def test_add_system_message_creates_correct_type(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = conv.add_system_message("Debate started.")
        assert msg.sender == "system"
        assert msg.content == "Debate started."
        assert msg.message_type == "system"

    def test_add_system_message_appended(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = conv.add_system_message("Round 2 started.")
        assert len(conv.messages) == 1
        assert conv.messages[0] is msg

    def test_add_system_message_returns_message(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        msg = conv.add_system_message("Test")
        assert isinstance(msg, ChatMessage)


class TestConversationToSummary:
    """Tests for to_summary method."""

    def test_summary_contains_all_fields(self):
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            status="active",
            debate_rounds=3,
            final_decision="Offer with Loading",
        )
        summary = conv.to_summary()
        assert summary["application_id"] == "abc123"
        assert summary["applicant_name"] == "Jane Doe"
        assert summary["status"] == "active"
        assert summary["debate_rounds"] == 3
        assert summary["final_decision"] == "Offer with Loading"
        assert summary["message_count"] == 0
        assert summary["created_at"] == conv.created_at
        assert summary["updated_at"] == conv.updated_at
        assert summary["agents_participating"] == []

    def test_summary_message_count(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        conv.add_message(ChatMessage(sender="Medical Agent", content="Hi"))
        conv.add_message(ChatMessage(sender="Financial Agent", content="Hi too"))
        summary = conv.to_summary()
        assert summary["message_count"] == 2

    def test_summary_agents(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        conv.add_message(ChatMessage(sender="Medical Agent", content="Hi"))
        conv.add_message(ChatMessage(sender="Financial Agent", content="Hi"))
        summary = conv.to_summary()
        assert summary["agents_participating"] == ["Medical Agent", "Financial Agent"]


class TestConversationStatus:
    """Tests for status field."""

    def test_default_status_active(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        assert conv.status == "active"

    def test_status_can_be_changed(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        conv.status = "completed"
        assert conv.status == "completed"
        conv.status = "archived"
        assert conv.status == "archived"

    def test_status_valid_values(self):
        for status in ("active", "completed", "archived"):
            conv = Conversation(
                application_id="abc123",
                applicant_name="Jane Doe",
                status=status,
            )
            assert conv.status == status


class TestConversationJSON:
    """Tests for Conversation JSON serialization/deserialization."""

    def test_to_dict(self):
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            status="completed",
            debate_rounds=2,
            final_decision="Standard Offer",
        )
        data = conv.model_dump()
        assert data["application_id"] == "abc123"
        assert data["applicant_name"] == "Jane Doe"
        assert data["status"] == "completed"
        assert data["debate_rounds"] == 2
        assert data["final_decision"] == "Standard Offer"
        assert data["messages"] == []

    def test_json_roundtrip(self):
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            status="active",
            debate_rounds=1,
        )
        conv.add_message(ChatMessage(sender="Medical Agent", content="Standard risk."))
        json_str = conv.model_dump_json()
        restored = Conversation.model_validate_json(json_str)
        assert restored.application_id == conv.application_id
        assert restored.applicant_name == conv.applicant_name
        assert restored.status == conv.status
        assert restored.debate_rounds == conv.debate_rounds
        assert len(restored.messages) == 1
        assert restored.messages[0].sender == "Medical Agent"
        assert restored.messages[0].content == "Standard risk."

    def test_json_serializable(self):
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            messages=[
                ChatMessage(sender="Medical Agent", content="Test message."),
            ],
            agents_participating=["Medical Agent"],
        )
        json_str = conv.model_dump_json()
        # Should not raise
        data = json.loads(json_str)
        assert data["application_id"] == "abc123"
        assert len(data["messages"]) == 1

    def test_full_roundtrip_with_all_fields(self):
        conv = Conversation(
            application_id="app-001",
            applicant_name="John Smith",
            status="completed",
            debate_rounds=5,
            final_decision="Offer with Loading",
            agents_participating=["Medical Agent", "Financial Agent", "Compliance Agent"],
        )
        conv.add_system_message("Debate started.")
        conv.add_message(ChatMessage(
            sender="Medical Agent",
            content="BMI 28, mild loading.",
            message_type="evidence",
            risk_tier_update="loading",
            reasoning="BMI slightly above 25.",
        ))
        json_str = conv.model_dump_json()
        restored = Conversation.model_validate_json(json_str)
        assert restored.application_id == "app-001"
        assert restored.applicant_name == "John Smith"
        assert restored.status == "completed"
        assert restored.debate_rounds == 5
        assert restored.final_decision == "Offer with Loading"
        assert restored.agents_participating == [
            "Medical Agent", "Financial Agent", "Compliance Agent",
        ]
        assert len(restored.messages) == 2
        assert restored.messages[0].sender == "system"
        assert restored.messages[1].sender == "Medical Agent"
        assert restored.messages[1].risk_tier_update == "loading"


class TestConversationIntegration:
    """Integration tests for Conversation with messages."""

    def test_full_debate_flow(self):
        """Simulate a complete debate conversation."""
        conv = Conversation(
            application_id="app-001",
            applicant_name="Jane Doe",
        )
        # System starts
        conv.add_system_message("Debate round 1 initiated.")
        # Medical agent speaks
        conv.add_message(ChatMessage(
            sender="Medical Agent",
            content="BMI 24.5, no conditions. Standard risk.",
            message_type="evidence",
            reasoning="All medical indicators normal.",
        ))
        # Financial agent speaks
        conv.add_message(ChatMessage(
            sender="Financial Agent",
            content="Income $120k, debt-to-income 30%. Standard.",
            message_type="evidence",
            reasoning="Financial position healthy.",
        ))
        # Compliance agent speaks
        conv.add_message(ChatMessage(
            sender="Compliance Agent",
            content="No red flags. Duty of disclosure acknowledged.",
            message_type="evidence",
        ))
        # User asks a question
        conv.add_message(ChatMessage(
            sender="user",
            content="What's my final tier?",
            is_user_input=True,
            message_type="question",
        ))
        # System responds
        conv.add_system_message("All agents agree: Standard Offer.")

        assert len(conv.messages) == 6
        assert conv.agents_participating == [
            "Medical Agent", "Financial Agent", "Compliance Agent",
        ]
        assert conv.status == "active"

        summary = conv.to_summary()
        assert summary["message_count"] == 6
        assert summary["agents_participating"] == [
            "Medical Agent", "Financial Agent", "Compliance Agent",
        ]

    def test_conversation_with_existing_messages(self):
        """Create conversation with pre-populated messages."""
        messages = [
            ChatMessage(sender="Medical Agent", content="Msg 1"),
            ChatMessage(sender="Financial Agent", content="Msg 2"),
        ]
        conv = Conversation(
            application_id="abc123",
            applicant_name="Test",
            messages=messages,
            status="completed",
            debate_rounds=2,
            final_decision="Standard Offer",
            agents_participating=["Medical Agent", "Financial Agent"],
        )
        assert len(conv.messages) == 2
        assert conv.status == "completed"
        assert conv.debate_rounds == 2
        assert conv.final_decision == "Standard Offer"
        assert conv.agents_participating == ["Medical Agent", "Financial Agent"]


# ---------------------------------------------------------------------------
# Conversation Agent Assessments
# ---------------------------------------------------------------------------


class TestConversationAgentAssessments:
    """Tests for the agent_assessments field on Conversation."""

    def test_default_agent_assessments_empty_dict(self):
        conv = Conversation(application_id="abc123", applicant_name="Jane Doe")
        assert conv.agent_assessments == {}

    def test_set_agent_assessments(self):
        assessments = {
            "Medical Agent": {
                "risk_tier": "loading",
                "flags": [{"rule_id": "MED-001", "severity": "high", "description": "High BMI"}],
                "recommendation": "loading",
                "loading_range": [1.2, 1.5],
                "confidence_score": 0.85,
                "reasoning_summary": "High BMI detected",
                "additional_evidence_required": [],
                "apra_references": [],
            },
        }
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            agent_assessments=assessments,
        )
        assert "Medical Agent" in conv.agent_assessments
        assert conv.agent_assessments["Medical Agent"]["risk_tier"] == "loading"
        assert conv.agent_assessments["Medical Agent"]["confidence_score"] == 0.85

    def test_json_roundtrip_with_assessments(self):
        assessments = {
            "Medical Agent": {
                "risk_tier": "loading",
                "flags": [{"rule_id": "MED-001", "severity": "high"}],
                "recommendation": "loading",
                "loading_range": [1.2, 1.5],
                "confidence_score": 0.85,
            },
            "Financial Agent": {
                "risk_tier": "standard",
                "flags": [],
                "recommendation": "standard",
                "loading_range": [1.0, 1.0],
                "confidence_score": 0.95,
            },
        }
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            agent_assessments=assessments,
        )
        json_str = conv.model_dump_json()
        restored = Conversation.model_validate_json(json_str)
        assert restored.agent_assessments == assessments
        assert restored.agent_assessments["Medical Agent"]["risk_tier"] == "loading"
        assert restored.agent_assessments["Financial Agent"]["confidence_score"] == 0.95

    def test_json_serializable_with_assessments(self):
        assessments = {
            "Medical Agent": {
                "risk_tier": "loading",
                "flags": [],
                "recommendation": "loading",
                "loading_range": [1.1, 1.3],
                "confidence_score": 0.8,
            },
        }
        conv = Conversation(
            application_id="abc123",
            applicant_name="Jane Doe",
            agent_assessments=assessments,
        )
        json_str = conv.model_dump_json()
        # Should not raise
        data = json.loads(json_str)
        assert "agent_assessments" in data
        assert data["agent_assessments"]["Medical Agent"]["risk_tier"] == "loading"
