"""Tests for ConversationStore JSON-file persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from underwriting.debate.chat_models import ChatMessage, Conversation
from underwriting.debate.persistence import ConversationStore


@pytest.fixture()
def store(tmp_path: Path) -> ConversationStore:
    """Return a ConversationStore backed by a temporary directory."""
    return ConversationStore(persist_dir=str(tmp_path))


@pytest.fixture()
def sample_conversation() -> Conversation:
    """Return a Conversation with a few messages for testing."""
    conv = Conversation(
        application_id="app-001",
        applicant_name="Jane Doe",
        status="active",
        debate_rounds=2,
        final_decision="",
    )
    conv.add_system_message("Debate started.")
    conv.add_message(
        ChatMessage(
            sender="Medical Agent",
            content="BMI 24.5, no conditions. Standard risk.",
        )
    )
    return conv


# ---------------------------------------------------------------------------
# test_save_and_load_round_trip
# ---------------------------------------------------------------------------


class TestSaveAndLoadRoundTrip:
    """Test saving a conversation and loading it back."""

    def test_save_and_load_round_trip(
        self, store: ConversationStore, sample_conversation: Conversation
    ):
        """Save conversation, load it back, verify all fields match."""
        store.save(sample_conversation)

        loaded = store.load("app-001")
        assert loaded is not None
        assert loaded.application_id == sample_conversation.application_id
        assert loaded.applicant_name == sample_conversation.applicant_name
        assert loaded.status == sample_conversation.status
        assert loaded.debate_rounds == sample_conversation.debate_rounds
        assert loaded.final_decision == sample_conversation.final_decision
        assert len(loaded.messages) == len(sample_conversation.messages)
        assert loaded.messages[0].sender == "system"
        assert loaded.messages[1].sender == "Medical Agent"

    def test_file_is_valid_json(self, store: ConversationStore, sample_conversation: Conversation):
        """Verify the saved file is valid, pretty-printed JSON."""
        store.save(sample_conversation)
        filepath = store.persist_dir / "app-001.json"

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["application_id"] == "app-001"
        assert data["applicant_name"] == "Jane Doe"
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 2


# ---------------------------------------------------------------------------
# test_load_nonexistent_returns_none
# ---------------------------------------------------------------------------


class TestLoadNonexistent:
    """Test loading a non-existent conversation."""

    def test_load_nonexistent_returns_none(self, store: ConversationStore):
        """Load non-existent ID returns None."""
        result = store.load("nonexistent-id")
        assert result is None

    def test_load_nonexistent_does_not_create_file(self, store: ConversationStore):
        """Loading a non-existent ID should not create any file."""
        store.load("nonexistent-id")
        files = list(store.persist_dir.glob("*.json"))
        assert len(files) == 0


# ---------------------------------------------------------------------------
# test_list_applications_empty
# ---------------------------------------------------------------------------


class TestListApplicationsEmpty:
    """Test listing applications when store is empty."""

    def test_list_applications_empty(self, store: ConversationStore):
        """Empty directory returns empty list."""
        result = store.list_applications()
        assert result == []

    def test_list_applications_empty_dir_not_created(self):
        """list_applications should not create the persist_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(persist_dir=tmpdir)
            # Remove the dir that __init__ created
            for f in Path(tmpdir).glob("*"):
                f.unlink()
            Path(tmpdir).rmdir()

            result = store.list_applications()
            assert result == []


# ---------------------------------------------------------------------------
# test_list_applications_with_data
# ---------------------------------------------------------------------------


class TestListApplicationsWithData:
    """Test listing applications when store has data."""

    def test_list_applications_with_data(self, store: ConversationStore):
        """Save 2 conversations, list returns both."""
        conv1 = Conversation(application_id="app-001", applicant_name="Jane Doe")
        conv2 = Conversation(application_id="app-002", applicant_name="John Smith")
        store.save(conv1)
        store.save(conv2)

        summaries = store.list_applications()
        assert len(summaries) == 2
        ids = [s["application_id"] for s in summaries]
        assert "app-001" in ids
        assert "app-002" in ids

    def test_list_applications_sorted_reverse(self, store: ConversationStore):
        """List should return conversations sorted by filename in reverse order."""
        conv1 = Conversation(application_id="app-001", applicant_name="Jane")
        conv2 = Conversation(application_id="app-002", applicant_name="John")
        conv3 = Conversation(application_id="app-003", applicant_name="Bob")
        store.save(conv1)
        store.save(conv2)
        store.save(conv3)

        summaries = store.list_applications()
        ids = [s["application_id"] for s in summaries]
        assert ids == ["app-003", "app-002", "app-001"]

    def test_list_applications_skips_corrupted_files(self, store: ConversationStore):
        """Corrupted JSON files should be skipped without raising."""
        conv = Conversation(application_id="app-001", applicant_name="Jane")
        store.save(conv)

        # Write a corrupted file
        corrupted_path = store.persist_dir / "app-999.json"
        with open(corrupted_path, "w", encoding="utf-8") as f:
            f.write("this is not valid json {{{")

        summaries = store.list_applications()
        assert len(summaries) == 1
        assert summaries[0]["application_id"] == "app-001"


# ---------------------------------------------------------------------------
# test_delete_conversation
# ---------------------------------------------------------------------------


class TestDeleteConversation:
    """Test deleting conversations."""

    def test_delete_conversation(self, store: ConversationStore, sample_conversation: Conversation):
        """Save, delete, verify file gone and load returns None."""
        store.save(sample_conversation)

        # Verify file exists before delete
        filepath = store.persist_dir / "app-001.json"
        assert filepath.exists()

        # Delete
        result = store.delete("app-001")
        assert result is True

        # Verify file is gone
        assert not filepath.exists()

        # Verify load returns None
        assert store.load("app-001") is None

    def test_delete_nonexistent_returns_false(self, store: ConversationStore):
        """Delete non-existent ID returns False."""
        result = store.delete("nonexistent-id")
        assert result is False


# ---------------------------------------------------------------------------
# test_append_message
# ---------------------------------------------------------------------------


class TestAppendMessage:
    """Test appending messages to existing conversations."""

    def test_append_message(self, store: ConversationStore, sample_conversation: Conversation):
        """Load conversation, append message, verify message count increased."""
        store.save(sample_conversation)
        initial_count = len(sample_conversation.messages)

        new_msg = ChatMessage(sender="Financial Agent", content="Income acceptable.")
        result = store.append_message("app-001", new_msg)

        assert result is True
        loaded = store.load("app-001")
        assert loaded is not None
        assert len(loaded.messages) == initial_count + 1
        assert loaded.messages[-1].sender == "Financial Agent"

    def test_append_message_nonexistent(self, store: ConversationStore):
        """Append to non-existent returns False."""
        msg = ChatMessage(sender="Medical Agent", content="Test")
        result = store.append_message("nonexistent-id", msg)
        assert result is False

    def test_append_message_updates_timestamp(
        self, store: ConversationStore, sample_conversation: Conversation
    ):
        """Appending a message should update the conversation's updated_at."""
        store.save(sample_conversation)
        original_updated = sample_conversation.updated_at

        import time
        time.sleep(0.01)

        new_msg = ChatMessage(sender="Medical Agent", content="Updated.")
        store.append_message("app-001", new_msg)

        loaded = store.load("app-001")
        assert loaded.updated_at > original_updated


# ---------------------------------------------------------------------------
# test_persist_dir_created_automatically
# ---------------------------------------------------------------------------


class TestPersistDirCreatedAutomatically:
    """Test that persist directory is created on initialization."""

    def test_persist_dir_created_automatically(self):
        """Non-existent dir is created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            assert not new_dir.exists()

            store = ConversationStore(persist_dir=str(new_dir))
            assert new_dir.exists()
            assert store.persist_dir == new_dir

    def test_init_with_existing_dir(self):
        """Initializing with an existing directory should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(persist_dir=tmpdir)
            assert store.persist_dir == Path(tmpdir)
