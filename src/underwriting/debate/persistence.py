"""JSON-file-based persistence for debate conversations."""

import json
from pathlib import Path
from typing import Any

from underwriting.debate.chat_models import ChatMessage, Conversation


class ConversationStore:
    """JSON-file-based persistence for debate conversations.

    Stores conversations as JSON files in a configurable directory.
    Each conversation is keyed by application_id.
    """

    def __init__(self, persist_dir: str = "data/chat_conversations") -> None:
        """Initialize the conversation store.

        Args:
            persist_dir: Directory to store conversation JSON files.
        """
        self.persist_dir: Path = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def save(self, conversation: Conversation) -> None:
        """Save a conversation to a JSON file.

        Args:
            conversation: The Conversation to save.
        """
        filepath = self.persist_dir / f"{conversation.application_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversation.model_dump(), f, indent=2, default=str)

    def load(self, application_id: str) -> Conversation | None:
        """Load a conversation from a JSON file.

        Args:
            application_id: The application ID to load.

        Returns:
            The Conversation object, or None if not found.
        """
        filepath = self.persist_dir / f"{application_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Conversation.model_validate(data)

    def list_applications(self) -> list[dict[str, Any]]:
        """List all saved conversations as summary dicts.

        Returns:
            List of dicts with application_id, applicant_name, created_at,
            status, debate_rounds, final_decision.
        """
        if not self.persist_dir.exists():
            return []

        summaries = []
        for filepath in sorted(self.persist_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                conv = Conversation.model_validate(data)
                summaries.append(conv.to_summary())
            except (json.JSONDecodeError, Exception):
                # Skip corrupted files
                continue
        return summaries

    def delete(self, application_id: str) -> bool:
        """Delete a conversation file.

        Args:
            application_id: The application ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        filepath = self.persist_dir / f"{application_id}.json"
        if not filepath.exists():
            return False
        filepath.unlink()
        return True

    def append_message(self, application_id: str, message: ChatMessage) -> bool:
        """Load a conversation, append a message, and save it.

        Args:
            application_id: The application ID.
            message: The ChatMessage to append.

        Returns:
            True if successful, False if conversation not found.
        """
        conversation = self.load(application_id)
        if conversation is None:
            return False

        conversation.add_message(message)
        self.save(conversation)
        return True
