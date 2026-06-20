"""ChromaDB-backed vector store for underwriting knowledge base."""

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
import yaml

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "chroma_db"
)


class VectorStore:
    """Wraps ChromaDB persistent client for underwriting guideline storage.

    Attributes:
        persist_directory: Absolute path to the ChromaDB data directory.
        client: Active ChromaDB persistent client instance.
    """

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        """Initialize the vector store.

        Loads *chroma_persist_dir* from *config.yaml* if present;
        otherwise falls back to the built-in default.

        Args:
            persist_directory: Override path for ChromaDB persistence.
        """
        self.persist_directory = self._resolve_path(persist_directory)

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info("VectorStore initialized at %s", self.persist_directory)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        docs: List[Dict[str, Any]],
        collection_name: str = "underwriting_guidelines",
    ) -> None:
        """Add documents to the target collection.

        Args:
            docs: List of dicts that must contain at minimum a ``"text"`` key.
            collection_name: Target collection identifier.

        Raises:
            ValueError: If any document lacks a ``"text"`` key.
        """
        collection = self.client.get_or_create_collection(name=collection_name)

        texts = [doc["text"] for doc in docs]
        ids = [
            doc.get("id", f"doc_{i}_{len(docs)}") for i, doc in enumerate(docs)
        ]
        metadatas = [
            {
                str(k): (
                    str(v) if not isinstance(v, (str, int, float, bool)) else v
                )
                for k, v in doc.items()
                if k != "text"
            }
            or {"_source": "document"}  # ChromaDB requires non-empty metadata
            for doc in docs
        ]

        collection.add(documents=texts, ids=ids, metadatas=metadatas)
        logger.info(
            "Ingested %d documents into collection '%s'",
            len(docs),
            collection_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = "underwriting_guidelines",
    ) -> List[Dict[str, Any]]:
        """Search the collection for the most similar documents.

        Args:
            query: Text query to search for.
            top_k: Maximum number of results to return.
            collection_name: Target collection identifier.

        Returns:
            List of dicts with keys ``"text"``, ``"id"``, and ``"distance"``.
        """
        collection = self.client.get_collection(name=collection_name)

        count = collection.count()
        n_results = min(top_k, count) if count > 0 else 1

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        records: List[Dict[str, Any]] = []
        for i, text in enumerate(results["documents"][0]):
            meta: Dict[str, Any] = {}
            if results["metadatas"] and results["metadatas"][0][i]:
                meta = dict(results["metadatas"][0][i])

            records.append(
                {
                    "text": text,
                    "id": results["ids"][0][i],
                    "distance": (
                        results["distances"][0][i]
                        if results["distances"]
                        else None
                    ),
                    **meta,
                }
            )

        return records

    def get_collection(self, name: str) -> Any:
        """Retrieve a collection by name.

        Args:
            name: Collection identifier.

        Returns:
            The ChromaDB collection object.
        """
        return self.client.get_collection(name=name)

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name.

        Args:
            name: Collection identifier.

        Raises:
            RuntimeError: If the collection does not exist.
        """
        existing = [c.name for c in self.client.list_collections()]
        if name not in existing:
            raise RuntimeError(f"Collection '{name}' does not exist")
        self.client.delete_collection(name=name)
        logger.info("Deleted collection '%s'", name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(persist_directory: Optional[str]) -> str:
        """Return the resolved persist directory.

        Tries *config.yaml* first, then falls back to the default.
        """
        if persist_directory:
            return os.path.abspath(persist_directory)

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config.yaml",
        )
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
            dir_path = config.get("paths", {}).get("chroma_persist_dir")
            if dir_path:
                return os.path.abspath(dir_path)

        return DEFAULT_PERSIST_DIRECTORY
