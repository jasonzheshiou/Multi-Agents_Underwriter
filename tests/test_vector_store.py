"""Tests for the vector store module."""

import os
import tempfile
from typing import Generator

import pytest

from underwriting.knowledge.vector_store import DEFAULT_PERSIST_DIRECTORY, VectorStore


class TestVectorStoreInit:
    """Test cases for VectorStore initialization."""

    def test_default_persist_directory(self) -> None:
        """Should resolve to the built-in default when no config or override is given."""
        store = VectorStore()
        assert store.persist_directory == DEFAULT_PERSIST_DIRECTORY

    def test_custom_persist_directory(self) -> None:
        """Should use the provided path when explicitly passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_directory=tmpdir)
            assert store.persist_directory == os.path.abspath(tmpdir)

    def test_creates_directory(self) -> None:
        """Should create the persist directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_chroma")
            VectorStore(persist_directory=new_dir)
            assert os.path.isdir(new_dir)

    def test_client_initialized(self) -> None:
        """Should create a valid ChromaDB client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_directory=tmpdir)
            assert store.client is not None


class TestVectorStoreIngest:
    """Test cases for document ingestion."""

    @pytest.fixture()
    def store(self) -> Generator[VectorStore, None, None]:
        """Provide a temporary VectorStore instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield VectorStore(persist_directory=tmpdir)

    def test_ingest_single_document(self, store: VectorStore) -> None:
        """Should ingest a single document successfully."""
        store.ingest_documents([{"text": "Hello world", "source": "test"}])
        collection = store.client.get_collection("underwriting_guidelines")
        assert collection.count() == 1

    def test_ingest_multiple_documents(self, store: VectorStore) -> None:
        """Should ingest multiple documents successfully."""
        docs = [
            {"text": "Rule A", "category": "A"},
            {"text": "Rule B", "category": "B"},
        ]
        store.ingest_documents(docs)
        collection = store.client.get_collection("underwriting_guidelines")
        assert collection.count() == 2

    def test_ingest_custom_collection(self, store: VectorStore) -> None:
        """Should create/use a custom collection name."""
        store.ingest_documents(
            [{"text": "Custom doc"}],
            collection_name="custom_rules",
        )
        collection = store.client.get_collection("custom_rules")
        assert collection.count() == 1

    def test_ingest_document_without_id(self, store: VectorStore) -> None:
        """Should auto-generate IDs when none are provided."""
        store.ingest_documents([{"text": "Auto-ID doc"}])
        collection = store.client.get_collection("underwriting_guidelines")
        assert collection.count() == 1

    def test_ingest_document_with_custom_id(self, store: VectorStore) -> None:
        """Should use the provided ID when present."""
        store.ingest_documents([{"id": "doc_1", "text": "Custom ID doc"}])
        collection = store.client.get_collection("underwriting_guidelines")
        results = collection.get(ids=["doc_1"])
        assert results is not None
        assert len(results["ids"]) == 1


class TestVectorStoreSearch:
    """Test cases for document search."""

    @pytest.fixture()
    def store_with_data(self) -> Generator[VectorStore, None, None]:
        """Provide a VectorStore pre-loaded with test documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_directory=tmpdir)
            docs = [
                {"text": "Life insurance covers death benefit", "category": "life"},
                {"text": "Term duration is 10 to 30 years", "category": "life"},
                {"text": "Critical illness pays on diagnosis", "category": "ci"},
                {"text": "Total permanent disability covers TPD", "category": "tpd"},
                {"text": "Trauma benefit covers heart attack", "category": "trauma"},
            ]
            store.ingest_documents(docs)
            yield store

    def test_search_returns_results(self, store_with_data: VectorStore) -> None:
        """Should return at least one result for a relevant query."""
        results = store_with_data.search("death benefit")
        assert len(results) > 0

    def test_search_respects_top_k(self, store_with_data: VectorStore) -> None:
        """Should not return more than top_k results."""
        results = store_with_data.search("insurance", top_k=2)
        assert len(results) <= 2

    def test_search_empty_collection(self) -> None:
        """Should handle search on an empty collection gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_directory=tmpdir)
            store.client.create_collection("empty_col")
            results = store.search("anything", collection_name="empty_col")
            assert results == []

    def test_search_custom_collection(self, store_with_data: VectorStore) -> None:
        """Should search within a specific collection."""
        store_with_data.ingest_documents(
            [{"text": "Custom collection doc"}],
            collection_name="custom_rules",
        )
        results = store_with_data.search(
            "Custom",
            collection_name="custom_rules",
        )
        assert len(results) == 1


class TestVectorStoreCollectionOps:
    """Test cases for collection management."""

    @pytest.fixture()
    def store(self) -> Generator[VectorStore, None, None]:
        """Provide a temporary VectorStore instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield VectorStore(persist_directory=tmpdir)

    def test_get_collection(self, store: VectorStore) -> None:
        """Should retrieve an existing collection."""
        store.client.create_collection("test_get")
        collection = store.get_collection("test_get")
        assert collection.name == "test_get"

    def test_get_nonexistent_collection_raises(self, store: VectorStore) -> None:
        """Should raise when retrieving a non-existent collection."""
        with pytest.raises(ValueError):
            store.get_collection("nonexistent_collection_xyz")

    def test_delete_collection(self, store: VectorStore) -> None:
        """Should delete an existing collection."""
        store.client.create_collection("to_delete")
        store.delete_collection("to_delete")
        collections = store.client.list_collections()
        names = {c.name for c in collections}
        assert "to_delete" not in names

    def test_delete_nonexistent_collection_raises(self, store: VectorStore) -> None:
        """Should raise RuntimeError when deleting a non-existent collection."""
        with pytest.raises(RuntimeError, match="does not exist"):
            store.delete_collection("nonexistent_collection_xyz")
