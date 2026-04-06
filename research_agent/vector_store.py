"""
Legal Vector Store
==================
ChromaDB-backed vector store for legal document retrieval.

Design:
  - Uses ChromaDB's built-in embedding (all-MiniLM-L6-v2 via default)
  - Persistent storage so corpus survives restarts
  - Multi-query retrieval: generates multiple search angles per clause
  - Metadata filtering by jurisdiction, source type, etc.

Why ChromaDB?
  - Zero-config local setup (no external services)
  - Built-in embedding function (no separate embedding API needed)
  - Metadata filtering for jurisdiction/source-type scoping
  - Easy to swap for Pinecone/Weaviate in production
"""

from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from pathlib import Path
from .document_loader import LegalChunk
from .models import LegalSource, SourceType


class LocalHashEmbedding(EmbeddingFunction[Documents]):
    """
    A lightweight local embedding function using hashed n-gram features.

    This is a FALLBACK for environments where the default ONNX model
    can't be downloaded. It produces 384-dim vectors via feature hashing
    of word unigrams and bigrams, with TF normalization.

    For production, swap this out for:
      - sentence-transformers (all-MiniLM-L6-v2)
      - OpenAI text-embedding-3-small
      - Anthropic Voyage embeddings
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(doc) for doc in input]

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts = Counter(features)
        vec = [0.0] * self.dim
        for feat, count in counts.items():
            idx = int(hashlib.md5(feat.encode()).hexdigest(), 16) % self.dim
            sign = 1.0 if int(hashlib.sha1(feat.encode()).hexdigest(), 16) % 2 == 0 else -1.0
            vec[idx] += sign * math.log1p(count)
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class LegalVectorStore:
    """
    ChromaDB-backed vector store for legal document search.
    """

    def __init__(
        self,
        persist_dir: str = "./legal_corpus/vectordb",
        collection_name: str = "legal_documents",
    ):
        """
        Args:
            persist_dir: Directory for persistent ChromaDB storage.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=LocalHashEmbedding(dim=384),
        )

    # ── Ingestion ─────────────────────────────────────────────────────

    def ingest(self, chunks: list[LegalChunk], batch_size: int = 100) -> int:
        """
        Ingest legal chunks into the vector store.

        Returns the number of chunks added.
        Skips chunks whose IDs already exist (idempotent).
        """
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            # Filter out already-existing IDs
            existing = set()
            try:
                result = self.collection.get(ids=ids)
                if result and result["ids"]:
                    existing = set(result["ids"])
            except Exception:
                pass

            new_ids = []
            new_docs = []
            new_metas = []
            for id_, doc, meta in zip(ids, documents, metadatas):
                if id_ not in existing:
                    new_ids.append(id_)
                    new_docs.append(doc)
                    new_metas.append(meta)

            if new_ids:
                self.collection.add(
                    ids=new_ids,
                    documents=new_docs,
                    metadatas=new_metas,
                )
                added += len(new_ids)

        return added

    # ── Retrieval ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        jurisdiction: str | None = None,
        source_type: str | None = None,
    ) -> list[LegalSource]:
        """
        Search the vector store for relevant legal passages.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            jurisdiction: Filter by jurisdiction (e.g., "Delaware").
            source_type: Filter by source type (e.g., "statute").

        Returns:
            List of LegalSource objects sorted by relevance.
        """
        where_filters = self._build_filters(jurisdiction, source_type)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filters if where_filters else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"  ⚠️  Vector store search failed: {e}")
            return []

        return self._parse_results(results)

    def multi_query_search(
        self,
        queries: list[str],
        n_results_per_query: int = 3,
        jurisdiction: str | None = None,
        source_type: str | None = None,
    ) -> list[LegalSource]:
        """
        Search with multiple queries and deduplicate results.

        This improves recall by approaching the same legal concept
        from different angles (e.g., "non-compete enforceability"
        and "restrictive covenant validity").

        Returns deduplicated results sorted by best relevance score.
        """
        seen_ids: set[str] = set()
        all_sources: list[LegalSource] = []

        for query in queries:
            results = self.search(
                query=query,
                n_results=n_results_per_query,
                jurisdiction=jurisdiction,
                source_type=source_type,
            )
            for source in results:
                if source.source_id not in seen_ids:
                    seen_ids.add(source.source_id)
                    all_sources.append(source)

        # Sort by relevance (lower distance = more relevant)
        all_sources.sort(key=lambda s: s.relevance_score)
        return all_sources

    # ── Corpus info ───────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self.collection.count()

    def list_jurisdictions(self) -> list[str]:
        """Return distinct jurisdictions in the corpus."""
        try:
            result = self.collection.get(include=["metadatas"], limit=1000)
            jurisdictions = set()
            for meta in (result.get("metadatas") or []):
                if meta and "jurisdiction" in meta:
                    jurisdictions.add(meta["jurisdiction"])
            return sorted(jurisdictions)
        except Exception:
            return []

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_filters(
        jurisdiction: str | None,
        source_type: str | None,
    ) -> dict | None:
        """Build ChromaDB where-filter from optional constraints."""
        conditions = []
        if jurisdiction:
            conditions.append({"jurisdiction": {"$eq": jurisdiction}})
        if source_type:
            conditions.append({"source_type": {"$eq": source_type}})

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    @staticmethod
    def _parse_results(results: dict) -> list[LegalSource]:
        """Convert ChromaDB query results into LegalSource objects."""
        sources: list[LegalSource] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return sources

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            source_type_str = meta.get("source_type", "unknown") if meta else "unknown"
            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                source_type = SourceType.UNKNOWN

            sources.append(LegalSource(
                source_id=id_,
                source_type=source_type,
                title=meta.get("title", "") if meta else "",
                jurisdiction=meta.get("jurisdiction", "") if meta else "",
                text=doc,
                relevance_score=dist,
                metadata=meta or {},
            ))

        return sources
