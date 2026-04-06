"""
Research Agent — Agentic Lawyer Pipeline Stage 2
=================================================
Retrieves legal sources via RAG and synthesizes research for each clause.
"""

from .agent import ResearchAgent
from .models import ResearchResult, ClauseResearch, LegalSource, SourceType
from .vector_store import LegalVectorStore
from .document_loader import LegalDocumentLoader
from .corpus_seeder import seed_corpus

__all__ = [
    "ResearchAgent",
    "ResearchResult",
    "ClauseResearch",
    "LegalSource",
    "SourceType",
    "LegalVectorStore",
    "LegalDocumentLoader",
    "seed_corpus",
]
