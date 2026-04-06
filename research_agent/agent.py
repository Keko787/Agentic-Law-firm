"""
Research Agent — Main Orchestrator
====================================
The Research Agent is Stage 2 of the Agentic Lawyer pipeline.

It takes an IntakeResult and produces a ResearchResult by:
  1. Loading/seeding the legal corpus into a vector store
  2. For each clause (prioritizing high/medium risk):
     a. Generating targeted search queries via LLM
     b. Retrieving relevant legal sources via RAG
     c. Synthesizing findings into structured research
  3. Packaging everything into a ResearchResult for downstream agents

Usage:
    from research_agent import ResearchAgent
    from intake_agent import IntakeAgent

    intake = IntakeAgent(api_key="sk-ant-...")
    intake_result = intake.process("contract.pdf")

    research = ResearchAgent(api_key="sk-ant-...")
    research_result = research.research(intake_result)
    research_result.save("output/research_result.json")
"""

from __future__ import annotations
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intake_agent.models import IntakeResult, Clause
from .models import ResearchResult, ClauseResearch
from .vector_store import LegalVectorStore
from .research_synthesizer import ResearchSynthesizer
from .corpus_seeder import seed_corpus


console = Console()


class ResearchAgent:
    """
    Orchestrates legal research for each clause in an IntakeResult.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        persist_dir: str = "./legal_corpus/vectordb",
        auto_seed: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            api_key: Anthropic API key.
            model: Claude model for query gen and synthesis.
            persist_dir: Directory for the ChromaDB vector store.
            auto_seed: Seed the corpus with sample data if empty.
            verbose: Print progress to console.
        """
        self.synthesizer = ResearchSynthesizer(api_key=api_key, model=model)
        self.verbose = verbose

        # Initialize vector store
        self._log("📚 Initializing legal corpus...")
        self.store = LegalVectorStore(persist_dir=persist_dir)

        if auto_seed and self.store.count() == 0:
            self._log("📥 Corpus is empty — seeding with sample legal provisions...")
            seed_corpus(persist_dir=persist_dir, verbose=verbose)
            # Refresh the store reference
            self.store = LegalVectorStore(persist_dir=persist_dir)

        self._log(f"   Corpus loaded: {self.store.count()} chunks across "
                   f"{', '.join(self.store.list_jurisdictions()) or 'no'} jurisdictions")

    def research(
        self,
        intake_result: IntakeResult,
        max_clauses: int | None = None,
        risk_threshold: str = "low",
    ) -> ResearchResult:
        """
        Run legal research on an IntakeResult.

        Args:
            intake_result: Output from the Intake Agent.
            max_clauses: Limit the number of clauses to research (None = all).
            risk_threshold: Minimum risk level to research.
                           "high" = only high-risk,
                           "medium" = high + medium,
                           "low" = all (default).

        Returns:
            ResearchResult with findings for each researched clause.
        """
        jurisdiction = intake_result.metadata.jurisdiction or ""
        doc_type = intake_result.metadata.document_type

        # Filter and prioritize clauses
        clauses = self._prioritize_clauses(
            intake_result.clauses, risk_threshold
        )
        if max_clauses:
            clauses = clauses[:max_clauses]

        self._log(f"\n🔍 Researching {len(clauses)} clauses from '{intake_result.metadata.filename}'")
        self._log(f"   Document type: {doc_type} | Jurisdiction: {jurisdiction or 'unspecified'}")

        # Research each clause
        clause_research: list[ClauseResearch] = []
        total_sources = 0

        for i, clause in enumerate(clauses):
            self._log(f"\n── Clause {clause.index}: {clause.heading} "
                       f"({clause.clause_type.value}, {clause.risk_level.value}) ──")

            research = self._research_clause(clause, jurisdiction)
            clause_research.append(research)
            total_sources += len(research.sources)

            self._log(f"   Retrieved {len(research.sources)} sources, "
                       f"generated {len(research.recommendations)} recommendations")

        # Build result
        result = ResearchResult(
            document_filename=intake_result.metadata.filename,
            document_type=doc_type,
            jurisdiction=jurisdiction,
            clause_research=clause_research,
            corpus_size=self.store.count(),
            total_sources_retrieved=total_sources,
        )

        if self.verbose:
            self._print_summary(result)

        return result

    # ── Per-clause research pipeline ──────────────────────────────────

    def _research_clause(
        self,
        clause: Clause,
        jurisdiction: str,
    ) -> ClauseResearch:
        """
        Full research pipeline for a single clause:
          1. Generate queries → 2. Retrieve sources → 3. Synthesize
        """
        # Step 1: Generate search queries
        self._log("   🔎 Generating search queries...")
        queries = self.synthesizer.generate_queries(
            clause_text=clause.text,
            clause_type=clause.clause_type.value,
            jurisdiction=jurisdiction,
            risk_level=clause.risk_level.value,
        )
        self._log(f"   → {len(queries)} queries: {queries[:3]}")

        # Step 2: Retrieve from vector store
        self._log("   📖 Retrieving legal sources...")
        sources = self.store.multi_query_search(
            queries=queries,
            n_results_per_query=3,
            jurisdiction=jurisdiction if jurisdiction else None,
        )

        # If jurisdiction-filtered search returned few results, broaden
        if len(sources) < 3 and jurisdiction:
            self._log("   🔄 Broadening search (removing jurisdiction filter)...")
            broader_sources = self.store.multi_query_search(
                queries=queries,
                n_results_per_query=2,
            )
            seen = {s.source_id for s in sources}
            for s in broader_sources:
                if s.source_id not in seen:
                    sources.append(s)
                    seen.add(s.source_id)

        self._log(f"   → {len(sources)} unique sources retrieved")

        # Step 3: Synthesize research
        self._log("   🧠 Synthesizing legal research...")
        synthesis = self.synthesizer.synthesize(
            clause_text=clause.text,
            clause_type=clause.clause_type.value,
            risk_level=clause.risk_level.value,
            sources=sources,
            jurisdiction=jurisdiction,
        )

        return ClauseResearch(
            clause_index=clause.index,
            clause_heading=clause.heading,
            clause_type=clause.clause_type.value,
            risk_level=clause.risk_level.value,
            sources=sources,
            legal_context=synthesis.get("legal_context", ""),
            applicable_law=synthesis.get("applicable_law", []),
            risk_analysis=synthesis.get("risk_analysis", ""),
            recommendations=synthesis.get("recommendations", []),
            enforceability_notes=synthesis.get("enforceability_notes", ""),
            queries_used=queries,
        )

    # ── Clause prioritization ─────────────────────────────────────────

    @staticmethod
    def _prioritize_clauses(
        clauses: list[Clause],
        risk_threshold: str,
    ) -> list[Clause]:
        """
        Filter and sort clauses by risk level.
        High-risk clauses are researched first.
        """
        risk_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        threshold_val = risk_order.get(risk_threshold, 2)

        filtered = [
            c for c in clauses
            if risk_order.get(c.risk_level.value, 3) <= threshold_val
            and c.clause_type.value not in ("preamble", "signature_block")
        ]

        filtered.sort(key=lambda c: risk_order.get(c.risk_level.value, 3))
        return filtered

    # ── Display ───────────────────────────────────────────────────────

    def _log(self, message: str, style: str = "cyan") -> None:
        if self.verbose:
            console.print(message, style=style)

    def _print_summary(self, result: ResearchResult) -> None:
        """Print a formatted summary of research findings."""
        console.print()
        console.print(
            Panel(
                f"[bold]{result.document_filename}[/bold]\n"
                f"Clauses researched: {len(result.clause_research)}  |  "
                f"Sources retrieved: {result.total_sources_retrieved}  |  "
                f"Corpus size: {result.corpus_size}",
                title="📚 Research Agent — Results",
                border_style="blue",
            )
        )

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", width=4, justify="center")
        table.add_column("Clause", width=24)
        table.add_column("Risk", width=8, justify="center")
        table.add_column("Sources", width=8, justify="center")
        table.add_column("Key Findings", width=50)

        risk_colors = {"high": "red", "medium": "yellow", "low": "green", "none": "dim"}

        for cr in result.clause_research:
            risk_style = risk_colors.get(cr.risk_level, "white")
            finding = cr.risk_analysis[:50] + "..." if len(cr.risk_analysis) > 50 else cr.risk_analysis

            table.add_row(
                str(cr.clause_index),
                cr.clause_heading[:24],
                f"[{risk_style}]{cr.risk_level.upper()}[/{risk_style}]",
                str(len(cr.sources)),
                finding,
            )

        console.print(table)

        # Recommendations for high-risk
        high_risk = result.high_risk_research()
        if high_risk:
            console.print(f"\n📋 [bold]Recommendations for high-risk clauses:[/bold]")
            for cr in high_risk:
                console.print(f"\n  [bold red]{cr.clause_heading}:[/bold red]")
                for rec in cr.recommendations[:3]:
                    console.print(f"    → {rec}")
