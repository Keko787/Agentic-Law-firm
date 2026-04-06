"""
main.py — Run the Agentic Lawyer Pipeline
============================================

Usage:
    # Run intake only:
    python main.py intake contract.pdf

    # Run intake + research:
    python main.py pipeline contract.pdf

    # Run with the sample NDA:
    python main.py pipeline --sample

    # Seed the legal corpus only:
    python main.py seed-corpus

Environment:
    ANTHROPIC_API_KEY must be set (or pass --api-key)
"""

import argparse
import sys
import os
from pathlib import Path

from intake_agent import IntakeAgent
from research_agent import ResearchAgent
from analysis_agent import AnalysisAgent
from evaluator_agent import EvaluatorAgent
from output_agent import OutputAgent


def cmd_intake(args):
    """Run the Intake Agent only."""
    pdf_path = _resolve_input(args)
    api_key = _get_api_key(args)

    agent = IntakeAgent(api_key=api_key, model=args.model)
    result = agent.process(pdf_path)

    output = Path(args.output or "output/intake_result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output))
    print(f"\n💾 Intake results saved to: {output}")


def cmd_pipeline(args):
    """Run the full pipeline: Intake → Research."""
    pdf_path = _resolve_input(args)
    api_key = _get_api_key(args)

    # Stage 1: Intake
    print("=" * 60)
    print("  STAGE 1: INTAKE AGENT")
    print("=" * 60)
    intake_agent = IntakeAgent(api_key=api_key, model=args.model)
    intake_result = intake_agent.process(pdf_path)

    intake_output = Path("output/intake_result.json")
    intake_output.parent.mkdir(parents=True, exist_ok=True)
    intake_result.save(str(intake_output))
    print(f"💾 Intake saved to: {intake_output}")

    # Stage 2: Research
    print("\n" + "=" * 60)
    print("  STAGE 2: RESEARCH AGENT")
    print("=" * 60)
    research_agent = ResearchAgent(api_key=api_key, model=args.model)
    research_result = research_agent.research(
        intake_result,
        risk_threshold=args.risk_threshold,
    )

    research_output = Path(args.output or "output/research_result.json")
    research_output.parent.mkdir(parents=True, exist_ok=True)
    research_result.save(str(research_output))
    print(f"\n💾 Research saved to: {research_output}")

    # Stage 3: Analysis
    print("\n" + "=" * 60)
    print("  STAGE 3: ANALYSIS AGENT")
    print("=" * 60)
    analysis_agent = AnalysisAgent(api_key=api_key, model=args.model)
    analysis_result = analysis_agent.analyze(intake_result, research_result)

    analysis_output = Path("output/analysis_result.json")
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis_result.save(str(analysis_output))
    print(f"\n💾 Analysis saved to: {analysis_output}")

    # Stage 4: Evaluator
    print("\n" + "=" * 60)
    print("  STAGE 4: EVALUATOR AGENT")
    print("=" * 60)
    evaluator_agent = EvaluatorAgent(api_key=api_key, model=args.model)
    evaluation_result = evaluator_agent.evaluate(analysis_result, research_result)

    eval_output = Path("output/evaluation_result.json")
    evaluation_result.save(str(eval_output))
    print(f"💾 Evaluation saved to: {eval_output}")

    # Reflection loop: re-analyze flagged clauses
    if evaluation_result.clauses_needing_reanalysis:
        print(f"\n🔄 Reflection loop: {len(evaluation_result.clauses_needing_reanalysis)} "
              f"clause(s) flagged for reanalysis")
        # In a full implementation, we would re-run the analysis agent
        # on flagged clauses with additional context from the evaluator.
        # For now, we note it and proceed.
        print("   (Reanalysis noted — proceeding with current results)")

    # Stage 5: Output
    print("\n" + "=" * 60)
    print("  STAGE 5: OUTPUT AGENT")
    print("=" * 60)
    output_agent = OutputAgent(api_key=api_key, model=args.model)
    final_report = output_agent.generate_report(analysis_result, evaluation_result)

    report_output = Path(args.output or "output/final_report.json")
    report_output.parent.mkdir(parents=True, exist_ok=True)
    final_report.save(str(report_output))

    md_output = Path("output/final_report.md")
    final_report.save_markdown(str(md_output))
    print(f"💾 Report saved to: {report_output}")
    print(f"💾 Markdown saved to: {md_output}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — ALL 5 STAGES")
    print(f"{'=' * 60}")
    print(f"  Document: {final_report.document_filename}")
    print(f"  Risk: {final_report.risk_level.upper()} ({final_report.risk_score}/100)")
    print(f"  Recommendation: {final_report.proceed_recommendation}")
    print(f"  Findings: {final_report.total_findings} total, "
          f"{final_report.critical_findings} critical")
    print(f"  Confidence: {final_report.pipeline_confidence}")
    print(f"{'=' * 60}")


def cmd_seed_corpus(args):
    """Seed the legal corpus vector store."""
    from research_agent.corpus_seeder import seed_corpus
    print("📚 Seeding legal corpus...")
    store = seed_corpus(verbose=True)
    print(f"\n✅ Corpus seeded: {store.count()} chunks")


def _resolve_input(args) -> Path:
    """Resolve input PDF path."""
    if args.sample:
        pdf_path = Path("sample_docs/sample_nda.pdf")
        if not pdf_path.exists():
            print("Generating sample NDA...")
            from generate_sample_nda import create_sample_nda
            create_sample_nda(str(pdf_path))
        return pdf_path
    elif args.pdf_path:
        p = Path(args.pdf_path)
        if not p.exists():
            print(f"❌ File not found: {p}")
            sys.exit(1)
        return p
    else:
        print("❌ Provide a PDF path or use --sample")
        sys.exit(1)


def _get_api_key(args) -> str:
    """Resolve API key."""
    key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("❌ No API key. Set ANTHROPIC_API_KEY or pass --api-key")
        sys.exit(1)
    return key


def main():
    parser = argparse.ArgumentParser(description="Agentic Lawyer Pipeline")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")

    sub = parser.add_subparsers(dest="command")

    # Intake command
    p_intake = sub.add_parser("intake", help="Run Intake Agent only")
    p_intake.add_argument("pdf_path", nargs="?")
    p_intake.add_argument("--sample", action="store_true")
    p_intake.add_argument("--output", "-o")

    # Pipeline command
    p_pipe = sub.add_parser("pipeline", help="Run full pipeline (Intake + Research)")
    p_pipe.add_argument("pdf_path", nargs="?")
    p_pipe.add_argument("--sample", action="store_true")
    p_pipe.add_argument("--output", "-o")
    p_pipe.add_argument("--risk-threshold", default="low",
                         choices=["high", "medium", "low"],
                         help="Minimum risk level to research (default: low)")

    # Seed corpus command
    sub.add_parser("seed-corpus", help="Seed the legal corpus")

    args = parser.parse_args()

    if args.command == "intake":
        cmd_intake(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "seed-corpus":
        cmd_seed_corpus(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
