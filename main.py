"""
Orchestrator for the research -> write -> critique -> revise pipeline.

Flow:
    Researcher -> Writer -> Critic -> decide()
        approved -> done
        not approved and revisions left -> Writer revises -> Critic checks again
        not approved and out of revisions -> return best draft with a warning
"""

import os
import sys
from dotenv import load_dotenv

from agents.researcher import research
from agents.writer import write
from agents.critic import critique
from agents.revision import build_revision_context, clean_approved_draft
from schemas.decision_rules import decide
from schemas.draft_schema import Draft
from schemas.metrics_schema import PipelineMetrics
from utils.storage import save_json
from utils.metrics import supported_pct, count_citations

MAX_REVISIONS = 2  # per the challenge guide: "limit revision to one or two rounds"


def format_open_questions(research_output) -> str:
    """Rendered directly from ResearchOutput.open_questions - no LLM involved,
    so this can never be flagged as 'unsupported' (nothing here claims to be
    a fact; it's explicitly labeled as what the research couldn't confirm)."""
    if not research_output.open_questions:
        return ""
    lines = "\n".join(f"- {q}" for q in research_output.open_questions)
    return f"\n\nOpen questions the research could not confirm:\n{lines}"


def print_metrics(metrics: PipelineMetrics) -> None:
    print("=== METRICS ===")
    print(f"Findings: {metrics.num_findings} | Sources: {metrics.num_sources}")
    print(f"Supported-claim %: {metrics.first_draft_supported_pct}% (first draft) -> {metrics.final_draft_supported_pct}% (final)")
    print(f"Citations in final draft: {metrics.num_citations_final}")
    print(f"Revision rounds: {metrics.revision_count}")
    print(f"Unsupported claims removed: {metrics.unsupported_claims_removed}")
    print(f"Final approved: {metrics.final_approved}")


def run_pipeline(question: str, api_key: str) -> None:
    print(f"=== QUESTION ===\n{question}\n")

    print("--- Researching ---")
    research_output = research(question, api_key)
    save_json(research_output, "research")
    print(f"Found {len(research_output.findings)} findings from {len(research_output.sources)} sources.\n")

    print("--- Writing first draft ---")
    draft_text = write(question, research_output, api_key)
    print(draft_text + "\n")

    revision_count = 0
    decision = None
    first_round_raw = None  # captured once, at revision_count == 0, for the "before" metric

    while True:
        print(f"--- Critic reviewing (revision {revision_count}) ---")
        raw = critique(draft_text, research_output, api_key)
        decision = decide(raw)

        if first_round_raw is None:
            first_round_raw = raw

        for v in raw.verdicts:
            print(f"  [{v.verdict.value} | {v.severity.value}] {v.claim}")
            print(f"      finding_ids: {v.finding_ids} | feedback: {v.feedback}")
        print(f"  Summary feedback: {raw.summary_feedback}")
        print(f"  Major issues: {decision.major_issues} | Minor issues: {decision.minor_issues} | Approved: {decision.approved}\n")

        save_json(Draft(text=draft_text, revision=revision_count), "draft")
        save_json(decision, "critic_decision")

        if decision.approved:
            cleaned_text, removed = clean_approved_draft(draft_text, decision)
            final_text = cleaned_text + format_open_questions(research_output)

            print("=== FINAL RESULT (approved) ===")
            if removed:
                print(f"(Removed {len(removed)} tolerated-but-unsupported claim(s) before final output:)")
                for r in removed:
                    print(f"  - \"{r}\"")
                print()
            print(final_text)

            metrics = PipelineMetrics(
                question=question,
                num_findings=len(research_output.findings),
                num_sources=len(research_output.sources),
                first_draft_supported_pct=supported_pct(first_round_raw),
                final_draft_supported_pct=supported_pct(raw),
                num_citations_final=count_citations(cleaned_text),
                revision_count=revision_count,
                unsupported_claims_removed=len(removed),
                final_approved=True,
            )
            save_json(metrics, "metrics")
            print()
            print_metrics(metrics)
            return

        if revision_count >= MAX_REVISIONS:
            final_text = draft_text + format_open_questions(research_output)

            print("=== FINAL RESULT (max revisions reached - unresolved issues remain) ===")
            print("WARNING: this draft was NOT fully approved by the Critic. "
                  f"{decision.major_issues} major issue(s) remain unresolved.")
            print(final_text)

            metrics = PipelineMetrics(
                question=question,
                num_findings=len(research_output.findings),
                num_sources=len(research_output.sources),
                first_draft_supported_pct=supported_pct(first_round_raw),
                final_draft_supported_pct=supported_pct(raw),
                num_citations_final=count_citations(draft_text),
                revision_count=revision_count,
                unsupported_claims_removed=0,
                final_approved=False,
            )
            save_json(metrics, "metrics")
            print()
            print_metrics(metrics)
            return

        print("--- Revising draft based on Critic feedback ---")
        revision_context = build_revision_context(decision)
        draft_text = write(question, research_output, api_key, revision_context=revision_context)
        print(draft_text + "\n")
        revision_count += 1


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    user_question = " ".join(sys.argv[1:]) or "What are the main benefits and risks of RAG (retrieval-augmented generation)?"
    run_pipeline(user_question, api_key)