from groq import Groq

from schemas.research_schema import ResearchOutput

MODEL = "llama-3.3-70b-versatile"


def build_prompt(question: str, research: ResearchOutput, revision_context: str = "") -> str:
    findings_block = "\n".join(
        f"[{f.id}] {f.claim} (evidence: {f.evidence})"
        + (f" [caution: {f.limitations}]" if f.limitations else "")
        for f in research.findings
    )

    open_questions_block = (
        "\n".join(f"- {q}" for q in research.open_questions)
        if research.open_questions else "(none)"
    )

    base = f"""You are a writer agent. Write a short, readable summary answering the question below,
using ONLY the findings provided. Do not add facts from your own knowledge.

Question: {question}

Findings:
{findings_block}

Things the research could NOT confirm (do not present these as fact):
{open_questions_block}

Rules:
- After every factual sentence, cite the finding(s) it's based on using its id in brackets, e.g. [f1].
  If a sentence merges two findings, cite both: [f1][f3].
- Do not state anything as fact that isn't backed by a finding above.
- If a finding has a [caution: ...] note, reflect that caution in your wording
  (e.g. "one source suggests..." instead of stating it flatly as settled fact).
- Do not mention the open_questions above anywhere in your summary, not even to say something is
  "unclear" or "still being explored". Any such meta-comment about the research itself cannot be
  cited to a finding, so the Critic will always reject it. Open questions are handled separately -
  just write the summary using only what the findings actually support.
- Write 3-6 sentences. Plain prose, no headers or bullet lists.
- Do not add a concluding or "wrap-up" sentence that combines multiple findings into a broader
  evaluative judgment (e.g. "this makes RAG beneficial/advantageous/valuable"). Report each
  finding's own content directly; do not add a summary-level opinion on top of what the findings state.
- When citing multiple findings together, you may list or compare facts that each finding
  directly states. Do NOT combine findings to infer a new broader conclusion, causal
  relationship, or general judgment, unless one of the cited findings explicitly states
  that conclusion itself.
  Self-check before writing a sentence that cites 2+ findings: does at least ONE single
  finding, on its own, already state this exact conclusion? If yes, it's fine to cite it
  alongside supporting context. If the conclusion only emerges by combining two findings
  together, do not write it - just state each finding's fact separately instead.
  
Return ONLY the summary text. No preamble, no explanation of what you're doing."""

    if revision_context:
        return f"{base}\n\n{revision_context}"
    return base


def write(question: str, research: ResearchOutput, api_key: str, revision_context: str = "") -> str:
    """Generate a draft. revision_context is empty for the first draft;
    Phase 5 will pass the Critic's feedback here for revision rounds."""
    client = Groq(api_key=api_key)
    prompt = build_prompt(question, research, revision_context)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()