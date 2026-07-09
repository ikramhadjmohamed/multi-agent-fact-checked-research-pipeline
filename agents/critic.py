"""
Critic agent.

Responsibility: read the Writer's draft, split it into individual factual
claims, and check each claim against the Researcher's findings. Never
invents new facts, never creates content - pure verification.

The LLM decides claim boundaries AND per-claim verdicts in one pass
(see design discussion: splitting requires the same reading comprehension
as verifying, so there's no benefit to doing it as a separate step).

Whether the draft AS A WHOLE is approved is NOT decided here - that's
schemas/decision_rules.py's job, applied to this agent's RawCriticOutput.
"""

import json
from groq import Groq
from pydantic import ValidationError

from schemas.research_schema import ResearchOutput
from schemas.critic_schema import RawCriticOutput

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 1


def build_prompt(draft: str, research: ResearchOutput) -> str:
    findings_block = "\n".join(
        f"[{f.id}] {f.claim} (evidence: {f.evidence}, source: {f.source_id})"
        for f in research.findings
    )

    return f"""You are a critic/fact-checking agent. Split the draft below into its individual
factual claims, and check each one against the findings it cites.

Draft to check:
{draft}

Available findings (what the draft is allowed to cite):
{findings_block}

For EACH factual claim in the draft (opinions/hedged statements like "some sources suggest"
still count as claims to check, just judge them against what they actually claim):

1. Extract the claim as it appears in the draft.
2. Decide a verdict:
   - "supported": the cited finding(s) clearly back this claim
   - "partially_supported": related evidence exists but the claim overstates or generalizes it
   - "unsupported": no cited finding actually backs this claim
   - "contradicted": a finding directly contradicts this claim
   - "opinion": not a factual claim at all (e.g. a transition sentence), no verification needed
3. Decide severity:
   - "major": one of the main answers to the user's question - a core benefit, core risk,
     core limitation, or central conclusion. If removing or changing this claim would
     change the main answer to the question, it is major.
   - "minor": a supporting detail whose removal would NOT change the main meaning of the draft.
   Do NOT label a claim "minor" just because it is hedged or cautiously worded - a hedged
   claim can still be central to the answer. Severity is about IMPORTANCE to the question,
   not about how confidently it's phrased.
4. List the finding_ids the claim cites or should cite.
5. Give brief feedback: what the Writer should do (remove, soften, add citation, or "fine as-is").

Return ONLY valid JSON (no markdown fences, no preamble) matching exactly this structure:
{{
  "verdicts": [
    {{"claim": "...", "verdict": "supported", "severity": "major", "finding_ids": ["f1"], "feedback": "fine as-is"}}
  ],
  "summary_feedback": "..."
}}
"""


def _call_llm(client: Groq, prompt: str, correction: str = "") -> str:
    full_prompt = prompt if not correction else f"{prompt}\n\nYour previous response was invalid: {correction}\nReturn corrected JSON only."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def _strip_code_fences(text: str) -> str:
    """Same defensive stripping as researcher.py - the LLM occasionally
    wraps JSON in ``` fences despite the prompt saying not to."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()

def critique(draft: str, research: ResearchOutput, api_key: str) -> RawCriticOutput:
    client = Groq(api_key=api_key)
    prompt = build_prompt(draft, research)
    correction = ""

    for attempt in range(MAX_RETRIES + 1):
        raw_text = _call_llm(client, prompt, correction)
        try:
            data = json.loads(_strip_code_fences(raw_text))
            return RawCriticOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            correction = str(e)
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Critic failed validation after {MAX_RETRIES + 1} attempts: {e}")

    raise RuntimeError("unreachable")