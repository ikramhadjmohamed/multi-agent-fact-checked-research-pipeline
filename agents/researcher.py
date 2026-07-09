import json
from ddgs import DDGS
from groq import Groq
from pydantic import ValidationError

from schemas.research_schema import ResearchOutput

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 1  # one correction attempt if the LLM returns invalid/fabricated JSON


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Real search, no API key needed. Returns raw DDGS results:
    each dict has 'title', 'href', 'body' (a short snippet)."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def build_prompt(question: str, search_results: list[dict]) -> str:
    # We number the sources here in the prompt itself (not left to the LLM
    # to invent numbering), so Source.id in the output is guaranteed to
    # match something real we actually gave it.
    sources_block = "\n".join(
        f"[s{i+1}] {r['title']} — {r['href']}\n    Snippet: {r['body']}"
        for i, r in enumerate(search_results)
    )

    return f"""You are a research agent. Answer the question using ONLY the search results below.

Question: {question}

Search results:
{sources_block}

Rules:
- Only use facts found in the search results above. Do NOT add anything from your own knowledge.
- Skip snippets that only contain titles, headings, questions, or teasers. Only create findings from concrete factual evidence — if a snippet has no real content, do not turn it into a finding.
- Do not create two separate findings from the same sentence of evidence. If one sentence supports multiple angles, pick the strongest single claim.
- If the search results don't clearly cover some aspect of the question, list it in "open_questions" instead of guessing.
- Every finding's "source_id" MUST be one of the ids shown above (s1, s2, ...) exactly as written.
- For each finding, add "limitations": a short note whenever the source is opinion-based, marketing/vendor content, anecdotal, narrowly scoped, or otherwise weak/biased. If the source is solid (e.g. established research, reputable technical documentation) and you see no real caveat, use an empty string "".

Return ONLY valid JSON (no markdown fences, no preamble) matching exactly this structure:
{{
  "findings": [
    {{"id": "f1", "claim": "...", "evidence": "...", "source_id": "s1", "confidence": 0.8}}
  ],
  "sources": [
    {{"id": "s1", "title": "...", "url": "..."}}
  ],
  "open_questions": ["..."],
  "confidence": 0.75
}}
"""


def _call_llm(client: Groq, prompt: str, correction: str = "") -> str:
    full_prompt = prompt if not correction else f"{prompt}\n\nYour previous response was invalid: {correction}\nReturn corrected JSON only."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,  # low temperature: we want consistent extraction, not creative writing
    )
    return response.choices[0].message.content


def _validate_no_fabricated_sources(research: ResearchOutput) -> None:
    """Extra check beyond Pydantic's type checking: every finding's
    source_id must point to a source that's actually in the sources list.
    A mismatch here means the LLM invented a finding not tied to real
    search results — exactly the failure mode our prompt tries to prevent."""
    valid_ids = {s.id for s in research.sources}
    for f in research.findings:
        if f.source_id not in valid_ids:
            raise ValueError(f"Finding {f.id} references unknown source_id '{f.source_id}'")

def _strip_code_fences(text: str) -> str:
    """The prompt says 'no markdown fences', but LLMs sometimes wrap JSON in
```json ... ``` or ``` ... ``` anyway. Prompting alone isn't reliable
    enough to guarantee this never happens (same lesson as the citation
    rules earlier), so we defensively strip fences in code before parsing."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()

def research(question: str, api_key: str) -> ResearchOutput:
    client = Groq(api_key=api_key)
    search_results = search_web(question)

    if not search_results:
        raise RuntimeError("No search results found — check network/query before blaming the LLM")

    prompt = build_prompt(question, search_results)
    correction = ""

    for attempt in range(MAX_RETRIES + 1):
        raw_text = _call_llm(client, prompt, correction)

        if not raw_text or not raw_text.strip():
            print(f"  [researcher] WARNING: empty response on attempt {attempt + 1}")
            correction = "Your previous response was completely empty. Return the JSON object described above."
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Researcher got an empty response from the LLM after {MAX_RETRIES + 1} attempts. "
                    "This usually means an API-side issue (rate limit, transient error) rather than a "
                    "prompt/schema problem - try re-running."
                )
            continue

        try:
            data = json.loads(_strip_code_fences(raw_text))
            output = ResearchOutput.model_validate(data)
            _validate_no_fabricated_sources(output)
            return output
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"  [researcher] WARNING: invalid response on attempt {attempt + 1}: {e}")
            print(f"  [researcher] raw response was: {raw_text[:300]}")
            correction = str(e)
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Researcher failed validation after {MAX_RETRIES + 1} attempts: {e}")

    raise RuntimeError("unreachable")

    raise RuntimeError("unreachable")