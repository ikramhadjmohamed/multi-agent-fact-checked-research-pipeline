# Multi-Agent Fact-Checked Research Pipeline

A multi-agent AI system where a Researcher gathers evidence, a Writer produces a cited
summary, and a Critic verifies every claim before the final answer is returned.

## Why this task

Most single-prompt LLM answers can't be checked against anything - you just have to trust
the output. This project splits the work into three roles specifically so that no single
agent is trusted to both generate an answer and verify it:

- The **Researcher** is the only agent allowed to touch the outside world (search) and
  introduce new facts into the system.
- The **Writer** can only use what the Researcher found - it's deliberately "blind" to
  its own training knowledge.
- The **Critic** never creates anything - it only compares the Writer's draft against
  the Researcher's evidence, sentence by sentence.

This separation of trust is what makes the collaboration meaningful: the final answer is
better not because three LLM calls happened, but because each role is prevented from
covering for its own mistakes.

I focused on AI/tech topics (RAG, model quantization, AI tutors, LLM reliability) because
I could actually evaluate by eye whether the Critic's fact-checking was accurate.

## Agents

| Agent | Input | Output | Tools |
|---|---|---|---|
| **Researcher** | User's question | `ResearchOutput`: findings (claim, evidence, source, confidence, limitations), sources, open_questions | DuckDuckGo search (`ddgs`, no API key), Groq LLaMA for extraction |
| **Writer** | Question + `ResearchOutput` | Plain-text summary, citing findings by `[f_id]` | Groq LLaMA only, no tools - works purely from the Researcher's data |
| **Critic** | Writer's draft + `ResearchOutput` | `RawCriticOutput`: a verdict per claim (supported/partially_supported/unsupported/contradicted/opinion), severity (major/minor), feedback | Groq LLaMA only - pure verification, no external calls |

A fourth piece, **`decide()`** (in `schemas/decision_rules.py`), is deliberately **not an
LLM agent** - it's plain code that turns the Critic's per-claim verdicts into a single
approve/reject decision. This was a specific design choice: letting the LLM set its own
"approved" flag risked it disagreeing with its own per-claim verdicts (e.g. marking two
claims "unsupported" but still setting `approved: true`). Keeping the rule in code means
there's exactly one place that decision is made, and it can't contradict itself.

## How agents communicate

All Researcher/Critic output is structured JSON, validated with **Pydantic v2** schemas
(`schemas/research_schema.py`, `schemas/critic_schema.py`). If the LLM returns invalid or
malformed JSON, the agent retries once with the actual validation error fed back as a
correction prompt.

Findings and citations use **`finding_id`** (e.g. `f1`), not `source_id`. This was a
deliberate choice: one source can back multiple distinct findings, so citing the specific
finding gives the Critic an exact claim+evidence pair to check, with no ambiguity about
which of a source's findings a sentence refers to.

```
Researcher
   -> ResearchOutput (findings, sources, open_questions, confidence)
Writer
   -> draft text, citing [f_id] after each factual claim
Critic
   -> RawCriticOutput (verdict + severity + feedback per claim)
decide() [code, not LLM]
   -> CriticDecision (approved: bool, computed from the verdicts)
```

## Workflow / revision loop

```
Researcher -> Writer -> Critic -> decide()
    approved -> clean_approved_draft() strips any tolerated-but-unsupported
                minor claims, then done
    not approved, revisions left -> Writer revises using Critic's feedback -> Critic re-checks
    not approved, out of revisions (max 2) -> return best draft with an explicit warning
```

**Approval rule** (`schemas/decision_rules.py`):
- Any **major** claim that's unsupported, contradicted, or only partially supported ->
  reject. A central claim being overstated still needs revision.
- **Minor** unsupported/contradicted claims are tolerated individually (up to a threshold
  of 2), but any minor claim that's tolerated is still stripped from the final text by
  `clean_approved_draft()` - a fact-checking system shouldn't silently ship a claim it
  knows isn't backed by evidence, even if it wasn't severe enough to block approval.

## How to run

```bash
git clone <this repo>
cd multi-agent-fact-checked-research-pipeline
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with:
```
GROQ_API_KEY=your_key_here
```

Run the full pipeline on a question:
```bash
python main.py What are the main benefits and risks of RAG?
```

Run the deterministic unit test (no API calls):
```bash
python -m tests.test_clean_approved_draft
```

Run individual agent checks (useful for debugging one stage without running the whole pipeline):
```bash
python -m scripts.check_researcher
python -m scripts.check_writer data\research_<timestamp>.json
python -m scripts.check_critic data\research_<timestamp>.json data\draft_<timestamp>.json
python -m scripts.check_direction_reversal
```
## Project structure

```
.
├── agents/
│   ├── critic.py        Verifies draft claims against research findings
│   ├── researcher.py     Searches + extracts structured findings
│   ├── revision.py       Builds revision feedback, cleans approved drafts
│   └── writer.py         Drafts summary, cites findings by [f_id]
├── schemas/
│   ├── critic_schema.py     Verdict/CriticDecision data shapes
│   ├── decision_rules.py    The one non-LLM decision: approve/reject
│   ├── draft_schema.py      Wraps Writer output for storage
│   ├── metrics_schema.py    Evaluation metrics data shape
│   └── research_schema.py   Finding/Source/ResearchOutput data shapes
├── utils/
│   ├── metrics.py        Pure calculations (supported %, citation count)
│   └── storage.py         Generic save/load for any Pydantic model
├── scripts/               Manual one-off checks (cost API calls, for debugging)
├── tests/                 Deterministic automated tests (no API calls)
├── demo/                  Full transcripts for each evaluation topic
├── data/                  Saved JSON per run (gitignored except .gitkeep)
├── main.py                Orchestrator - the actual pipeline loop
├── requirements.txt
├── .env.example
└── README.md
```

## Demo

A complete end-to-end execution of the system, including a Critic-triggered revision
round, is in [`demo/risks_ai_tutors_education.txt`](./demo/risks_ai_tutors_education.txt).

The demo contains:
- User question
- Researcher findings
- Writer first draft
- Critic verdicts
- Revision loop
- Final approved draft
- Evaluation metrics

## Tools, models, frameworks

- **LLM**: Groq API, `llama-3.3-70b-versatile`, temperature 0.2 (Researcher/Critic, extraction
  and verification need consistency) and 0.4 (Writer, needs slightly more natural phrasing)
- **Search**: `ddgs` (DuckDuckGo), free, no API key
- **Validation**: Pydantic v2 for all structured agent outputs
- **Storage**: JSON files in `data/`, via generic `save_json`/`load_json` helpers
- No orchestration framework (LangGraph/CrewAI/AutoGen) - implemented the orchestration manually with plain
  function calls, per the challenge's suggested approach of understanding the flow manually
  before reaching for a framework

## Evaluation
 
Metrics tracked per run (`schemas/metrics_schema.py`, computed in `utils/metrics.py`):
first-draft vs. final-draft supported-claim %, citation count, revision count, and count of
claims stripped from an approved draft.
 
Tested across 5 topics on the current version of the pipeline (after all fixes below).
 
| Question | Findings | First draft % | Final % | Citations | Revisions | Approved |
|---|---|---|---|---|---|---|
| Benefits and risks of RAG | 4 | 100.0% | 100.0% | 5 | 0 | Yes |
| Limitations of model quantization | 3 | 100.0% | 100.0% | 6 | 0 | Yes |
| Limitations of RAG | 2 | 100.0% | 100.0% | 6 | 0 | Yes |
| Reliability risks of LLMs | 4 | 80.0% | 100.0% | 5 | 1 | Yes |
| Risks of AI tutors in education | 3 | 60.0% | 100.0% | 6 | 1 | Yes |
 
The last two rows are the clearest evidence of the revision loop doing real work: the
"Risks of AI tutors" run started at only 60% supported (two overstated claims, one
generalizing across sources it wasn't backed by) and reached 100% after exactly one
revision round - a concrete, measurable improvement, not just "the system ran."
 
Full transcripts for each run above are in [`demo/`](./demo/).
 
The "Reliability risks of LLMs" run that failed to converge is kept in this table
deliberately - it's an honest example of the system correctly refusing to approve a draft
it couldn't fully verify, rather than always reporting success.

## Challenges and what I'd improve

Several real bugs were found by running the same question multiple times and comparing
results, not by inspecting code alone:

1. **Headline-echo findings**: early on, the Researcher sometimes extracted a search
   result's *title* as if it were a fact (e.g. "There are limitations of RAG"). Fixed by
   explicitly instructing the Researcher to skip snippets with no concrete content.

2. **Critic severity drift**: the same draft, critiqued twice, sometimes classified the
   same claims as "major" and sometimes as "minor" - because the original severity
   instruction ("central to the question") was too vague, and hedged/cautious wording was
   apparently being conflated with "unimportant." Fixed with an explicit rule: importance
   is about impact on the answer, not confidence of phrasing. Verified stable across 3
   repeated runs on the same input.

3. **Open-questions contradiction**: the Writer was told to "note what's unclear," but any
   such meta-comment about the research itself can never be backed by a finding_id, so the
   Critic would always reject it - a self-defeating instruction that caused 3 full revision
   rounds without ever converging. Fixed by moving open questions into a separate section,
   generated directly from data (no LLM), appended after the Critic-checked body.

4. **Evaluative wrap-up sentences**: the Writer would sometimes end a summary with a
   sentence like "this makes RAG beneficial," adding judgment beyond what any single
   finding stated. Fixed by explicitly forbidding concluding/evaluative sentences.

5. **Cross-finding synthesis**: combining two narrow findings into an inferred general
   conclusion (e.g. "hallucinations exist" + "one specific shortcoming" -> "therefore
   LLMs are unreliable") that neither finding stated alone. Fixed with a self-check rule:
   only combine findings into one sentence if at least one finding already states that
   exact conclusion; otherwise list facts separately.

6. **Direction/polarity reversal**: the Writer occasionally paraphrased a finding
   inaccurately in a way that reversed its meaning (a finding saying RAG "reduces" the
   need for retraining became "RAG may require... retraining" in the draft) - topically
   related, but backwards. The Critic was originally only checking topic overlap, not
   direction, so this slipped through as "supported." Fixed by having both the Writer
   avoid reversing direction when paraphrasing, and the Critic explicitly check direction
   and mark reversed claims as "contradicted."

7. **Evidence filter false positives silently dropping relevant findings**: a filter added
   to drop findings with unclear evidence (see Known limitations below) originally flagged
   ANY evidence starting with "it", "this", "they" etc. as too vague - including perfectly
   clear sentences like "This comprehensive analysis examines seven fundamental advantages
   of RAG...". On one run, this caused the ONLY benefit-related finding to be dropped,
   so a "benefits AND risks of RAG" question got answered with a risks-only draft that
   still passed as 100% supported (every remaining claim genuinely was well-cited - the
   system had no way to notice half the question went unanswered). Fixed by checking the
   word immediately AFTER the pronoun: "this COMPREHENSIVE analysis" (followed by an
   adjective/noun) is clear and kept; "it DOES not remove it entirely" (followed by a verb,
   nothing anchoring "it") is genuinely ambiguous and still correctly dropped.

**Known limitations, not fixed (deliberately scoped out):**

- **Duplicate/redundant claims** aren't detected. The Critic sometimes explicitly flags a
  claim as a duplicate in its feedback, but `decide()` only checks evidence-support, not
  redundancy, so a duplicated-but-accurate claim can still be shipped in the final draft.
- **Incomplete evidence snippets**: DuckDuckGo returns short snippets, not full article
  text. A code-level filter now drops evidence that's too short or opens with an unresolved
  pronoun ("it", "this", "they"...), but this is a blunt heuristic, not true source-quality
  filtering.
- **No source-quality tiering**: sources currently aren't ranked by reliability (e.g.
  official docs/research papers vs. vendor blogs vs. YouTube/LinkedIn posts), though the
  Researcher does flag known-weak sources per-finding in the `limitations` field. A proper
  Tier 1/2/3 source ranking, with the Researcher preferring higher tiers, would strengthen
  this further.
- **Live search nondeterminism**: the same question can return different search results
  (and therefore different findings, drafts, and revision counts) between runs, since
  DuckDuckGo results aren't fixed. This makes exact reproducibility harder to guarantee
  than with a static, curated source set.