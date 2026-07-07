import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from agents.critic import critique
from schemas.research_schema import ResearchOutput
from schemas.draft_schema import Draft
from schemas.decision_rules import decide
from utils.storage import load_json, save_json

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# python -m scripts.check_critic <research_json> <draft_json>
research_path = sys.argv[1]
draft_path = sys.argv[2]

research = load_json(Path(research_path), ResearchOutput)
draft = load_json(Path(draft_path), Draft)

raw = critique(draft.text, research, api_key)
decision = decide(raw)

print("=== VERDICTS ===")
for v in raw.verdicts:
    print(f"[{v.verdict.value} | {v.severity.value}] {v.claim}")
    print(f"    finding_ids: {v.finding_ids}")
    print(f"    feedback: {v.feedback}")

print(f"\nSummary feedback: {raw.summary_feedback}")
print(f"\nMajor issues: {decision.major_issues}")
print(f"Minor issues: {decision.minor_issues}")
print(f"APPROVED: {decision.approved}")

saved_path = save_json(decision, "critic_decision")
print(f"\nSaved to: {saved_path}")