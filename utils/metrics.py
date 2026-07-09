"""
Pure computation functions for Phase 6 metrics. Kept separate from
schemas/metrics_schema.py (the data shape) and main.py (the orchestration) -
these are just calculations over data the pipeline already produced.
"""

import re
from schemas.critic_schema import RawCriticOutput, VerdictLabel


def supported_pct(raw: RawCriticOutput) -> float:
    """% of checkable claims (excluding 'opinion', which isn't a factual
    claim at all) that were marked 'supported'. Returns 0.0 if there were
    no checkable claims, to avoid a divide-by-zero on an all-opinion draft."""
    checkable = [v for v in raw.verdicts if v.verdict != VerdictLabel.OPINION]
    if not checkable:
        return 0.0
    supported = sum(1 for v in checkable if v.verdict == VerdictLabel.SUPPORTED)
    return round(100 * supported / len(checkable), 1)


def count_citations(draft_text: str) -> int:
    """Counts [f1], [f2], etc. tags in the draft text. Uses regex here
    (not string.count) because a claim can cite multiple findings back to
    back like [f1][f3], and we want each bracket group counted once."""
    return len(re.findall(r"\[f\d+\]", draft_text))