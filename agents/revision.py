"""
Turns a CriticDecision into the revision_context string that agents.writer.write()
accepts. Lives separately from both agents so neither the Critic nor Writer needs
to know about the other's internal format directly - this is the "translation
layer" between them.

Also handles cleanup of approved drafts: stripping any claim the Critic marked
unsupported/contradicted, even if it was tolerated as a minor issue.
"""

from schemas.critic_schema import CriticDecision, VerdictLabel


OK_VERDICTS = {VerdictLabel.SUPPORTED, VerdictLabel.OPINION}


def build_revision_context(decision: CriticDecision) -> str:
    problem_verdicts = [v for v in decision.raw.verdicts if v.verdict not in OK_VERDICTS]

    if not problem_verdicts:
        issues_block = "(No specific claim issues, but overall the draft needs polishing.)"
    else:
        issues_block = "\n".join(
            f"- \"{v.claim}\" -> {v.verdict.value} ({v.severity.value}): {v.feedback}"
            for v in problem_verdicts
        )

    return f"""The previous draft was reviewed and needs revision. Here is what to fix:

{issues_block}

Overall feedback: {decision.raw.summary_feedback}

Rewrite the summary addressing these specific issues. Keep any claims not mentioned
above as they are - they were already approved. Follow all the original rules
(cite by finding_id, hedge per limitations, do not invent facts)."""


STRIP_VERDICTS = {VerdictLabel.UNSUPPORTED, VerdictLabel.CONTRADICTED}


def clean_approved_draft(draft_text: str, decision: CriticDecision) -> tuple[str, list[str]]:
    """Remove any sentence the Critic marked unsupported/contradicted, even
    if it was a tolerated minor issue that didn't block overall approval.

    Returns (cleaned_text, removed_claims) so the caller can log what was cut.
    """
    removed = []
    cleaned = draft_text

    for v in decision.raw.verdicts:
        if v.verdict in STRIP_VERDICTS and v.claim in cleaned:
            cleaned = cleaned.replace(v.claim, "").strip()
            removed.append(v.claim)

    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace(" .", ".").replace("..", ".")

    return cleaned, removed