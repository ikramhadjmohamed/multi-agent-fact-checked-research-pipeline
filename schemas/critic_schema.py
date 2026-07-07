"""
Schema for the Critic agent's output.

The Critic's job is verification, not creation — it never invents new facts,
it only compares the Writer's draft against the Researcher's ResearchOutput
(see research_schema.py) and reports what it finds.
"""

from enum import Enum
from pydantic import BaseModel, Field


class VerdictLabel(str, Enum):
    """Using an Enum (not a plain str) means Pydantic rejects invalid values
    at validation time. If the LLM returns "unsuported" (typo) or invents
    a new label, this raises immediately instead of the bug surviving
    silently until someone notices the pipeline behaving oddly downstream.

    Inheriting from `str` as well as `Enum` means it still serializes to
    plain JSON strings (e.g. "unsupported"), which keeps main.py and any
    logging/printing simple — you're not stuck unwrapping Enum objects
    everywhere.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    OPINION = "opinion"


class Severity(str, Enum):
    """How central this claim is to the draft's core argument.

    Why the Critic (not code) decides this: judging "is this claim central
    to the topic or a side detail" needs actual reading comprehension —
    there's no reliable heuristic (sentence length, position) that
    substitutes for that.
    """

    MAJOR = "major"
    MINOR = "minor"


class Verdict(BaseModel):
    """The Critic's judgment on ONE claim extracted from the Writer's draft."""

    claim: str = Field(..., description="The exact sentence/claim from the draft being checked")
    verdict: VerdictLabel
    severity: Severity = Field(..., description="Used by main.py's approval rule: major unsupported/contradicted claims block approval, minor ones don't (unless too many accumulate)")
    finding_ids: list[str] = Field(
        default_factory=list,
        description="Which Researcher findings support/contradict this claim. "
                    "A list, not a single id — one sentence can merge multiple findings."
    )
    feedback: str = Field(
        ..., description="What the Writer should do: remove, soften wording, add citation, etc. "
                          "Required even for SUPPORTED claims — can just say 'fine as-is'."
    )


class RawCriticOutput(BaseModel):
    """What the Critic LLM itself returns — no 'approved' field here.

    The LLM judges each claim (that needs its reading comprehension).
    Whether the DRAFT AS A WHOLE is approved is a judgment call with a
    rule you designed (major unsupported claims block approval; minor
    ones only block it past a threshold) — that rule lives in code, in
    one place, so it can't disagree with itself the way an LLM-set
    'approved' flag could disagree with its own per-claim verdicts.
    """

    verdicts: list[Verdict]
    summary_feedback: str = Field(
        default="", description="High-level notes for the Writer, on top of per-claim feedback"
    )


class CriticDecision(BaseModel):
    """Wraps RawCriticOutput with the code-computed approval decision.

    This is what main.py actually branches on. Built by a function like
    `decide(raw: RawCriticOutput) -> CriticDecision`, not by the LLM.
    """

    raw: RawCriticOutput
    approved: bool = Field(..., description="Computed by code, not the LLM. main.py's stopping condition.")
    major_issues: int = Field(..., description="Count of major claims with verdict UNSUPPORTED or CONTRADICTED")
    minor_issues: int = Field(..., description="Count of minor claims with verdict UNSUPPORTED or CONTRADICTED")