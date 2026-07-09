"""
Metrics captured for one full pipeline run, per the Phase 6 requirements:
claim count, supported-claim %, citations, revision count, claims removed.

Computed by code from data already produced by the pipeline (RawCriticOutput,
Draft, CriticDecision) - no new LLM calls needed for evaluation itself.
"""

from pydantic import BaseModel, Field


class PipelineMetrics(BaseModel):
    question: str
    num_findings: int
    num_sources: int

    first_draft_supported_pct: float = Field(
        ..., description="% of checkable (non-opinion) claims marked 'supported' in round 0"
    )
    final_draft_supported_pct: float = Field(
        ..., description="% of checkable (non-opinion) claims marked 'supported' in the final round"
    )

    num_citations_final: int = Field(..., description="Count of [f_id] citation tags in the final draft text")
    revision_count: int = Field(..., description="How many revision rounds actually ran (0 = approved on first try)")
    unsupported_claims_removed: int = Field(
        default=0, description="Count of claims stripped by clean_approved_draft (tolerated-minor unsupported claims)"
    )
    final_approved: bool