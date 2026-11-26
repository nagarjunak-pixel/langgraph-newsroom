from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class SourceDocument(BaseModel):
    source_id: str = Field(..., min_length=2)
    publisher: str = Field(..., min_length=2)
    headline: str = Field(..., min_length=5)
    body: str = Field(..., min_length=15)
    published_iso: str = Field(..., min_length=10)

    def sha256_digest(self) -> str:
        canonical = f"{self.publisher}|{self.headline}|{self.body.strip()}".encode()
        return hashlib.sha256(canonical).hexdigest()


class ExtractedClaim(BaseModel):
    claim_id: str
    statement: str
    source_ids: list[str]
    numeric_tokens: list[str]


class ArticleDraft(BaseModel):
    headline: str
    nut_graf: str
    paragraphs: list[str]
    cited_source_ids: list[str]
    revision_number: int = 1


class VerificationReport(BaseModel):
    passed: bool
    citation_coverage: float
    unverified_numbers: list[str]
    missing_sources: list[str]
    risk_flags: list[str]


class EditorialReview(BaseModel):
    decision: ApprovalDecision
    reviewer: str
    notes: str = ""


class NewsroomState(TypedDict, total=False):
    topic: str
    sources: list[dict[str, str]]
    deduplicated_digests: list[str]
    claims: list[dict[str, object]]
    draft: dict[str, object]
    verification: dict[str, object]
    editorial_decision: str
    editorial_reviewer: str
    editorial_notes: str
    revision_count: int
    max_revisions: int
    error_log: list[str]
    recovery_attempts: int
    published_package: dict[str, object] | None
    status: str
