from __future__ import annotations

import re

from newsroom.models import ArticleDraft, SourceDocument, VerificationReport

CITATION_RE = re.compile(r"\[(SRC-[A-Za-z0-9_-]+)\]")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|(?=\b))")


def extract_numeric_tokens(text: str) -> list[str]:
    return sorted(set(NUMBER_RE.findall(text)))


def verify_draft_grounding(
    draft: ArticleDraft,
    sources: list[SourceDocument],
) -> VerificationReport:
    """Checks that every paragraph carries valid [SRC-*] citations and grounded numbers."""
    known_ids = {s.source_id for s in sources}
    corpus_numbers: set[str] = set()
    for src in sources:
        corpus_numbers.update(extract_numeric_tokens(f"{src.headline} {src.body}"))

    cited_paragraphs = 0
    referenced_ids: set[str] = set()
    draft_numbers: set[str] = set()

    for para in draft.paragraphs:
        tags = CITATION_RE.findall(para)
        if tags:
            cited_paragraphs += 1
            referenced_ids.update(tags)
        cleaned = CITATION_RE.sub("", para)
        draft_numbers.update(extract_numeric_tokens(cleaned))

    total_paras = max(1, len(draft.paragraphs))
    coverage = round(cited_paragraphs / total_paras, 4)
    missing_sources = sorted(referenced_ids - known_ids)
    unverified_numbers = sorted(draft_numbers - corpus_numbers)

    risk_flags: list[str] = []
    if len(sources) < 2:
        risk_flags.append("single_source_corroboration_warning")
    if coverage < 1.0:
        risk_flags.append("uncited_paragraph_detected")
    if missing_sources:
        risk_flags.append(f"unknown_citations:{','.join(missing_sources)}")
    if unverified_numbers:
        risk_flags.append(f"unverified_numbers:{','.join(unverified_numbers)}")

    passed = coverage == 1.0 and not missing_sources and not unverified_numbers
    return VerificationReport(
        passed=passed,
        citation_coverage=coverage,
        unverified_numbers=unverified_numbers,
        missing_sources=missing_sources,
        risk_flags=risk_flags,
    )
