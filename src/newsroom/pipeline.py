from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from newsroom.models import (
    ApprovalDecision,
    ArticleDraft,
    ExtractedClaim,
    NewsroomState,
    SourceDocument,
)
from newsroom.verifier import extract_numeric_tokens, verify_draft_grounding

DraftGenerator = Callable[[str, list[ExtractedClaim], int, str], ArticleDraft]


def default_draft_generator(
    topic: str,
    claims: list[ExtractedClaim],
    revision: int,
    editor_notes: str,
) -> ArticleDraft:
    """Deterministic citation-grounded synthesis over verified source claims."""
    paragraphs: list[str] = []
    cited_ids: list[str] = []
    for claim in claims:
        tags = " ".join(f"[{sid}]" for sid in claim.source_ids)
        cited_ids.extend(claim.source_ids)
        sentence = f"{claim.statement.rstrip('.')} {tags}."
        if editor_notes and revision > 1 and len(paragraphs) == 0:
            sentence = f"Updated ({editor_notes.strip()}): {sentence}"
        paragraphs.append(sentence)

    nut_graf = (
        f"Synthesis across {len(set(cited_ids))} primary source filing(s) covering {topic} "
        f"(revision {revision})."
    )
    return ArticleDraft(
        headline=f"{topic} — Verified Briefing",
        nut_graf=nut_graf,
        paragraphs=paragraphs,
        cited_source_ids=sorted(set(cited_ids)),
        revision_number=revision,
    )


class NewsroomPipeline:
    """3-stage LangGraph newsroom with SQLite/Memory checkpointing, HITL gates & error recovery."""

    def __init__(
        self,
        sqlite_path: str | None = None,
        draft_generator: DraftGenerator | None = None,
        max_revisions: int = 3,
    ) -> None:
        self._draft_generator = draft_generator or default_draft_generator
        self._max_revisions = max_revisions
        self._conn: sqlite3.Connection | None = None
        if sqlite_path:
            self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            self._checkpointer: Any = SqliteSaver(self._conn)
        else:
            self._checkpointer = MemorySaver()
        self._graph = self._compile_graph()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _stage1_research(self, state: NewsroomState) -> NewsroomState:
        raw_sources = [SourceDocument.model_validate(s) for s in state.get("sources", [])]
        if not raw_sources:
            raise ValueError("Stage 1 research requires at least one primary SourceDocument.")

        seen_digests: set[str] = set()
        unique_sources: list[SourceDocument] = []
        for src in raw_sources:
            digest = src.sha256_digest()
            if digest not in seen_digests:
                seen_digests.add(digest)
                unique_sources.append(src)

        claims: list[dict[str, object]] = []
        for idx, src in enumerate(unique_sources, start=1):
            lead_sentence = src.body.strip().split(". ")[0].strip()
            claim = ExtractedClaim(
                claim_id=f"CLM-{idx:02d}",
                statement=f"{src.publisher} reported that {lead_sentence}",
                source_ids=[src.source_id],
                numeric_tokens=extract_numeric_tokens(f"{src.headline} {src.body}"),
            )
            claims.append(claim.model_dump())

        return {
            "sources": [s.model_dump() for s in unique_sources],
            "deduplicated_digests": sorted(seen_digests),
            "claims": claims,
            "revision_count": state.get("revision_count", 0),
            "max_revisions": state.get("max_revisions", self._max_revisions),
            "error_log": list(state.get("error_log", [])),
            "recovery_attempts": state.get("recovery_attempts", 0),
            "status": "researched",
        }

    def _stage2_draft_and_verify(self, state: NewsroomState) -> NewsroomState:
        revision = state.get("revision_count", 0) + 1
        claims = [ExtractedClaim.model_validate(c) for c in state.get("claims", [])]
        sources = [SourceDocument.model_validate(s) for s in state.get("sources", [])]
        notes = state.get("editorial_notes", "")
        errors = list(state.get("error_log", []))
        recoveries = state.get("recovery_attempts", 0)

        try:
            draft = self._draft_generator(state["topic"], claims, revision, notes)
        except Exception as exc:
            errors.append(f"draft_generator_error:r{revision}:{exc}")
            recoveries += 1
            draft = default_draft_generator(state["topic"], claims, revision, notes)

        report = verify_draft_grounding(draft, sources)
        if not report.passed and revision < state.get("max_revisions", self._max_revisions):
            errors.append(f"verification_retry:r{revision}:{','.join(report.risk_flags)}")

        return {
            "draft": draft.model_dump(),
            "verification": report.model_dump(),
            "revision_count": revision,
            "error_log": errors,
            "recovery_attempts": recoveries,
            "status": "awaiting_human_approval" if report.passed else "verification_failed",
        }

    def _route_after_verification(self, state: NewsroomState) -> str:
        verification = state.get("verification", {})
        passed = bool(verification.get("passed", False))
        rev = state.get("revision_count", 1)
        max_rev = state.get("max_revisions", self._max_revisions)
        if passed or rev >= max_rev:
            return "editorial_gate"
        return "draft_and_verify"

    def _editorial_gate(self, state: NewsroomState) -> NewsroomState:
        decision = state.get("editorial_decision", "")
        if not decision:
            return {"status": "awaiting_human_approval"}
        return {"status": f"decision_recorded:{decision}"}

    def _route_after_editorial_gate(self, state: NewsroomState) -> str:
        decision = state.get("editorial_decision", ApprovalDecision.APPROVE.value)
        if decision == ApprovalDecision.REVISE.value:
            return "draft_and_verify"
        if decision == ApprovalDecision.REJECT.value:
            return "archive_rejected"
        return "publish"

    def _stage3_publish(self, state: NewsroomState) -> NewsroomState:
        verification = state.get("verification", {})
        if not verification.get("passed", False):
            return {
                "published_package": None,
                "status": "blocked_unverified_draft",
            }
        package: dict[str, object] = {
            "topic": state["topic"],
            "draft": state["draft"],
            "verification": verification,
            "approved_by": state.get("editorial_reviewer", "managing-editor"),
            "editorial_notes": state.get("editorial_notes", ""),
            "published_at": datetime.now(UTC).isoformat(),
        }
        return {
            "published_package": package,
            "status": "published",
        }

    def _archive_rejected(self, state: NewsroomState) -> NewsroomState:
        return {
            "published_package": None,
            "status": "rejected",
        }

    def _compile_graph(self) -> Any:
        builder: StateGraph[Any] = StateGraph(NewsroomState)
        builder.add_node("research", self._stage1_research)
        builder.add_node("draft_and_verify", self._stage2_draft_and_verify)
        builder.add_node("editorial_gate", self._editorial_gate)
        builder.add_node("publish", self._stage3_publish)
        builder.add_node("archive_rejected", self._archive_rejected)

        builder.add_edge(START, "research")
        builder.add_edge("research", "draft_and_verify")
        builder.add_conditional_edges(
            "draft_and_verify",
            self._route_after_verification,
            {"draft_and_verify": "draft_and_verify", "editorial_gate": "editorial_gate"},
        )
        builder.add_conditional_edges(
            "editorial_gate",
            self._route_after_editorial_gate,
            {
                "draft_and_verify": "draft_and_verify",
                "publish": "publish",
                "archive_rejected": "archive_rejected",
            },
        )
        builder.add_edge("publish", END)
        builder.add_edge("archive_rejected", END)

        return builder.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["editorial_gate"],
        )

    def start_run(
        self,
        thread_id: str,
        topic: str,
        sources: list[SourceDocument],
    ) -> NewsroomState:
        config = {"configurable": {"thread_id": thread_id}}
        initial: NewsroomState = {
            "topic": topic,
            "sources": [s.model_dump() for s in sources],
            "revision_count": 0,
            "max_revisions": self._max_revisions,
            "error_log": [],
            "recovery_attempts": 0,
            "published_package": None,
            "status": "initialized",
        }
        result: NewsroomState = self._graph.invoke(initial, config=config)
        return result

    def get_checkpoint_state(self, thread_id: str) -> NewsroomState:
        config = {"configurable": {"thread_id": thread_id}}
        snap = self._graph.get_state(config)
        values: NewsroomState = dict(snap.values)  # type: ignore[assignment]
        return values

    def submit_human_decision(
        self,
        thread_id: str,
        decision: ApprovalDecision,
        reviewer: str,
        notes: str = "",
    ) -> NewsroomState:
        config = {"configurable": {"thread_id": thread_id}}
        self._graph.update_state(
            config,
            {
                "editorial_decision": decision.value,
                "editorial_reviewer": reviewer,
                "editorial_notes": notes,
            },
        )
        resumed: NewsroomState = self._graph.invoke(None, config=config)
        return resumed
