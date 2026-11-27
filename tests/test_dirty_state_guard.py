from __future__ import annotations

from newsroom.dirty_state_guard import TransactionalStageGuard, compute_state_hash
from newsroom.models import NewsroomState, SourceDocument


def test_transactional_stage_guard_zero_dirty_state_across_20_fault_runs() -> None:
    sources = [
        SourceDocument(
            source_id="SRC-1",
            publisher="Grid Wire",
            headline="Capacity rose 18% across 14 substations",
            body="Primary filing confirms 18% increase across 14 substations.",
            published_iso="2026-11-01",
        ).model_dump()
    ]
    base_state: NewsroomState = {
        "topic": "Substation Audit",
        "sources": sources,
        "status": "researched",
        "error_log": [],
    }
    guard = TransactionalStageGuard(require_verified_draft=True)

    # 20 fault-injected runs (10 runtime exceptions + 10 ungrounded/fabricated drafts)
    for idx in range(20):
        baseline_topic = base_state["topic"]
        baseline_status = base_state["status"]

        def faulty_stage(working: NewsroomState, i: int = idx) -> NewsroomState:
            # Mutate working copy mid-flight before failing
            working["status"] = "corrupted_partial_mutation"
            if i % 2 == 0:
                raise RuntimeError(f"interrupted_tool_write_{i}")
            return {
                "status": "corrupted_draft",
                "draft": {
                    "headline": "Bad Draft",
                    "nut_graf": "Bad",
                    "paragraphs": ["Fabricated 999% metric [SRC-UNKNOWN]."],
                    "cited_source_ids": ["SRC-UNKNOWN"],
                    "revision_number": 1,
                },
            }

        recovered, ok, _ = guard.execute_atomic(base_state, faulty_stage)
        assert ok is False
        assert recovered["topic"] == baseline_topic
        assert recovered["status"] == baseline_status
        assert "draft" not in recovered

    assert guard.rollback_count == 20

    # Valid grounded mutation commits cleanly
    def clean_stage(working: NewsroomState) -> NewsroomState:
        return {
            "status": "awaiting_human_approval",
            "draft": {
                "headline": "Substation Audit — Verified Briefing",
                "nut_graf": "Summary",
                "paragraphs": ["Capacity rose 18% across 14 substations [SRC-1]."],
                "cited_source_ids": ["SRC-1"],
                "revision_number": 1,
            },
        }

    committed, ok2, post_hash = guard.execute_atomic(base_state, clean_stage)
    assert ok2 is True
    assert committed["status"] == "awaiting_human_approval"
    assert post_hash == compute_state_hash(committed)
