from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from newsroom.api import create_app
from newsroom.cli import app as cli_app
from newsroom.models import ApprovalDecision, ArticleDraft, ExtractedClaim, SourceDocument
from newsroom.pipeline import NewsroomPipeline, default_draft_generator
from newsroom.verifier import verify_draft_grounding


def sample_sources() -> list[SourceDocument]:
    return [
        SourceDocument(
            source_id="SRC-1",
            publisher="Enterprise Infrastructure Wire",
            headline="Cloud database failover latency drops 42% in benchmark audit",
            body=(
                "Multi-region consensus logs showed failover latency fell 42% "
                "across 120 test clusters."
            ),
            published_iso="2026-10-15",
        ),
        SourceDocument(
            source_id="SRC-2",
            publisher="Systems Engineering Digest",
            headline="Independent audit validates zero packet loss during regional switchover",
            body=(
                "Auditors confirmed 120 test clusters completed switchover with "
                "0% committed transaction loss."
            ),
            published_iso="2026-10-16",
        ),
    ]


def test_source_dedup_and_happy_path_sqlite_checkpoint(tmp_path: Path) -> None:
    db_file = str(tmp_path / "checkpoints.sqlite")
    pipeline = NewsroomPipeline(sqlite_path=db_file)
    try:
        duped = sample_sources() + [sample_sources()[0]]
        paused = pipeline.start_run("run-101", "Database Failover Audit", duped)
        assert paused["status"] == "awaiting_human_approval"
        assert len(paused["sources"]) == 2
        assert paused["verification"]["passed"] is True
        assert paused["published_package"] is None

        # Re-open from SQLite checkpoint to prove durable state persistence across restarts
        pipeline.close()
        reopened = NewsroomPipeline(sqlite_path=db_file)
        try:
            restored = reopened.get_checkpoint_state("run-101")
            assert restored["topic"] == "Database Failover Audit"

            published = reopened.submit_human_decision(
                "run-101",
                ApprovalDecision.APPROVE,
                reviewer="desk-editor",
                notes="All numeric claims verified against SRC-1 and SRC-2.",
            )
            assert published["status"] == "published"
            assert published["published_package"] is not None
            assert published["published_package"]["approved_by"] == "desk-editor"
        finally:
            reopened.close()
    finally:
        pipeline.close()


def test_human_revision_loop_and_rejection_gate() -> None:
    pipeline = NewsroomPipeline()
    paused = pipeline.start_run("run-rev", "Failover Telemetry", sample_sources())
    assert paused["revision_count"] == 1

    # Editor requests revision -> loops back to Stage 2 and pauses again at editorial_gate
    revised = pipeline.submit_human_decision(
        "run-rev",
        ApprovalDecision.REVISE,
        reviewer="managing-editor",
        notes="Clarify cluster sample size",
    )
    assert revised["revision_count"] == 2
    paras = revised["draft"]["paragraphs"]
    assert isinstance(paras, list)
    assert "Clarify cluster sample size" in str(paras[0])

    # Editor rejects -> transitions cleanly to archive_rejected without publishing
    rejected = pipeline.submit_human_decision(
        "run-rev",
        ApprovalDecision.REJECT,
        reviewer="managing-editor",
        notes="Hold pending Q4 addendum",
    )
    assert rejected["status"] == "rejected"
    assert rejected["published_package"] is None


def test_error_recovery_when_upstream_generator_raises_or_hallucinates() -> None:
    calls = {"n": 0}

    def flaky_generator(
        topic: str,
        claims: list[ExtractedClaim],
        revision: int,
        notes: str,
    ) -> ArticleDraft:
        calls["n"] += 1
        if calls["n"] == 1:
            # Inject an unverified number + uncited paragraph so verifier triggers bounded retry
            return ArticleDraft(
                headline="Uncited Draft",
                nut_graf="Summary",
                paragraphs=["Latency dropped 999% with no source tag."],
                cited_source_ids=[],
                revision_number=revision,
            )
        if calls["n"] == 2:
            raise RuntimeError("transient upstream model timeout")
        return default_draft_generator(topic, claims, revision, notes)

    pipeline = NewsroomPipeline(draft_generator=flaky_generator, max_revisions=3)
    paused = pipeline.start_run("run-recover", "Resilience Test", sample_sources())
    assert paused["revision_count"] == 2
    assert paused["recovery_attempts"] == 1
    assert any("verification_retry" in e for e in paused["error_log"])
    assert any("draft_generator_error" in e for e in paused["error_log"])
    assert paused["verification"]["passed"] is True


def test_unverified_draft_cannot_publish_even_if_forced() -> None:
    def bad_generator(
        topic: str,
        claims: list[ExtractedClaim],
        revision: int,
        notes: str,
    ) -> ArticleDraft:
        return ArticleDraft(
            headline="Bad Draft",
            nut_graf="Bad",
            paragraphs=["Fabricated 777% metric [SRC-999]."],
            cited_source_ids=["SRC-999"],
            revision_number=revision,
        )

    pipeline = NewsroomPipeline(draft_generator=bad_generator, max_revisions=1)
    paused = pipeline.start_run("run-block", "Blocked Story", [sample_sources()[0]])
    assert paused["verification"]["passed"] is False
    flags = paused["verification"]["risk_flags"]
    assert isinstance(flags, list)
    assert "single_source_corroboration_warning" in flags

    blocked = pipeline.submit_human_decision(
        "run-block",
        ApprovalDecision.APPROVE,
        reviewer="editor",
    )
    assert blocked["status"] == "blocked_unverified_draft"
    assert blocked["published_package"] is None


def test_empty_sources_rejected_and_verifier_details() -> None:
    pipeline = NewsroomPipeline()
    with pytest.raises(ValueError, match="at least one primary SourceDocument"):
        pipeline.start_run("empty", "Empty", [])

    report = verify_draft_grounding(
        ArticleDraft(
            headline="Test",
            nut_graf="Nut",
            paragraphs=["Uncited claim with 88% metric."],
            cited_source_ids=[],
        ),
        sample_sources(),
    )
    assert not report.passed
    assert "88%" in report.unverified_numbers


def test_fastapi_endpoints_and_cli_demo(tmp_path: Path) -> None:
    client = TestClient(create_app(NewsroomPipeline()))
    bad = client.post("/runs", json={"thread_id": "t0", "topic": "Empty", "sources": []})
    assert bad.status_code == 422

    missing = client.get("/runs/nonexistent")
    assert missing.status_code == 404
    missing_dec = client.post(
        "/runs/nonexistent/decision",
        json={"decision": "approve", "reviewer": "ed"},
    )
    assert missing_dec.status_code == 404

    res = client.post(
        "/runs",
        json={
            "thread_id": "api-01",
            "topic": "Failover Audit",
            "sources": [s.model_dump() for s in sample_sources()],
        },
    )
    assert res.status_code == 200
    assert res.json()["status"] == "awaiting_human_approval"

    inspected = client.get("/runs/api-01")
    assert inspected.status_code == 200

    approved = client.post(
        "/runs/api-01/decision",
        json={"decision": "approve", "reviewer": "api-editor", "notes": "LGTM"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "published"

    runner = CliRunner()
    cli_res = runner.invoke(cli_app, ["--db-path", str(tmp_path / "cli.sqlite")])
    assert cli_res.exit_code == 0
    assert '"status": "published"' in cli_res.stdout
