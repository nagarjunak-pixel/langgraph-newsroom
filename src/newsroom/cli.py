from __future__ import annotations

import json

import typer

from newsroom.models import ApprovalDecision, SourceDocument
from newsroom.pipeline import NewsroomPipeline

app = typer.Typer(
    help="3-stage LangGraph newsroom CLI with SQLite checkpointing and HITL approval gates."
)


@app.command("demo")
def run_demo(db_path: str = "newsroom_checkpoints.sqlite3") -> None:
    """Runs an end-to-end 3-stage checkpointed newsroom cycle with human approval."""
    pipeline = NewsroomPipeline(sqlite_path=db_path)
    try:
        sources = [
            SourceDocument(
                source_id="SRC-1",
                publisher="Grid Reliability Wire",
                headline="Regional grid storage capacity rises 18% in Q3 filing",
                body=(
                    "Utility filings show battery storage dispatch rose 18% "
                    "across 14 substations during peak load."
                ),
                published_iso="2026-11-01",
            ),
            SourceDocument(
                source_id="SRC-2",
                publisher="Energy Systems Bulletin",
                headline="Substation telemetry confirms peak curtailment drop",
                body=(
                    "Independent market monitor data recorded a 12% drop in "
                    "curtailment across 14 substations."
                ),
                published_iso="2026-11-02",
            ),
        ]
        paused = pipeline.start_run("demo-thread-01", "Grid Storage Dispatch", sources)
        typer.echo(f"Paused at gate: status={paused['status']}")
        final = pipeline.submit_human_decision(
            "demo-thread-01",
            ApprovalDecision.APPROVE,
            reviewer="senior-editor",
            notes="Verified primary utility filings.",
        )
        typer.echo(json.dumps({"status": final["status"], "draft": final["draft"]}, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    app()
