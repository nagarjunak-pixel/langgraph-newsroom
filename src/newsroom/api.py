from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from newsroom.models import ApprovalDecision, SourceDocument
from newsroom.pipeline import NewsroomPipeline


class StartRunPayload(BaseModel):
    thread_id: str
    topic: str
    sources: list[SourceDocument]


class DecisionPayload(BaseModel):
    decision: ApprovalDecision
    reviewer: str
    notes: str = ""


def create_app(pipeline: NewsroomPipeline | None = None) -> FastAPI:
    engine = pipeline or NewsroomPipeline()
    app = FastAPI(
        title="LangGraph Newsroom API",
        version="1.0.0",
        description=(
            "3-stage news production pipeline with human approval gates, "
            "checkpointing, and error recovery."
        ),
    )

    @app.post("/runs")
    def create_run(payload: StartRunPayload) -> dict[str, object]:
        try:
            state = engine.start_run(payload.thread_id, payload.topic, payload.sources)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return dict(state)

    @app.get("/runs/{thread_id}")
    def inspect_run(thread_id: str) -> dict[str, object]:
        state = engine.get_checkpoint_state(thread_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Unknown thread_id '{thread_id}'")
        return dict(state)

    @app.post("/runs/{thread_id}/decision")
    def submit_decision(thread_id: str, payload: DecisionPayload) -> dict[str, object]:
        existing = engine.get_checkpoint_state(thread_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Unknown thread_id '{thread_id}'")
        updated = engine.submit_human_decision(
            thread_id=thread_id,
            decision=payload.decision,
            reviewer=payload.reviewer,
            notes=payload.notes,
        )
        return dict(updated)

    return app


app = create_app()
