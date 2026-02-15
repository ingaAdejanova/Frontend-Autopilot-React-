import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from sqlalchemy import select, desc
from repoops.db.session import SessionLocal
from repoops.db.models import Run, Artifact
from repoops.shared.schemas import RunCreate, RunOut, ArtifactOut

router = APIRouter()

@router.post("/runs", response_model=dict)
def create_run(payload: RunCreate):
    session = SessionLocal()
    try:
        run = Run(
            repo_url=payload.repo_url,
            ref=payload.ref,
            task_text=payload.task_text,
            status="PENDING",
            commands_profile=json.dumps(payload.commands_profile) if payload.commands_profile else None,
            updated_at=datetime.utcnow(),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return {"run_id": run.id}
    finally:
        session.close()

@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str):
    session = SessionLocal()
    try:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return RunOut(
            id=run.id,
            repo_url=run.repo_url,
            ref=run.ref,
            task_text=run.task_text,
            status=run.status,
            created_at=run.created_at.isoformat(),
            updated_at=run.updated_at.isoformat(),
            error=run.error,
        )
    finally:
        session.close()

@router.get("/runs/{run_id}/artifacts", response_model=list[ArtifactOut])
def get_artifacts(run_id: str):
    session = SessionLocal()
    try:
        artifacts = (
            session.execute(select(Artifact).where(Artifact.run_id == run_id).order_by(desc(Artifact.created_at)))
            .scalars()
            .all()
        )
        return [
            ArtifactOut(kind=artifact.kind, content=artifact.content, created_at=artifact.created_at.isoformat())
            for artifact in artifacts
        ]
    finally:
        session.close()
