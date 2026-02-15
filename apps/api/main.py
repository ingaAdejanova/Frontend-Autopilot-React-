from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.routes_runs import router as runs_router
from repoops.common.logging import setup_logging
from repoops.db.init_db import init_db
from repoops.indexing.semantic_index import ensure_pgvector

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        ensure_pgvector()
    except Exception as exc:
        # API should still run even if pgvector init fails
        print("[api] pgvector init warning:", exc)
    yield


app = FastAPI(title="RepoOps Autopilot (Backend)", version="0.1.0", lifespan=lifespan)
app.include_router(runs_router, prefix="")


@app.get("/health")
def health():
    return {"ok": True}
