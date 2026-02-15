from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

from langsmith import traceable
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from repoops.common.logging import setup_logging
from repoops.common.config import WORKSPACE_ROOT
from repoops.db.session import SessionLocal
from repoops.db.models import Run, Artifact, ToolCall
from repoops.db.init_db import init_db

from repoops.indexing.semantic_index import upsert_repo_semantic, delete_repo_semantic
from repoops.core.llm_router import LLMRouter
from repoops.core.orchestrator_graph import build_graph, Deps
from repoops.core.runner import build_context, git_changed_paths

from repoops.eval.rag_eval import compute_rag_metrics, metrics_to_dict

POLL_INTERVAL = 2.0

RUN_STALE_SECONDS = 60 * 30  # 30 minutes
HEARTBEAT_EVERY_SECONDS = 10


def main():
    setup_logging()
    init_db()
    print("[worker] started")

    while True:
        try:
            session = SessionLocal()
            try:
                _mark_stale_runs(session)
            finally:
                session.close()

            run_one()
        except Exception as exc:
            print("[worker] loop error:", exc)

        time.sleep(POLL_INTERVAL)


def _mark_stale_runs(session: Session) -> None:
    cutoff = datetime.utcnow() - timedelta(seconds=RUN_STALE_SECONDS)
    with session.begin():
        session.execute(
            update(Run)
            .where(Run.status == "RUNNING")
            .where(Run.updated_at < cutoff)
            .values(
                status="FAILED",
                error="Worker stale timeout (no heartbeat)",
                updated_at=datetime.utcnow(),
            )
        )


def _claim_next_run(session: Session) -> Run | None:
    """
    Atomic claim:
    - locks one PENDING row
    - SKIP LOCKED prevents other workers from taking the same run
    """
    with session.begin():
        run = (
            session.execute(
                select(Run)
                .where(Run.status == "PENDING")
                .order_by(Run.created_at.asc())
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .first()
        )
        if not run:
            return None

        run.status = "RUNNING"
        run.updated_at = datetime.utcnow()
        return run


@traceable(name="repoops.run", run_type="chain")
def _invoke_graph(graph, state_in):
    return graph.invoke(state_in)


def run_one():
    session = SessionLocal()
    run: Run | None = None

    try:
        run = _claim_next_run(session)
        if not run:
            return

        last_hb = time.time()

        def heartbeat(force: bool = False):
            nonlocal last_hb
            if force or (time.time() - last_hb >= HEARTBEAT_EVERY_SECONDS):
                run.updated_at = datetime.utcnow()
                session.commit()
                last_hb = time.time()

        def checkpoint(note: str | None = None):
            if note:
                _artifact(session, run.id, "notes", note[:20000])
            heartbeat(force=True)

        ws_root = Path(WORKSPACE_ROOT) / run.id
        repo_dir = ws_root / "repo"

        if ws_root.exists():
            shutil.rmtree(ws_root)
        repo_dir.mkdir(parents=True, exist_ok=True)

        _artifact(session, run.id, "notes", f"Workspace: {repo_dir}")
        checkpoint()

        _clone_repo(session, run.id, run.repo_url, run.ref, str(repo_dir))
        checkpoint("Cloned repo.")

        # Semantic index (best effort)
        try:
            try:
                delete_repo_semantic(repo_id=run.id)
            except Exception:
                pass

            upsert_repo_semantic(repo_id=run.id, repo_root=str(repo_dir), max_files=600)
            _artifact(session, run.id, "notes", "Semantic index updated (pgvector).")
        except Exception as exc:
            _artifact(session, run.id, "notes", f"Semantic indexing skipped: {exc}")
        checkpoint()

        llm = LLMRouter()
        if not llm.llm_enabled():
            run.status = "NEEDS_CONFIG"
            run.error = "LLM not configured. Set OPENAI_BASE_URL + OPENAI_MODEL (and OPENAI_API_KEY if OpenAI)."
            run.updated_at = datetime.utcnow()
            session.commit()
            _artifact(session, run.id, "pr_text", "LLM disabled. Configure LLM settings to generate edits.")
            session.commit()
            return

        # Build context (hybrid + optional rerank)
        ctx = build_context(run.id, str(repo_dir), run.task_text, llm=llm)
        _artifact(session, run.id, "repo_overview", json.dumps(ctx, ensure_ascii=False)[:20000])
        try:
            _artifact(session, run.id, "rag_trace", json.dumps(ctx.get("rag_trace") or {}, ensure_ascii=False)[:20000])
        except Exception:
            pass
        checkpoint("Context built.")

        def artifact_cb(kind: str, content: str):
            _artifact(session, run.id, kind, content[:20000])

        deps = Deps(llm=llm, artifact_cb=artifact_cb)
        graph = build_graph(deps)

        state_in = {
            "run_id": run.id,
            "repo_id": run.id,
            "repo_root": str(repo_dir),
            "task_text": run.task_text,
            "ctx": ctx,
            "commands_profile": json.loads(run.commands_profile) if run.commands_profile else {},
            "prefer_strong": False,
            "fix_attempt": 0,
        }

        state_out = _invoke_graph(graph, state_in)
        checkpoint("Graph completed.")

        # -------------------------
        # Automatic RAG evaluation
        # -------------------------
        try:
            retrieved = [
                (snippet.get("path") or "").replace("\\", "/")
                for snippet in (ctx.get("snippets") or [])
                if isinstance(snippet, dict) and snippet.get("path")
            ]
            edited = git_changed_paths(str(repo_dir))
            metrics = compute_rag_metrics(retrieved_paths=retrieved, edited_paths=edited, k=20)
            _artifact(session, run.id, "rag_eval", json.dumps(metrics_to_dict(metrics), ensure_ascii=False)[:20000])
        except Exception as exc:
            _artifact(session, run.id, "notes", f"[rag_eval] skipped: {str(exc)[:400]}")

        logs = state_out.get("verification_logs") or {}
        failed = "failed_step" in logs

        run.status = "FAILED" if failed else "DONE"
        run.error = f"Verification failed at step: {logs.get('failed_step')}" if failed else None
        run.updated_at = datetime.utcnow()
        session.commit()

    except Exception as exc:
        if run is not None:
            run.status = "FAILED"
            run.error = str(exc)[:2000]
            run.updated_at = datetime.utcnow()
            session.commit()
            _artifact(session, run.id, "error", str(exc)[:20000])
            session.commit()
        raise
    finally:
        session.close()


def _clone_repo(session, run_id: str, repo_url: str, ref: str, dest: str):
    _tool(session, run_id, "git.clone", {"repo_url": repo_url, "ref": ref, "dest": dest})

    clone_proc = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if clone_proc.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{(clone_proc.stdout or '')[:4000]}")

    checkout_proc = subprocess.run(
        ["git", "checkout", ref],
        cwd=dest,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if checkout_proc.returncode != 0:
        raise RuntimeError(f"git checkout {ref} failed:\n{(checkout_proc.stdout or '')[:4000]}")

    subprocess.run(
        ["git", "checkout", "-b", "repoops/run"],
        cwd=dest,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _artifact(session, run_id: str, kind: str, content: str):
    session.add(Artifact(run_id=run_id, kind=kind, content=content))
    session.flush()


def _tool(session, run_id: str, tool: str, args: dict, result_summary: str = "ok", duration_ms: int = 0):
    session.add(
        ToolCall(
            run_id=run_id,
            tool=tool,
            args_json=json.dumps(args),
            result_summary=result_summary,
            duration_ms=duration_ms,
        )
    )
    session.flush()


if __name__ == "__main__":
    main()
