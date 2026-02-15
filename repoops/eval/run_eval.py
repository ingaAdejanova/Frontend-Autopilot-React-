from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from repoops.common.config import (
    EMBEDDINGS_MODEL,
    OPENAI_MODEL,
    RAG_RERANK_MODEL,
    LEXICAL_PRIMARY,
    ZOEK_URL,
)
from repoops.core.llm_router import LLMRouter
from repoops.eval.rag_eval import compute_rag_metrics, metrics_to_dict
from repoops.indexing.semantic_index import upsert_repo_semantic
from repoops.retrieval.hybrid import hybrid_search


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = (line or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            continue
    return items


def _unique_paths(snips: List[Any], k: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for snippet in snips:
        path = None
        if isinstance(snippet, dict):
            path = snippet.get("path")
        else:
            path = getattr(snippet, "path", None)
        if not isinstance(path, str):
            continue
        normalized_path = path.replace("\\", "/")
        if normalized_path in seen:
            continue
        seen.add(normalized_path)
        out.append(normalized_path)
        if len(out) >= k:
            break
    return out


def _repo_slug(url: str) -> str:
    base = url.rstrip("/").split("/")[-1] or "repo"
    if base.endswith(".git"):
        base = base[:-4]
    safe = "".join([char if (char.isalnum() or char in "-_") else "-" for char in base])
    return safe or "repo"


def _clone_repo(repo_url: str, ref: str, clone_root: str, *, force: bool) -> str:
    root = Path(clone_root)
    root.mkdir(parents=True, exist_ok=True)

    slug = _repo_slug(repo_url)
    hash_prefix = hashlib.sha1(f"{repo_url}@{ref}".encode("utf-8")).hexdigest()[:8]
    target = root / f"{slug}-{hash_prefix}"

    if target.exists():
        if force:
            shutil.rmtree(target)
        else:
            if (target / ".git").exists():
                # best-effort update + checkout
                subprocess.run(["git", "fetch", "--depth", "1", "origin", ref], cwd=str(target), check=False)
                subprocess.run(["git", "checkout", ref], cwd=str(target), check=False)
                return str(target)
            raise RuntimeError(f"Target exists and is not a git repo: {target}. Use --force-clone.")

    clone_proc = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if clone_proc.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{(clone_proc.stdout or '')[:1200]}")

    checkout_proc = subprocess.run(
        ["git", "checkout", ref],
        cwd=str(target),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if checkout_proc.returncode != 0:
        # fallback to default branch
        default_ref_proc = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
            cwd=str(target),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        default_ref = ""
        if default_ref_proc.returncode == 0:
            out = (default_ref_proc.stdout or "").strip()
            if out.startswith("refs/remotes/origin/"):
                default_ref = out.replace("refs/remotes/origin/", "", 1)

        if default_ref:
            fallback_proc = subprocess.run(
                ["git", "checkout", default_ref],
                cwd=str(target),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if fallback_proc.returncode == 0:
                return str(target)

        raise RuntimeError(
            f"git checkout {ref} failed:\n{(checkout_proc.stdout or '')[:1200]}\n"
            f"Default branch not detected (or checkout failed). Try --ref master or the repo default branch."
        )

    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval eval on a JSONL dataset.")
    parser.add_argument("--dataset", default="repoops/eval/data/sample_tasks.jsonl")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo-url", default="", help="Optional git URL to clone for evaluation.")
    parser.add_argument("--ref", default="main", help="Git ref/branch/sha to checkout when cloning.")
    parser.add_argument("--clone-root", default="/workspaces/eval", help="Where to clone the repo when using --repo-url.")
    parser.add_argument("--force-clone", action="store_true", help="Delete existing clone target before cloning.")
    parser.add_argument("--repo-id", default="local-eval")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--index", action="store_true", help="Rebuild semantic index before running.")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM rerank during eval.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--label", default="", help="Label for this run (e.g. model name).")
    args = parser.parse_args()

    if args.repo_url:
        repo_root = _clone_repo(args.repo_url, args.ref, args.clone_root, force=args.force_clone)
    else:
        repo_root = os.path.abspath(args.repo_root)
    dataset_path = os.path.abspath(args.dataset)

    items = _load_jsonl(dataset_path)
    if not items:
        print("No dataset items found.")
        return 2

    if args.index:
        print("Rebuilding semantic index...")
        upsert_repo_semantic(repo_id=args.repo_id, repo_root=repo_root, max_files=600)

    llm = LLMRouter() if args.use_llm else None

    results: List[Dict[str, Any]] = []
    totals = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    counted = 0

    start_time = time.time()
    for item in items:
        task = item.get("task") or ""
        expected = item.get("expected_paths") or []
        if not isinstance(expected, list) or not isinstance(task, str) or not task.strip():
            continue

        hybrid_result = hybrid_search(
            repo_id=args.repo_id,
            repo_root=repo_root,
            query=task,
            k=args.k,
            llm=llm,
        )

        retrieved_paths = _unique_paths(hybrid_result.get("snippets") or [], args.k)
        metrics = compute_rag_metrics(
            retrieved_paths=retrieved_paths,
            edited_paths=[str(path) for path in expected if path],
            k=args.k,
        )

        totals["recall"] += metrics.recall_files_at_k
        totals["precision"] += metrics.precision_files_at_k
        totals["mrr"] += metrics.mrr_files
        totals["ndcg"] += metrics.ndcg_files
        counted += 1

        results.append(
            {
                "id": item.get("id"),
                "task": task,
                "expected_paths": expected,
                "retrieved_paths": retrieved_paths,
                "metrics": metrics_to_dict(metrics),
                "trace": hybrid_result.get("trace") or {},
            }
        )

    avg = {}
    if counted:
        avg = {
            "recall_files_at_k": round(totals["recall"] / counted, 4),
            "precision_files_at_k": round(totals["precision"] / counted, 4),
            "mrr_files": round(totals["mrr"] / counted, 4),
            "ndcg_files": round(totals["ndcg"] / counted, 4),
            "count": counted,
        }

    meta = {
        "label": args.label or "",
        "dataset": dataset_path,
        "repo_root": repo_root,
        "repo_id": args.repo_id,
        "repo_url": args.repo_url or "",
        "ref": args.ref if args.repo_url else "",
        "k": args.k,
        "embeddings_model": EMBEDDINGS_MODEL,
        "llm_model": OPENAI_MODEL,
        "rerank_model": RAG_RERANK_MODEL,
        "lexical_primary": LEXICAL_PRIMARY,
        "zoekt_url": ZOEK_URL or "",
        "use_llm": bool(args.use_llm),
        "elapsed_s": round(time.time() - start_time, 2),
    }

    output = {"meta": meta, "avg": avg, "results": results}
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
