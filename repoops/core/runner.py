from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from repoops.common.config import WORKSPACE_ROOT, RAG_MAX_CONTEXT_CHARS
from repoops.tools.exec_sandbox import ExecError, exec_cmd
from repoops.tools.repo_fs import read_range, tree
from repoops.retrieval.hybrid import hybrid_search
from repoops.tools.ts_graph import collect_ts_graph

try:
    from repoops.core.llm_router import LLMRouter
except Exception:  # pragma: no cover
    LLMRouter = Any  # type: ignore


@dataclass
class CommandsProfile:
    install: Optional[str] = None
    lint: Optional[str] = None
    typecheck: Optional[str] = None
    test: Optional[str] = None
    build: Optional[str] = None


def build_context(repo_id: str, repo_root: str, task_text: str, llm: Optional[LLMRouter] = None) -> Dict[str, Any]:
    files = tree(repo_root, max_entries=200)

    query = _pick_query(task_text)
    hybrid_result = hybrid_search(
        repo_id=repo_id,
        repo_root=repo_root,
        query=query,
        k=12,
        filters=None,
        llm=llm,
    )

    context_snips: List[Dict[str, Any]] = []
    for doc in (hybrid_result.get("docs") or []):
        txt = doc.get("text") or ""
        if not txt:
            continue
        context_snips.append(
            {
                "path": doc["path"],
                "range": [int(doc["start_line"]), int(doc["end_line"])],
                "source": doc.get("source") or "semantic",
                "score": float(doc.get("base_score") or 0.0),
                "note": doc.get("reason") or doc.get("symbol") or "",
                "text": txt,
            }
        )

    # TypeScript graph expansion (defs/refs/import closure)
    try:
        trace = hybrid_result.get("trace") or {}
        top_files = trace.get("top_files") or []
        focus_files: List[str] = []
        for path in (top_files or []):
            if isinstance(path, str) and path not in focus_files:
                focus_files.append(path)
        for doc in (hybrid_result.get("docs") or []):
            path = doc.get("path")
            if isinstance(path, str) and path not in focus_files:
                focus_files.append(path)

        ts_spans = collect_ts_graph(repo_root, focus_files)
        seen = {(snippet["path"], int(snippet["range"][0]), int(snippet["range"][1])) for snippet in context_snips if snippet.get("path") and snippet.get("range")}
        total_chars = sum(len(snippet.get("text") or "") for snippet in context_snips)

        for span in ts_spans:
            key = (span.path, int(span.start_line), int(span.end_line))
            if key in seen:
                continue
            if total_chars >= RAG_MAX_CONTEXT_CHARS:
                break
            try:
                txt = read_range(repo_root, span.path, span.start_line, span.end_line)
            except Exception:
                continue
            if not txt:
                continue
            context_snips.append(
                {
                    "path": span.path,
                    "range": [int(span.start_line), int(span.end_line)],
                    "source": "ts_graph",
                    "score": 0.86,
                    "note": span.reason or "tsserver graph",
                    "text": txt,
                }
            )
            total_chars += len(txt)
            seen.add(key)
    except Exception:
        pass

    low = (task_text or "").lower()
    repo_root_path = Path(repo_root)

    def _force_include(rel: str, start: int = 1, end: int = 240, note: str = ""):
        try:
            file_path = repo_root_path / rel
            if file_path.exists() and file_path.is_file():
                full = read_range(repo_root, rel, start, end)
                context_snips.append(
                    {
                        "path": rel,
                        "range": [start, end],
                        "source": "forced",
                        "score": 0.92,
                        "note": note or "SPA wiring",
                        "text": full,
                    }
                )
        except Exception:
            return

    # React SPA wiring files (high value)
    for rel in ("src/main.tsx", "src/index.tsx", "src/main.jsx", "src/index.jsx"):
        _force_include(rel, 1, 220, "Entry point")

    for rel in ("src/App.tsx", "src/App.jsx", "src/app.tsx", "src/app.jsx"):
        _force_include(rel, 1, 260, "App root")

    # Router hints
    if any(x in low for x in ("router", "route", "react-router", "navigation", "navigate", "pathname")):
        # best-effort: include likely router files
        for cand in ("src/router.tsx", "src/router.ts", "src/routes.tsx", "src/routes.ts", "src/navigation.tsx", "src/navigation.ts"):
            _force_include(cand, 1, 260, "Routing")

    # State manager hints
    if any(x in low for x in ("redux", "zustand", "store", "slice", "reducer", "selector", "dispatch", "context", "provider")):
        for cand in ("src/store.ts", "src/store.tsx", "src/state/store.ts", "src/app/store.ts", "src/redux/store.ts"):
            _force_include(cand, 1, 260, "State wiring")

    # Tooling hints
    if any(x in low for x in ("vite", "webpack", "build", "bundle", "dev server")):
        for cand in ("vite.config.ts", "vite.config.js", "webpack.config.js", "craco.config.js"):
            _force_include(cand, 1, 240, "Build config")

    if "typescript" in low or "typecheck" in low or "tsc" in low or "types" in low:
        _force_include("tsconfig.json", 1, 220, "TypeScript config")

    # README booster remains
    if "readme" in low:
        try:
            full = read_range(repo_root, "README.md", 1, 400)
            context_snips.append(
                {
                    "path": "README.md",
                    "range": [1, 400],
                    "source": "forced",
                    "score": 1.0,
                    "note": "Full README included for safe rewriting",
                    "text": full,
                }
            )
        except Exception:
            pass

    return {
        "file_sample": files,
        "snippets": context_snips,
        "rag_trace": hybrid_result.get("trace") or {},
    }


def run_verification(repo_root: str, profile: CommandsProfile) -> Dict[str, str]:
    logs: Dict[str, str] = {}
    repo_rootp = Path(repo_root)

    env = _build_pm_cache_env(repo_rootp)

    steps = [
        ("install", profile.install),
        ("lint", profile.lint),
        ("typecheck", profile.typecheck),
        ("test", profile.test),
        ("build", profile.build),
    ]

    for name, cmd in steps:
        if not cmd:
            continue

        if name == "install" and (repo_rootp / "node_modules").exists():
            logs[name] = "SKIPPED: node_modules/ exists"
            continue

        try:
            logs[name] = exec_cmd(cmd, cwd=repo_root, timeout_s=1800, env=env)
        except ExecError as err:
            logs[name] = str(err)
            logs["failed_step"] = name
            break

    return logs


def _build_pm_cache_env(repo_root: Path) -> Dict[str, str]:
    base_env = dict(os.environ)

    cache_root = Path(WORKSPACE_ROOT or "/workspaces") / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    repo_fingerprint = _repo_fingerprint(repo_root)
    repo_cache = cache_root / "pm" / repo_fingerprint
    repo_cache.mkdir(parents=True, exist_ok=True)

    npm_cache = repo_cache / "npm"
    npm_cache.mkdir(parents=True, exist_ok=True)
    base_env["npm_config_cache"] = str(npm_cache)

    yarn_cache = repo_cache / "yarn"
    yarn_cache.mkdir(parents=True, exist_ok=True)
    base_env["YARN_CACHE_FOLDER"] = str(yarn_cache)

    pnpm_store = repo_cache / "pnpm-store"
    pnpm_store.mkdir(parents=True, exist_ok=True)
    base_env["PNPM_STORE_DIR"] = str(pnpm_store)

    base_env.setdefault("CI", "1")
    return base_env


def _repo_fingerprint(repo_root: Path) -> str:
    candidates = [
        "package.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "package-lock.json",
        "npm-shrinkwrap.json",
        "turbo.json",
        "pnpm-workspace.yaml",
        "tsconfig.json",
        "vite.config.ts",
        "vite.config.js",
    ]

    hash_obj = hashlib.sha256()
    any_file = False
    for rel in candidates:
        candidate_path = repo_root / rel
        if candidate_path.exists() and candidate_path.is_file():
            any_file = True
            try:
                hash_obj.update(rel.encode("utf-8"))
                hash_obj.update(b"\0")
                hash_obj.update(candidate_path.read_bytes())
                hash_obj.update(b"\0")
            except Exception:
                continue

    if not any_file:
        hash_obj.update(str(repo_root).encode("utf-8"))

    return hash_obj.hexdigest()[:24]


def _pick_query(task_text: str) -> str:
    import re

    text = (task_text or "").strip()
    if not text:
        return "TODO"

    norm = re.sub(r"\s+", " ", text).strip()
    low = norm.lower()

    first = re.split(r"[.\n;]+", norm, maxsplit=1)[0].strip()
    if len(first) > 220:
        first = first[:220].rstrip()

    quoted = re.findall(r"['\"`]{1}([^'\"`]{3,80})['\"`]{1}", text)
    quoted = [quoted_text.strip() for quoted_text in quoted if quoted_text and quoted_text.strip()]

    text_slash = text.replace("\\", "/")
    paths = re.findall(
        r"(?:^|[\s:(])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.[A-Za-z0-9]+)",
        text_slash,
    )
    top_files = re.findall(r"(?:^|[\s:(])([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,6})(?:$|[\s):,])", text_slash)
    paths = [path_item.strip().strip("):,") for path_item in (paths + top_files)]
    seen_p = set()
    paths_d: list[str] = []
    for path_item in paths:
        if not path_item:
            continue
        cleaned_path = path_item.strip()
        if cleaned_path in seen_p:
            continue
        seen_p.add(cleaned_path)
        paths_d.append(cleaned_path)
    paths = paths_d

    endpoints = re.findall(r"(?:/[\w.\-]+)+", text_slash)
    endpoints = [endpoint for endpoint in endpoints if len(endpoint) >= 6 and not endpoint.startswith(("//", "/mnt", "/home"))]
    seen_e = set()
    endpoints_d: list[str] = []
    for endpoint in endpoints:
        if endpoint in seen_e:
            continue
        seen_e.add(endpoint)
        endpoints_d.append(endpoint)
        if len(endpoints_d) >= 6:
            break
    endpoints = endpoints_d

    error_phrases: list[str] = []
    for code in re.findall(r"\b([1-5][0-9]{2})\b", norm):
        if code in {"200", "201", "204"}:
            continue
        error_phrases.append(code)

    for tok in re.findall(r"\b[a-z][a-z0-9]+(?:[-_][a-z0-9]+){1,4}\b", low):
        if any(x in tok for x in ("error", "fail", "timeout", "exception", "traceback", "unknown", "invalid", "denied", "already", "taken")):
            error_phrases.append(tok)

    for error_name in re.findall(r"\b([A-Z][A-Za-z]+(?:Error|Exception))\b", norm):
        error_phrases.append(error_name)

    seen_err = set()
    err_d: list[str] = []
    for phrase in error_phrases:
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase or cleaned_phrase in seen_err:
            continue
        seen_err.add(cleaned_phrase)
        err_d.append(cleaned_phrase)
        if len(err_d) >= 8:
            break
    error_phrases = err_d

    stop = {
        "the", "and", "with", "from", "that", "this", "into", "when", "then", "than",
        "also", "just", "only", "more", "most", "some", "any", "all", "each",
        "make", "made", "does", "did", "done", "doing",
        "need", "needs", "needed", "should", "would", "could",
        "please", "plz", "help",
        "add", "added", "adding", "implement", "implemented", "implementing",
        "fix", "fixed", "fixing", "refactor", "refactored", "refactoring",
        "update", "updated", "updating",
        "create", "created", "creating",
    }
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}", norm)
    cleaned: list[str] = []
    seen = set()
    for token in toks:
        token_lower = token.lower()
        if token_lower in stop:
            continue
        if token_lower.startswith(("http://", "https://")):
            continue
        if token_lower in {"true", "false", "none", "null", "undefined"}:
            continue

        looks_useful = (
            ("_" in token) or ("." in token) or ("/" in token) or ("-" in token)
            or (token.isupper() and len(token) <= 24)
            or (token[:1].isupper() and any(char.islower() for char in token[1:]))
            or (token.startswith("use") and len(token) > 3)
        )
        if not looks_useful:
            continue

        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= 14:
            break

    intent = ""
    if any(keyword in low for keyword in ("fix", "bug", "broken", "fails", "error")):
        intent = "fix bug"
    elif any(keyword in low for keyword in ("refactor", "cleanup", "restructure")):
        intent = "refactor"
    elif any(keyword in low for keyword in ("add", "implement", "support", "feature")):
        intent = "add feature"
    elif any(keyword in low for keyword in ("test", "coverage", "jest", "vitest", "rtl", "testing-library")):
        intent = "tests"

    parts: list[str] = []
    if intent:
        parts.append(intent)
    if first:
        parts.append(first)
    parts.extend(paths[:3])
    parts.extend(endpoints[:3])
    parts.extend(quoted[:3])
    parts.extend(error_phrases[:4])
    parts.extend(cleaned)

    query_text = " ".join([part for part in parts if part]).strip()
    query_text = re.sub(r"\s+", " ", query_text)
    if len(query_text) > 420:
        query_text = query_text[:420].rstrip()
    return query_text or "TODO"


def git_changed_paths(repo_root: str) -> List[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    return [line_text.strip().replace("\\", "/") for line_text in out.splitlines() if line_text.strip()]
