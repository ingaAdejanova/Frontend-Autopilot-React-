from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int
    text: str


def rg_search(
    repo_root: str,
    query: str,
    globs: Optional[List[str]] = None,
    max_hits: int = 40,
    *,
    follow_symlinks: bool = False,
) -> list[SearchHit]:
    query_text = (query or "").strip()
    if not query_text:
        return []

    if len(query_text) > 600:
        query_text = query_text[:600]

    max_hits = max(1, int(max_hits))
    max_per_file = max(1, min(8, max_hits))

    cmd: list[str] = [
        "rg",
        "--line-number",
        "--no-heading",
        "--smart-case",
        "--fixed-strings",
        "--max-count-per-file",
        str(max_per_file),
        "--max-columns",
        "400",
        "--trim",
        "--no-mmap",
        "--hidden",
        "--glob", "!.git/**",
        "--glob", "!**/node_modules/**",
        "--glob", "!**/dist/**",
        "--glob", "!**/build/**",
        "--glob", "!**/.next/**",
        "--glob", "!**/.turbo/**",
        "--glob", "!**/.cache/**",
        "--glob", "!**/.venv/**",
        "--glob", "!**/coverage/**",
        "--glob", "!**/.pytest_cache/**",
    ]

    if follow_symlinks:
        cmd.append("--follow")

    if globs:
        for glob_pattern in globs:
            glob_value = (glob_pattern or "").strip()
            if glob_value:
                cmd.extend(["--glob", glob_value])

    cmd.extend([query_text, repo_root])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    out = (proc.stdout or "").strip()
    if not out:
        return []

    hits: list[SearchHit] = []
    for line in out.splitlines():
        try:
            path_str, line_str, line_text = line.split(":", 2)
            hits.append(SearchHit(path=path_str.replace("\\", "/"), line=int(line_str), text=(line_text or "").strip()))
            if len(hits) >= max_hits:
                break
        except Exception:
            continue

    return hits


def rg_search_many(
    repo_root: str,
    queries: List[str],
    *,
    globs: Optional[List[str]] = None,
    max_hits_total: int = 220,
    max_hits_per_query: int = 50,
) -> list[SearchHit]:
    """
    Run ripgrep multiple times for high-signal tokens and merge results.
    Dedup by (path,line) keeping earliest hit.
    """
    query_list = []
    seen_queries = set()
    for raw_query in (queries or []):
        cleaned_query = (raw_query or "").strip()
        if not cleaned_query:
            continue
        if len(cleaned_query) > 120:
            cleaned_query = cleaned_query[:120]
        query_key = cleaned_query.lower()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        query_list.append(cleaned_query)
        if len(query_list) >= 14:
            break

    merged: list[SearchHit] = []
    seen: set[Tuple[str, int]] = set()

    for query_text in query_list:
        hits = rg_search(repo_root, query_text, globs=globs, max_hits=max_hits_per_query)
        for hit in hits:
            key = (hit.path, int(hit.line))
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if len(merged) >= max_hits_total:
                return merged

    return merged
