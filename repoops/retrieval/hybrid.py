from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import math
import time
import re
import os
from pathlib import Path
import requests

from repoops.tools.repo_search import rg_search_many
from repoops.indexing.semantic_index import semantic_search
from repoops.tools.repo_fs import read_range
from repoops.common.config import (
    RAG_CANDIDATES_KEYWORD,
    RAG_CANDIDATES_SEMANTIC,
    RAG_TOPK_CONTEXT,
    RAG_RERANK,
    RAG_RERANK_MAX_DOCS,
    RAG_FILTER_LANGS,
    RAG_RERANK_ON_AMBIGUITY,
    LEXICAL_PRIMARY,
    ZOEK_URL,
    ZOEK_TIMEOUT_S,
    ZOEK_K,
    RRF_K,
    RAG_MAX_FILES,
    RAG_MAX_SNIPS_PER_FILE,
    RAG_MAX_CONTEXT_CHARS,
    WORKSPACE_ROOT,
)
from repoops.core.llm_router import LLMRouter
from repoops.retrieval.rerank import llm_rerank
from repoops.indexing.chunker import chunk_file, detect_lang

@dataclass
class Snippet:
    path: str
    start_line: int
    end_line: int
    score: float
    source: str  # keyword|semantic
    note: str = ""
    lang: str = ""
    symbol: str = ""

@dataclass
class LexicalHit:
    path: str
    line: int
    text: str
    kind: str  # zoekt|rg|file

@dataclass
class Candidate:
    path: str
    start_line: int
    end_line: int
    score: float
    source: str  # lexical|semantic|fused
    note: str = ""
    symbol: str = ""

_ZOEK_LAST_ERROR = ""

_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".cache",
    ".venv",
    "coverage",
    ".pytest_cache",
}

def _is_ignored_path(path: str) -> bool:
    normalized_path = (path or "").replace("\\", "/")
    if not normalized_path:
        return True
    path_parts = [part for part in normalized_path.split("/") if part]
    if any(ignored_dir in path_parts for ignored_dir in _IGNORE_DIRS):
        return True
    if normalized_path.endswith(".map"):
        return True
    return False

def _normalize_ident(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())

def _query_hints(query: str, kw_tokens: List[str]) -> Dict[str, Any]:
    query_lower = (query or "").lower()
    wants_tests = any(x in query_lower for x in ("test", "tests", "spec", "specs", "jest", "vitest", "unit test"))
    wants_styles = any(x in query_lower for x in ("css", "scss", "style", "styles", "stylesheet"))

    idents: set[str] = set()
    filenames: set[str] = set()
    path_tokens: set[str] = set()
    for token in kw_tokens or []:
        token_text = (token or "").strip()
        if not token_text:
            continue
        token_lower = token_text.lower()
        if "/" in token_text or "." in token_text:
            filenames.add(token_lower)
            path_tokens.add(token_lower)
            stem = Path(token_text).stem
            if stem:
                idents.add(_normalize_ident(stem))
        else:
            idents.add(_normalize_ident(token_text))

    return {
        "wants_tests": wants_tests,
        "wants_styles": wants_styles,
        "idents": idents,
        "filenames": filenames,
        "path_tokens": path_tokens,
    }

def _repo_scope_terms(repo_root: str) -> List[str]:
    """
    Build Zoekt file-scope filters that bias results to the active repo.
    We return multiple terms (OR'ed) to be robust across different path layouts.
    """
    terms: List[str] = []
    if not repo_root:
        return terms

    try:
        repo_name = Path(repo_root).name
    except Exception:
        repo_name = ""
    if repo_name:
        # Prefer repo scoping when Zoekt indexes repositories by name.
        terms.append(f"r:{repo_name}")

    if not WORKSPACE_ROOT:
        return terms

    repo_root_abs = os.path.abspath(repo_root)
    ws_root_abs = os.path.abspath(WORKSPACE_ROOT)
    if not repo_root_abs.startswith(ws_root_abs):
        return terms

    rel = os.path.relpath(repo_root_abs, ws_root_abs).replace("\\", "/").strip("./")
    if rel and rel != ".":
        terms.append(f"file:{rel}")
    return terms

def _scoped_queries(query: str, repo_root: str) -> List[str]:
    terms = _repo_scope_terms(repo_root)
    if not terms:
        return [query]
    # Avoid OR/parentheses for broad compatibility with Zoekt query parser.
    return [f"{query} {term}" for term in terms]

def _norm(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))

def _calibrate_scores(
    keyword_snips: List[Snippet],
    semantic_snips: List[Snippet],
) -> Tuple[List[Snippet], List[Snippet]]:
    if keyword_snips:
        for snip in keyword_snips:
            base = 0.54
            if snip.note and snip.note.strip():
                base += 0.06
            snip.score = base

    if semantic_snips:
        score_values = [semantic_snip.score for semantic_snip in semantic_snips]
        lower, upper = min(score_values), max(score_values)
        for snip in semantic_snips:
            snip.score = 0.60 + 0.40 * _norm(snip.score, lower, upper)

    return keyword_snips, semantic_snips

def _dedupe_key(path: str, start_line: int, end_line: int) -> tuple:
    normalized_path = (path or "").replace("\\", "/").strip()
    start = int(start_line or 1)
    end = int(end_line or start)
    if end < start:
        end = start
    bucket_start = max(1, (start // 25) * 25)
    bucket_end = max(bucket_start, (end // 25) * 25)
    return (normalized_path, bucket_start, bucket_end)

def _rrf_score(rank: int, k: int) -> float:
    return 1.0 / float(k + max(1, int(rank)))

def _maybe_int(x: Any, default: int = 1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _extract_zoekt_hits(data: Any, max_hits: int) -> List[LexicalHit]:
    hits: List[LexicalHit] = []

    def add_hit(path: str, line: int, text: str):
        if not path:
            return
        hits.append(LexicalHit(path=path.replace("\\", "/"), line=max(1, int(line or 1)), text=text or "", kind="zoekt"))

    def walk(obj: Any):
        if isinstance(obj, dict):
            # Common Zoekt JSON keys
            if ("FileName" in obj or "File" in obj or "Path" in obj) and ("LineMatches" in obj or "Matches" in obj):
                path = obj.get("FileName") or obj.get("File") or obj.get("Path") or ""
                line_matches = obj.get("LineMatches") or obj.get("Matches") or []
                if isinstance(line_matches, list) and line_matches:
                    for line_match in line_matches:
                        if not isinstance(line_match, dict):
                            continue
                        line = line_match.get("LineNumber") or line_match.get("LineNumberStart") or line_match.get("Line") or line_match.get("line")
                        text = line_match.get("Line") or line_match.get("Preview") or line_match.get("Text") or ""
                        add_hit(path, _maybe_int(line, 1), str(text))
                        if len(hits) >= max_hits:
                            return
                else:
                    add_hit(path, 1, "")
            for value in obj.values():
                if len(hits) >= max_hits:
                    return
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                if len(hits) >= max_hits:
                    return
                walk(item)

    walk(data)
    return hits[:max_hits]

def _zoekt_search(query: str, *, max_hits: int) -> List[LexicalHit]:
    global _ZOEK_LAST_ERROR
    _ZOEK_LAST_ERROR = ""
    if not ZOEK_URL:
        _ZOEK_LAST_ERROR = "ZOEK_URL not set"
        return []
    base = ZOEK_URL.rstrip("/")

    # Prefer JSON API if available (requires zoekt-webserver -rpc)
    headers = {"Accept": "application/json"}

    # Prefer /search?format=json (works even when /api/search returns HTML).
    for path, params in [
        ("/search", {"q": query, "num": str(max_hits), "limit": str(max_hits), "format": "json"}),
        ("/search", {"query": query, "num": str(max_hits), "limit": str(max_hits), "format": "json"}),
        ("/api/search", {"q": query, "num": str(max_hits), "limit": str(max_hits)}),
        ("/api/search", {"query": query, "num": str(max_hits), "limit": str(max_hits)}),
    ]:
        url = base + path
        try:
            response = requests.get(url, params=params, timeout=ZOEK_TIMEOUT_S, headers=headers, allow_redirects=False)
        except Exception as err:
            _ZOEK_LAST_ERROR = f"GET {url} failed: {type(err).__name__}: {err}"
            continue
        if not response.ok:
            _ZOEK_LAST_ERROR = f"GET {url} status {response.status_code}"
            continue
        ctype = (response.headers.get("Content-Type") or "").lower()
        if "html" in ctype:
            body = (response.text or "")[:200].replace("\n", " ").strip()
            _ZOEK_LAST_ERROR = f"GET {url} returned HTML; body={body}"
            continue
        try:
            data = response.json()
        except Exception as err:
            body = (response.text or "")[:200].replace("\n", " ").strip()
            _ZOEK_LAST_ERROR = f"GET {url} json error: {type(err).__name__}: {err}; body={body}"
            continue
        # Detect empty index (helpful for debug)
        try:
            stats = (data.get("result") or {}).get("Stats") or {}
            if stats.get("FileCount") == 0:
                _ZOEK_LAST_ERROR = f"GET {url} ok but index empty (FileCount=0)"
        except Exception:
            pass

        hits = _extract_zoekt_hits(data, max_hits=max_hits)
        if hits:
            return hits
        _ZOEK_LAST_ERROR = f"GET {url} ok but 0 hits"

    # POST fallback for API variants
    for path, payload in [
        ("/api/search", {"query": query, "num": max_hits}),
        ("/api/search", {"q": query, "num": max_hits}),
    ]:
        url = base + path
        try:
            response = requests.post(url, json=payload, timeout=ZOEK_TIMEOUT_S, headers=headers, allow_redirects=False)
        except Exception as err:
            _ZOEK_LAST_ERROR = f"POST {url} failed: {type(err).__name__}: {err}"
            continue
        if not response.ok:
            _ZOEK_LAST_ERROR = f"POST {url} status {response.status_code}"
            continue
        ctype = (response.headers.get("Content-Type") or "").lower()
        if "html" in ctype:
            body = (response.text or "")[:200].replace("\n", " ").strip()
            _ZOEK_LAST_ERROR = f"POST {url} returned HTML; body={body}"
            continue
        try:
            data = response.json()
        except Exception as err:
            body = (response.text or "")[:200].replace("\n", " ").strip()
            _ZOEK_LAST_ERROR = f"POST {url} json error: {type(err).__name__}: {err}; body={body}"
            continue
        hits = _extract_zoekt_hits(data, max_hits=max_hits)
        if hits:
            return hits
        _ZOEK_LAST_ERROR = f"POST {url} ok but 0 hits"
    return []

def _rg_search(repo_root: str, kw_tokens: List[str], *, max_hits: int) -> List[LexicalHit]:
    hits = rg_search_many(repo_root, kw_tokens, max_hits_total=max_hits, max_hits_per_query=50)
    out: List[LexicalHit] = []
    for hit in hits:
        if _is_ignored_path(hit.path):
            continue
        out.append(LexicalHit(path=hit.path, line=hit.line, text=hit.text, kind="rg"))
    return out

def _file_name_search(repo_root: str, kw_tokens: List[str], *, max_hits: int) -> List[LexicalHit]:
    if not repo_root or not kw_tokens:
        return []

    patterns: list[str] = []
    for token in kw_tokens:
        token_text = (token or "").strip()
        if not token_text:
            continue
        if "/" in token_text or "." in token_text:
            patterns.append(token_text)
        elif token_text[:1].isupper() and token_text.isalpha():
            patterns.extend([token_text + ext for ext in (".tsx", ".ts", ".jsx", ".js")])

    if not patterns:
        return []

    patterns_lower = [pattern.lower() for pattern in patterns]
    hits: List[LexicalHit] = []
    skip_dirs = {
        ".git", "node_modules", "dist", "build", ".next", ".turbo", ".cache",
        ".venv", "coverage", ".pytest_cache",
    }

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [dir_name for dir_name in dirs if dir_name not in skip_dirs and not dir_name.startswith(".")]
        for filename in files:
            filename_lower = filename.lower()
            if not any(pattern in filename_lower for pattern in patterns_lower):
                continue
            rel = os.path.relpath(os.path.join(root, filename), repo_root).replace("\\", "/")
            if _is_ignored_path(rel):
                continue
            hits.append(LexicalHit(path=rel, line=1, text=f"filename:{filename}", kind="file"))
            if len(hits) >= max_hits:
                return hits

    return hits

def _normalize_zoekt_hits(hits: List[LexicalHit], repo_root: str) -> List[LexicalHit]:
    if not hits or not repo_root:
        return hits

    repo_root_abs = os.path.abspath(repo_root)
    ws_root_abs = os.path.abspath(WORKSPACE_ROOT) if WORKSPACE_ROOT else ""
    prefix = ""
    if ws_root_abs and repo_root_abs.startswith(ws_root_abs):
        prefix = os.path.relpath(repo_root_abs, ws_root_abs).replace("\\", "/")

    in_repo: List[LexicalHit] = []
    for hit in hits:
        normalized_path = (hit.path or "").replace("\\", "/").lstrip("./")
        if not normalized_path or _is_ignored_path(normalized_path):
            continue

        # Case 1: zoekt returns workspace-prefixed paths
        if prefix and (normalized_path == prefix or normalized_path.startswith(prefix + "/")):
            rel_path = normalized_path[len(prefix) + 1 :] if normalized_path.startswith(prefix + "/") else ""
            if rel_path and not _is_ignored_path(rel_path):
                in_repo.append(LexicalHit(path=rel_path, line=hit.line, text=hit.text, kind=hit.kind))
            continue

        # Case 2: zoekt returns repo-relative paths
        if Path(repo_root_abs, normalized_path).exists():
            in_repo.append(LexicalHit(path=normalized_path, line=hit.line, text=hit.text, kind=hit.kind))

    return in_repo

def _lexical_search(repo_root: str, query: str, kw_tokens: List[str], *, max_hits: int) -> Tuple[List[LexicalHit], str, bool]:
    """
    Returns: (hits, engine_used, used_fallback)
    """
    engine = "rg"
    used_fallback = True

    if LEXICAL_PRIMARY == "zoekt" and ZOEK_URL:
        try:
            hits: List[LexicalHit] = []
            for scoped_query in _scoped_queries(query, repo_root):
                hits = _zoekt_search(scoped_query, max_hits=max_hits)
                if hits:
                    break
            if not hits and kw_tokens:
                # Zoekt treats whitespace as AND; long NL queries often yield 0 hits.
                # Fall back to high-signal tokens one-by-one.
                for token in kw_tokens[:8]:
                    for scoped_query in _scoped_queries(token, repo_root):
                        hits = _zoekt_search(scoped_query, max_hits=max_hits)
                        if hits:
                            break
                    if hits:
                        break
            if hits:
                hits = _normalize_zoekt_hits(hits, repo_root)
                if hits:
                    return hits, "zoekt", False
        except Exception:
            pass

    hits = _rg_search(repo_root, kw_tokens, max_hits=max_hits)
    if hits:
        return hits, engine, used_fallback

    file_hits = _file_name_search(repo_root, kw_tokens, max_hits=max_hits)
    if file_hits:
        return file_hits, "file", True

    return hits, engine, used_fallback

def _rrf_fuse(lex: List[Candidate], sem: List[Candidate], *, rrf_k: int) -> List[Candidate]:
    fused: Dict[tuple, Candidate] = {}
    source_sets: Dict[tuple, set[str]] = {}

    def apply_list(items: List[Candidate], source: str, key_ranked: List[Candidate]):
        seen: set = set()
        for rank, candidate in enumerate(key_ranked, start=1):
            key = _dedupe_key(candidate.path, candidate.start_line, candidate.end_line)
            if key in seen:
                continue
            seen.add(key)
            score = _rrf_score(rank, rrf_k)
            if key not in fused:
                fused[key] = Candidate(
                    path=candidate.path,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    score=score,
                    source="fused",
                    note=candidate.note,
                    symbol=candidate.symbol,
                )
                source_sets[key] = {source}
            else:
                fused[key].score += score
                source_sets[key].add(source)

    lex_ranked = list(lex)
    sem_ranked = sorted(sem, key=lambda candidate: float(candidate.score), reverse=True)

    apply_list(lex_ranked, "lexical", lex_ranked)
    apply_list(sem_ranked, "semantic", sem_ranked)

    out = list(fused.values())
    for candidate in out:
        key = _dedupe_key(candidate.path, candidate.start_line, candidate.end_line)
        sources = sorted(source_sets.get(key, []))
        if sources:
            candidate.note = (candidate.note + " | " if candidate.note else "") + f"sources={','.join(sources)}"

    out.sort(key=lambda candidate_item: float(candidate_item.score), reverse=True)
    return out

def _select_top_files(
    candidates: List[Candidate],
    *,
    max_files: int,
    query_hints: Optional[Dict[str, Any]] = None,
) -> List[str]:
    scores: Dict[str, Dict[str, float]] = {}
    symbols_by_path: Dict[str, set[str]] = {}
    for candidate in candidates:
        path = (candidate.path or "").replace("\\", "/")
        if not path:
            continue
        score_entry = scores.setdefault(path, {"score": 0.0, "lex": 0.0, "sem": 0.0})
        score_entry["score"] += float(candidate.score)
        if "lexical" in (candidate.note or "") or "sources=lexical" in (candidate.note or ""):
            score_entry["lex"] += 1.0
        if "semantic" in (candidate.note or "") or "sources=semantic" in (candidate.note or ""):
            score_entry["sem"] += 1.0
        if candidate.symbol:
            symbol_norm = _normalize_ident(candidate.symbol)
            if symbol_norm:
                symbols_by_path.setdefault(path, set()).add(symbol_norm)

    ranked = []
    hints = query_hints or {}
    wants_tests = bool(hints.get("wants_tests"))
    wants_styles = bool(hints.get("wants_styles"))
    idents = hints.get("idents") or set()
    filenames = hints.get("filenames") or set()
    path_tokens = hints.get("path_tokens") or set()
    for path, score_entry in scores.items():
        pri = _file_priority(path)
        bonus = 0.12 * score_entry["lex"] + 0.08 * score_entry["sem"] + max(0.0, 0.06 * (1.0 - min(1.0, pri / 50.0)))

        # Boost exact filename or identifier matches.
        name = Path(path).name.lower()
        stem_norm = _normalize_ident(Path(path).stem)
        if name in filenames:
            bonus += 0.45
        if stem_norm and stem_norm in idents:
            bonus += 0.35
        if any(path_token in path.lower() for path_token in path_tokens if path_token):
            bonus += 0.25

        # Boost symbol matches (from semantic index)
        symset = symbols_by_path.get(path) or set()
        if symset and idents and symset.intersection(idents):
            bonus += 0.35

        # Penalize test/style files unless explicitly requested.
        path_lower = path.lower()
        is_test = (".test." in path_lower) or (".spec." in path_lower) or ("/__tests__/" in path_lower)
        is_style = path_lower.endswith((".css", ".scss", ".sass", ".less"))
        if is_test and not wants_tests:
            bonus -= 0.35
        if is_style and not wants_styles:
            bonus -= 0.40
        ranked.append((path, score_entry["score"] + bonus, pri))

    ranked.sort(key=lambda x: (-x[1], x[2]))
    return [path for (path, score_value, priority) in ranked[: max(1, int(max_files))]]

def _expand_to_symbol_range(repo_root: str, path: str, start: int, end: int, cache: Dict[str, list]) -> Tuple[int, int, str]:
    rel_path = (path or "").replace("\\", "/")
    if rel_path not in cache:
        try:
            cache[rel_path] = chunk_file(repo_root, rel_path)
        except Exception:
            cache[rel_path] = []

    for chunk in cache.get(rel_path, []):
        if chunk.start_line <= start <= chunk.end_line:
            return chunk.start_line, chunk.end_line, chunk.symbol
    return start, end, ""

def _should_rerank(lex_files: List[str], sem_files: List[str]) -> Tuple[bool, float]:
    if not RAG_RERANK_ON_AMBIGUITY:
        return False, 0.0
    if not lex_files or not sem_files:
        return True, 1.0
    lex_set, sem_set = set(lex_files), set(sem_files)
    union = lex_set.union(sem_set)
    inter = lex_set.intersection(sem_set)
    jaccard = (len(inter) / len(union)) if union else 1.0
    ambiguity = 1.0 - jaccard
    return ambiguity >= 0.55, round(ambiguity, 3)

def _extract_keyword_tokens(query: str) -> List[str]:
    """
    React SPA oriented token extraction:
    - file paths, quoted strings
    - identifiers (PascalCase/camelCase/snake_case)
    - state manager signals (slice/store/useStore)
    - error tokens
    """
    query_text = (query or "").strip()
    if not query_text:
        return []

    text_slash = query_text.replace("\\", "/")

    paths = re.findall(
        r"(?:^|[\s:(])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.[A-Za-z0-9]+)",
        text_slash,
    )
    top_files = re.findall(r"(?:^|[\s:(])([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,6})(?:$|[\s):,])", text_slash)
    paths = [path_token.strip().strip("):,") for path_token in (paths + top_files)]

    quoted = re.findall(r"['\"`]{1}([^'\"`]{3,80})['\"`]{1}", query_text)
    quoted = [quoted_text.strip() for quoted_text in quoted if quoted_text.strip()]

    # Build composite tokens from TitleCase sequences (e.g., Movie Detail Page -> MovieDetailPage)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", query_text)
    composites: list[str] = []

    def is_title(word: str) -> bool:
        if not word:
            return False
        if word[:1].isupper() and any(char.islower() for char in word[1:]):
            return True
        if word.isupper() and len(word) <= 6:
            return True
        return False

    i = 0
    while i < len(words):
        word = words[i]
        word_lower = word.lower()
        if word_lower == "use" and i + 1 < len(words) and is_title(words[i + 1]):
            seq: list[str] = []
            j = i + 1
            while j < len(words) and is_title(words[j]):
                seq.append(words[j])
                j += 1
            if seq:
                composites.append("use" + "".join(seq))
                i = j
                continue
        if is_title(word):
            seq = [word]
            j = i + 1
            while j < len(words) and is_title(words[j]):
                seq.append(words[j])
                j += 1
            if len(seq) >= 2:
                composites.append("".join(seq))
            i = j
            continue
        i += 1

    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}", query_text)
    stop = {
        "the", "and", "with", "from", "that", "this", "into", "when", "then", "than",
        "also", "just", "only", "more", "most", "some", "any", "all", "each",
        "please", "plz", "help",
        "add", "fix", "update", "implement", "create", "refactor",
    }

    keep: list[str] = []
    seen = set()

    for token in identifiers:
        token_lower = token.lower()
        if token_lower in stop:
            continue
        if token_lower.startswith(("http://", "https://")):
            continue
        if token_lower in {"true", "false", "none", "null", "undefined"}:
            continue

        useful = (
            ("_" in token)
            or ("-" in token)
            or ("." in token)
            or ("/" in token)
            or (token[:1].isupper() and any(char.islower() for char in token[1:]))  # Class/Component-ish
            or (token.startswith("use") and len(token) > 3)
            or any(x in token_lower for x in ("redux", "zustand", "store", "slice", "reducer", "selector", "dispatch", "context", "provider"))
            or any(x in token_lower for x in ("error", "failed", "timeout", "exception", "traceback"))
        )
        if not useful:
            continue
        if token_lower in seen:
            continue
        seen.add(token_lower)
        keep.append(token)
        if len(keep) >= 12:
            break

    out: list[str] = []
    out.extend(paths[:5])
    out.extend(quoted[:4])
    out.extend(composites[:4])
    out.extend(keep)

    # Add a few common SPA anchors
    has_specific = bool(paths or quoted or composites or any(
        (token[:1].isupper() and any(char.islower() for char in token[1:]))
        or token.startswith("use")
        or ("/" in token or "." in token)
        for token in keep
    ))
    if not has_specific:
        anchors = ["App.tsx", "main.tsx", "index.tsx", "routes", "router", "store", "slice", "useStore", "Provider", "createContext"]
        for anchor in anchors:
            if len(out) >= 14:
                break
            if anchor.lower() not in {x.lower() for x in out}:
                out.append(anchor)

    # dedupe preserve order
    final: list[str] = []
    seen_lower = set()
    for item in out:
        item_lower = item.lower()
        if item_lower in seen_lower:
            continue
        seen_lower.add(item_lower)
        final.append(item)
    return final[:14]

def _file_priority(path: str) -> int:
    """
    Prefer React SPA files:
      - src/** higher
      - tsx over ts over jsx/js
      - tests get moderate priority
    Lower number = better.
    """
    normalized_path = (path or "").replace("\\", "/").lower()
    ext = Path(normalized_path).suffix.lower()
    is_src = normalized_path.startswith("src/")
    is_test = (".test." in normalized_path) or (".spec." in normalized_path) or ("/__tests__/" in normalized_path)
    if ext == ".tsx":
        ext_rank = 0
    elif ext == ".ts":
        ext_rank = 1
    elif ext == ".jsx":
        ext_rank = 2
    elif ext == ".js":
        ext_rank = 3
    elif ext in (".json", ".css", ".scss", ".md"):
        ext_rank = 4
    else:
        ext_rank = 5
    pri = 0
    if not is_src:
        pri += 2
    if is_test:
        pri += 1
    return pri * 10 + ext_rank

def hybrid_search(
    *,
    repo_id: str,
    repo_root: str,
    query: str,
    k: int = RAG_TOPK_CONTEXT,
    filters: Optional[Dict[str, Any]] = None,
    llm: Optional[LLMRouter] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    filters = filters or {}

    allowed_langs = set()
    if RAG_FILTER_LANGS.strip():
        allowed_langs = {x.strip() for x in RAG_FILTER_LANGS.split(",") if x.strip()}

    primary_q = (query or "").strip() or "TODO"
    kw_tokens = _extract_keyword_tokens(primary_q)
    q_hints = _query_hints(primary_q, kw_tokens)

    query_start_time = time.time()
    lex_hits, lex_engine, used_fallback = _lexical_search(
        repo_root, primary_q, kw_tokens, max_hits=ZOEK_K or RAG_CANDIDATES_KEYWORD
    )

    # Build lexical candidates
    lex_cands: List[Candidate] = []
    for hit in lex_hits:
        start = max(1, int(hit.line) - 10)
        end = int(hit.line) + 18
        lex_cands.append(
            Candidate(
                path=hit.path,
                start_line=start,
                end_line=end,
                score=0.0,
                source="lexical",
                note=(hit.text or "")[:200],
            )
        )

    # Semantic candidates
    sem_raw = []
    try:
        sem_raw = semantic_search(repo_id, primary_q, k=RAG_CANDIDATES_SEMANTIC)
    except Exception:
        sem_raw = []

    sem_cands: List[Candidate] = [
        Candidate(
            path=sem_item["path"],
            start_line=int(sem_item["start_line"]),
            end_line=int(sem_item["end_line"]),
            score=float(sem_item["score"]),
            source="semantic",
            note=str(sem_item.get("symbol") or ""),
            symbol=str(sem_item.get("symbol") or ""),
        )
        for sem_item in sem_raw
    ]

    if allowed_langs:
        sem_cands = [candidate for candidate in sem_cands if not detect_lang(candidate.path) or detect_lang(candidate.path) in allowed_langs]

    fused = _rrf_fuse(lex_cands, sem_cands, rrf_k=RRF_K)

    # Select work files
    top_files = _select_top_files(fused, max_files=RAG_MAX_FILES, query_hints=q_hints)
    top_file_set = set(top_files)

    # Gather candidates by file
    by_file: Dict[str, List[Candidate]] = {}
    for candidate in fused:
        path = (candidate.path or "").replace("\\", "/")
        if path not in top_file_set:
            continue
        by_file.setdefault(path, []).append(candidate)

    # Build docs with symbol expansion and context budget
    docs: List[Dict[str, Any]] = []
    read_fail = 0
    total_chars = 0
    chunk_cache: Dict[str, list] = {}

    for path in top_files:
        snippets = sorted(by_file.get(path, []), key=lambda snippet: float(snippet.score), reverse=True)[: RAG_MAX_SNIPS_PER_FILE]
        if not snippets:
            snippets = [Candidate(path=path, start_line=1, end_line=160, score=0.1, source="lexical", note="file header")]

        for snippet in snippets:
            start, end, symbol = _expand_to_symbol_range(repo_root, snippet.path, snippet.start_line, snippet.end_line, chunk_cache)
            try:
                txt = read_range(repo_root, snippet.path, start, end)
            except Exception:
                read_fail += 1
                continue
            if not txt:
                continue
            if total_chars + len(txt) > RAG_MAX_CONTEXT_CHARS:
                break
            docs.append(
                {
                    "path": snippet.path,
                    "start_line": start,
                    "end_line": end,
                    "text": txt,
                    "source": snippet.source,
                    "base_score": float(snippet.score),
                    "lang": detect_lang(snippet.path),
                    "symbol": symbol or snippet.symbol,
                    "reason": snippet.note or "",
                }
            )
            total_chars += len(txt)
        if total_chars >= RAG_MAX_CONTEXT_CHARS:
            break

    rerank_used = False
    rerank_latency_ms = 0

    # Ambiguity check for rerank
    lex_top_files = [hit.path for hit in lex_hits[:6]]
    sem_top_files = [candidate.path for candidate in sem_cands[:6]]
    should_rerank, ambiguity = _should_rerank(lex_top_files, sem_top_files)

    if RAG_RERANK and llm is not None and should_rerank and len(docs) >= 8:
        try:
            rerank_result = llm_rerank(llm, query=primary_q, docs=docs)
            rerank_used = True
            rerank_latency_ms = rerank_result.latency_ms
            docs = [docs[i] for i in rerank_result.order]
        except Exception:
            docs = sorted(docs, key=lambda doc: float(doc.get("base_score") or 0.0), reverse=True)

    final_docs = docs[:k]

    out_snips: List[Snippet] = []
    for doc in final_docs:
        out_snips.append(
            Snippet(
                path=doc["path"],
                start_line=int(doc["start_line"]),
                end_line=int(doc["end_line"]),
                score=float(doc.get("base_score") or 0.0),
                source=doc.get("source") or "semantic",
                note=str(doc.get("reason") or ""),
                lang=doc.get("lang") or "",
                symbol=doc.get("symbol") or "",
            )
        )

    trace = {
        "primary_query": primary_q,
        "kw_tokens": kw_tokens,
        "lex_engine": lex_engine,
        "lex_fallback": bool(used_fallback),
        "lex_hits_total": int(len(lex_hits)),
        "sem_snips_total": len(sem_cands),
        "fused_total": len(fused),
        "top_files": top_files,
        "docs_for_rerank": len(docs),
        "read_fail": int(read_fail),
        "rerank_used": rerank_used,
        "rerank_latency_ms": int(rerank_latency_ms),
        "ambiguity": float(ambiguity),
        "latency_ms": int((time.time() - query_start_time) * 1000),
        "total_latency_ms": int((time.time() - start_time) * 1000),
    }
    if LEXICAL_PRIMARY == "zoekt" and lex_engine != "zoekt" and _ZOEK_LAST_ERROR:
        trace["zoekt_error"] = _ZOEK_LAST_ERROR
    return {"snippets": out_snips, "docs": final_docs, "trace": trace}
