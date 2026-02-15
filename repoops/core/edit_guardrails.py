from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

FORBIDDEN_PREFIXES = (
    ".git/",
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".turbo/",
    ".cache/",
    ".venv/",
)

DEFAULT_ALLOWED_PREFIXES = (
    "src/",
    "public/",
)

CONFIG_ALLOWLIST = (
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
    "jest.config.js",
    "jest.config.ts",
    "vitest.config.ts",
    "vitest.config.js",
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.json",
    ".prettierrc",
    ".prettierrc.js",
    ".prettierrc.cjs",
    ".prettierrc.json",
)

LOCKFILES = (
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
)

@dataclass(frozen=True)
class GuardrailResult:
    is_ok: bool
    reasons: List[str]

def _norm_path(path: str) -> str:
    return (path or "").strip().lstrip("/").replace("\\", "/")

def _is_forbidden(path: str) -> bool:
    normalized = _norm_path(path).lower()
    return any(normalized.startswith(x) for x in FORBIDDEN_PREFIXES)

def _is_lockfile(path: str) -> bool:
    normalized = _norm_path(path)
    return normalized in LOCKFILES

def _looks_allowed(path: str) -> bool:
    normalized = _norm_path(path)
    if normalized in CONFIG_ALLOWLIST:
        return True
    if any(normalized.startswith(x) for x in DEFAULT_ALLOWED_PREFIXES):
        return True
    # allow tests anywhere under src or common test dirs
    if "/__tests__/" in normalized or normalized.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx", ".test.js", ".test.jsx", ".spec.js", ".spec.jsx")):
        return True
    return False

def validate_edits(
    edits: Dict[str, str],
    *,
    task_text: str,
    retrieved_paths: Optional[List[str]] = None,
    max_files: int = 6,
    max_total_chars: int = 180_000,
) -> GuardrailResult:
    reasons: List[str] = []
    if not edits:
        return GuardrailResult(is_ok=True, reasons=[])

    # Basic caps
    if len(edits) > max_files:
        reasons.append(f"Too many files changed ({len(edits)} > {max_files}). Keep changes minimal.")

    total_chars = sum(len(content_text or "") for content_text in edits.values())
    if total_chars > max_total_chars:
        reasons.append(f"Edits too large ({total_chars} chars). Avoid broad rewrites.")

    # Path checks
    for path, content in edits.items():
        normalized_path = _norm_path(path)
        if not normalized_path:
            reasons.append("Empty path in edits.")
            continue
        if _is_forbidden(normalized_path):
            reasons.append(f"Forbidden path edited: {normalized_path}")
        if not _looks_allowed(normalized_path):
            # Soft block: allow but warn. For 9/10 we block by default unless task hints config.
            low = (task_text or "").lower()
            if "config" in low or "build" in low or "lint" in low or "typescript" in low or "jest" in low or "vitest" in low:
                # allow configs in these cases
                pass
            else:
                reasons.append(f"Edited file outside expected SPA scope: {normalized_path} (prefer src/ or known configs).")

        # JSON must parse
        if normalized_path.lower().endswith(".json"):
            try:
                json.loads(content or "")
            except Exception:
                reasons.append(f"Invalid JSON produced for {normalized_path}.")

        if (content or "").strip() == "":
            reasons.append(f"Empty content for {normalized_path}.")

        # Lockfiles only if explicitly requested
        if _is_lockfile(normalized_path):
            low = (task_text or "").lower()
            if not any(x in low for x in ("lockfile", "pnpm-lock", "yarn.lock", "package-lock", "dependency", "deps", "upgrade", "update package")):
                reasons.append(f"Lockfile modified without request: {normalized_path}")

    # Relevance: intersect with retrieved paths (soft but helpful)
    if retrieved_paths:
        retrieved_set = {(_norm_path(x)) for x in retrieved_paths if x}
        edited_set = {_norm_path(x) for x in edits.keys()}
        # require some overlap unless task explicitly names a different file
        overlap = edited_set.intersection(retrieved_set)
        if not overlap:
            reasons.append("Edits do not overlap retrieved context files; likely off-target.")

    return GuardrailResult(is_ok=(len(reasons) == 0), reasons=reasons)
