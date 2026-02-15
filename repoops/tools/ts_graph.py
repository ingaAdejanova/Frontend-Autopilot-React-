from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from repoops.common.config import (
    TS_GRAPH_ENABLED,
    TS_GRAPH_MAX_FILES,
    TS_GRAPH_MAX_REFS,
    TS_GRAPH_MAX_DEFS,
    TS_GRAPH_MAX_IMPORTS,
)


@dataclass(frozen=True)
class TSGraphSpan:
    path: str
    start_line: int
    end_line: int
    reason: str


def collect_ts_graph(
    repo_root: str,
    focus_files: List[str],
    *,
    max_files: int = TS_GRAPH_MAX_FILES,
    max_refs: int = TS_GRAPH_MAX_REFS,
    max_defs: int = TS_GRAPH_MAX_DEFS,
    max_imports: int = TS_GRAPH_MAX_IMPORTS,
    timeout_s: int = 30,
) -> List[TSGraphSpan]:
    if not TS_GRAPH_ENABLED:
        return []

    script = Path(__file__).with_name("ts_graph.js")
    if not script.exists():
        return []

    payload = {
        "repoRoot": repo_root,
        "files": focus_files or [],
        "maxFiles": int(max_files),
        "maxRefs": int(max_refs),
        "maxDefs": int(max_defs),
        "maxImports": int(max_imports),
    }

    try:
        proc = subprocess.run(
            ["node", str(script)],
            input=json.dumps(payload),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    try:
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return []

    spans = data.get("spans") if isinstance(data, dict) else None
    if not isinstance(spans, list):
        return []

    out: List[TSGraphSpan] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        path = span.get("path")
        start = span.get("start_line")
        end = span.get("end_line")
        reason = span.get("reason") or ""
        if not isinstance(path, str):
            continue
        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:
            continue
        if start_i <= 0 or end_i <= 0:
            continue
        if end_i < start_i:
            end_i = start_i
        out.append(TSGraphSpan(path=path.replace("\\", "/"), start_line=start_i, end_line=end_i, reason=str(reason)))

    return out
