from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from repoops.common.config import RAG_RERANK_MODEL, RAG_RERANK_TIMEOUT_S
from repoops.core.llm_router import LLMRouter

@dataclass(frozen=True)
class RerankResult:
    order: List[int]
    latency_ms: int
    model: str

def llm_rerank(
    llm: LLMRouter,
    *,
    query: str,
    docs: List[Dict[str, Any]],
    timeout_s: int = RAG_RERANK_TIMEOUT_S,
) -> RerankResult:
    """
    docs: [{path,start_line,end_line,text,source,base_score,lang,symbol}]
    Returns order: indices into docs sorted best->worst
    """
    start_time = time.time()

    # Keep payload small, but informative
    compact = []
    for i, doc in enumerate(docs):
        txt = (doc.get("text") or "")
        if len(txt) > 1200:
            txt = txt[:1200] + "\n...<truncated>..."
        compact.append(
            {
                "i": i,
                "path": doc.get("path"),
                "range": [doc.get("start_line"), doc.get("end_line")],
                "lang": doc.get("lang"),
                "symbol": doc.get("symbol"),
                "text": txt,
                "source": doc.get("source"),
                "base_score": float(doc.get("base_score") or 0.0),
            }
        )

    system = (
        "You are a code retrieval reranker.\n"
        "Task: rank the snippets by how useful they are for answering the query / making the required code change.\n"
        "Return ONLY JSON: {\"order\": [int, int, ...]} with all indices exactly once.\n"
        "No explanations."
    )
    user = json.dumps({"query": query, "snippets": compact}, ensure_ascii=False)

    resp = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout_s=timeout_s,
        prefer_strong=False,
        force_json=True,
    )

    raw = llm.strip_fences(resp["text"]).strip()
    obj = llm.loads_json_object(raw)
    order = obj.get("order")

    if not isinstance(order, list) or len(order) != len(docs):
        raise RuntimeError("Rerank output invalid: missing/incorrect 'order'")

    seen = set()
    out: List[int] = []
    for x in order:
        if not isinstance(x, int):
            continue
        if x < 0 or x >= len(docs):
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)

    if len(out) != len(docs):
        raise RuntimeError("Rerank output invalid: indices not a full permutation")

    return RerankResult(order=out, latency_ms=int((time.time() - start_time) * 1000), model=RAG_RERANK_MODEL)
