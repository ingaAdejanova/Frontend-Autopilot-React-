from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import math

@dataclass(frozen=True)
class RAGMetrics:
    k: int
    recall_files_at_k: float
    precision_files_at_k: float
    mrr_files: float
    ndcg_files: float
    retrieved_files: int
    edited_files: int
    timestamp: str

def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").strip()

def _unique_preserve(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in paths or []:
        normalized = _norm(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out

def compute_rag_metrics(
    *,
    retrieved_paths: List[str],
    edited_paths: List[str],
    k: int = 20,
) -> RAGMetrics:
    # De-dupe to avoid double-counting repeated snippets from the same file.
    topk = _unique_preserve(retrieved_paths)[:k]
    edited = [_norm(path) for path in (edited_paths or []) if _norm(path)]

    retrieved_set: Set[str] = set(topk)
    edited_set: Set[str] = set(edited)

    # Recall/Precision
    hits = [path for path in edited_set if path in retrieved_set]
    recall = (len(hits) / len(edited_set)) if edited_set else 1.0
    precision = (len(hits) / len(retrieved_set)) if retrieved_set else 1.0

    # MRR: first relevant rank
    mrr = 0.0
    for i, path in enumerate(topk, start=1):
        if path in edited_set:
            mrr = 1.0 / i
            break

    # nDCG (binary relevance)
    def dcg(lst: List[str]) -> float:
        score = 0.0
        for i, path in enumerate(lst, start=1):
            rel = 1.0 if path in edited_set else 0.0
            score += rel / math.log2(i + 1)
        return score

    ideal = list(edited_set)[:k]
    ndcg = (dcg(topk) / dcg(ideal)) if edited_set else 1.0

    return RAGMetrics(
        k=k,
        recall_files_at_k=round(recall, 4),
        precision_files_at_k=round(precision, 4),
        mrr_files=round(mrr, 4),
        ndcg_files=round(ndcg, 4),
        retrieved_files=len(retrieved_set),
        edited_files=len(edited_set),
        timestamp=datetime.utcnow().isoformat(),
    )

def metrics_to_dict(metrics: RAGMetrics) -> Dict[str, Any]:
    return {
        "k": metrics.k,
        "recall_files_at_k": metrics.recall_files_at_k,
        "precision_files_at_k": metrics.precision_files_at_k,
        "mrr_files": metrics.mrr_files,
        "ndcg_files": metrics.ndcg_files,
        "retrieved_files": metrics.retrieved_files,
        "edited_files": metrics.edited_files,
        "timestamp": metrics.timestamp,
        "ts": metrics.timestamp,
    }
