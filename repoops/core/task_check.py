from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from repoops.core.llm_router import LLMRouter

@dataclass(frozen=True)
class TaskCheckResult:
    satisfied: bool
    reasons: List[str]

def _extract_required_literals(task_text: str) -> List[str]:
    """
    If user provides quoted strings, treat them as must-have literals.
    """
    import re
    task_raw = task_text or ""
    quoted = re.findall(r"['\"`]{1}([^'\"`]{2,120})['\"`]{1}", task_raw)
    out: List[str] = []
    for quoted_text in quoted:
        cleaned_quote = (quoted_text or "").strip()
        if cleaned_quote and cleaned_quote not in out:
            out.append(cleaned_quote)
        if len(out) >= 6:
            break
    return out

def task_satisfaction_check(
    llm: LLMRouter,
    *,
    task_text: str,
    diff_text: str,
    verification_logs: Dict[str, str],
) -> TaskCheckResult:
    """
    Lightweight correctness gate:
      - deterministic literal check for quoted requirements
      - LLM judgement for overall satisfaction (fast, strict)
    """
    required = _extract_required_literals(task_text)
    missing: List[str] = []
    for lit in required:
        if lit not in (diff_text or "") and lit not in json.dumps(verification_logs or {}, ensure_ascii=False):
            missing.append(lit)

    if missing:
        return TaskCheckResult(satisfied=False, reasons=[f"Missing required literal(s) from task: {missing}"])

    system = (
        "You are a strict PR reviewer for a React single-page app.\n"
        "Decide if the diff satisfies the task.\n"
        "Return ONLY JSON: {\"satisfied\": true|false, \"reasons\": [\"...\"]}\n"
        "Rules:\n"
        "- If the diff seems unrelated or too broad, satisfied=false.\n"
        "- If task requires a specific behavior, it must be implemented in code/tests.\n"
        "- Be conservative: if unsure, satisfied=false.\n"
    )

    payload = {
        "task": task_text,
        "diff": (diff_text or "")[:12000],
        "verification_logs": (verification_logs or {}),
    }

    resp = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        force_json=True,
        timeout_s=60,
    )

    raw = llm.strip_fences(resp["text"]).strip()
    obj = llm.loads_json_object(raw)

    is_satisfied = bool(obj.get("satisfied"))
    reasons = obj.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    reasons_out = [str(reason) for reason in reasons if str(reason).strip()][:6]

    if is_satisfied and not reasons_out:
        reasons_out = ["Task appears satisfied by the diff."]
    if not is_satisfied and not reasons_out:
        reasons_out = ["Diff does not clearly satisfy the task."]

    return TaskCheckResult(satisfied=is_satisfied, reasons=reasons_out)
