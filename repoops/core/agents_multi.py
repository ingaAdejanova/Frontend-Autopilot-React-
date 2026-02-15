import json
from typing import Any, Dict, List


def _compact_context(ctx: Dict[str, Any], max_chars: int = 22000) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "file_sample": (ctx.get("file_sample") or [])[:250],
        "snippets": [],
    }

    for snippet in (ctx.get("snippets") or [])[:20]:
        snippet_copy = dict(snippet)
        txt = snippet_copy.get("text") or ""
        if len(txt) > 2500:
            snippet_copy["text"] = txt[:2500] + "\n...<truncated>..."
        out["snippets"].append(snippet_copy)

    raw = json.dumps(out, ensure_ascii=False)
    if len(raw) <= max_chars:
        return out

    out2: Dict[str, Any] = {"file_sample": out["file_sample"], "snippets": []}
    for snippet in out["snippets"][:12]:
        snippet_copy = dict(snippet)
        if "text" in snippet_copy:
            snippet_copy["text"] = (snippet_copy.get("text") or "")[:800]
        out2["snippets"].append(snippet_copy)
    return out2


def planner_prompt(task_text: str, ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    compact_context = _compact_context(ctx, max_chars=14000)
    system = (
        "You are an Engineering Manager / Planner for an autonomous code-change agent.\n"
        "Goal: produce a concise plan that maximizes correctness and minimizes changes.\n"
        "Return plain text only."
    )
    user = (
        "Task:\n"
        f"{task_text}\n\n"
        "Repo glimpses (files + snippets):\n"
        f"{json.dumps(compact_context, ensure_ascii=False)}\n\n"
        "Return a plan with max 8 bullets. No extra commentary."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def engineer_edits_prompt(
    task_text: str,
    ctx: Dict[str, Any],
    *,
    plan_text: str = "",
    extra_instructions: str = "",
) -> List[Dict[str, str]]:
    compact_context = _compact_context(ctx, max_chars=22000)

    system = (
        "You are a Senior Software Engineer agent.\n"
        "Return ONLY valid JSON. No markdown. No explanations.\n\n"
        "Schema:\n"
        "{\n"
        '  "edits": [\n'
        '    {"path": "relative/path", "content": "FULL updated file content"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Only include files that must change.\n"
        "- 'content' must be the COMPLETE final file, not a diff.\n"
        "- Keep changes minimal and localized.\n"
        "- Preserve formatting; do not reflow unrelated text.\n"
        "- If no changes needed, return exactly: {\"edits\": []}\n\n"
        "CRITICAL ACCEPTANCE CRITERIA:\n"
        "- The edits MUST directly and explicitly satisfy the task text.\n"
        "- Do NOT add generic or unrelated content.\n"
        "- If the task mentions specific names or facts, they MUST appear verbatim in the output.\n"
        "- Do NOT invent new facts.\n"
        "- If you cannot satisfy the task exactly, return {\"edits\": []}.\n"
    )

    user_payload = {
        "task": task_text,
        "plan": plan_text,
        "repo_glimpses": compact_context,
        "extra_instructions": extra_instructions,
        "requirements": [
            "Keep changes minimal.",
            "Do not modify unrelated files.",
            "Prefer adding/adjusting tests when requested.",
        ],
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def fe_engineer_edits_prompt(
    task_text: str,
    ctx: Dict[str, Any],
    *,
    plan_text: str = "",
    extra_instructions: str = "",
) -> List[Dict[str, str]]:
    msgs = engineer_edits_prompt(task_text, ctx, plan_text=plan_text, extra_instructions=extra_instructions)
    msgs[0]["content"] = (
        "You are a Frontend Engineer for a React single-page application (SPA).\n"
        "Scope: src/, components, hooks, state managers (Redux/Zustand/Context), routing (react-router), Vite/Webpack.\n"
        "Rules:\n"
        "- Keep diffs small and targeted.\n"
        "- If tests exist, update/add unit tests using the repo's framework (Jest or Vitest) and Testing Library when present.\n"
        "- Do not change build tooling unless explicitly required by the task.\n"
        "Return ONLY valid JSON edits (Pattern A). No markdown.\n\n"
        + msgs[0]["content"]
    )
    return msgs



def qa_failure_analysis_prompt(
    task_text: str,
    plan_text: str,
    verification_logs: Dict[str, str],
    diff_text: str,
) -> List[Dict[str, str]]:
    system = (
        "You are a QA / Release Engineer for a React SPA.\n"
        "Your job: diagnose verification failures and propose concrete fix instructions.\n"
        "Return plain text only.\n"
        "Be specific: name files to change, what to change, and why.\n"
        "If logs suggest missing deps, incorrect scripts, or config mismatch, call it out.\n"
        "IMPORTANT:\n"
        "- Do NOT propose broad refactors.\n"
        "- If install/lint/typecheck/test/build fails, propose the smallest fix.\n"
    )

    user = (
        "Task:\n"
        f"{task_text}\n\n"
        "Plan:\n"
        f"{plan_text}\n\n"
        "Verification logs:\n"
        f"{json.dumps(verification_logs, ensure_ascii=False)[:18000]}\n\n"
        "Current diff:\n"
        f"{diff_text[:18000]}\n\n"
        "Return: a short diagnosis + bullet list of fix instructions."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def pr_summary_prompt(
    task_text: str,
    plan_text: str,
    verification_logs: Dict[str, str],
    diff_stat: str,
) -> List[Dict[str, str]]:
    system = (
        "You are a Staff Engineer writing a PR description for a React SPA.\n"
        "Return a short PR-ready summary:\n"
        "- What changed\n"
        "- Why\n"
        "- How verified\n"
        "- Notes/risks\n"
        "Return markdown text."
    )
    user = (
        f"Task:\n{task_text}\n\n"
        f"Plan:\n{plan_text}\n\n"
        f"Verification logs:\n{json.dumps(verification_logs, ensure_ascii=False)[:14000]}\n\n"
        f"Diff stat:\n{diff_stat}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_edits_json(raw: str) -> Dict[str, str]:
    raw = raw.strip()
    obj = json.loads(raw)

    edits = obj.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("Invalid JSON: 'edits' must be a list")

    out: Dict[str, str] = {}
    for item in edits:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if isinstance(path, str) and isinstance(content, str) and path.strip():
            rel = path.strip().lstrip("/").replace("\\", "/")
            out[rel] = content

    return out
