import json
import time
import subprocess
from dataclasses import dataclass
from langsmith import traceable
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict, List

from langgraph.graph import StateGraph, END

from repoops.core.llm_router import LLMRouter
from repoops.core.commands_detect import detect_commands_profile
from repoops.core import agents_multi
from repoops.tools.repo_fs import write_text
from repoops.core.runner import CommandsProfile, run_verification
from repoops.tools.git_ops import diff as git_diff
from repoops.common.redaction import redact

from repoops.core.edit_guardrails import validate_edits as validate_edits_guardrails
from repoops.core.task_check import task_satisfaction_check

class GraphState(TypedDict, total=False):
    run_id: str
    repo_id: str
    repo_root: str
    task_text: str
    ctx: Dict[str, Any]
    commands_profile: Dict[str, Optional[str]]

    prefer_strong: bool
    fix_attempt: int

    plan_text: str
    edits: Dict[str, str]
    verification_logs: Dict[str, str]
    pr_text: str

    last_error: str
    fix_instructions: str


@dataclass
class Deps:
    llm: LLMRouter
    artifact_cb: Callable[[str, str], None]


def traced(name: str, node_func, deps: Deps):
    def _run(state: GraphState) -> GraphState:
        start_time = time.time()

        msg = f"[graph] ▶ {name}"
        print(msg, flush=True)
        _artifact(deps, "notes", msg)

        try:
            out = node_func(state, deps)
        except Exception as err:
            err_msg = f"[graph] ❌ {name}: {redact(str(err))[:1200]}"
            print(err_msg, flush=True)
            _artifact(deps, "notes", err_msg)
            raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        done = f"[graph] ✅ {name} ({elapsed_ms}ms)"
        print(done, flush=True)
        _artifact(deps, "notes", done)

        if out.get("last_error"):
            warn = f"[graph] ⚠ last_error: {redact(out['last_error'])[:1200]}"
            print(warn, flush=True)
            _artifact(deps, "notes", warn)

        return out
    return _run


def _artifact(deps: Deps, kind: str, content: str) -> None:
    try:
        deps.artifact_cb(kind, content)
    except Exception:
        pass


def _safe_apply_edits(repo_root: str, edits: Dict[str, str]) -> None:
    repo_rootp = Path(repo_root).resolve()

    for rel_path, content in edits.items():
        rel_norm = rel_path.lstrip("/").replace("\\", "/")
        abs_path = (repo_rootp / rel_norm).resolve()

        if repo_rootp not in abs_path.parents and abs_path != repo_rootp:
            raise RuntimeError(f"Refusing to write outside repo: {rel_path}")

        write_text(str(repo_rootp), rel_norm, content)


def _git_diff_stat(repo_root: str) -> str:
    proc = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return redact(proc.stdout or "")


@traceable(run_type="tool", name="detect_profile")
def node_detect_profile(state: GraphState, deps: Deps) -> GraphState:
    prof = state.get("commands_profile") or {}
    if any(prof.get(k) for k in ["install", "lint", "typecheck", "test", "build"]):
        return state

    detected = detect_commands_profile(state["repo_root"])
    if detected:
        _artifact(deps, "notes", f"Detected commands_profile: {json.dumps(detected)}")
        state["commands_profile"] = detected
    return state


@traceable(run_type="tool", name="plan")
def node_plan(state: GraphState, deps: Deps) -> GraphState:
    msgs = agents_multi.planner_prompt(state["task_text"], state.get("ctx") or {})
    resp = deps.llm.chat(msgs, prefer_strong=bool(state.get("prefer_strong")))
    plan_text = resp["text"].strip()
    state["plan_text"] = plan_text
    _artifact(deps, "notes", "PLAN\n" + plan_text)
    return state


@traceable(run_type="tool", name="propose")
def node_engineer_propose(state: GraphState, deps: Deps) -> GraphState:
    extra = state.get("fix_instructions", "") or ""

    role = "FE"
    msgs = agents_multi.fe_engineer_edits_prompt(
        state["task_text"],
        state.get("ctx") or {},
        plan_text=state.get("plan_text", ""),
        extra_instructions=extra,
    )

    resp = deps.llm.chat(
        msgs,
        prefer_strong=bool(state.get("prefer_strong")),
        force_json=True,
    )

    raw = deps.llm.strip_fences(resp["text"])

    try:
        edits = agents_multi.parse_edits_json(raw)
    except Exception as err:
        state["last_error"] = f"Engineer returned invalid JSON: {err}"
        _artifact(deps, "notes", f"[engineer] invalid JSON, escalating attempt={state.get('fix_attempt',0)}")
        state["prefer_strong"] = True
        state["edits"] = {}
        return state

    state["edits"] = edits
    _artifact(deps, "notes", f"[engineer] role={role} proposed edits: {list(edits.keys())}")
    return state


@traceable(run_type="tool", name="validate_edits")
def node_validate_edits(state: GraphState, deps: Deps) -> GraphState:
    edits = state.get("edits") or {}
    if not edits:
        return state

    # retrieved paths for relevance check
    retrieved = []
    for snippet in (state.get("ctx") or {}).get("snippets") or []:
        if isinstance(snippet, dict) and snippet.get("path"):
            retrieved.append(str(snippet.get("path")))

    res = validate_edits_guardrails(
        edits,
        task_text=state.get("task_text") or "",
        retrieved_paths=retrieved[:25],
        max_files=6,
        max_total_chars=180_000,
    )

    if res.is_ok:
        _artifact(deps, "notes", "[guardrails] edits accepted.")
        return state

    # Reject edits and provide strict instructions for next propose
    msg = "[guardrails] edits rejected:\n- " + "\n- ".join(res.reasons[:12])
    _artifact(deps, "notes", msg)

    state["edits"] = {}
    state["fix_instructions"] = (
        (state.get("fix_instructions") or "")
        + "\n\nGUARDRAILS REJECTION:\n"
        + "\n".join(res.reasons[:12])
        + "\n\nRewrite the solution with minimal, targeted edits within src/ or known configs. "
          "Do not touch lockfiles unless explicitly requested. Ensure changes overlap retrieved files."
    )
    # force escalation quickly after a rejection
    state["prefer_strong"] = True
    return state


@traceable(run_type="tool", name="apply")
def node_apply(state: GraphState, deps: Deps) -> GraphState:
    edits = state.get("edits") or {}
    if not edits:
        return state

    _safe_apply_edits(state["repo_root"], edits)
    _artifact(deps, "diff", git_diff(state["repo_root"])[:20000])
    return state


@traceable(run_type="tool", name="verify")
def node_verify(state: GraphState, deps: Deps) -> GraphState:
    # Docs-only change? Skip verification to avoid toolchain noise.
    try:
        diff_text = git_diff(state["repo_root"])
        changed = []
        for line_text in diff_text.splitlines():
            if line_text.startswith("+++ b/"):
                changed.append(line_text[len("+++ b/"):].strip())

        if changed and all(
            changed_path.lower().endswith((".md", ".txt")) or changed_path.lower().startswith(("docs/",))
            for changed_path in changed
        ):
            logs = {"skipped": "Docs-only change detected; verification skipped."}
            state["verification_logs"] = logs
            _artifact(deps, "test_log", json.dumps(logs, ensure_ascii=False)[:20000])
            return state
    except Exception:
        pass

    prof = state.get("commands_profile") or {}
    commands_profile = CommandsProfile(
        install=prof.get("install"),
        lint=prof.get("lint"),
        typecheck=prof.get("typecheck"),
        test=prof.get("test"),
        build=prof.get("build"),
    )
    logs = run_verification(state["repo_root"], commands_profile)
    state["verification_logs"] = logs
    _artifact(deps, "test_log", json.dumps(logs, ensure_ascii=False)[:20000])
    return state


@traceable(run_type="tool", name="task_check")
def node_task_check(state: GraphState, deps: Deps) -> GraphState:
    logs = state.get("verification_logs") or {}
    diff_text = git_diff(state["repo_root"])[:20000]

    # If verification failed already, don't run satisfaction check.
    if "failed_step" in logs:
        return state

    # If no edits, likely no-op; mark as not satisfied unless task clearly says no change.
    if not (state.get("edits") or {}):
        state["verification_logs"] = dict(logs)
        state["verification_logs"]["failed_step"] = "task_check"
        state["verification_logs"]["task_check"] = "No edits produced; task likely not satisfied."
        return state

    res = task_satisfaction_check(
        deps.llm,
        task_text=state.get("task_text") or "",
        diff_text=diff_text,
        verification_logs=logs,
    )

    _artifact(deps, "notes", f"[task_check] satisfied={res.satisfied} reasons={res.reasons}")

    if not res.satisfied:
        state["verification_logs"] = dict(logs)
        state["verification_logs"]["failed_step"] = "task_check"
        state["verification_logs"]["task_check"] = "; ".join(res.reasons[:4])
        state["fix_instructions"] = (
            (state.get("fix_instructions") or "")
            + "\n\nTASK SATISFACTION CHECK FAILED:\n"
            + "\n".join(f"- {reason}" for reason in res.reasons[:6])
            + "\n\nRevise edits to directly satisfy the task with minimal changes."
        )
        state["prefer_strong"] = True

    return state


@traceable(run_type="tool", name="qa_analyze")
def node_qa_analyze(state: GraphState, deps: Deps) -> GraphState:
    logs = state.get("verification_logs") or {}
    diff_text = git_diff(state["repo_root"])
    msgs = agents_multi.qa_failure_analysis_prompt(
        state["task_text"],
        state.get("plan_text", ""),
        logs,
        diff_text,
    )
    resp = deps.llm.chat(msgs, prefer_strong=bool(state.get("prefer_strong")))
    instr = resp["text"].strip()
    state["fix_instructions"] = instr
    _artifact(deps, "notes", "[qa] fix instructions:\n" + instr[:18000])
    return state


def node_bump_attempt(state: GraphState, deps: Deps) -> GraphState:
    state["fix_attempt"] = int(state.get("fix_attempt") or 0) + 1
    if state["fix_attempt"] >= 1:
        state["prefer_strong"] = True
    return state


@traceable(run_type="tool", name="final_pr")
def node_pr_text(state: GraphState, deps: Deps) -> GraphState:
    logs = state.get("verification_logs") or {}
    stat = _git_diff_stat(state["repo_root"])
    msgs = agents_multi.pr_summary_prompt(
        state["task_text"],
        state.get("plan_text", ""),
        logs,
        stat,
    )
    resp = deps.llm.chat(msgs, prefer_strong=bool(state.get("prefer_strong")))
    pr_text = resp["text"].strip()
    state["pr_text"] = pr_text
    _artifact(deps, "pr_text", pr_text[:20000])
    return state


def should_repair(state: GraphState) -> str:
    logs = state.get("verification_logs") or {}
    failed = "failed_step" in logs
    attempts = int(state.get("fix_attempt") or 0)

    if not logs:
        return "finalize"

    if failed and attempts < 2:
        return "repair"

    return "finalize"


def build_graph(deps: Deps):
    graph = StateGraph(GraphState)

    graph.add_node("detect_profile", traced("detect_profile", node_detect_profile, deps))

    graph.add_node("plan", traced("plan", node_plan, deps))
    graph.add_node("propose", traced("propose", node_engineer_propose, deps))
    graph.add_node("validate_edits", traced("validate_edits", node_validate_edits, deps))
    graph.add_node("apply", traced("apply", node_apply, deps))
    graph.add_node("verify", traced("verify", node_verify, deps))
    graph.add_node("task_check", traced("task_check", node_task_check, deps))

    graph.add_node("bump_attempt", traced("bump_attempt", node_bump_attempt, deps))
    graph.add_node("qa_analyze", traced("qa_analyze", node_qa_analyze, deps))

    graph.add_node("final_pr", traced("final_pr", node_pr_text, deps))

    graph.set_entry_point("detect_profile")
    graph.add_edge("detect_profile", "plan")
    graph.add_edge("plan", "propose")
    graph.add_edge("propose", "validate_edits")
    graph.add_edge("validate_edits", "apply")
    graph.add_edge("apply", "verify")
    graph.add_edge("verify", "task_check")

    graph.add_conditional_edges(
        "task_check",
        should_repair,
        {"repair": "bump_attempt", "finalize": "final_pr"},
    )

    graph.add_edge("bump_attempt", "qa_analyze")
    graph.add_edge("qa_analyze", "propose")
    graph.add_edge("final_pr", END)

    return graph.compile()
