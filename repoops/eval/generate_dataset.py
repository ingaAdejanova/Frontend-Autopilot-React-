from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

IGNORE_DIRS = {
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


def _iter_source_files(repo_root: str) -> List[str]:
    out: List[str] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [dir_name for dir_name in dirs if dir_name not in IGNORE_DIRS and not dir_name.startswith(".")]
        for filename in files:
            if filename.startswith("."):
                continue
            ext = Path(filename).suffix.lower()
            if ext not in {".ts", ".tsx", ".js", ".jsx"}:
                continue
            if filename.endswith(".map"):
                continue
            rel = os.path.relpath(os.path.join(root, filename), repo_root).replace("\\", "/")
            out.append(rel)
    return out


def _split_camel(name: str) -> str:
    if not name:
        return name
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    parts = parts.replace("_", " ").replace("-", " ")
    return " ".join(parts.split())


def _strip_test_suffix(stem: str) -> str:
    if stem.endswith(".test"):
        return stem[:-5]
    if stem.endswith(".spec"):
        return stem[:-5]
    return stem

def _identifier_for_path(path: str) -> str:
    stem = Path(path).stem
    if stem in {"index", "main"}:
        parent = Path(path).parent.name
        if parent:
            return parent
    return stem

def _label_for_path(path: str) -> str:
    ident = _identifier_for_path(path)
    return _split_camel(ident)


def _categorize(files: List[str]) -> Dict[str, List[str]]:
    cats: Dict[str, List[str]] = {
        "components": [],
        "pages": [],
        "hooks": [],
        "routes": [],
        "store": [],
        "api": [],
        "tests": [],
        "other": [],
    }

    for rel_path in files:
        path_lower = rel_path.lower()
        stem = Path(rel_path).stem
        if "/__tests__/" in path_lower or ".test." in path_lower or ".spec." in path_lower:
            cats["tests"].append(rel_path)
        elif "/components/" in path_lower or (stem[:1].isupper() and rel_path.endswith((".tsx", ".jsx"))):
            cats["components"].append(rel_path)
        elif "/pages/" in path_lower:
            cats["pages"].append(rel_path)
        elif "/hooks/" in path_lower or stem.startswith("use"):
            cats["hooks"].append(rel_path)
        elif "/routes" in path_lower or "/router" in path_lower or "route" in stem.lower():
            cats["routes"].append(rel_path)
        elif "/redux/" in path_lower or "/store/" in path_lower or "slice" in stem.lower():
            cats["store"].append(rel_path)
        elif "/api/" in path_lower or "/services/" in path_lower or stem.lower().endswith("api"):
            cats["api"].append(rel_path)
        else:
            cats["other"].append(rel_path)

    return cats


def _add_task(tasks: List[Dict], task: str, paths: List[str]) -> None:
    if not task or not paths:
        return
    item = {"id": f"t{len(tasks) + 1}", "task": task, "expected_paths": paths}
    tasks.append(item)


def generate(repo_root: str, max_tasks: int, seed: int) -> List[Dict]:
    files = _iter_source_files(repo_root)
    if not files:
        return []

    cats = _categorize(files)
    rnd = random.Random(seed)

    # deterministically shuffle each category
    for category_name in cats:
        rnd.shuffle(cats[category_name])

    tasks: List[Dict] = []

    # App.tsx baseline if present
    if "src/App.tsx" in files:
        _add_task(
            tasks,
            "Update the main App component to show a loading state while data is being fetched.",
            ["src/App.tsx"],
        )

    # Pair tests with components when possible
    comp_by_stem = {Path(comp_path).stem.lower(): comp_path for comp_path in cats["components"]}
    for test_path in cats["tests"]:
        stem = _strip_test_suffix(Path(test_path).stem).lower()
        comp = comp_by_stem.get(stem)
        if comp:
            ident = _identifier_for_path(comp)
            _add_task(
                tasks,
                f"Add a basic unit test for the `{ident}` component render.",
                [comp, test_path],
            )
        if len(tasks) >= max_tasks:
            return tasks

    # Hooks
    for hook_path in cats["hooks"][:5]:
        ident = _identifier_for_path(hook_path)
        _add_task(
            tasks,
            f"Refactor the `{ident}` hook to support an optional leading flag.",
            [hook_path],
        )
        if len(tasks) >= max_tasks:
            return tasks

    # Pages
    for page_path in cats["pages"][:5]:
        ident = _identifier_for_path(page_path)
        _add_task(
            tasks,
            f"Add a new UI section to the `{ident}` page.",
            [page_path],
        )
        if len(tasks) >= max_tasks:
            return tasks

    # Components
    for component_path in cats["components"][:6]:
        ident = _identifier_for_path(component_path)
        _add_task(
            tasks,
            f"Update the `{ident}` component to accept a new optional prop.",
            [component_path],
        )
        if len(tasks) >= max_tasks:
            return tasks

    # Store / Redux
    for store_path in cats["store"][:5]:
        ident = _identifier_for_path(store_path)
        _add_task(
            tasks,
            f"Improve the `{ident}` selector to handle missing state safely.",
            [store_path],
        )
        if len(tasks) >= max_tasks:
            return tasks

    # Routes
    for routes_path in cats["routes"][:4]:
        ident = _identifier_for_path(routes_path)
        _add_task(
            tasks,
            f"Wire an existing page into the `{ident}` router config.",
            [routes_path],
        )
        if len(tasks) >= max_tasks:
            return tasks

    # API
    for api_path in cats["api"][:4]:
        ident = _identifier_for_path(api_path)
        _add_task(
            tasks,
            f"Refactor the `{ident}` API module to include a shared base URL.",
            [api_path],
        )
        if len(tasks) >= max_tasks:
            return tasks

    # Fill remaining from other files
    for other_path in cats["other"]:
        if len(tasks) >= max_tasks:
            break
        ident = _identifier_for_path(other_path)
        _add_task(
            tasks,
            f"Clean up or refactor `{ident}` to improve readability.",
            [other_path],
        )

    return tasks[:max_tasks]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a repo-specific retrieval eval dataset.")
    parser.add_argument("--repo-root", default=".", help="Path to the repo to scan.")
    parser.add_argument("--out", default="repoops/eval/data/generated_tasks.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    tasks = generate(repo_root, max_tasks=args.max_tasks, seed=args.seed)
    if not tasks:
        print("No tasks generated (no source files found).")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        for task_item in tasks:
            out_file.write(json.dumps(task_item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(tasks)} tasks to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
