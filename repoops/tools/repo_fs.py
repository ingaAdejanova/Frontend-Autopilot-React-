import os
from pathlib import Path

def tree(root: str, max_entries: int = 300) -> list[str]:
    root_path = Path(root)
    out: list[str] = []
    for path_obj in root_path.rglob("*"):
        if path_obj.is_dir():
            continue
        rel = str(path_obj.relative_to(root_path))
        if rel.startswith(".git/") or rel.startswith("node_modules/") or rel.startswith("dist/") or rel.startswith("build/"):
            continue
        out.append(rel)
        if len(out) >= max_entries:
            break
    return out

def read_range(root: str, path: str, start_line: int, end_line: int) -> str:
    file_path = Path(root) / path
    if not file_path.exists():
        raise FileNotFoundError(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)
    snippet = lines[start-1:end]
    return "\n".join(snippet)

def write_text(root: str, path: str, content: str) -> None:
    file_path = Path(root) / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
