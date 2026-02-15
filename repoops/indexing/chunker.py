from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int
    end_line: int
    kind: str           # file|symbol|block
    symbol: str
    lang: str           # ts|tsx|js|jsx|py|md|json|css|scss|unknown
    text: str
    content_hash: str

_CODE_EXT_TO_LANG = {
    ".ts": "ts",
    ".tsx": "tsx",
    ".js": "js",
    ".jsx": "jsx",
    ".py": "py",
    ".md": "md",
    ".json": "json",
    ".css": "css",
    ".scss": "scss",
}

def detect_lang(rel_path: str) -> str:
    ext = Path(rel_path).suffix.lower()
    return _CODE_EXT_TO_LANG.get(ext, "unknown")

def iter_code_files(repo_root: str) -> Iterable[str]:
    root = Path(repo_root)
    for path_obj in root.rglob("*"):
        if path_obj.is_dir():
            continue
        rel = str(path_obj.relative_to(root)).replace("\\", "/")
        if rel.startswith(".git/"):
            continue
        if any(seg in rel for seg in ("node_modules/", "dist/", "build/", ".next/", ".turbo/", ".cache/", ".venv/")):
            continue

        # React SPA focus: include code + configs + styles.
        if rel.endswith((".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".json", ".css", ".scss")):
            yield rel

def chunk_file(repo_root: str, rel_path: str, max_lines: int = 220) -> list[Chunk]:
    file_path = Path(repo_root) / rel_path
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lang = detect_lang(rel_path)

    if len(lines) <= max_lines:
        return [Chunk(rel_path, 1, max(1, len(lines)), "file", rel_path, lang, text, _sha1(text))]

    ast_chunks = _try_ast_chunk(rel_path, text, lang)
    if ast_chunks:
        return ast_chunks

    chunks = _heuristic_symbol_chunks(rel_path, lines, lang, max_chunks=240)
    if not chunks:
        return [Chunk(rel_path, 1, len(lines), "file", rel_path, lang, text, _sha1(text))]
    return chunks

def _try_ast_chunk(rel_path: str, text: str, lang: str) -> Optional[list[Chunk]]:
    # Only for code languages
    if lang not in {"ts", "tsx", "js", "jsx", "py"}:
        return None

    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except Exception:
        return None

    try:
        parser = get_parser("python" if lang == "py" else "typescript" if lang in {"ts", "tsx"} else "javascript")
        tree = parser.parse(bytes(text, "utf-8"))
        root = tree.root_node
    except Exception:
        return None

    lines = text.splitlines()
    out: list[Chunk] = []

    def node_to_range(node):
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return start_line, max(start_line, end_line)

    top = root.children
    for node in top:
        node_type = node.type
        if lang == "py" and node_type in {"function_definition", "class_definition"}:
            start_line, end_line = node_to_range(node)
            part = "\n".join(lines[start_line - 1 : end_line])
            sym = _guess_symbol_from_text(part) or node_type
            if end_line - start_line + 1 >= 5:
                out.append(Chunk(rel_path, start_line, end_line, "symbol", sym, lang, part, _sha1(part)))
        elif lang in {"ts", "tsx", "js", "jsx"} and node_type in {
            "function_declaration",
            "class_declaration",
            "lexical_declaration",
            "export_statement",
        }:
            start_line, end_line = node_to_range(node)
            part = "\n".join(lines[start_line - 1 : end_line])
            sym = _guess_symbol_from_text(part) or node_type
            if end_line - start_line + 1 >= 5:
                out.append(Chunk(rel_path, start_line, end_line, "symbol", sym, lang, part, _sha1(part)))

        if len(out) >= 240:
            break

    if len(out) < 2:
        return None

    overview = "\n".join(lines[: min(120, len(lines))])
    out.insert(0, Chunk(rel_path, 1, min(120, len(lines)), "block", "__file_overview__", lang, overview, _sha1(overview)))
    return out

def _heuristic_symbol_chunks(rel_path: str, lines: list[str], lang: str, *, max_chunks: int) -> list[Chunk]:
    boundaries = [1]

    if lang in {"ts", "tsx", "js", "jsx"}:
        # General JS/TS boundaries + React SPA extras
        pat = re.compile(
            r"^\s*(export\s+)?(default\s+)?(async\s+)?(function|class)\s+"
            r"|^\s*(export\s+)?(const|let|var)\s+[A-Za-z0-9_]+\s*="
            r"|^\s*(export\s+)?(type|interface)\s+[A-Za-z0-9_]+"
            r"|^\s*import\s+.+\s+from\s+['\"][^'\"]+['\"]\s*;?\s*$"
        )

        # React-ish anchors (components/hooks/context)
        react_pat = re.compile(
            r"\bcreateContext\s*\(|\buse[A-Z][A-Za-z0-9_]*\b|"
            r"^\s*(export\s+)?(default\s+)?function\s+[A-Z][A-Za-z0-9_]*\s*\(|"
            r"^\s*(export\s+)?(const|let|var)\s+[A-Z][A-Za-z0-9_]*\s*=\s*\("
        )
    elif lang == "py":
        pat = re.compile(r"^\s*(def|class)\s+[A-Za-z0-9_]+\s*[\(:]")
        react_pat = None
    else:
        # markdown/json/css/scss: chunk by headings/comments or big blocks
        pat = re.compile(r"^\s*#\s+|^\s*##\s+|^\s*/\*|^\s*//|^\s*\.")
        react_pat = None

    for line_number, line_text in enumerate(lines, start=1):
        if pat.search(line_text):
            boundaries.append(line_number)
        elif react_pat is not None and react_pat.search(line_text):
            boundaries.append(line_number)

    boundaries = sorted(set(boundaries))
    out: list[Chunk] = []

    for boundary_index, start in enumerate(boundaries):
        end = (boundaries[boundary_index + 1] - 1) if (boundary_index + 1) < len(boundaries) else len(lines)
        if end - start + 1 < 5:
            continue
        part = "\n".join(lines[start - 1 : end])
        sym = _guess_symbol_from_text(lines[start - 1]) or "block"
        out.append(Chunk(rel_path, start, end, "block", sym, lang, part, _sha1(part)))
        if len(out) >= max_chunks:
            break

    return out

def _guess_symbol_from_text(text: str) -> Optional[str]:
    match = re.search(r"\b(class|function|def)\s+([A-Za-z0-9_]+)", text)
    if match:
        return match.group(2)
    match_decl = re.search(r"\b(const|let|var)\s+([A-Za-z0-9_]+)", text)
    if match_decl:
        return match_decl.group(2)
    match_type = re.search(r"\b(type|interface)\s+([A-Za-z0-9_]+)", text)
    if match_type:
        return match_type.group(2)
    return None

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
