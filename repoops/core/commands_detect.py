import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()

def _detect_js_package_manager(root: Path, pkg_json: Dict[str, Any]) -> Tuple[str, str]:
    has_pnpm_lock = _exists(root, "pnpm-lock.yaml")
    has_yarn_lock = _exists(root, "yarn.lock")
    has_npm_lock = _exists(root, "package-lock.json") or _exists(root, "npm-shrinkwrap.json")

    pm_field = (pkg_json.get("packageManager") or "").lower()
    if pm_field.startswith("pnpm"):
        return "pnpm", "pnpm install --frozen-lockfile"
    if pm_field.startswith("yarn"):
        if _exists(root, ".yarnrc.yml"):
            return "yarn", "yarn install --immutable"
        return "yarn", "yarn install"
    if pm_field.startswith("npm"):
        return "npm", "npm ci" if has_npm_lock else "npm install"

    if has_pnpm_lock or _exists(root, "pnpm-workspace.yaml"):
        return "pnpm", "pnpm install --frozen-lockfile"
    if has_yarn_lock:
        if _exists(root, ".yarnrc.yml"):
            return "yarn", "yarn install --immutable"
        return "yarn", "yarn install"

    return "npm", "npm ci" if has_npm_lock else "npm install"

def _script_cmd(package_manager: str, script: str) -> str:
    if package_manager == "npm":
        return f"npm run {script}"
    return f"{package_manager} {script}"

def _pick_script(scripts: Dict[str, Any], preferred: list[str]) -> Optional[str]:
    if not isinstance(scripts, dict):
        return None
    for name in preferred:
        if name in scripts and isinstance(scripts[name], str) and scripts[name].strip():
            return name
    return None

def detect_commands_profile(repo_root: str) -> Dict[str, str]:
    root = Path(repo_root)

    pkg = root / "package.json"
    if pkg.exists():
        data = _read_json(pkg) or {}
        scripts = data.get("scripts") or {}
        package_manager, install_cmd = _detect_js_package_manager(root, data)

        out: Dict[str, str] = {"install": install_cmd}

        lint_script = _pick_script(
            scripts,
            preferred=["lint", "lint:check", "lint:ci", "eslint"],
        )
        if lint_script:
            out["lint"] = _script_cmd(package_manager, lint_script)

        # Add typecheck if present
        typecheck_script = _pick_script(
            scripts,
            preferred=["typecheck", "check", "tsc", "types", "type-check", "type:check"],
        )
        if typecheck_script:
            out["typecheck"] = _script_cmd(package_manager, typecheck_script)

        test_script = _pick_script(
            scripts,
            preferred=["test", "test:unit", "unit", "test:ci", "jest", "vitest"],
        )
        if test_script:
            out["test"] = _script_cmd(package_manager, test_script)

        build_script = _pick_script(
            scripts,
            preferred=["build", "build:prod", "build:production", "bundle"],
        )
        if build_script:
            out["build"] = _script_cmd(package_manager, build_script)

        return out

    # Python fallbacks unchanged
    pyproject = root / "pyproject.toml"
    requirements = root / "requirements.txt"

    if pyproject.exists() and _exists(root, "poetry.lock"):
        return {"install": "poetry install --no-interaction --no-ansi"}

    if pyproject.exists() and (_exists(root, "uv.lock") or _exists(root, ".python-version")):
        return {"install": "python -m pip install -e ."}

    if requirements.exists():
        return {"install": "python -m pip install -r requirements.txt"}

    return {}
