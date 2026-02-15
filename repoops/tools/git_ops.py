import subprocess
from repoops.common.redaction import redact

def run_git(args: list[str], cwd: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return redact(proc.stdout or "")

def diff(cwd: str) -> str:
    return run_git(["diff"], cwd)

def status(cwd: str) -> str:
    return run_git(["status", "--porcelain=v1"], cwd)

def commit(cwd: str, message: str) -> str:
    run_git(["add", "-A"], cwd)
    return run_git(["commit", "-m", message], cwd)
