import subprocess
import time
from typing import Optional, Dict

from repoops.common.config import EXEC_ALLOWLIST
from repoops.common.redaction import redact


class ExecError(RuntimeError):
    pass


def exec_cmd(cmd: str, cwd: str, timeout_s: int = 900, env: Optional[Dict[str, str]] = None) -> str:
    cmd = cmd.strip()
    if not cmd:
        raise ExecError("Empty command")

    first = cmd.split()[0]
    if first not in EXEC_ALLOWLIST:
        raise ExecError(f"Command not allowlisted: {first}. Allowed: {EXEC_ALLOWLIST}")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
            env=env,
        )
    except subprocess.TimeoutExpired as err:
        raise ExecError(f"Timeout after {timeout_s}s: {cmd}") from err

    out = redact(proc.stdout or "")
    dur = int((time.time() - start) * 1000)
    if proc.returncode != 0:
        raise ExecError(f"Command failed ({proc.returncode}) in {dur}ms:\n{out[-8000:]}")
    return out[-8000:]
