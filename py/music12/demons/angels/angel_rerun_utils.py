from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_command_tokens(command: Any) -> List[str]:
    """
    Accepts:
    - list[str]
    - string command line
    Returns normalized argv list.
    """
    if isinstance(command, list):
        return [str(x) for x in command]
    if isinstance(command, str):
        return shlex.split(command, posix=False)
    return []


def extract_rerun_command(report: Dict[str, Any], project_root: Path) -> Optional[List[str]]:
    """
    Correct priority:
    1. explicit full command fields
    2. reconstruct from target_module + argv/target_args
    3. never use bare argv as executable command
    """

    for key in ("rerun_command", "command_tokens", "target_command"):
        cmd = normalize_command_tokens(report.get(key))
        if cmd:
            return cmd

    target_module = report.get("target_module")
    target_args = report.get("target_args")

    if not isinstance(target_args, list):
        target_args = report.get("argv")

    if isinstance(target_module, str) and target_module.strip():
        cmd = [sys.executable, "-m", target_module.strip()]
        if isinstance(target_args, list):
            cmd.extend(str(x) for x in target_args)
        return cmd

    return None


def run_command(
    cmd: List[str],
    project_root: Path,
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_sec: int = 180,
) -> Dict[str, Any]:
    import os
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "py")
    if env_overrides:
        env.update(env_overrides)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "cmd": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "ok": completed.returncode == 0,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "ok": False,
            "timed_out": True,
            "timeout_sec": timeout_sec,
        }


def classify_rerun_result(result: Dict[str, Any]) -> str:
    if result.get("timed_out"):
        return "timeout_after_repair"

    stderr = (result.get("stderr") or "").lower()
    stdout = (result.get("stdout") or "").lower()

    if result.get("ok"):
        return "pass"

    if "invalid int value" in stderr or "invalid int value" in stdout:
        return "same_cli_failure"

    if "--octave_max" in stderr or "--octave_min" in stderr:
        return "octave_cli_failure"

    return "new_or_other_failure"