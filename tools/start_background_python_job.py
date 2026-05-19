from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Start a long Python job in the background on Windows and save "
            "a small state file with pid, command, cwd and start time."
        )
    )
    ap.add_argument("--python-exe", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--state-json", required=True)
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cwd = Path(args.cwd).resolve()
    state_json = Path(args.state_json).resolve()
    state_json.parent.mkdir(parents=True, exist_ok=True)

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    cmd = [args.python_exe, args.script, *script_args]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
    )

    state = {
        "pid": int(proc.pid),
        "cwd": str(cwd),
        "command": cmd,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "state": "running",
    }
    state_json.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
