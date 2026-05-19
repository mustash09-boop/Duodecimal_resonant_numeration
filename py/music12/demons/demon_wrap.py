from __future__ import annotations

import argparse
import io
import json
import os
import runpy
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _build_log_paths(*, logdir: Path, tag: str) -> tuple[Path, Path]:
    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(tag))
    return (
        logdir / f"{safe_tag}.json",
        logdir / f"{safe_tag}.txt",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _format_txt_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("DEMON WRAP REPORT")
    lines.append("=" * 80)
    lines.append(f"status           : {payload.get('status')}")
    lines.append(f"tag              : {payload.get('tag')}")
    lines.append(f"module           : {payload.get('module')}")
    lines.append(f"started_utc      : {payload.get('started_utc')}")
    lines.append(f"finished_utc     : {payload.get('finished_utc')}")
    lines.append(f"duration_seconds : {payload.get('duration_seconds')}")
    lines.append(f"cwd              : {payload.get('cwd')}")
    lines.append(f"pythonpath       : {payload.get('pythonpath')}")
    lines.append(f"argv             : {payload.get('argv')}")
    lines.append(f"exit_code        : {payload.get('exit_code')}")
    lines.append(f"error_type       : {payload.get('error_type')}")
    lines.append(f"error_message    : {payload.get('error_message')}")
    lines.append("")

    stdout_text = payload.get("stdout") or ""
    stderr_text = payload.get("stderr") or ""

    if stdout_text.strip():
        lines.append("STDOUT")
        lines.append("-" * 80)
        lines.append(stdout_text.rstrip())
        lines.append("")

    if stderr_text.strip():
        lines.append("STDERR")
        lines.append("-" * 80)
        lines.append(stderr_text.rstrip())
        lines.append("")

    tb = payload.get("traceback")
    if tb:
        lines.append("TRACEBACK")
        lines.append("-" * 80)
        lines.append(tb.rstrip())
        lines.append("")

    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run_module_under_demon(
    *,
    logdir: str | Path,
    tag: str,
    module: str,
    module_args: list[str] | None = None,
) -> dict:
    """
    Programmatic API for Maxwell Demon and future Angel layers.

    Runs a target module exactly the same way as CLI main(),
    writes json/txt logs, and returns the payload augmented with log paths.

    Additionally captures stdout/stderr so argparse failures become visible.

    Returned dict keys:
      status
      started_utc
      finished_utc
      duration_seconds
      tag
      module
      cwd
      pythonpath
      argv
      exit_code
      error_type
      error_message
      traceback
      stdout
      stderr
      log_json
      log_txt
    """
    logdir = Path(logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    json_path, txt_path = _build_log_paths(logdir=logdir, tag=tag)

    started_utc = _utc_now_iso()
    t0 = time.perf_counter()

    original_argv = sys.argv[:]
    exit_code = 0
    status = "ok"
    error_type = None
    error_message = None
    tb_text = None

    forwarded_args = list(module_args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    payload = {
        "status": "running",
        "started_utc": started_utc,
        "finished_utc": None,
        "duration_seconds": None,
        "tag": tag,
        "module": module,
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "argv": forwarded_args,
        "exit_code": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "stdout": "",
        "stderr": "",
    }

    try:
        sys.argv = [module] + forwarded_args
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            runpy.run_module(module, run_name="__main__")

    except SystemExit as e:
        code = e.code
        if code is None:
            exit_code = 0
            status = "ok"
        elif isinstance(code, int):
            exit_code = code
            status = "ok" if code == 0 else "system_exit"
        else:
            exit_code = 1
            status = "system_exit"
            error_type = "SystemExit"
            error_message = _safe_text(code)

        stderr_text = stderr_buf.getvalue().strip()
        stdout_text = stdout_buf.getvalue().strip()

        if status != "ok":
            if stderr_text:
                error_type = error_type or "SystemExit"
                error_message = stderr_text
            elif stdout_text:
                error_type = error_type or "SystemExit"
                error_message = stdout_text

    except Exception as e:
        exit_code = 1
        status = "exception"
        error_type = type(e).__name__
        error_message = _safe_text(e)
        tb_text = traceback.format_exc()

    finally:
        sys.argv = original_argv

    t1 = time.perf_counter()
    finished_utc = _utc_now_iso()

    payload["status"] = status
    payload["finished_utc"] = finished_utc
    payload["duration_seconds"] = round(t1 - t0, 6)
    payload["exit_code"] = exit_code
    payload["error_type"] = error_type
    payload["error_message"] = error_message
    payload["traceback"] = tb_text
    payload["stdout"] = stdout_buf.getvalue()
    payload["stderr"] = stderr_buf.getvalue()

    _write_json(json_path, payload)
    _write_text(txt_path, _format_txt_report(payload))

    payload["log_json"] = str(json_path)
    payload["log_txt"] = str(txt_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal wrapper demon for music12 modules"
    )
    parser.add_argument("--logdir", required=True, help="Directory for demon logs")
    parser.add_argument("--tag", required=True, help="Human-readable run tag")
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        help="Python module to run, e.g. music12.blocks.Block002_audio_recogn.some_cli",
    )
    parser.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the target module after '--'",
    )
    args = parser.parse_args()

    payload = run_module_under_demon(
        logdir=args.logdir,
        tag=args.tag,
        module=args.module,
        module_args=list(args.module_args),
    )

    print(f"demon log json: {payload['log_json']}")
    print(f"demon log txt : {payload['log_txt']}")

    if int(payload.get("exit_code", 1)) != 0:
        raise SystemExit(int(payload["exit_code"]))


if __name__ == "__main__":
    main()