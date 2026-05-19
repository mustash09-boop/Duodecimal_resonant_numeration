from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from music12.demons.angels.angel_patch_utils import write_json, write_txt
from music12.demons.angels.angel_registry import select_angel


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_txt_report(payload: Dict[str, Any]) -> str:
    lines = [
        "ANGEL RUN REPORT",
        "",
        f"angel_name: {payload.get('angel_name')}",
        f"status: {payload.get('status')}",
        f"failure_class_before: {payload.get('failure_class_before')}",
        f"target_file: {payload.get('target_file')}",
        f"backup_file: {payload.get('backup_file')}",
        f"verify_status: {payload.get('verify_status')}",
        f"rerun_class: {payload.get('rerun_class')}",
        f"rerun_returncode: {payload.get('rerun_returncode')}",
        "",
        "PATCHED REGIONS:",
    ]
    for item in payload.get("patched_regions", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("NOTES:")
    for item in payload.get("notes", []):
        lines.append(f"- {item}")

    if payload.get("rerun_command"):
        lines.append("")
        lines.append("RERUN COMMAND:")
        lines.append(" ".join(str(x) for x in payload["rerun_command"]))

    if payload.get("stderr_tail"):
        lines.append("")
        lines.append("STDERR TAIL:")
        lines.append(payload["stderr_tail"])

    if payload.get("stdout_tail"):
        lines.append("")
        lines.append("STDOUT TAIL:")
        lines.append(payload["stdout_tail"])

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run angel repair based on Maxwell JSON report."
    )
    parser.add_argument("--report_json", required=True, help="Path to Maxwell JSON report.")
    parser.add_argument("--project_root", required=True, help="Project root directory.")
    parser.add_argument("--out_json", default="_demon_logs/angel_last_run.json")
    parser.add_argument("--out_txt", default="_demon_logs/angel_last_run.txt")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    report_json = Path(args.report_json).resolve()
    out_json = (project_root / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_txt = (project_root / args.out_txt).resolve() if not Path(args.out_txt).is_absolute() else Path(args.out_txt)

    report = load_json(report_json)

    angel = select_angel(report)
    if angel is None:
        payload = {
            "angel_name": None,
            "status": "no_matching_angel",
            "failure_class_before": report.get("failure_class"),
            "target_file": None,
            "backup_file": None,
            "patched_regions": [],
            "verify_status": "not_run",
            "rerun_class": None,
            "rerun_command": [],
            "rerun_returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "notes": ["No angel matched the Maxwell failure class / report pattern."],
        }
        write_json(out_json, payload)
        write_txt(out_txt, build_txt_report(payload))
        return 2

    analysis = angel.analyze(report, project_root=project_root)
    repair_result = angel.repair(report, analysis, project_root=project_root)
    verify_result = angel.verify(repair_result, report, project_root=project_root)

    payload = {
        "angel_name": repair_result.angel_name,
        "status": repair_result.status,
        "failure_class_before": report.get("failure_class"),
        "target_file": repair_result.target_file,
        "backup_file": repair_result.backup_file,
        "patched_regions": repair_result.patched_regions,
        "verify_status": verify_result.get("verify_status"),
        "rerun_class": verify_result.get("rerun_class"),
        "rerun_command": verify_result.get("rerun_command", []),
        "rerun_returncode": verify_result.get("rerun_returncode"),
        "stdout_tail": verify_result.get("stdout_tail", ""),
        "stderr_tail": verify_result.get("stderr_tail", ""),
        "notes": repair_result.notes + verify_result.get("notes", []),
        "details": {
            "analysis": {
                "can_handle": analysis.can_handle,
                "repair_strategy": analysis.repair_strategy,
                "reasons": analysis.reasons,
            },
            "verify": verify_result,
        },
    }

    write_json(out_json, payload)
    write_txt(out_txt, build_txt_report(payload))

    if payload["verify_status"] == "pass_after_repair":
        return 0
    if payload["verify_status"] == "repair_failed_same_cli_issue":
        return 1
    if repair_result.status in {"patched", "no_change"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())