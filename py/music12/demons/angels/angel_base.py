from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from music12.demons.angels.angel_rerun_utils import classify_rerun_result, extract_rerun_command, run_command


@dataclass
class AngelAnalysis:
    angel_name: str
    can_handle: bool
    target_file: Optional[str] = None
    failure_class: Optional[str] = None
    repair_strategy: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AngelRepairResult:
    angel_name: str
    status: str
    target_file: Optional[str] = None
    backup_file: Optional[str] = None
    patched_regions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class AngelBase:
    angel_name = "base_angel"
    supported_failure_classes: List[str] = []

    def can_handle(self, report: Dict[str, Any]) -> bool:
        failure_class = str(report.get("failure_class", "")).strip()
        return failure_class in self.supported_failure_classes

    def analyze(self, report: Dict[str, Any], project_root: Path) -> AngelAnalysis:
        raise NotImplementedError

    def repair(self, report: Dict[str, Any], analysis: AngelAnalysis, project_root: Path) -> AngelRepairResult:
        raise NotImplementedError

    def verify(self, repair_result: AngelRepairResult, report: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
        """
        Phase 2 verify:
        - reconstruct rerun command
        - rerun target module
        - compare whether original CLI failure disappeared
        """
        target_ok = False
        if repair_result.target_file:
            target_ok = Path(repair_result.target_file).exists()

        if repair_result.status not in {"patched", "no_change"}:
            return {
                "angel_name": self.angel_name,
                "verify_status": "not_run",
                "target_exists": target_ok,
                "notes": ["Verify skipped because repair did not reach patched/no_change state."],
            }

        rerun_cmd = extract_rerun_command(report, project_root=project_root)
        if not rerun_cmd:
            return {
                "angel_name": self.angel_name,
                "verify_status": "needs_manual_check",
                "target_exists": target_ok,
                "notes": [
                    "Could not reconstruct rerun command from Maxwell report.",
                    "Add rerun_command or target_module/target_args to Maxwell JSON for full automation.",
                ],
            }

        rerun_result = run_command(rerun_cmd, project_root=project_root)
        rerun_class = classify_rerun_result(rerun_result)

        if rerun_class == "pass":
            verify_status = "pass_after_repair"
        elif rerun_class in {"same_cli_failure", "octave_cli_failure"}:
            verify_status = "repair_failed_same_cli_issue"
        else:
            verify_status = "rerun_revealed_next_failure"

        return {
            "angel_name": self.angel_name,
            "verify_status": verify_status,
            "target_exists": target_ok,
            "rerun_class": rerun_class,
            "rerun_ok": rerun_result.get("ok", False),
            "rerun_command": rerun_result.get("cmd", []),
            "rerun_returncode": rerun_result.get("returncode"),
            "stdout_tail": (rerun_result.get("stdout") or "")[-4000:],
            "stderr_tail": (rerun_result.get("stderr") or "")[-4000:],
            "notes": [
                "Phase 2 verify executed target module again after repair.",
            ],
        }