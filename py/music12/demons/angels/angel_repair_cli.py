from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from music12.demons.angels.angel_base import AngelAnalysis, AngelBase, AngelRepairResult
from music12.demons.angels.angel_patch_utils import backup_file, read_text, write_text


class CliNotationRepairAngel(AngelBase):
    angel_name = "cli_notation_repair_angel"
    supported_failure_classes = ["notation_failure", "value_failure", "runtime_exception"]

    TARGET_MODULE_PATH = Path("py/music12/blocks/Block002_audio_recogn/resonance_probe12_scan_cli.py")

    def _extract_exception_text(self, report: Dict[str, Any]) -> str:
        parts: List[str] = []

        for key in ("exception", "exception_text", "traceback", "notes"):
            value = report.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(x) for x in value)

        return "\n".join(parts)

    def _looks_like_duodecimal_cli_issue(self, report: Dict[str, Any], exception_text: str) -> bool:
        module_text = str(report.get("target_module", ""))

        signals = [
            "invalid int value 'A'",
            "invalid int value",
            "--octave_max",
            "--octave_min",
            "argparse",
            "int(",
        ]

        has_signal = any(s.lower() in exception_text.lower() for s in signals)
        right_module = ("resonance_probe12_scan_cli" in module_text) or ("resonance_probe12_scan_cli" in exception_text)

        # MVP: allow either explicit module hit or strong octave/argparse symptoms
        return has_signal or right_module

    def analyze(self, report: Dict[str, Any], project_root: Path) -> AngelAnalysis:
        failure_class = str(report.get("failure_class", "")).strip()
        exception_text = self._extract_exception_text(report)
        target_file = project_root / self.TARGET_MODULE_PATH

        reasons: List[str] = []
        can_handle = self.can_handle(report) and self._looks_like_duodecimal_cli_issue(report, exception_text)

        if failure_class in self.supported_failure_classes:
            reasons.append(f"failure_class={failure_class}")
        if "invalid int value" in exception_text.lower():
            reasons.append("argparse rejected octave argument as decimal int")
        if "--octave_max" in exception_text or "--octave_min" in exception_text:
            reasons.append("octave CLI parameter found in exception text")
        if not target_file.exists():
            reasons.append(f"target file not found: {target_file}")

        if can_handle and target_file.exists():
            return AngelAnalysis(
                angel_name=self.angel_name,
                can_handle=True,
                target_file=str(target_file),
                failure_class=failure_class,
                repair_strategy="patch_cli_octave_parser_to_duodecimal",
                reasons=reasons,
                details={
                    "target_module_path": str(self.TARGET_MODULE_PATH),
                },
            )

        return AngelAnalysis(
            angel_name=self.angel_name,
            can_handle=False,
            target_file=str(target_file),
            failure_class=failure_class,
            repair_strategy=None,
            reasons=reasons,
            details={},
        )

    def _build_patch_block(self) -> str:
        return '''
DUODECIMAL_OCTAVE_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
}

def parse_duodecimal_octave_arg(value: str) -> int:
    s = str(value).strip().upper()
    if s not in DUODECIMAL_OCTAVE_MAP:
        raise argparse.ArgumentTypeError(
            f"Invalid duodecimal octave '{value}'. Allowed: 1..9 A B C"
        )
    return DUODECIMAL_OCTAVE_MAP[s]
'''.strip("\n")

    def _apply_targeted_patch(self, source: str) -> tuple[str, List[str], List[str]]:
        patched_regions: List[str] = []
        notes: List[str] = []
        text = source

        patch_block = self._build_patch_block()

        if "def parse_duodecimal_octave_arg(" not in text:
            if "import argparse" in text:
                text = text.replace(
                    "import argparse",
                    "import argparse\n\n# ANGEL_PATCH: duodecimal octave CLI parser\n" + patch_block,
                    1,
                )
                patched_regions.append("inserted parse_duodecimal_octave_arg after import argparse")
            else:
                notes.append("Could not locate 'import argparse' anchor for parser insertion.")

        replacements = [
            ('parser.add_argument("--octave_min", type=int', 'parser.add_argument("--octave_min", type=parse_duodecimal_octave_arg'),
            ('parser.add_argument("--octave_max", type=int', 'parser.add_argument("--octave_max", type=parse_duodecimal_octave_arg'),
            ("parser.add_argument('--octave_min', type=int", "parser.add_argument('--octave_min', type=parse_duodecimal_octave_arg"),
            ("parser.add_argument('--octave_max', type=int", "parser.add_argument('--octave_max', type=parse_duodecimal_octave_arg"),
        ]

        for old, new in replacements:
            if old in text:
                text = text.replace(old, new, 1)
                patched_regions.append(f"replaced: {old} -> {new}")

        if not patched_regions:
            notes.append("No parser.add_argument(type=int) pattern matched; manual review may be needed.")

        return text, patched_regions, notes

    def repair(self, report: Dict[str, Any], analysis: AngelAnalysis, project_root: Path) -> AngelRepairResult:
        if not analysis.can_handle or not analysis.target_file:
            return AngelRepairResult(
                angel_name=self.angel_name,
                status="skipped",
                target_file=analysis.target_file,
                notes=["Analysis says angel cannot safely handle this report."],
            )

        target_path = Path(analysis.target_file)
        backup_root = project_root / "_demon_logs" / "angel_backups"

        source = read_text(target_path)
        new_text, patched_regions, notes = self._apply_targeted_patch(source)

        if new_text == source:
            return AngelRepairResult(
                angel_name=self.angel_name,
                status="no_change",
                target_file=str(target_path),
                patched_regions=patched_regions,
                notes=notes,
            )

        backup_path = backup_file(target_path, backup_root)
        write_text(target_path, new_text)

        return AngelRepairResult(
            angel_name=self.angel_name,
            status="patched",
            target_file=str(target_path),
            backup_file=str(backup_path),
            patched_regions=patched_regions,
            notes=notes,
            details={
                "repair_strategy": analysis.repair_strategy,
            },
        )