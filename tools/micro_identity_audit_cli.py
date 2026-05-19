# -*- coding: utf-8 -*-
"""
music12.tools.micro_identity_audit_cli

Soft audit for possible micro-token identity collapse in Python source files.

IMPORTANT:
- This is NOT a Maxwell demon.
- This tool does NOT block execution.
- This tool does NOT patch code.
- This tool only reports suspicious places for manual review.

Purpose:
    Find code patterns that may collapse full 12-radix micro tokens like:
        9.A'i3, 8.7'a2, A.C'i3A
    into coarse tokens like:
        9.A'-

Typical dangerous patterns:
    split("'")
    + "'-"
    _normalize_note(...)
    coarse_note
    pitch_class
    base_pitch
    note_token normalization without preserving micro token fields

Recommended location:
    E:\Duodecimal_resonant_numeration\py\music12\tools\micro_identity_audit_cli.py

Example:
    cd /d E:\Duodecimal_resonant_numeration && set PYTHONPATH=%CD%\py && python -m music12.tools.micro_identity_audit_cli --root py\music12\blocks\Block002_audio_recogn --out _demon_logs\micro_identity_audit_Block002.txt --csv _demon_logs\micro_identity_audit_Block002.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class AuditPattern:
    name: str
    regex: re.Pattern[str]
    severity: str
    reason: str
    recommendation: str


@dataclass(frozen=True)
class Finding:
    file: str
    line_no: int
    severity: str
    pattern: str
    line: str
    reason: str
    recommendation: str


PATTERNS: list[AuditPattern] = [
    AuditPattern(
        name="apostrophe_split",
        regex=re.compile(r"\.split\s*\(\s*([\"'])'\1\s*,?"),
        severity="HIGH",
        reason="Code splits token at apostrophe. This can discard micro suffix after inch mark.",
        recommendation="Check whether full micro token is preserved in a separate field before coarse extraction.",
    ),
    AuditPattern(
        name="force_micro_dash_concat",
        regex=re.compile(r"\+\s*([\"'])'-\1|([\"'])'-\2\s*\+"),
        severity="HIGH",
        reason="Code explicitly constructs plain exact suffix '-. This may collapse micro identity.",
        recommendation="Use a separate coarse token field instead of overwriting the full micro token.",
    ),
    AuditPattern(
        name="normalize_note_function",
        regex=re.compile(r"\b_?normalize_?note\b|\bcanonical_?note\b|\bnormalize_?pitch\b", re.IGNORECASE),
        severity="HIGH",
        reason="Note normalization function found. Such functions often collapse micro suffix to coarse note.",
        recommendation="Verify function preserves note_token_micro / token_raw and only writes coarse token to a separate field.",
    ),
    AuditPattern(
        name="coarse_identity_terms",
        regex=re.compile(r"\b(coarse_note|coarse_token|base_pitch|pitch_class|pc_token|pitchclass)\b", re.IGNORECASE),
        severity="MEDIUM",
        reason="Coarse identity terminology found. This is valid only if micro identity is kept separately.",
        recommendation="Confirm there are paired fields such as note_token_micro, observed_token_micro, raw_token, freq_hz, delta_cents.",
    ),
    AuditPattern(
        name="token_field_write",
        regex=re.compile(r"\b(note_token|candidate_note|resolved_note|scene_note|final_note|root_note|observed_token|theoretical_token)\b", re.IGNORECASE),
        severity="LOW",
        reason="Token-related field found. Not dangerous by itself, but should be checked in stages that transform identity.",
        recommendation="Check whether this field is exact micro identity or coarse display identity.",
    ),
    AuditPattern(
        name="replace_apostrophe_suffix",
        regex=re.compile(r"\.replace\s*\(.*'|re\.sub\s*\(.*'", re.IGNORECASE),
        severity="HIGH",
        reason="Replacement near apostrophe detected. This may rewrite micro suffix.",
        recommendation="Manually inspect replacement logic and ensure micro suffix is not erased.",
    ),
    AuditPattern(
        name="micro_unaware_groupby",
        regex=re.compile(r"groupby\s*\(.*(note|token|pitch)|sort_values\s*\(.*(note|token|pitch)", re.IGNORECASE),
        severity="MEDIUM",
        reason="Grouping/sorting by note/token/pitch may merge micro-distinct identities.",
        recommendation="If grouping is coarse, preserve micro-level rows or add group key fields explicitly.",
    ),
]


EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "venv",
    ".venv",
    "env",
}


def iter_py_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".py":
            yield root
        return

    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        yield path


def audit_file(path: Path, project_root: Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        rel = str(path)
        if project_root:
            try:
                rel = str(path.relative_to(project_root))
            except Exception:
                pass
        return [
            Finding(
                file=rel,
                line_no=0,
                severity="ERROR",
                pattern="read_error",
                line="",
                reason=f"Could not read file: {exc}",
                recommendation="Check file encoding or permissions.",
            )
        ]

    rel = str(path)
    if project_root:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            pass

    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pat in PATTERNS:
            if pat.regex.search(line):
                findings.append(
                    Finding(
                        file=rel,
                        line_no=i,
                        severity=pat.severity,
                        pattern=pat.name,
                        line=stripped[:300],
                        reason=pat.reason,
                        recommendation=pat.recommendation,
                    )
                )
    return findings


def severity_rank(sev: str) -> int:
    return {"ERROR": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(sev.upper(), 0)


def write_txt(findings: Sequence[Finding], out_path: Path, scanned_files: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    ordered = sorted(
        findings,
        key=lambda f: (-severity_rank(f.severity), f.file.lower(), f.line_no, f.pattern),
    )

    lines: list[str] = []
    lines.append("MICRO IDENTITY AUDIT")
    lines.append("====================")
    lines.append("")
    lines.append("Soft source-code audit for possible 12-radix micro-token collapse.")
    lines.append("This tool does not fail execution and does not patch code.")
    lines.append("")
    lines.append(f"Scanned .py files: {scanned_files}")
    lines.append(f"Findings total: {len(findings)}")
    for sev in ("ERROR", "HIGH", "MEDIUM", "LOW"):
        lines.append(f"{sev}: {counts.get(sev, 0)}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- HIGH means manual review is strongly recommended.")
    lines.append("- MEDIUM means the code may be valid if exact micro identity is preserved elsewhere.")
    lines.append("- LOW marks token-related places that are useful for tracing identity flow.")
    lines.append("")
    lines.append("Findings")
    lines.append("--------")

    if not ordered:
        lines.append("No suspicious patterns found.")
    else:
        for f in ordered:
            lines.append("")
            lines.append(f"[{f.severity}] {f.file}:{f.line_no}  pattern={f.pattern}")
            lines.append(f"LINE: {f.line}")
            lines.append(f"WHY : {f.reason}")
            lines.append(f"DO  : {f.recommendation}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(findings: Sequence[Finding], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "severity",
                "file",
                "line_no",
                "pattern",
                "line",
                "reason",
                "recommendation",
            ],
        )
        writer.writeheader()
        for item in sorted(
            findings,
            key=lambda x: (-severity_rank(x.severity), x.file.lower(), x.line_no, x.pattern),
        ):
            writer.writerow(
                {
                    "severity": item.severity,
                    "file": item.file,
                    "line_no": item.line_no,
                    "pattern": item.pattern,
                    "line": item.line,
                    "reason": item.reason,
                    "recommendation": item.recommendation,
                }
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Soft audit for possible micro-token identity collapse in music12 Python source files."
    )
    p.add_argument(
        "--root",
        required=True,
        help="File or directory to scan, e.g. py\\music12\\blocks\\Block002_audio_recogn",
    )
    p.add_argument(
        "--project-root",
        default=".",
        help="Project root for relative paths. Default: current directory.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="TXT report output path.",
    )
    p.add_argument(
        "--csv",
        default="",
        help="Optional CSV report output path.",
    )
    p.add_argument(
        "--min-severity",
        default="LOW",
        choices=["LOW", "MEDIUM", "HIGH", "ERROR"],
        help="Minimum severity to include in reports. Default: LOW.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    project_root = Path(args.project_root).resolve()
    root = Path(args.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()

    min_rank = severity_rank(args.min_severity)

    files = list(iter_py_files(root))
    findings: list[Finding] = []
    for py_file in files:
        findings.extend(audit_file(py_file, project_root=project_root))

    findings = [f for f in findings if severity_rank(f.severity) >= min_rank]

    write_txt(findings, Path(args.out), scanned_files=len(files))
    if args.csv:
        write_csv(findings, Path(args.csv))

    high = sum(1 for f in findings if f.severity in {"HIGH", "ERROR"})
    print("MICRO_IDENTITY_AUDIT_DONE")
    print(f"scanned_files={len(files)}")
    print(f"findings={len(findings)}")
    print(f"high_or_error={high}")
    print(f"txt={args.out}")
    if args.csv:
        print(f"csv={args.csv}")
    print("verdict=SOFT_AUDIT_ONLY_MANUAL_REVIEW")

    # Always return 0: this is not a blocking demon.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
