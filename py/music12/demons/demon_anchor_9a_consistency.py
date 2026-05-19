from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DEFAULT_SCAN_ROOT = Path("py/music12")


@dataclass
class DemonIssue:
    severity: str
    file: str
    line: int
    kind: str
    snippet: str
    message: str


def _iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.is_file():
            yield path


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _strip_inline_comment(line: str) -> str:
    if "#" in line:
        return line.split("#", 1)[0]
    return line


def _is_probably_safe_context(line: str) -> bool:
    lowered = line.lower()
    safe_markers = [
        "traceback",
        "error_message",
        "exception",
        "demo",
        "example",
        "test_",
        "pytest",
        "unittest",
        "docstring",
        "commentary only",
    ]
    return any(x in lowered for x in safe_markers)


def _check_a4_literals(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    code = _strip_inline_comment(stripped)

    patterns = [
        r'["\']A4["\']',
        r'["\']A#4["\']',
        r'["\']Bb4["\']',
        r'["\']C4["\']',
        r'["\']C#4["\']',
        r'["\']D4["\']',
        r'["\']E4["\']',
        r'["\']F4["\']',
        r'["\']G4["\']',
    ]

    if any(re.search(p, code) for p in patterns):
        if _is_probably_safe_context(code):
            return issues
        issues.append(
            DemonIssue(
                severity="warning",
                file=str(path),
                line=lineno,
                kind="western_note_literal",
                snippet=stripped,
                message='Найдена западная буквенная нота (например A4/C4). Проверить, не обходит ли она якорь 9.A.',
            )
        )

    return issues


def _check_440_literals(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    code = _strip_inline_comment(stripped)

    if re.search(r'(^|[^\d])440(\.0+)?([^\d]|$)', code):
        if _is_probably_safe_context(code):
            return issues

        if "9.A" in code or "9.a" in code.lower():
            return issues

        issues.append(
            DemonIssue(
                severity="warning",
                file=str(path),
                line=lineno,
                kind="bare_440_literal",
                snippet=stripped,
                message='Найдено число 440 без явной привязки к 9.A. Проверить якорную согласованность.',
            )
        )

    return issues


def _check_anchor_words(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    lowered = stripped.lower()

    suspicious_words = [
        "a4 = 440",
        "a4=440",
        "concert pitch",
        "middle c",
        "equal temperament anchor",
        "midi note 69",
        "note 69",
        "concert a",
    ]

    for word in suspicious_words:
        if word in lowered:
            issues.append(
                DemonIssue(
                    severity="warning",
                    file=str(path),
                    line=lineno,
                    kind="foreign_anchor_wording",
                    snippet=stripped,
                    message="Найдена формулировка старого якоря/строя. Проверить, не противоречит ли она принципу 9.A.",
                )
            )
            break

    return issues


def _check_direct_anchor_override(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    code = _strip_inline_comment(stripped).lower()

    suspicious_patterns = [
        r'anchor_note\s*=',
        r'anchor_frequency',
        r'base_note\s*=',
        r'reference_note\s*=',
        r'concert_a\s*=',
        r'midi_note_69',
    ]

    if any(re.search(p, code) for p in suspicious_patterns):
        if "9.a" in code:
            return issues
        issues.append(
            DemonIssue(
                severity="warning",
                file=str(path),
                line=lineno,
                kind="anchor_override_pattern",
                snippet=stripped,
                message="Обнаружен паттерн ручного задания якоря. Проверить, не вводится ли якорь в обход 9.A.",
            )
        )

    return issues


def _check_missing_9a_near_440(path: Path, lines: list[str]) -> list[DemonIssue]:
    issues: list[DemonIssue] = []

    text = "\n".join(lines)
    lowered = text.lower()

    mentions_440 = "440" in lowered
    mentions_9a = "9.a" in lowered or "9.A" in text

    if mentions_440 and not mentions_9a:
        issues.append(
            DemonIssue(
                severity="info",
                file=str(path),
                line=1,
                kind="440_without_9a_context",
                snippet=path.name,
                message='В файле встречается 440, но не найден контекст 9.A. Проверить вручную.',
            )
        )

    return issues


def scan_file(path: Path) -> list[DemonIssue]:
    lines = _read_lines(path)
    issues: list[DemonIssue] = []

    for idx, line in enumerate(lines, start=1):
        issues.extend(_check_a4_literals(path, idx, line))
        issues.extend(_check_440_literals(path, idx, line))
        issues.extend(_check_anchor_words(path, idx, line))
        issues.extend(_check_direct_anchor_override(path, idx, line))

    issues.extend(_check_missing_9a_near_440(path, lines))
    return issues


def build_report(issues: list[DemonIssue]) -> tuple[dict, str]:
    by_severity = {"error": 0, "warning": 0, "info": 0}
    by_kind: dict[str, int] = {}

    for issue in issues:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        by_kind[issue.kind] = by_kind.get(issue.kind, 0) + 1

    payload = {
        "issue_count": len(issues),
        "by_severity": by_severity,
        "by_kind": by_kind,
        "issues": [asdict(x) for x in issues],
    }

    lines = []
    lines.append("MUSIC12 ANCHOR 9.A CONSISTENCY REPORT")
    lines.append("=" * 72)
    lines.append(f"issue_count : {len(issues)}")
    lines.append(f"errors      : {by_severity.get('error', 0)}")
    lines.append(f"warnings    : {by_severity.get('warning', 0)}")
    lines.append(f"info        : {by_severity.get('info', 0)}")
    lines.append("")

    if by_kind:
        lines.append("BY KIND")
        lines.append("-" * 72)
        for kind, count in sorted(by_kind.items()):
            lines.append(f"{kind:<32} {count}")
        lines.append("")

    if issues:
        lines.append("ISSUES")
        lines.append("-" * 72)
        for idx, issue in enumerate(issues, start=1):
            lines.append(f"{idx:>3}. [{issue.severity.upper()}] {issue.kind}")
            lines.append(f"     file    : {issue.file}")
            lines.append(f"     line    : {issue.line}")
            lines.append(f"     snippet : {issue.snippet}")
            lines.append(f"     message : {issue.message}")
            lines.append("")
    else:
        lines.append("No issues found.")
        lines.append("")

    return payload, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demon for checking anchor 9.A consistency across the project"
    )
    parser.add_argument("--root", default=str(DEFAULT_SCAN_ROOT), help="Root directory to scan")
    parser.add_argument("--out_txt", required=True, help="Output TXT report")
    parser.add_argument("--out_json", required=True, help="Output JSON report")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_json = Path(args.out_json).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Scan root does not exist: {root}")

    all_issues: list[DemonIssue] = []

    for py_file in _iter_py_files(root):
        all_issues.extend(scan_file(py_file))

    payload, txt = build_report(all_issues)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_txt.write_text(txt, encoding="utf-8")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report txt : {out_txt}")
    print(f"report json: {out_json}")
    print(f"issues     : {len(all_issues)}")


if __name__ == "__main__":
    main()