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


def _is_probably_safe_numeric_context(line: str) -> bool:
    safe_markers = [
        "sample_rate",
        "44100",
        "48000",
        "96000",
        "dpi=",
        "max_results",
        "http",
        "https",
        "status_code",
        "exit_code",
        "probe_index",
        "frame_index",
    ]
    lowered = line.lower()
    return any(x.lower() in lowered for x in safe_markers)


def _check_explicit_hundred_division(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    code = _strip_inline_comment(stripped)

    patterns = [
        (r"/\s*100(\D|$)", 'Обнаружено деление на 100. Для аналитической временной сетки проекта это подозрительно.'),
        (r"\*\s*100(\D|$)", 'Обнаружено умножение на 100. Проверить, не внедряется ли centi-time логика.'),
        (r"0\.01(\D|$)", 'Обнаружен шаг 0.01 секунды. Для проекта базовой сеткой является 1/60.'),
        (r"0\.02(\D|$)", 'Обнаружен шаг 0.02 секунды. Проверить, не используется ли деление секунды на 50/100.'),
        (r"1\.0\s*/\s*100(\D|$)", 'Обнаружено явное 1/100. Это противоречит time60-принципу, если речь об аналитической сетке.'),
    ]

    for pattern, message in patterns:
        if re.search(pattern, code):
            if _is_probably_safe_numeric_context(code):
                continue
            issues.append(
                DemonIssue(
                    severity="warning",
                    file=str(path),
                    line=lineno,
                    kind="time_division_by_100",
                    snippet=stripped,
                    message=message,
                )
            )
            break

    return issues


def _check_hundred_words(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    lowered = stripped.lower()

    suspicious_words = [
        "centisecond",
        "centiseconds",
        "milliseconds",
        "millisecond",
        "divide second by 100",
        "100 frames per second",
        "100 fps",
        "seconds / 100",
        "time*100",
        "time * 100",
        "sec*100",
        "sec * 100",
    ]

    for word in suspicious_words:
        if word in lowered:
            issues.append(
                DemonIssue(
                    severity="warning",
                    file=str(path),
                    line=lineno,
                    kind="time100_wording",
                    snippet=stripped,
                    message="Найдена текстовая/комментарная логика времени, тяготеющая к 100, а не к time60.",
                )
            )
            break

    return issues


def _check_time_step_literals(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []
    stripped = line.strip()
    code = _strip_inline_comment(stripped).lower()

    if "time_step" in code or "step_seconds" in code or "window_seconds" in code or "frame_length" in code:
        if "1.0 / 60" in code or "1/60" in code or "0.016666" in code:
            return issues

        if "0.01" in code or "0.02" in code or "/100" in code:
            issues.append(
                DemonIssue(
                    severity="error",
                    file=str(path),
                    line=lineno,
                    kind="time_step_not_60_based",
                    snippet=stripped,
                    message="Параметр временной сетки выглядит не как time60-основанный.",
                )
            )

    return issues


def _check_time60_absence_hint(path: Path, lines: list[str]) -> list[DemonIssue]:
    """
    Мягкая эвристика: если файл активно работает со временем,
    но нигде не видно /60, можно напомнить проверить.
    """
    issues: list[DemonIssue] = []

    text = "\n".join(lines).lower()

    time_related = any(x in text for x in ["time_step", "step_seconds", "window_seconds", "frame_length", "time grid"])
    has_60 = any(x in text for x in ["1.0 / 60", "1/60", "0.016666", "time60"])

    if time_related and not has_60:
        issues.append(
            DemonIssue(
                severity="info",
                file=str(path),
                line=1,
                kind="time60_not_explicit",
                snippet=path.name,
                message="Файл работает со временем, но явная опора на 60 не обнаружена. Проверить вручную.",
            )
        )

    return issues


def scan_file(path: Path) -> list[DemonIssue]:
    lines = _read_lines(path)
    issues: list[DemonIssue] = []

    for idx, line in enumerate(lines, start=1):
        issues.extend(_check_explicit_hundred_division(path, idx, line))
        issues.extend(_check_hundred_words(path, idx, line))
        issues.extend(_check_time_step_literals(path, idx, line))

    issues.extend(_check_time60_absence_hint(path, lines))
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
    lines.append("MUSIC12 TIME60 CONSISTENCY REPORT")
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
        description="Demon for checking time60 consistency across the project"
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