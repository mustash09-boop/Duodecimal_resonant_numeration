from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


# ============================================================
# DEMON: NOTATION ALPHABET 12 CONSISTENCY (STRICT)
# ------------------------------------------------------------
# ЖЁСТКАЯ версия:
# - запрещает любые локальные _DIGITS12
# - запрещает "0" в нотации
# - требует SSOT через harmonic_alphabet12.py
# ============================================================


ALLOWED_ALPHABET_SYMBOLS = set("123456789ABC")
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
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _check_digits12(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []

    literal = re.search(r'=\s*([\'"])(.*?)\1', line)
    value = literal.group(2) if literal else None

    # 🚫 ЛЮБОЙ _DIGITS12 — уже нарушение архитектуры
    issues.append(
        DemonIssue(
            severity="error",
            file=str(path),
            line=lineno,
            kind="digits12_forbidden_definition",
            snippet=line.strip(),
            message="Локальное определение _DIGITS12 запрещено. Использовать SSOT из harmonic_alphabet12.py.",
        )
    )

    if value:
        if "0" in value:
            issues.append(
                DemonIssue(
                    severity="error",
                    file=str(path),
                    line=lineno,
                    kind="digits12_contains_zero",
                    snippet=line.strip(),
                    message='В нотационной азбуке найден символ "0". Это недопустимо.',
                )
            )

        if set(value) != ALLOWED_ALPHABET_SYMBOLS:
            issues.append(
                DemonIssue(
                    severity="error",
                    file=str(path),
                    line=lineno,
                    kind="digits12_wrong_alphabet",
                    snippet=line.strip(),
                    message='Азбука не соответствует "123456789ABC".',
                )
            )

    return issues


def _check_degree12_render(path: Path, lineno: int, line: str) -> list[DemonIssue]:
    issues: list[DemonIssue] = []

    if "degree12" in line and "{" in line:
        if "_format_degree12_external" not in line:
            issues.append(
                DemonIssue(
                    severity="warning",
                    file=str(path),
                    line=lineno,
                    kind="direct_degree12_render",
                    snippet=line.strip(),
                    message="Прямой вывод degree12. Используй алфавитный мост.",
                )
            )

    return issues


def scan_file(path: Path) -> list[DemonIssue]:
    lines = _read_lines(path)
    issues: list[DemonIssue] = []

    for i, line in enumerate(lines):
        lineno = i + 1
        stripped = line.strip()

        if re.search(r'^\s*_DIGITS12\s*=', stripped):
            issues.extend(_check_digits12(path, lineno, line))

        issues.extend(_check_degree12_render(path, lineno, line))

    return issues


def build_report(issues: list[DemonIssue]) -> tuple[dict, str]:
    by_severity = {"error": 0, "warning": 0}
    by_kind: dict[str, int] = {}

    for issue in issues:
        by_severity[issue.severity] += 1
        by_kind[issue.kind] = by_kind.get(issue.kind, 0) + 1

    payload = {
        "issue_count": len(issues),
        "by_severity": by_severity,
        "by_kind": by_kind,
        "issues": [asdict(x) for x in issues],
    }

    lines = []
    lines.append("MUSIC12 NOTATION STRICT REPORT")
    lines.append("=" * 60)
    lines.append(f"errors   : {by_severity['error']}")
    lines.append(f"warnings : {by_severity['warning']}")
    lines.append("")

    return payload, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_SCAN_ROOT))
    parser.add_argument("--out_txt", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    issues = []

    for f in _iter_py_files(root):
        issues.extend(scan_file(f))

    payload, txt = build_report(issues)

    Path(args.out_txt).write_text(txt, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"errors={payload['by_severity']['error']}")


if __name__ == "__main__":
    main()