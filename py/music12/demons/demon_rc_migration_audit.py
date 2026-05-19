from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple


FALSE_F0_TOKENS = [
    "chosen_f0",
    "chosen_f0_note",
    "chosen_f0_hz",
    "chosen_f0_energy",
    "has_f0",
    "stable_f0",
    "stabilize_f0",
    "infer_f0",
    "compare_f0",
    "f0_chain",
    "true_f0",
]

FALSE_FD_TOKENS = [
    "chosen_fd",
    "chosen_fd_note",
    "chosen_fd_hz",
    "chosen_fd_energy",
    "has_fd",
    "stable_fd",
    "stabilize_fd",
    "infer_fd",
    "compare_fd",
    "fd_chain",
]

RC_TOKENS = [
    "chosen_rc",
    "chosen_rc_note",
    "chosen_rc_hz",
    "chosen_rc_energy",
    "has_rc",
    "stable_rc",
    "stabilize_rc",
    "infer_rc",
    "compare_rc",
    "rc_chain",
    "rc_chain_score",
    "representative_rc",
]

ALL_TOKENS = FALSE_F0_TOKENS + FALSE_FD_TOKENS + RC_TOKENS


@dataclass
class MatchRecord:
    file_path: str
    line_no: int
    token: str
    token_group: str
    context_type: str
    migration_status: str
    target_name: str
    line_text: str


def iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except Exception:
            return path.read_text(encoding="latin-1", errors="replace")
    except Exception:
        return ""


def classify_token_group(token: str) -> str:
    if token in FALSE_F0_TOKENS:
        return "false_f0"
    if token in FALSE_FD_TOKENS:
        return "false_fd"
    if token in RC_TOKENS:
        return "rc"
    return "unknown"


def target_name_for(token: str) -> str:
    if token.startswith("chosen_f0"):
        return token.replace("chosen_f0", "chosen_rc")
    if token.startswith("chosen_fd"):
        return token.replace("chosen_fd", "chosen_rc")
    if token == "has_f0":
        return "has_rc"
    if token == "has_fd":
        return "has_rc"
    if token == "stable_f0":
        return "stable_rc"
    if token == "stable_fd":
        return "stable_rc"
    if token == "stabilize_f0":
        return "stabilize_rc"
    if token == "stabilize_fd":
        return "stabilize_rc"
    if token == "infer_f0":
        return "infer_rc"
    if token == "infer_fd":
        return "infer_rc"
    if token == "compare_f0":
        return "compare_rc"
    if token == "compare_fd":
        return "compare_rc"
    if token == "f0_chain":
        return "rc_chain"
    if token == "fd_chain":
        return "rc_chain"
    return token


def detect_context_type(line: str) -> str:
    s = line.strip()
    lower = s.lower()

    if re.search(r"^\s*(class|def)\s+", s):
        return "logic_assertion"

    if any(k in lower for k in ["description=", "help=", "print(", "deprecated", "semantic", "meta"]):
        return "meta_text"

    if re.search(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*', s):
        return "create"

    if re.search(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*', s):
        return "create"

    if any(k in s for k in [".get(", "r.get(", "row.", "res.", "reader", "writer", "[", "]"]):
        return "read"

    return "mention"


def classify_migration_status(token_group: str, context_type: str) -> str:
    if token_group == "rc":
        return "OK_RC"
    if context_type in {"logic_assertion", "create", "read"}:
        return "REWRITE_REQUIRED"
    if context_type in {"meta_text", "mention"}:
        return "RENAME_SAFE"
    return "REVIEW"


def find_token_matches(text: str) -> List[Tuple[int, str, str]]:
    out: List[Tuple[int, str, str]] = []
    lines = text.splitlines()

    token_patterns = [
        (tok, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])"))
        for tok in ALL_TOKENS
    ]

    for i, line in enumerate(lines, start=1):
        for token, pattern in token_patterns:
            if pattern.search(line):
                out.append((i, token, line))
    return out


def scan_root(root: Path) -> List[MatchRecord]:
    records: List[MatchRecord] = []

    for path in iter_py_files(root):
        text = safe_read_text(path)
        if not text:
            continue

        matches = find_token_matches(text)
        if not matches:
            continue

        for line_no, token, line in matches:
            token_group = classify_token_group(token)
            context_type = detect_context_type(line)
            migration_status = classify_migration_status(token_group, context_type)
            target_name = target_name_for(token)

            records.append(
                MatchRecord(
                    file_path=str(path),
                    line_no=line_no,
                    token=token,
                    token_group=token_group,
                    context_type=context_type,
                    migration_status=migration_status,
                    target_name=target_name,
                    line_text=line.strip(),
                )
            )

    return records


def build_summary(records: List[MatchRecord]) -> dict:
    by_token = Counter(r.token for r in records)
    by_group = Counter(r.token_group for r in records)
    by_context = Counter(r.context_type for r in records)
    by_status = Counter(r.migration_status for r in records)
    by_file = Counter(r.file_path for r in records)

    rewrite_required = Counter(r.file_path for r in records if r.migration_status == "REWRITE_REQUIRED")
    rename_safe = Counter(r.file_path for r in records if r.migration_status == "RENAME_SAFE")
    ok_rc = Counter(r.file_path for r in records if r.migration_status == "OK_RC")

    return {
        "summary": {
            "total_matches": len(records),
            "files_with_hits_count": len(set(r.file_path for r in records)),
        },
        "by_token": dict(by_token.most_common()),
        "by_group": dict(by_group.most_common()),
        "by_context": dict(by_context.most_common()),
        "by_status": dict(by_status.most_common()),
        "rewrite_required_top": [
            {"file_path": p, "matches": c} for p, c in rewrite_required.most_common(100)
        ],
        "rename_safe_top": [
            {"file_path": p, "matches": c} for p, c in rename_safe.most_common(100)
        ],
        "ok_rc_top": [
            {"file_path": p, "matches": c} for p, c in ok_rc.most_common(100)
        ],
        "top_files": [
            {"file_path": p, "matches": c} for p, c in by_file.most_common(100)
        ],
    }


def write_txt_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("RC MIGRATION AUDIT REPORT (PY ONLY)")
    lines.append("=" * 80)
    lines.append(f"total_matches: {summary['summary']['total_matches']}")
    lines.append(f"files_with_hits_count: {summary['summary']['files_with_hits_count']}")
    lines.append("")

    lines.append("BY GROUP")
    for k, v in summary["by_group"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("BY CONTEXT")
    for k, v in summary["by_context"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("BY STATUS")
    for k, v in summary["by_status"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("TOP FILES REWRITE_REQUIRED")
    for item in summary["rewrite_required_top"][:50]:
        lines.append(f"  {item['matches']:4d}  {item['file_path']}")
    lines.append("")

    lines.append("TOP FILES RENAME_SAFE")
    for item in summary["rename_safe_top"][:50]:
        lines.append(f"  {item['matches']:4d}  {item['file_path']}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_report(path: Path, records: List[MatchRecord], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **summary,
        "records": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_report(path: Path, records: List[MatchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "line_no",
                "token",
                "token_group",
                "context_type",
                "migration_status",
                "target_name",
                "line_text",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast audit of Python code for migration from false f0/fd semantics to rc."
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    records = scan_root(root)
    summary = build_summary(records)

    write_txt_report(Path(args.out_txt).resolve(), summary)
    write_json_report(Path(args.out_json).resolve(), records, summary)
    write_csv_report(Path(args.out_csv).resolve(), records)

    print("rc migration audit complete (py only)")
    print(json.dumps(
        {
            "root": str(root),
            "total_matches": summary["summary"]["total_matches"],
            "files_with_hits_count": summary["summary"]["files_with_hits_count"],
            "out_txt": str(Path(args.out_txt).resolve()),
            "out_json": str(Path(args.out_json).resolve()),
            "out_csv": str(Path(args.out_csv).resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()