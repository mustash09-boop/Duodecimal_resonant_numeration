from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


TARGET_PATTERNS = [
    "chosen_f0_note",
    "chosen_f0_hz",
    "chosen_f0",
    "out_framewise_csv",
    "out_framewise_readable_csv",
    "stabilized",
]


CREATE_HINT_PATTERNS = [
    re.compile(r"chosen_f0_note\s*[:=]"),
    re.compile(r"chosen_f0_hz\s*[:=]"),
    re.compile(r"DictWriter"),
    re.compile(r"writerow"),
    re.compile(r"writerows"),
    re.compile(r"fieldnames"),
]


READ_HINT_PATTERNS = [
    re.compile(r"get\(\s*['\"]chosen_f0_note['\"]"),
    re.compile(r"\[\s*['\"]chosen_f0_note['\"]\s*\]"),
    re.compile(r"get\(\s*['\"]chosen_f0_hz['\"]"),
    re.compile(r"\[\s*['\"]chosen_f0_hz['\"]\s*\]"),
]


@dataclass
class MatchRecord:
    file_path: str
    line_no: int
    line_text: str
    matched_token: str
    context_type: str  # create / read / mention / csv_header / unknown


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".py", ".csv", ".json", ".md", ".txt", ".yaml", ".yml"
    }


def iter_project_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if "__pycache__" in path.parts:
            continue
        if not is_text_file(path):
            continue
        yield path


def classify_line(line: str, suffix: str) -> str:
    if suffix == ".csv":
        return "csv_header"

    for rx in CREATE_HINT_PATTERNS:
        if rx.search(line):
            return "create"

    for rx in READ_HINT_PATTERNS:
        if rx.search(line):
            return "read"

    return "mention"


def scan_file(path: Path) -> list[MatchRecord]:
    records: list[MatchRecord] = []

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return records

    lines = text.splitlines()

    # Special case for CSV header
    if path.suffix.lower() == ".csv" and lines:
        header = lines[0]
        for token in TARGET_PATTERNS:
            if token in header:
                records.append(
                    MatchRecord(
                        file_path=str(path),
                        line_no=1,
                        line_text=header[:500],
                        matched_token=token,
                        context_type="csv_header",
                    )
                )

    for idx, line in enumerate(lines, start=1):
        for token in TARGET_PATTERNS:
            if token in line:
                records.append(
                    MatchRecord(
                        file_path=str(path),
                        line_no=idx,
                        line_text=line[:500],
                        matched_token=token,
                        context_type=classify_line(line, path.suffix.lower()),
                    )
                )

    return records


def summarize(records: list[MatchRecord]) -> dict:
    files_with_hits = sorted({r.file_path for r in records})

    by_token: dict[str, int] = {}
    by_context: dict[str, int] = {}
    probable_creators: dict[str, int] = {}
    probable_readers: dict[str, int] = {}
    csv_headers: dict[str, int] = {}

    for r in records:
        by_token[r.matched_token] = by_token.get(r.matched_token, 0) + 1
        by_context[r.context_type] = by_context.get(r.context_type, 0) + 1

        if r.context_type == "create":
            probable_creators[r.file_path] = probable_creators.get(r.file_path, 0) + 1
        elif r.context_type == "read":
            probable_readers[r.file_path] = probable_readers.get(r.file_path, 0) + 1
        elif r.context_type == "csv_header":
            csv_headers[r.file_path] = csv_headers.get(r.file_path, 0) + 1

    return {
        "total_matches": len(records),
        "files_with_hits_count": len(files_with_hits),
        "files_with_hits": files_with_hits,
        "by_token": by_token,
        "by_context": by_context,
        "probable_creators": sorted(
            [{"file": k, "score": v} for k, v in probable_creators.items()],
            key=lambda x: (-x["score"], x["file"]),
        ),
        "probable_readers": sorted(
            [{"file": k, "score": v} for k, v in probable_readers.items()],
            key=lambda x: (-x["score"], x["file"]),
        ),
        "csv_headers": sorted(
            [{"file": k, "score": v} for k, v in csv_headers.items()],
            key=lambda x: (-x["score"], x["file"]),
        ),
    }


def write_txt_report(
    out_txt: Path,
    root: Path,
    records: list[MatchRecord],
    summary: dict,
) -> None:
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("CHOSEN_F0 FIELD AUDIT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"root: {root}\n")
        f.write(f"total_matches: {summary['total_matches']}\n")
        f.write(f"files_with_hits_count: {summary['files_with_hits_count']}\n\n")

        f.write("BY TOKEN\n")
        for k, v in sorted(summary["by_token"].items()):
            f.write(f"  {k}: {v}\n")

        f.write("\nBY CONTEXT\n")
        for k, v in sorted(summary["by_context"].items()):
            f.write(f"  {k}: {v}\n")

        f.write("\nPROBABLE CREATORS\n")
        if summary["probable_creators"]:
            for item in summary["probable_creators"]:
                f.write(f"  {item['score']:>3}  {item['file']}\n")
        else:
            f.write("  (none)\n")

        f.write("\nPROBABLE READERS\n")
        if summary["probable_readers"]:
            for item in summary["probable_readers"]:
                f.write(f"  {item['score']:>3}  {item['file']}\n")
        else:
            f.write("  (none)\n")

        f.write("\nCSV HEADERS\n")
        if summary["csv_headers"]:
            for item in summary["csv_headers"]:
                f.write(f"  {item['score']:>3}  {item['file']}\n")
        else:
            f.write("  (none)\n")

        f.write("\nDETAILED MATCHES\n")
        for r in records:
            f.write(
                f"[{r.context_type}] {r.file_path}:{r.line_no} "
                f"[{r.matched_token}] {r.line_text}\n"
            )


def write_csv_report(out_csv: Path, records: list[MatchRecord]) -> None:
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "line_no",
                "matched_token",
                "context_type",
                "line_text",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit where chosen_f0_* fields are created/read/mentioned."
    )
    parser.add_argument("--root", required=True, help="Project root")
    parser.add_argument("--out_txt", required=True, help="Human-readable report")
    parser.add_argument("--out_json", required=True, help="Machine-readable report")
    parser.add_argument("--out_csv", required=True, help="Detailed CSV report")
    args = parser.parse_args()

    root = Path(args.root)
    out_txt = Path(args.out_txt)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    records: list[MatchRecord] = []
    for path in iter_project_files(root):
        records.extend(scan_file(path))

    records.sort(key=lambda r: (r.file_path, r.line_no, r.matched_token))
    summary = summarize(records)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_txt_report(out_txt, root, records, summary)
    write_csv_report(out_csv, records)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "records": [asdict(r) for r in records],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("chosen_f0 field audit complete")
    print(json.dumps({
        "out_txt": str(out_txt),
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "total_matches": summary["total_matches"],
        "files_with_hits_count": summary["files_with_hits_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()