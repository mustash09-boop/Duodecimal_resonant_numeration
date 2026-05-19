from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# HELPERS
# ============================================================

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_root_from_folder(folder_name: str) -> str:
    parts = folder_name.split("__")
    if len(parts) >= 3:
        return parts[2].strip()
    return folder_name.strip()


def get_note_token_from_row(row: dict[str, str]) -> str:
    for c in [
        "note_token",
        "best_theoretical_root_token",
        "representative_rc_note",
        "target_root_token",
    ]:
        v = (row.get(c, "") or "").strip()
        if v:
            return v
    return ""


# ============================================================
# SIGNATURE SELECTION
# ============================================================

def build_signature_tokens(
    global_csv: Path,
    per_band_csv: Path,
    *,
    min_global_count: int = 250,
    min_band_count: int = 2,
) -> tuple[list[dict], set[str]]:
    """
    Select instrument-signature response tokens.

    Rule:
    - token must have strong global count
    - token must appear in at least N coarse bands (low/mid/high)
    """
    global_rows = load_csv(global_csv)
    band_rows = load_csv(per_band_csv)

    global_count_by_note: dict[str, int] = {}
    for r in global_rows:
        note = (r.get("response_note", "") or "").strip()
        count = safe_int(r.get("count", 0))
        if note:
            global_count_by_note[note] = count

    band_presence: dict[str, set[str]] = defaultdict(set)
    for r in band_rows:
        band = (r.get("band", "") or "").strip()
        note = (r.get("response_note", "") or "").strip()
        if band and note:
            band_presence[note].add(band)

    selected_rows: list[dict] = []
    selected_tokens: set[str] = set()

    for note, count in sorted(global_count_by_note.items(), key=lambda kv: (-kv[1], kv[0])):
        bands = band_presence.get(note, set())
        if count >= min_global_count and len(bands) >= min_band_count:
            selected_tokens.add(note)
            selected_rows.append(
                {
                    "response_note": note,
                    "global_count": count,
                    "bands_detected": len(bands),
                    "bands": " ".join(sorted(bands)),
                }
            )

    return selected_rows, selected_tokens


# ============================================================
# FILTER REPORTS
# ============================================================

def filter_reports(
    input_reports_dir: Path,
    output_reports_dir: Path,
    signature_tokens: set[str],
) -> tuple[list[dict], dict[str, Counter]]:
    """
    Mirror folder structure and remove rows whose note token belongs
    to the instrument signature set.
    """
    manifest_rows: list[dict] = []
    dominant_remaining_by_root: dict[str, Counter] = defaultdict(Counter)

    folders = [p for p in input_reports_dir.iterdir() if p.is_dir()]

    for folder in sorted(folders):
        root = extract_root_from_folder(folder.name)
        out_folder = output_reports_dir / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        csv_files = list(folder.glob("*__stabilized__with_phase.csv"))

        # copy everything else untouched
        for item in folder.iterdir():
            if item.is_file() and not item.name.endswith("__stabilized__with_phase.csv"):
                shutil.copy2(item, out_folder / item.name)

        for csv_file in csv_files:
            rows = load_csv(csv_file)
            if not rows:
                out_path = out_folder / csv_file.name
                ensure_parent(out_path)
                with out_path.open("w", encoding="utf-8", newline="") as f:
                    f.write("")
                manifest_rows.append(
                    {
                        "root": root,
                        "source_csv": str(csv_file),
                        "output_csv": str(out_path),
                        "rows_before": 0,
                        "rows_after": 0,
                        "removed_rows": 0,
                    }
                )
                continue

            fieldnames = list(rows[0].keys())
            kept_rows: list[dict] = []
            removed = 0

            for row in rows:
                note = get_note_token_from_row(row)
                if note in signature_tokens:
                    removed += 1
                    continue

                kept_rows.append(row)

                # for quick post-analysis: what dominates after filtering
                if note:
                    dominant_remaining_by_root[root][note] += 1

            out_path = out_folder / csv_file.name
            write_csv(out_path, fieldnames, kept_rows)

            manifest_rows.append(
                {
                    "root": root,
                    "source_csv": str(csv_file),
                    "output_csv": str(out_path),
                    "rows_before": len(rows),
                    "rows_after": len(kept_rows),
                    "removed_rows": removed,
                }
            )

    return manifest_rows, dominant_remaining_by_root


# ============================================================
# WRITE DERIVED REPORTS
# ============================================================

def write_dominant_remaining(path: Path, dominant_remaining_by_root: dict[str, Counter]) -> None:
    rows: list[dict] = []
    for root in sorted(dominant_remaining_by_root.keys()):
        for note, count in dominant_remaining_by_root[root].most_common():
            rows.append(
                {
                    "root": root,
                    "remaining_note": note,
                    "count": count,
                }
            )
    write_csv(path, ["root", "remaining_note", "count"], rows)


def write_txt_report(
    path: Path,
    *,
    signature_rows: list[dict],
    manifest_rows: list[dict],
    dominant_remaining_by_root: dict[str, Counter],
) -> None:
    ensure_parent(path)

    total_before = sum(safe_int(r["rows_before"]) for r in manifest_rows)
    total_after = sum(safe_int(r["rows_after"]) for r in manifest_rows)
    total_removed = sum(safe_int(r["removed_rows"]) for r in manifest_rows)

    with path.open("w", encoding="utf-8") as f:
        f.write("FILTER REPORTS BY INSTRUMENT SIGNATURE\n")
        f.write("=" * 80 + "\n")
        f.write(f"signature_token_count: {len(signature_rows)}\n")
        f.write(f"csv_file_count: {len(manifest_rows)}\n")
        f.write(f"rows_before_total: {total_before}\n")
        f.write(f"rows_after_total:  {total_after}\n")
        f.write(f"rows_removed_total:{total_removed}\n\n")

        f.write("SELECTED INSTRUMENT SIGNATURE TOKENS\n")
        for r in signature_rows:
            f.write(
                f"  {r['response_note']} | "
                f"global_count={r['global_count']} | "
                f"bands={r['bands']}\n"
            )

        f.write("\nTOP REMAINING TOKENS PER ROOT\n")
        for root in sorted(dominant_remaining_by_root.keys()):
            f.write(f"\n[{root}]\n")
            for note, count in dominant_remaining_by_root[root].most_common(10):
                f.write(f"  {note}: {count}\n")

        f.write("\nINTERPRETATION\n")
        f.write("  - these filtered reports remove tokens that behave like instrument signature.\n")
        f.write("  - next step is to inspect which chains become dominant after removal.\n")
        f.write("  - if chains clarify strongly, hypothesis A gets support.\n")
        f.write("  - if they still look mixed, hypothesis B (PSM representation issue) gets stronger.\n")


def write_meta_json(
    path: Path,
    *,
    input_reports_dir: Path,
    output_reports_dir: Path,
    global_csv: Path,
    per_band_csv: Path,
    min_global_count: int,
    min_band_count: int,
    signature_rows: list[dict],
    manifest_rows: list[dict],
) -> None:
    ensure_parent(path)
    data = {
        "inputs": {
            "input_reports_dir": str(input_reports_dir),
            "global_csv": str(global_csv),
            "per_band_csv": str(per_band_csv),
        },
        "outputs": {
            "output_reports_dir": str(output_reports_dir),
        },
        "params": {
            "min_global_count": min_global_count,
            "min_band_count": min_band_count,
        },
        "summary": {
            "signature_token_count": len(signature_rows),
            "csv_file_count": len(manifest_rows),
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Filter stabilized_with_phase reports by instrument-signature spiral tokens."
    )
    ap.add_argument("--input_reports_dir", required=True)
    ap.add_argument("--global_csv", required=True)
    ap.add_argument("--per_band_csv", required=True)
    ap.add_argument("--output_reports_dir", required=True)
    ap.add_argument("--out_signature_csv", required=True)
    ap.add_argument("--out_manifest_csv", required=True)
    ap.add_argument("--out_dominant_remaining_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--min_global_count", type=int, default=250)
    ap.add_argument("--min_band_count", type=int, default=2)
    args = ap.parse_args()

    input_reports_dir = Path(args.input_reports_dir).resolve()
    global_csv = Path(args.global_csv).resolve()
    per_band_csv = Path(args.per_band_csv).resolve()
    output_reports_dir = Path(args.output_reports_dir).resolve()

    signature_rows, signature_tokens = build_signature_tokens(
        global_csv,
        per_band_csv,
        min_global_count=args.min_global_count,
        min_band_count=args.min_band_count,
    )

    manifest_rows, dominant_remaining_by_root = filter_reports(
        input_reports_dir=input_reports_dir,
        output_reports_dir=output_reports_dir,
        signature_tokens=signature_tokens,
    )

    write_csv(
        Path(args.out_signature_csv),
        ["response_note", "global_count", "bands_detected", "bands"],
        signature_rows,
    )
    write_csv(
        Path(args.out_manifest_csv),
        ["root", "source_csv", "output_csv", "rows_before", "rows_after", "removed_rows"],
        manifest_rows,
    )
    write_dominant_remaining(Path(args.out_dominant_remaining_csv), dominant_remaining_by_root)
    write_txt_report(
        Path(args.out_txt),
        signature_rows=signature_rows,
        manifest_rows=manifest_rows,
        dominant_remaining_by_root=dominant_remaining_by_root,
    )
    write_meta_json(
        Path(args.out_meta_json),
        input_reports_dir=input_reports_dir,
        output_reports_dir=output_reports_dir,
        global_csv=global_csv,
        per_band_csv=per_band_csv,
        min_global_count=args.min_global_count,
        min_band_count=args.min_band_count,
        signature_rows=signature_rows,
        manifest_rows=manifest_rows,
    )

    print("filter by instrument signature complete")
    print(f"signature_token_count={len(signature_rows)}")
    print(f"csv_file_count={len(manifest_rows)}")


if __name__ == "__main__":
    main()