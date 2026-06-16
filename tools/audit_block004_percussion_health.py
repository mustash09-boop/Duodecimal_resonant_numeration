from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERC_ROOT = PROJECT_ROOT / "Block004_data" / "percussion"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"

MANIFEST_CSV = PERC_ROOT / "20_manifest" / "percussion_manifest_events.csv"
REPORTS10_DIR = PERC_ROOT / "10_reports"
SPIRAL50_DIR = PERC_ROOT / "50_spiral3d"
LINEAGE55_DIR = PERC_ROOT / "55_cluster_lineage_spiral3d"
PASSPORT40_DIR = PERC_ROOT / "40_passports"


TODAY = date.today().isoformat()
OUT_EVENTS = REPORTS_DIR / f"block004_percussion_audit_events_{TODAY}.csv"
OUT_INSTRUMENTS = REPORTS_DIR / f"block004_percussion_audit_instruments_{TODAY}.csv"
OUT_SUMMARY = REPORTS_DIR / f"block004_percussion_audit_summary_{TODAY}.txt"


REQUIRED_10_SUFFIXES = [
    "__percussion_dense.csv",
    "__percussion_event_summary.json",
    "__percussion_event_summary.txt",
    "__percussion_frequency_clusters.csv",
    "__percussion_spectrum.png",
    "__percussion_spiral.png",
]

REQUIRED_50_SUFFIXES = [
    "__percussion_spiral3d_points.csv",
    "__percussion_spiral3d.png",
    "__percussion_spiral3d.html",
]

REQUIRED_55_SUFFIXES = [
    "__percussion_cluster_lineage_points.csv",
    "__percussion_cluster_lineage.png",
    "__percussion_cluster_lineage.html",
]


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def to_bool_text(value: bool) -> str:
    return "1" if value else "0"


def check_required(base_dir: Path, stem: str, suffixes: list[str]) -> tuple[dict[str, bool], list[str]]:
    flags: dict[str, bool] = {}
    missing: list[str] = []
    for suffix in suffixes:
        path = base_dir / f"{stem}{suffix}"
        ok = path.exists()
        flags[suffix] = ok
        if not ok:
            missing.append(path.name)
    return flags, missing


def main() -> int:
    manifest_rows = load_manifest()
    by_instrument: dict[str, list[dict[str, str]]] = defaultdict(list)
    event_rows_out: list[dict[str, str]] = []

    for row in manifest_rows:
        instrument_name = row["instrument_name"]
        stem = Path(row["original_filename"]).stem
        by_instrument[instrument_name].append(row)

        report_dir = REPORTS10_DIR / stem
        has_report_dir = report_dir.exists()
        missing: list[str] = []

        flags10 = {suffix: False for suffix in REQUIRED_10_SUFFIXES}
        if has_report_dir:
            for suffix in REQUIRED_10_SUFFIXES:
                path = report_dir / f"{stem}{suffix}"
                ok = path.exists()
                flags10[suffix] = ok
                if not ok:
                    missing.append(path.name)
        else:
            missing.append(f"{stem}/")

        flags50, missing50 = check_required(SPIRAL50_DIR, stem, REQUIRED_50_SUFFIXES)
        flags55, missing55 = check_required(LINEAGE55_DIR, stem, REQUIRED_55_SUFFIXES)
        missing.extend(missing50)
        missing.extend(missing55)

        status = "OK" if not missing else "MISSING_ASSETS"

        out_row = {
            "instrument_name": instrument_name,
            "event_stem": stem,
            "event_id": row.get("event_id", ""),
            "dynamic": row.get("dynamic", ""),
            "articulation": row.get("articulation", ""),
            "gesture_type": row.get("gesture_type", ""),
            "has_report_dir": to_bool_text(has_report_dir),
            "has_10_dense_csv": to_bool_text(flags10["__percussion_dense.csv"]),
            "has_10_summary_json": to_bool_text(flags10["__percussion_event_summary.json"]),
            "has_10_summary_txt": to_bool_text(flags10["__percussion_event_summary.txt"]),
            "has_10_clusters_csv": to_bool_text(flags10["__percussion_frequency_clusters.csv"]),
            "has_10_spectrum_png": to_bool_text(flags10["__percussion_spectrum.png"]),
            "has_10_spiral_png": to_bool_text(flags10["__percussion_spiral.png"]),
            "has_50_points_csv": to_bool_text(flags50["__percussion_spiral3d_points.csv"]),
            "has_50_png": to_bool_text(flags50["__percussion_spiral3d.png"]),
            "has_50_html": to_bool_text(flags50["__percussion_spiral3d.html"]),
            "has_55_points_csv": to_bool_text(flags55["__percussion_cluster_lineage_points.csv"]),
            "has_55_png": to_bool_text(flags55["__percussion_cluster_lineage.png"]),
            "has_55_html": to_bool_text(flags55["__percussion_cluster_lineage.html"]),
            "status": status,
            "issues": "; ".join(missing),
        }
        event_rows_out.append(out_row)

    instrument_rows_out: list[dict[str, str]] = []
    instrument_status_counts: dict[str, int] = defaultdict(int)

    for instrument_name, rows in sorted(by_instrument.items()):
        passport_json = PASSPORT40_DIR / f"{instrument_name}__percussion_passport.json"
        passport_md = PASSPORT40_DIR / f"{instrument_name}__percussion_passport.md"
        has_passport_json = passport_json.exists()
        has_passport_md = passport_md.exists()

        passport_data = {}
        if has_passport_json:
            passport_data = json.loads(passport_json.read_text(encoding="utf-8"))

        summary = passport_data.get("summary") or {}
        events_list = passport_data.get("events") or []
        event_morphology = passport_data.get("event_morphology_compare")
        cluster_lineage = passport_data.get("cluster_lineage_spiral3d")

        manifest_event_count = len(rows)
        missing_event_assets_count = sum(
            1 for r in event_rows_out if r["instrument_name"] == instrument_name and r["status"] != "OK"
        )

        issues: list[str] = []
        if not has_passport_json:
            issues.append("missing passport json")
        if not has_passport_md:
            issues.append("missing passport md")
        if has_passport_json:
            if summary.get("event_count") != manifest_event_count:
                issues.append(
                    f"summary.event_count={summary.get('event_count')} != manifest={manifest_event_count}"
                )
            if len(events_list) != manifest_event_count:
                issues.append(f"events.len={len(events_list)} != manifest={manifest_event_count}")
            if not event_morphology:
                issues.append("missing event_morphology_compare")
            if not cluster_lineage:
                issues.append("missing cluster_lineage_spiral3d")
            if summary.get("event_morphology_event_count") not in (None, manifest_event_count):
                issues.append(
                    "event_morphology_event_count mismatch"
                )
            if summary.get("cluster_lineage_event_count") not in (None, manifest_event_count):
                issues.append(
                    "cluster_lineage_event_count mismatch"
                )
        if missing_event_assets_count:
            issues.append(f"missing event asset rows={missing_event_assets_count}")

        status = "OK" if not issues else "REVIEW"
        instrument_status_counts[status] += 1

        instrument_rows_out.append(
            {
                "instrument_name": instrument_name,
                "manifest_event_count": str(manifest_event_count),
                "has_passport_json": to_bool_text(has_passport_json),
                "has_passport_md": to_bool_text(has_passport_md),
                "summary_event_count": str(summary.get("event_count", "")),
                "passport_events_len": str(len(events_list) if has_passport_json else ""),
                "event_morphology_event_count": str(summary.get("event_morphology_event_count", "")),
                "cluster_lineage_event_count": str(summary.get("cluster_lineage_event_count", "")),
                "missing_event_assets_count": str(missing_event_assets_count),
                "status": status,
                "issues": "; ".join(issues),
            }
        )

    event_fieldnames = list(event_rows_out[0].keys()) if event_rows_out else []
    with OUT_EVENTS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=event_fieldnames)
        writer.writeheader()
        writer.writerows(sorted(event_rows_out, key=lambda r: (r["instrument_name"], r["event_stem"])))

    instrument_fieldnames = list(instrument_rows_out[0].keys()) if instrument_rows_out else []
    with OUT_INSTRUMENTS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=instrument_fieldnames)
        writer.writeheader()
        writer.writerows(instrument_rows_out)

    missing_event_rows = sum(1 for row in event_rows_out if row["status"] != "OK")
    review_instruments = sum(1 for row in instrument_rows_out if row["status"] != "OK")

    summary_lines = [
        f"Block004 percussion audit {TODAY}",
        "",
        f"manifest_event_rows = {len(manifest_rows)}",
        f"instrument_count = {len(by_instrument)}",
        f"event_rows_ok = {len(manifest_rows) - missing_event_rows}",
        f"event_rows_missing_assets = {missing_event_rows}",
        f"instruments_ok = {len(by_instrument) - review_instruments}",
        f"instruments_review = {review_instruments}",
        "",
        f"events_csv = {OUT_EVENTS}",
        f"instruments_csv = {OUT_INSTRUMENTS}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(OUT_SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
