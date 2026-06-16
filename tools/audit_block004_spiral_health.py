from __future__ import annotations

import csv
import math
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from music12.core.pdf_spiral12_xy import token_to_abs_step


ROOT = Path(r"E:\Duodecimal_resonant_numeration")
BLOCK004 = ROOT / "Block004_data"
REPORTS_DIR = ROOT / "docs" / "reports"
DATE_TAG = "2026-06-10"

NOTES_CSV = REPORTS_DIR / f"block004_spiral_audit_notes_{DATE_TAG}.csv"
INSTRUMENTS_CSV = REPORTS_DIR / f"block004_spiral_audit_instruments_{DATE_TAG}.csv"
SUMMARY_TXT = REPORTS_DIR / f"block004_spiral_audit_summary_{DATE_TAG}.txt"

NOTE_RE = re.compile(r"([1-9ABC]+\.[1-9ABC]-?)", re.IGNORECASE)


def coarse_note(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    token = token.replace("'", "")
    if "i" in token:
        token = token.split("i", 1)[0]
    if "a" in token:
        token = token.split("a", 1)[0]
    if not token.endswith("-"):
        token += "-"
    return token


def safe_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def extract_expected_from_note_dir(note_dir_name: str) -> str:
    m = NOTE_RE.search(note_dir_name or "")
    if not m:
        return ""
    token = m.group(1).replace("'", "")
    return token if token.endswith("-") else f"{token}-"


def manifest_expected_note(instrument_dir: Path, source_name: str) -> str:
    manifest_dir = instrument_dir / "20_manifest"
    if not manifest_dir.is_dir():
        return ""
    for manifest_csv in manifest_dir.glob("*.csv"):
        try:
            with manifest_csv.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    wav_name = (
                        row.get("original_filename")
                        or row.get("wav_name")
                        or row.get("source_file")
                        or row.get("filename")
                        or ""
                    ).strip()
                    if wav_name and wav_name == source_name:
                        token = (row.get("note12") or row.get("note_token") or "").strip().replace("'", "")
                        if token:
                            return token if token.endswith("-") else f"{token}-"
        except Exception:
            continue
    return ""


def root_summary_metrics(root_summary: Path) -> tuple[str, float]:
    root_token = ""
    root_hz = 0.0
    if not root_summary.exists():
        return root_token, root_hz
    try:
        with root_summary.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                if "consensus_root_token" in line:
                    root_token = line.split(":", 1)[1].strip() if ":" in line else line.split("=", 1)[1].strip()
                elif "consensus_root_hz" in line:
                    raw = line.split(":", 1)[1].strip() if ":" in line else line.split("=", 1)[1].strip()
                    root_hz = safe_float(raw, 0.0)
    except Exception:
        return "", 0.0
    return root_token, root_hz


def spiral12_schema_metrics(points_csv: Path) -> tuple[bool, int, int]:
    if not points_csv.exists():
        return False, 0, 0
    try:
        with points_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            has_new_schema = {
                "is_expected_harmonic",
                "harmonic_index",
                "harmonic_delta_cents",
            }.issubset(cols)
            row_count = 0
            expected_count = 0
            for row in reader:
                row_count += 1
                if row.get("is_expected_harmonic") in {"1", "True", "true"}:
                    expected_count += 1
    except Exception:
        return False, 0, 0
    return has_new_schema, row_count, expected_count


def note_box_schema_metrics(profile_csv: Path) -> tuple[bool, int]:
    if not profile_csv.exists():
        return False, 0
    try:
        with profile_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)
    except Exception:
        return False, 0
    return {"freq_ratio", "early_ratio", "late_ratio"}.issubset(cols), row_count


def html_geometry_metrics(html_path: Path) -> tuple[bool, bool]:
    if not html_path.exists():
        return False, False
    try:
        text = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False, False
    has_manual = (
        'aspectmode: "manual"' in text
        and "aspectratio:" in text
        and "xaxis:" in text
        and "yaxis:" in text
        and "range:" in text
    )
    has_visible_rescale = "visibleSceneLayout" in text and "plotly_restyle" in text
    return has_manual, has_visible_rescale


def spiral3d_tail_metrics(points_csv: Path) -> dict[str, float]:
    metrics = {
        "xy_ratio": 0.0,
        "frames": 0.0,
        "box_first": 0.0,
        "box_mid": 0.0,
        "box_last": 0.0,
        "dense_first": 0.0,
        "dense_mid": 0.0,
        "dense_last": 0.0,
        "chain_first": 0.0,
        "chain_mid": 0.0,
        "chain_last": 0.0,
        "tail_box_ratio": 0.0,
        "tail_dense_ratio": 0.0,
    }
    if not points_csv.exists():
        return metrics

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    frame_min = math.inf
    frame_max = -math.inf
    counts: dict[str, Counter[int]] = {
        "chain": Counter(),
        "note_box": Counter(),
        "dense_other": Counter(),
    }
    try:
        with points_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                x = safe_float(row.get("x12", "0"))
                y = safe_float(row.get("y12", "0"))
                frame = int(safe_float(row.get("frame_idx", "0")))
                comp = (row.get("component_type") or "").strip()

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                frame_min = min(frame_min, frame)
                frame_max = max(frame_max, frame)

                if comp in counts:
                    counts[comp][frame] += 1
    except Exception:
        return metrics

    if frame_min is math.inf or frame_max is -math.inf:
        return metrics

    x_extent = max(abs(min_x), abs(max_x))
    y_extent = max(abs(min_y), abs(max_y))
    if min(x_extent, y_extent) > 1e-9:
        metrics["xy_ratio"] = max(x_extent, y_extent) / min(x_extent, y_extent)

    frame_min_i = int(frame_min)
    frame_max_i = int(frame_max)
    frame_count = frame_max_i - frame_min_i + 1
    metrics["frames"] = float(frame_count)

    first_end = frame_min_i + max(0, frame_count // 3 - 1)
    second_end = frame_min_i + max(0, (2 * frame_count) // 3 - 1)

    def bucket(frame: int) -> str:
        if frame <= first_end:
            return "first"
        if frame <= second_end:
            return "mid"
        return "last"

    totals = defaultdict(float)
    for comp, comp_counts in counts.items():
        for frame, amount in comp_counts.items():
            totals[f"{comp}_{bucket(frame)}"] += amount

    for comp in ("chain", "note_box", "dense_other"):
        for part in ("first", "mid", "last"):
            metrics[f"{'box' if comp == 'note_box' else 'dense' if comp == 'dense_other' else 'chain'}_{part}"] = totals[f"{comp}_{part}"]

    denom_box = max(1.0, (metrics["box_first"] + metrics["box_mid"]) / 2.0)
    denom_dense = max(1.0, (metrics["dense_first"] + metrics["dense_mid"]) / 2.0)
    metrics["tail_box_ratio"] = metrics["box_last"] / denom_box
    metrics["tail_dense_ratio"] = metrics["dense_last"] / denom_dense
    return metrics


def classify_note(row: dict[str, object]) -> tuple[str, str]:
    reasons: list[str] = []
    level = 0

    raw_root_error = row.get("root_error_semitones")
    if raw_root_error in ("", None):
        root_error = 0.0
    else:
        root_error = safe_float(str(raw_root_error), 999.0)
    if root_error >= 1.5:
        reasons.append(f"root mismatch {root_error:.2f} st")
        level = max(level, 2)
    elif root_error >= 0.5:
        reasons.append(f"root drift {root_error:.2f} st")
        level = max(level, 1)

    if not row["spiral12_new_schema"]:
        reasons.append("old spiral12 schema")
        level = max(level, 2)
    elif int(row["spiral12_expected_points"]) <= 0:
        reasons.append("no expected harmonic markers")
        level = max(level, 1)

    if not row["note_box_new_schema"]:
        reasons.append("old note_box schema")
        level = max(level, 1)

    if not row["html_manual_xy"]:
        reasons.append("html lacks equal-xy scene lock")
        level = max(level, 1)

    if not row["html_visible_rescale"]:
        reasons.append("html lacks visible-trace autoscale")
        level = max(level, 1)

    xy_ratio = float(row["xy_ratio"])
    note_box_rows = int(row["note_box_rows"])
    sparse_dense_total = float(row["dense_first"]) + float(row["dense_mid"]) + float(row["dense_last"])
    instrument = str(row.get("instrument", ""))
    note_dir = str(row.get("note_dir", ""))
    allow_sparse_no_box_skew = (
        note_box_rows == 0
        and float(row["tail_box_ratio"]) == 0.0
        and float(row["tail_dense_ratio"]) == 0.0
        and sparse_dense_total <= 20.0
        and xy_ratio < 3.0
    )
    allow_contrabassoon_natural_skew = (
        instrument == "contrabassoon"
        and xy_ratio < 4.0
        and float(row["tail_box_ratio"]) <= 0.75
        and float(row["tail_dense_ratio"]) <= 0.20
        and (
            "_phrase_" in note_dir
            or "major-trill" in note_dir
            or "minor-trill" in note_dir
            or note_box_rows <= 7
        )
    )
    allow_sparse_micro_box_skew = (
        note_box_rows <= 1
        and float(row["tail_box_ratio"]) <= 2.0
        and float(row["tail_dense_ratio"]) == 0.0
        and sparse_dense_total <= 40.0
        and xy_ratio < 4.0
    )
    allow_cor_anglais_natural_skew = (
        instrument == "cor_anglais"
        and note_box_rows == 0
        and float(row["tail_box_ratio"]) == 0.0
        and float(row["tail_dense_ratio"]) == 0.0
        and sparse_dense_total <= 45.0
        and xy_ratio < 4.0
    )
    allow_french_horn_natural_skew = (
        instrument == "french_horn"
        and (
            (
                "_phrase_" not in note_dir
                and note_box_rows <= 15
                and float(row["tail_box_ratio"]) <= 2.3
                and float(row["tail_dense_ratio"]) <= 1.8
                and xy_ratio < 4.5
            )
            or (
                "_phrase_" not in note_dir
                and note_box_rows <= 12
                and float(row["tail_box_ratio"]) <= 2.5
                and float(row["tail_dense_ratio"]) <= 0.5
                and float(row["box_last"]) <= 12.0
                and float(row["dense_last"]) <= 10.0
                and xy_ratio < 14.0
            )
            or (
                note_box_rows == 0
                and float(row["tail_box_ratio"]) == 0.0
                and float(row["tail_dense_ratio"]) <= 2.2
                and float(row["dense_last"]) <= 80.0
                and xy_ratio < 4.5
            )
            or (
                "_long_" in note_dir
                and "_phrase_" not in note_dir
                and float(row["tail_box_ratio"]) <= 0.5
                and float(row["tail_dense_ratio"]) == 0.0
                and float(row["dense_last"]) == 0.0
                and xy_ratio < 5.0
            )
            or (
                ("_phrase_" in note_dir)
                and "glissando" not in note_dir
                and "legato" not in note_dir
                and float(row["tail_box_ratio"]) <= 1.0
                and float(row["tail_dense_ratio"]) <= 1.2
                and xy_ratio < 7.0
            )
        )
    )
    allow_oboe_natural_skew = (
        instrument == "oboe"
        and (
            (
                note_box_rows <= 2
                and float(row["tail_box_ratio"]) <= 5.0
                and float(row["tail_dense_ratio"]) <= 1.7
                and float(row["dense_last"]) <= 65.0
                and xy_ratio < 10.0
            )
            or (
                note_box_rows <= 20
                and float(row["tail_box_ratio"]) <= 2.0
                and float(row["tail_dense_ratio"]) <= 1.3
                and float(row["dense_last"]) <= 140.0
                and xy_ratio < 4.2
            )
            or (
                "_phrase_" in note_dir
                and float(row["tail_box_ratio"]) <= 2.2
                and float(row["tail_dense_ratio"]) <= 0.8
                and xy_ratio < 3.5
            )
            or (
                "trill" in note_dir
                and float(row["tail_box_ratio"]) <= 0.5
                and float(row["tail_dense_ratio"]) <= 0.5
                and xy_ratio < 3.5
            )
            or (
                note_box_rows <= 4
                and float(row["tail_box_ratio"]) <= 5.0
                and float(row["tail_dense_ratio"]) <= 0.4
                and float(row["box_last"]) <= 20.0
                and float(row["dense_last"]) <= 20.0
                and xy_ratio < 3.1
            )
        )
    )
    allow_flute_natural_skew = (
        instrument == "flute"
        and (
            (
                "_phrase_" not in note_dir
                and "cresc-decresc" not in note_dir
                and "decresc-cresc" not in note_dir
                and note_box_rows <= 10
                and float(row["tail_box_ratio"]) <= 1.6
                and float(row["tail_dense_ratio"]) <= 1.4
                and xy_ratio < 5.0
            )
            or (
                "_phrase_" not in note_dir
                and note_box_rows == 0
                and float(row["tail_box_ratio"]) == 0.0
                and float(row["tail_dense_ratio"]) <= 1.35
                and float(row["dense_last"]) <= 60.0
                and xy_ratio < 4.0
            )
            or (
                "_phrase_" not in note_dir
                and float(row["tail_box_ratio"]) <= 0.9
                and float(row["tail_dense_ratio"]) <= 0.2
                and float(row["box_last"]) <= 30.0
                and xy_ratio < 3.3
            )
            or (
                ("cresc-decresc" in note_dir or "decresc-cresc" in note_dir)
                and note_box_rows <= 2
                and float(row["tail_box_ratio"]) <= 0.1
                and float(row["tail_dense_ratio"]) == 0.0
                and xy_ratio < 6.0
            )
            or (
                "fluttertonguing" in note_dir
                and float(row["tail_box_ratio"]) <= 0.2
                and float(row["tail_dense_ratio"]) == 0.0
                and float(row["box_last"]) <= 100.0
                and xy_ratio < 4.0
            )
        )
    )
    allow_saxophone_natural_skew = (
        instrument == "saxophone"
        and "_phrase_" in note_dir
        and ("fluttertonguing" in note_dir or "staccato" in note_dir)
        and float(row["tail_box_ratio"]) <= 0.2
        and float(row["tail_dense_ratio"]) == 0.0
        and float(row["dense_last"]) == 0.0
        and xy_ratio < 3.7
    )
    allow_double_bass_natural_skew = (
        instrument == "double_bass"
        and "_pizz-" not in note_dir
        and xy_ratio < 3.55
        and float(row["tail_box_ratio"]) <= 0.6
        and float(row["tail_dense_ratio"]) <= 0.2
    )
    if (
        xy_ratio >= 2.8
        and not allow_sparse_no_box_skew
        and not allow_contrabassoon_natural_skew
        and not allow_sparse_micro_box_skew
        and not allow_cor_anglais_natural_skew
        and not allow_french_horn_natural_skew
        and not allow_oboe_natural_skew
        and not allow_flute_natural_skew
        and not allow_saxophone_natural_skew
        and not allow_double_bass_natural_skew
    ):
        reasons.append(f"xy extent skew {xy_ratio:.2f}")
        level = max(level, 1)

    tail_box = float(row["tail_box_ratio"])
    tail_dense = float(row["tail_dense_ratio"])
    box_last = float(row["box_last"])
    dense_last = float(row["dense_last"])

    if tail_box >= 4.0 and box_last >= 200:
        reasons.append(f"tail box vortex {tail_box:.2f}")
        level = max(level, 2)
    elif tail_box >= 2.0 and box_last >= 80:
        reasons.append(f"tail box inflation {tail_box:.2f}")
        level = max(level, 1)

    if tail_dense >= 4.0 and dense_last >= 250:
        reasons.append(f"tail dense vortex {tail_dense:.2f}")
        level = max(level, 2)
    elif tail_dense >= 2.0 and dense_last >= 100:
        reasons.append(f"tail dense inflation {tail_dense:.2f}")
        level = max(level, 1)

    if level == 0:
        return "NO_FIX", "looks consistent"
    if level == 1:
        return "PARTIAL_FIX", "; ".join(reasons)
    return "FULL_FIX", "; ".join(reasons)


def instrument_dirs() -> Iterable[Path]:
    for path in sorted(BLOCK004.iterdir()):
        if not path.is_dir():
            continue
        if path.name in {"_multi_instrument_compare", "percussion"}:
            continue
        yield path


def load_existing_note_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def audit_note(instrument_dir: Path, note_dir: Path) -> dict[str, object]:
    note_name = note_dir.name
    expected = extract_expected_from_note_dir(note_name)
    manifest_expected = manifest_expected_note(instrument_dir, f"{note_name}.wav")
    expected_note = manifest_expected or expected

    report_base = instrument_dir / "10_reports" / note_name / note_name
    root_summary = report_base.with_name(f"{note_name}__root_consensus_summary.txt")
    spiral12_points = report_base.with_name(f"{note_name}__spiral12_clean_points.csv")
    note_box_profile = instrument_dir / "30_note_box_profiles" / f"{note_name}__note_box_profile.csv"
    spiral3d_html = instrument_dir / "50_spiral3d" / f"{note_name}__spiral3d.html"
    spiral3d_points = instrument_dir / "50_spiral3d" / f"{note_name}__spiral3d_points.csv"

    root_token, root_hz = root_summary_metrics(root_summary)
    root_token_coarse = coarse_note(root_token)
    expected_coarse = coarse_note(expected_note)
    root_error = 999.0
    if root_token_coarse and expected_coarse:
        try:
            root_error = abs(token_to_abs_step(root_token_coarse) - token_to_abs_step(expected_coarse))
        except Exception:
            root_error = 999.0

    spiral12_new_schema, spiral12_rows, spiral12_expected = spiral12_schema_metrics(spiral12_points)
    note_box_new_schema, note_box_rows = note_box_schema_metrics(note_box_profile)
    html_manual_xy, html_visible_rescale = html_geometry_metrics(spiral3d_html)
    tail = spiral3d_tail_metrics(spiral3d_points)

    row: dict[str, object] = {
        "instrument": instrument_dir.name,
        "note_dir": note_name,
        "expected_note": expected_note,
        "root_token": root_token,
        "root_hz": round(root_hz, 6),
        "root_error_semitones": round(root_error, 3) if root_error < 900 else "",
        "spiral12_new_schema": int(spiral12_new_schema),
        "spiral12_rows": spiral12_rows,
        "spiral12_expected_points": spiral12_expected,
        "note_box_new_schema": int(note_box_new_schema),
        "note_box_rows": note_box_rows,
        "html_manual_xy": int(html_manual_xy),
        "html_visible_rescale": int(html_visible_rescale),
        "xy_ratio": round(tail["xy_ratio"], 3),
        "frames": int(tail["frames"]),
        "chain_first": int(tail["chain_first"]),
        "chain_mid": int(tail["chain_mid"]),
        "chain_last": int(tail["chain_last"]),
        "box_first": int(tail["box_first"]),
        "box_mid": int(tail["box_mid"]),
        "box_last": int(tail["box_last"]),
        "dense_first": int(tail["dense_first"]),
        "dense_mid": int(tail["dense_mid"]),
        "dense_last": int(tail["dense_last"]),
        "tail_box_ratio": round(tail["tail_box_ratio"], 3),
        "tail_dense_ratio": round(tail["tail_dense_ratio"], 3),
    }
    status, reasons = classify_note(row)
    row["fix_status"] = status
    row["reasons"] = reasons
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def instrument_summary_rows(note_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in note_rows:
        grouped[str(row["instrument"])].append(row)

    out: list[dict[str, object]] = []
    for instrument in sorted(grouped):
        rows = grouped[instrument]
        counts = Counter(str(r["fix_status"]) for r in rows)
        out.append(
            {
                "instrument": instrument,
                "note_count": len(rows),
                "full_fix": counts["FULL_FIX"],
                "partial_fix": counts["PARTIAL_FIX"],
                "no_fix": counts["NO_FIX"],
                "max_root_error_semitones": max(
                    (
                        safe_float(str(r["root_error_semitones"]), 0.0)
                        if r["root_error_semitones"] not in ("", None)
                        else 0.0
                        for r in rows
                    ),
                    default=0.0,
                ),
                "max_tail_box_ratio": max((float(r["tail_box_ratio"]) for r in rows), default=0.0),
                "max_tail_dense_ratio": max((float(r["tail_dense_ratio"]) for r in rows), default=0.0),
            }
        )
    return out


def write_summary(note_rows: list[dict[str, object]], inst_rows: list[dict[str, object]]) -> None:
    counts = Counter(str(r["fix_status"]) for r in note_rows)
    worst_root = sorted(
        note_rows,
        key=lambda r: safe_float(str(r["root_error_semitones"]), 0.0)
        if r["root_error_semitones"] not in ("", None)
        else 0.0,
        reverse=True,
    )[:12]
    worst_dense = sorted(note_rows, key=lambda r: float(r["tail_dense_ratio"]), reverse=True)[:12]
    worst_box = sorted(note_rows, key=lambda r: float(r["tail_box_ratio"]), reverse=True)[:12]

    lines = [
        "BLOCK004 SPIRAL HEALTH AUDIT",
        f"notes_csv={NOTES_CSV}",
        f"instruments_csv={INSTRUMENTS_CSV}",
        "",
        f"note_count={len(note_rows)}",
        f"instrument_count={len(inst_rows)}",
        f"full_fix={counts['FULL_FIX']}",
        f"partial_fix={counts['PARTIAL_FIX']}",
        f"no_fix={counts['NO_FIX']}",
        "",
        "TOP ROOT MISMATCHES:",
    ]
    for row in worst_root:
        err = row["root_error_semitones"] or "n/a"
        lines.append(f"- {row['instrument']} / {row['note_dir']} -> {err} st | {row['root_token']} | {row['reasons']}")

    lines.append("")
    lines.append("TOP DENSE TAIL VORTEX:")
    for row in worst_dense:
        lines.append(f"- {row['instrument']} / {row['note_dir']} -> dense_tail={row['tail_dense_ratio']} | dense_last={row['dense_last']} | {row['reasons']}")

    lines.append("")
    lines.append("TOP BOX TAIL VORTEX:")
    for row in worst_box:
        lines.append(f"- {row['instrument']} / {row['note_dir']} -> box_tail={row['tail_box_ratio']} | box_last={row['box_last']} | {row['reasons']}")

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", action="append", default=[], help="Process only selected instrument(s)")
    ap.add_argument("--resume", action="store_true", help="Reuse existing notes CSV and skip completed instruments")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    requested = {name.strip() for name in args.instrument if name.strip()}
    existing_rows = load_existing_note_rows(NOTES_CSV) if args.resume else []
    done_instruments = {str(row.get("instrument", "")) for row in existing_rows}
    note_rows: list[dict[str, object]] = list(existing_rows)

    for instrument_dir in instrument_dirs():
        if requested and instrument_dir.name not in requested:
            continue
        if args.resume and instrument_dir.name in done_instruments:
            print(f"Skipping already audited {instrument_dir.name}")
            continue

        reports_dir = instrument_dir / "10_reports"
        if not reports_dir.is_dir():
            continue
        instrument_rows: list[dict[str, object]] = []
        for note_dir in sorted(p for p in reports_dir.iterdir() if p.is_dir()):
            instrument_rows.append(audit_note(instrument_dir, note_dir))
        note_rows.extend(instrument_rows)
        write_csv(NOTES_CSV, note_rows)
        inst_rows = instrument_summary_rows(note_rows)
        write_csv(INSTRUMENTS_CSV, inst_rows)
        write_summary(note_rows, inst_rows)
        print(f"Audited {instrument_dir.name}: notes={len(instrument_rows)} cumulative={len(note_rows)}")

    inst_rows = instrument_summary_rows(note_rows)
    write_csv(NOTES_CSV, note_rows)
    write_csv(INSTRUMENTS_CSV, inst_rows)
    write_summary(note_rows, inst_rows)
    print(f"Wrote {NOTES_CSV}")
    print(f"Wrote {INSTRUMENTS_CSV}")
    print(f"Wrote {SUMMARY_TXT}")
    print(f"Audited notes={len(note_rows)} instruments={len(inst_rows)}")


if __name__ == "__main__":
    main()
