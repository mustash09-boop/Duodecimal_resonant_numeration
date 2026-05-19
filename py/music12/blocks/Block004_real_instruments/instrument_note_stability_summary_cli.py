from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")


# ============================================================
# HELPERS
# ============================================================

def _safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return int(s)
    except Exception:
        return default


def _read_passport_txt(path: Path) -> dict:
    """
    Reads simple key: value passport format.
    Supports one nested block:
      bias_histogram:
        exact_match: 12
        fragmented: 5
    """
    data: dict = {}
    current_nested_key = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith("  ") and current_nested_key:
            nested = line.strip()
            if ":" in nested:
                k, v = nested.split(":", 1)
                data[current_nested_key][k.strip()] = _parse_scalar(v.strip())
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            data[key] = {}
            current_nested_key = key
        else:
            data[key] = _parse_scalar(value)
            current_nested_key = None

    return data


def _parse_scalar(v: str):
    if v == "":
        return ""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"

    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def _extract_note_from_folder_name(folder_name: str) -> str:
    """
    Example:
      001__RealPiano_1__5.A-__structure_report
      RealPiano_1__9.A-__structure_report
    """
    m = re.search(r"__([1-9ABC]+\.[1-9ABC]+[^_]*)__structure_report$", folder_name, flags=re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).upper()


def _range_regime_from_note_token(note_token: str) -> str:
    """
    Simple regime by octave token, without fake radial geometry.
    """
    m = re.match(r"^([1-9ABC]+)\.", note_token, flags=re.IGNORECASE)
    if not m:
        return "unknown"

    octave_token = m.group(1).upper()

    low = {"1", "2", "3", "4", "5"}
    mid = {"6", "7", "8", "9"}
    # A/B/C and multi-digit octaves treated as high by default
    if octave_token in low:
        return "low"
    if octave_token in mid:
        return "mid"
    return "high"


# ============================================================
# LOAD REPORTS
# ============================================================

def find_structure_report_dirs(reports_root: Path) -> List[Path]:
    dirs = [p for p in reports_root.iterdir() if p.is_dir() and p.name.endswith("__structure_report")]
    dirs.sort(key=lambda p: p.name)
    return dirs


def load_report_dir(report_dir: Path) -> dict:
    passport_txt = report_dir / "passport.txt"
    meta_json = report_dir / "meta.json"

    if not passport_txt.exists():
        raise FileNotFoundError(f"Missing passport.txt: {passport_txt}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_json}")

    passport = _read_passport_txt(passport_txt)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))

    expected_note = _safe_str(passport.get("expected_note", "")) or _extract_note_from_folder_name(report_dir.name)
    dominant_note = _safe_str(passport.get("dominant_note", ""))
    chain_status = _safe_str(passport.get("chain_status", ""))
    bias_type = _safe_str(passport.get("bias_type", ""))

    return {
        "report_dir": str(report_dir),
        "report_name": report_dir.name,
        "expected_note": expected_note,
        "dominant_note": dominant_note,
        "chain_status": chain_status,
        "bias_type": bias_type,
        "range_regime": _range_regime_from_note_token(expected_note),

        "row_count": _safe_int(passport.get("row_count", 0), 0),
        "dominant_note_count": _safe_int(passport.get("dominant_note_count", 0), 0),
        "dominant_note_ratio": _safe_float(passport.get("dominant_note_ratio", 0.0), 0.0),

        "support_hits_mean": _safe_float(passport.get("support_hits_mean", 0.0), 0.0),
        "stabilization_score_mean": _safe_float(passport.get("stabilization_score_mean", 0.0), 0.0),
        "window_chain_match_score_mean": _safe_float(passport.get("window_chain_match_score_mean", 0.0), 0.0),
        "spiral_consistency_score_mean": _safe_float(passport.get("spiral_consistency_score_mean", 0.0), 0.0),

        "theoretical_chain_verdict_mode": _safe_str(passport.get("theoretical_chain_verdict_mode", "")),
        "stabilization_role_mode": _safe_str(passport.get("stabilization_role_mode", "")),
        "best_theoretical_chain_string_mode": _safe_str(passport.get("best_theoretical_chain_string_mode", "")),

        "first_exact_match_segment": _safe_str(passport.get("first_exact_match_segment", "")),

        "source_csv": _safe_str(meta.get("source_csv", "")),
        "structure_full_csv": _safe_str(meta.get("outputs", {}).get("structure_full_csv", "")),
        "passport_txt": _safe_str(meta.get("outputs", {}).get("passport_txt", "")),
        "meta_json": str(meta_json),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary_rows(report_items: List[dict]) -> List[dict]:
    rows: List[dict] = []

    for item in report_items:
        rows.append(
            {
                "expected_note": item["expected_note"],
                "dominant_note": item["dominant_note"],
                "chain_status": item["chain_status"],
                "bias_type": item["bias_type"],
                "range_regime": item["range_regime"],

                "row_count": item["row_count"],
                "dominant_note_count": item["dominant_note_count"],
                "dominant_note_ratio": round(item["dominant_note_ratio"], 6),

                "support_hits_mean": round(item["support_hits_mean"], 6),
                "stabilization_score_mean": round(item["stabilization_score_mean"], 6),
                "window_chain_match_score_mean": round(item["window_chain_match_score_mean"], 6),
                "spiral_consistency_score_mean": round(item["spiral_consistency_score_mean"], 6),

                "theoretical_chain_verdict_mode": item["theoretical_chain_verdict_mode"],
                "stabilization_role_mode": item["stabilization_role_mode"],
                "best_theoretical_chain_string_mode": item["best_theoretical_chain_string_mode"],

                "first_exact_match_segment": item["first_exact_match_segment"],

                "report_name": item["report_name"],
                "source_csv": item["source_csv"],
                "structure_full_csv": item["structure_full_csv"],
            }
        )

    rows.sort(key=lambda r: r["expected_note"])
    return rows


def build_text_summary(report_items: List[dict], *, instrument_name: str) -> str:
    total = len(report_items)

    chain_counter = Counter(item["chain_status"] for item in report_items)
    bias_counter = Counter(item["bias_type"] for item in report_items)
    regime_counter = Counter((item["range_regime"], item["chain_status"]) for item in report_items)

    exact_like = sum(1 for item in report_items if item["bias_type"] in {"exact_match", "same_core_micro_shift"})
    harmonic_bias = sum(1 for item in report_items if item["chain_status"] == "stable_harmonic_bias")
    fragmented = sum(1 for item in report_items if item["chain_status"] == "fragmented")

    mean_stability = 0.0
    if report_items:
        mean_stability = sum(item["stabilization_score_mean"] for item in report_items) / len(report_items)

    lines: List[str] = []
    lines.append("INSTRUMENT NOTE STABILITY SUMMARY")
    lines.append("=" * 80)
    lines.append(f"instrument_name: {instrument_name}")
    lines.append(f"notes_analyzed: {total}")
    lines.append(f"mean_stabilization_score: {mean_stability:.6f}")
    lines.append("")
    lines.append("CHAIN STATUS")
    for k, v in chain_counter.most_common():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("BIAS TYPE")
    for k, v in bias_counter.most_common():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("RANGE REGIME x CHAIN STATUS")
    for (regime, status), count in sorted(regime_counter.items()):
        lines.append(f"  {regime} | {status}: {count}")
    lines.append("")
    lines.append("KEY TOTALS")
    lines.append(f"  exact_or_micro_shift_like: {exact_like}")
    lines.append(f"  stable_harmonic_bias: {harmonic_bias}")
    lines.append(f"  fragmented: {fragmented}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  stable_fundamental            -> note structure is centered on expected token")
    lines.append("  stable_core_micro_shift       -> note core is stable, micro-layer drifts")
    lines.append("  stable_harmonic_bias          -> note repeatedly locks to harmonic neighbor")
    lines.append("  stable_structure_nonfundamental -> chain is stable but not centered on expected root")
    lines.append("  fragmented                    -> note has no dominant stable structure")
    lines.append("  mixed                         -> no single structural regime dominates")

    return "\n".join(lines) + "\n"


def build_meta(report_items: List[dict], *, instrument_name: str, reports_root: Path, outputs: dict) -> dict:
    return {
        "instrument_name": instrument_name,
        "reports_root": str(reports_root),
        "report_count": len(report_items),
        "outputs": outputs,
        "semantic_note": (
            "Block004 instrument-wide note stability summary. "
            "Aggregates per-note structure reports from 10_reports and summarizes "
            "how stable the expected note is, where harmonic bias appears, and where structure fragments."
        ),
    }


# ============================================================
# WRITE
# ============================================================

def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build instrument-wide summary from Block004 per-note structure reports. "
            "Reads reports from Block004_data/<instrument>/10_reports."
        )
    )
    ap.add_argument("--instrument_name", required=True, help="Example: RealPiano_1 or piano_midi")
    ap.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    instrument_name = _safe_str(args.instrument_name)

    reports_root = project_root / "Block004_data" / instrument_name / "10_reports"
    if not reports_root.exists():
        raise FileNotFoundError(f"Reports root does not exist: {reports_root}")

    report_dirs = find_structure_report_dirs(reports_root)
    if not report_dirs:
        raise ValueError(f"No *__structure_report directories found in {reports_root}")

    report_items = [load_report_dir(d) for d in report_dirs]
    summary_rows = build_summary_rows(report_items)

    out_csv = reports_root / f"{instrument_name}__note_stability_summary.csv"
    out_txt = reports_root / f"{instrument_name}__note_stability_summary.txt"
    out_json = reports_root / f"{instrument_name}__note_stability_summary.json"

    write_csv(out_csv, summary_rows)
    write_txt(out_txt, build_text_summary(report_items, instrument_name=instrument_name))
    write_json(
        out_json,
        build_meta(
            report_items,
            instrument_name=instrument_name,
            reports_root=reports_root,
            outputs={
                "summary_csv": str(out_csv),
                "summary_txt": str(out_txt),
                "summary_json": str(out_json),
            },
        ),
    )

    print("instrument note stability summary complete")
    print(json.dumps(
        {
            "instrument_name": instrument_name,
            "report_count": len(report_items),
            "out_csv": str(out_csv),
            "out_txt": str(out_txt),
            "out_json": str(out_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()