from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================
# PATH RULES
# ============================================================

DEFAULT_PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
DEFAULT_BLOCK004_ROOT = DEFAULT_PROJECT_ROOT / "Block004_data"


# ============================================================
# NOTE MODEL
# ============================================================

@dataclass(frozen=True)
class NoteToken:
    token: str

    @property
    def core(self) -> str:
        m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", self.token, flags=re.IGNORECASE)
        if not m:
            return self.token
        return f"{m.group(1).upper()}.{m.group(2).upper()}"

    @property
    def octave(self) -> str:
        m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", self.token, flags=re.IGNORECASE)
        if not m:
            return ""
        return m.group(1).upper()

    @property
    def degree(self) -> str:
        m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", self.token, flags=re.IGNORECASE)
        if not m:
            return ""
        return m.group(2).upper()


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


def _mode(values: list[str]) -> str:
    cleaned = [v for v in (_safe_str(x) for x in values) if v]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_expected_note_from_filename(path: Path) -> str:
    """
    Expected examples:
      001__RealPiano_1__5.A-__chain_stabilized.csv
      049__RealPiano_1__9.A-__chain_stabilized.csv
      087__RealPiano_1__C.C-__chain_stabilized.csv

    Fallback:
      try to read any __<token>__*.csv pattern.
    """
    name = path.name

    patterns = [
        r"__([1-9ABC]+\.[1-9ABC]+[^_]*)__chain_stabilized\.csv$",
        r"__([1-9ABC]+\.[1-9ABC]+[^_]*)__stabilized\.csv$",
        r"__([1-9ABC]+\.[1-9ABC]+[^_]*)__.*\.csv$",
    ]

    for pattern in patterns:
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    raise ValueError(f"Cannot parse expected note token from filename: {name}")


def _bias_type(expected: str, detected: str) -> str:
    """
    Conservative structural bias classifier.

    It avoids fake phase/radial geometry and uses only token-core relation.
    """
    if not expected or not detected:
        return "no_decision"

    exp = NoteToken(expected)
    det = NoteToken(detected)

    if exp.token == det.token:
        return "exact_match"

    if exp.core == det.core:
        return "same_core_micro_shift"

    if exp.degree == det.degree and exp.octave != det.octave:
        return "same_degree_other_octave"

    if exp.octave == det.octave and exp.degree != det.degree:
        return "same_octave_other_degree"

    return "different_core"


def _classify_chain_status(
    *,
    dominant_note: str,
    expected_note: str,
    chain_verdict_mode: str,
    support_hits_mean: float,
    stabilization_score_mean: float,
    window_chain_match_score_mean: float,
    dominant_note_ratio: float,
) -> str:
    bias = _bias_type(expected_note, dominant_note)

    if (
        bias == "exact_match"
        and dominant_note_ratio >= 0.45
        and stabilization_score_mean >= 0.35
        and window_chain_match_score_mean >= 0.25
    ):
        return "stable_fundamental"

    if (
        bias == "same_core_micro_shift"
        and dominant_note_ratio >= 0.35
        and stabilization_score_mean >= 0.30
    ):
        return "stable_core_micro_shift"

    if (
        bias == "same_degree_other_octave"
        and dominant_note_ratio >= 0.35
        and support_hits_mean >= 2.0
    ):
        return "stable_harmonic_bias"

    if (
        chain_verdict_mode in {"CHAIN_CONFIRMED", "CHAIN_PHASE_CONFIRMED"}
        and dominant_note_ratio >= 0.25
    ):
        return "stable_structure_nonfundamental"

    if dominant_note_ratio < 0.15:
        return "fragmented"

    return "mixed"


# ============================================================
# LOAD
# ============================================================

def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# REPORT BUILD
# ============================================================

def build_per_note_rows(
    rows: list[dict],
    *,
    expected_note: str,
) -> list[dict]:
    out: list[dict] = []

    for r in rows:
        representative_rc_note = _safe_str(r.get("representative_rc_note", ""))
        best_theoretical_root_token = _safe_str(r.get("best_theoretical_root_token", ""))

        chosen_note = best_theoretical_root_token or representative_rc_note
        if not chosen_note:
            continue

        bias_type = _bias_type(expected_note, chosen_note)

        out.append(
            {
                "segment_index": _safe_int(r.get("segment_index", ""), 0),
                "window_start_frame": _safe_int(r.get("window_start_frame", ""), 0),
                "window_end_frame": _safe_int(r.get("window_end_frame", ""), 0),
                "window_start_sec": _safe_float(r.get("window_start_sec", ""), 0.0),
                "window_end_sec": _safe_float(r.get("window_end_sec", ""), 0.0),

                "expected_note": expected_note,
                "expected_note_core": NoteToken(expected_note).core,

                "representative_rc_note": representative_rc_note,
                "best_theoretical_root_token": best_theoretical_root_token,
                "chosen_note": chosen_note,
                "chosen_note_core": NoteToken(chosen_note).core,

                "bias_type": bias_type,

                "support_hits": _safe_int(r.get("support_hits", ""), 0),
                "spiral_match_count": _safe_int(r.get("spiral_match_count", ""), 0),
                "spiral_consistency_score": _safe_float(r.get("spiral_consistency_score", ""), 0.0),
                "window_chain_match_score": _safe_float(r.get("window_chain_match_score", ""), 0.0),
                "best_theoretical_root_score": _safe_float(r.get("best_theoretical_root_score", ""), 0.0),

                "stabilization_score": _safe_float(r.get("stabilization_score", ""), 0.0),
                "stabilization_role": _safe_str(r.get("stabilization_role", "")),
                "stabilization_reason": _safe_str(r.get("stabilization_reason", "")),
                "theoretical_chain_verdict": _safe_str(r.get("theoretical_chain_verdict", "")),
                "best_theoretical_chain_string": _safe_str(r.get("best_theoretical_chain_string", "")),
            }
        )

    out.sort(key=lambda x: (x["window_start_sec"], x["segment_index"]))
    return out


def build_passport(per_note_rows: list[dict], *, expected_note: str, source_csv: Path) -> dict:
    if not per_note_rows:
        return {
            "source_csv": str(source_csv),
            "expected_note": expected_note,
            "row_count": 0,
            "status": "no_valid_rows",
        }

    chosen_notes = [r["chosen_note"] for r in per_note_rows if r["chosen_note"]]
    dominant_note = _mode(chosen_notes)

    dominant_note_count = sum(1 for x in chosen_notes if x == dominant_note)
    dominant_note_ratio = dominant_note_count / len(chosen_notes) if chosen_notes else 0.0

    bias_counter = Counter(r["bias_type"] for r in per_note_rows)
    verdict_mode = _mode([r["theoretical_chain_verdict"] for r in per_note_rows])
    role_mode = _mode([r["stabilization_role"] for r in per_note_rows])
    chain_mode = _mode([r["best_theoretical_chain_string"] for r in per_note_rows])

    support_hits_mean = _mean([float(r["support_hits"]) for r in per_note_rows])
    stabilization_score_mean = _mean([float(r["stabilization_score"]) for r in per_note_rows])
    window_chain_match_score_mean = _mean([float(r["window_chain_match_score"]) for r in per_note_rows])
    spiral_consistency_score_mean = _mean([float(r["spiral_consistency_score"]) for r in per_note_rows])

    chain_status = _classify_chain_status(
        dominant_note=dominant_note,
        expected_note=expected_note,
        chain_verdict_mode=verdict_mode,
        support_hits_mean=support_hits_mean,
        stabilization_score_mean=stabilization_score_mean,
        window_chain_match_score_mean=window_chain_match_score_mean,
        dominant_note_ratio=dominant_note_ratio,
    )

    bias_type = _bias_type(expected_note, dominant_note)

    first_match_segment = None
    for r in per_note_rows:
        if r["chosen_note"] == expected_note:
            first_match_segment = r["segment_index"]
            break

    return {
        "source_csv": str(source_csv),
        "expected_note": expected_note,
        "expected_note_core": NoteToken(expected_note).core,

        "row_count": len(per_note_rows),
        "dominant_note": dominant_note,
        "dominant_note_core": NoteToken(dominant_note).core if dominant_note else "",
        "dominant_note_count": dominant_note_count,
        "dominant_note_ratio": round(dominant_note_ratio, 6),

        "bias_type": bias_type,
        "chain_status": chain_status,

        "theoretical_chain_verdict_mode": verdict_mode,
        "stabilization_role_mode": role_mode,
        "best_theoretical_chain_string_mode": chain_mode,

        "support_hits_mean": round(support_hits_mean, 6),
        "stabilization_score_mean": round(stabilization_score_mean, 6),
        "window_chain_match_score_mean": round(window_chain_match_score_mean, 6),
        "spiral_consistency_score_mean": round(spiral_consistency_score_mean, 6),

        "first_exact_match_segment": first_match_segment,
        "bias_histogram": dict(bias_counter),
    }


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_note_report_dir(
    *,
    project_root: Path,
    instrument_name: str,
    expected_note: str,
    source_csv: Path,
) -> Path:
    """
    Output path policy:

    Block004_data/<instrument_name>/10_reports/<note_prefix>__structure_report/
    """
    reports_root = project_root / "Block004_data" / instrument_name / "10_reports"

    m = re.match(r"^(\d+__)?", source_csv.name)
    prefix_num = ""
    if m and m.group(0):
        prefix_num = m.group(0)

    folder_name = f"{prefix_num}{instrument_name}__{expected_note}__structure_report"
    return reports_root / folder_name


# ============================================================
# WRITE
# ============================================================

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_passport_txt(path: Path, passport: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in passport.items():
            if isinstance(v, dict):
                f.write(f"{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


def write_meta_json(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build per-note chain structure report for Block004. "
            "This stage analyzes note stability and structural bias, "
            "not phase/radial pseudo-geometry."
        )
    )
    ap.add_argument("--source_csv", required=True, help="Input chain-stabilized CSV for one note")
    ap.add_argument("--instrument_name", required=True, help="Example: RealPiano_1 or piano_midi")
    ap.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--expected_note", default="", help="Optional explicit expected note token; otherwise parsed from filename")

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    source_csv = Path(args.source_csv).resolve()
    instrument_name = _safe_str(args.instrument_name)
    expected_note = _safe_str(args.expected_note) or _extract_expected_note_from_filename(source_csv)

    rows = load_rows(source_csv)
    per_note_rows = build_per_note_rows(rows, expected_note=expected_note)
    passport = build_passport(per_note_rows, expected_note=expected_note, source_csv=source_csv)

    out_dir = resolve_note_report_dir(
        project_root=project_root,
        instrument_name=instrument_name,
        expected_note=expected_note,
        source_csv=source_csv,
    )

    out_full_csv = out_dir / "structure_full.csv"
    out_passport_txt = out_dir / "passport.txt"
    out_meta_json = out_dir / "meta.json"

    write_csv(out_full_csv, per_note_rows)
    write_passport_txt(out_passport_txt, passport)

    meta = {
        "source_csv": str(source_csv),
        "instrument_name": instrument_name,
        "expected_note": expected_note,
        "output_dir": str(out_dir),
        "outputs": {
            "structure_full_csv": str(out_full_csv),
            "passport_txt": str(out_passport_txt),
            "meta_json": str(out_meta_json),
        },
        "semantic_note": (
            "Per-note structure analysis for Block004. "
            "Outputs are written to Block004_data/<instrument>/10_reports. "
            "No phase/radial pseudo-geometry is used."
        ),
    }
    write_meta_json(out_meta_json, meta)

    print("per-note chain structure report complete")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()