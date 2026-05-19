from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from music12.core.notation12 import bij12_to_int, parse_token, token_to_abs_semitone_index


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@dataclass
class ReadableRow:
    source: str
    chain_id: int
    segment_no: int
    event_index: int
    time_start_sec: float
    time_end_sec: float
    duration_sec: float
    start_frame60: int
    end_frame60: int
    duration_frames60: int
    onset_group: int
    onset_polyphony: int
    f0_note: str
    f0_hz: float
    harmonics: Dict[int, str]
    harmonics_hz: Dict[int, float]

    @property
    def f0_abs(self) -> int:
        return token_to_abs_semitone_index(self.f0_note)

    @property
    def octave_token(self) -> str:
        return parse_token(self.f0_note).oct

    @property
    def step_token(self) -> str:
        return parse_token(self.f0_note).step

    @property
    def octave_index1(self) -> int:
        return bij12_to_int(self.octave_token)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def _parse_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def _parse_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def load_readable_csv(path: Path, source: str) -> List[ReadableRow]:
    rows: List[ReadableRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "chain_id",
            "segment_no",
            "event_index",
            "time_start_sec",
            "time_end_sec",
            "duration_sec",
            "start_frame60",
            "end_frame60",
            "duration_frames60",
            "onset_group",
            "onset_polyphony",
            "f0_note",
            "f0_hz",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing required CSV columns: {sorted(missing)}")

        for r in reader:
            harmonics: Dict[int, str] = {}
            harmonics_hz: Dict[int, float] = {}

            # читаем h1..h12 (или сколько есть)
            for k, v in r.items():
                if not k:
                    continue
                if k.startswith("h") and k.endswith("_note"):
                    try:
                        idx = int(k[1:-5])
                        harmonics[idx] = (v or "").strip()
                    except Exception:
                        pass
                elif k.startswith("h") and k.endswith("_hz"):
                    try:
                        idx = int(k[1:-3])
                        harmonics_hz[idx] = float(v)
                    except Exception:
                        pass

            rows.append(
                ReadableRow(
                    source=source,
                    chain_id=_parse_int(r, "chain_id"),
                    segment_no=_parse_int(r, "segment_no"),
                    event_index=_parse_int(r, "event_index"),
                    time_start_sec=_parse_float(r, "time_start_sec"),
                    time_end_sec=_parse_float(r, "time_end_sec"),
                    duration_sec=_parse_float(r, "duration_sec"),
                    start_frame60=_parse_int(r, "start_frame60"),
                    end_frame60=_parse_int(r, "end_frame60"),
                    duration_frames60=_parse_int(r, "duration_frames60"),
                    onset_group=_parse_int(r, "onset_group"),
                    onset_polyphony=_parse_int(r, "onset_polyphony"),
                    f0_note=(r.get("f0_note", "") or "").strip(),
                    f0_hz=_parse_float(r, "f0_hz"),
                    harmonics=harmonics,
                    harmonics_hz=harmonics_hz,
                )
            )

    rows.sort(key=lambda x: (x.time_start_sec, x.time_end_sec, x.chain_id, x.segment_no))
    return rows


# ------------------------------------------------------------
# MATCHING
# ------------------------------------------------------------

def overlap_seconds(a: ReadableRow, b: ReadableRow) -> float:
    left = max(a.time_start_sec, b.time_start_sec)
    right = min(a.time_end_sec, b.time_end_sec)
    return max(0.0, right - left)


def choose_best_etalon(det: ReadableRow, etalon_rows: List[ReadableRow]) -> Optional[ReadableRow]:
    """
    Берём эталонную строку с максимальным перекрытием.
    При равенстве — минимальная разница центров.
    """
    best: Optional[ReadableRow] = None
    best_key: Optional[Tuple[float, float]] = None

    det_center = 0.5 * (det.time_start_sec + det.time_end_sec)

    for et in etalon_rows:
        ov = overlap_seconds(det, et)
        if ov <= 0:
            continue

        et_center = 0.5 * (et.time_start_sec + et.time_end_sec)
        center_diff = abs(det_center - et_center)

        # max overlap, then min center diff
        key = (ov, -center_diff)
        if best_key is None or key > best_key:
            best_key = key
            best = et

    return best


# ------------------------------------------------------------
# DIAGNOSTICS
# ------------------------------------------------------------

def harmonic_match_index(det_note: str, etalon: ReadableRow) -> Optional[int]:
    for idx, tok in sorted(etalon.harmonics.items()):
        if det_note == tok:
            return idx
    return None


def nearest_harmonic(det_abs: int, etalon: ReadableRow) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    best_idx: Optional[int] = None
    best_tok: Optional[str] = None
    best_delta: Optional[int] = None

    for idx, tok in sorted(etalon.harmonics.items()):
        try:
            abs_h = token_to_abs_semitone_index(tok)
        except Exception:
            continue
        delta = det_abs - abs_h
        if best_delta is None or abs(delta) < abs(best_delta):
            best_idx = idx
            best_tok = tok
            best_delta = delta

    return best_idx, best_tok, best_delta


def classify_mismatch(det: ReadableRow, et: Optional[ReadableRow]) -> str:
    if et is None:
        return "no_time_match"

    if det.f0_note == et.f0_note:
        return "exact_f0"

    h_idx = harmonic_match_index(det.f0_note, et)
    if h_idx is not None:
        return f"matched_h{h_idx}_instead_of_f0"

    det_abs = det.f0_abs
    et_abs = et.f0_abs
    abs_delta = det_abs - et_abs

    # если съехало ровно по октавам
    if abs_delta != 0 and abs_delta % 12 == 0:
        return "octave_shift"

    # если выше даже h12 эталона
    try:
        h12_abs = token_to_abs_semitone_index(et.harmonics.get(12, et.f0_note))
        if det_abs > h12_abs:
            return "above_h12_range"
    except Exception:
        pass

    # если попали ближе к какой-то верхней гармонике, но не точно
    nearest_idx, _, nearest_delta = nearest_harmonic(det_abs, et)
    if nearest_idx is not None and nearest_idx >= 2 and nearest_delta is not None and abs(nearest_delta) <= 2:
        return f"near_h{nearest_idx}_region"

    # если у наблюдения многозначная октава, а у эталона нет, и ошибка большая —
    # это не доказательство, но сигнал проверить правило октавной оси.
    if len(det.octave_token) > 1 and abs(abs_delta) >= 12:
        return "multi_digit_octave_suspicion"

    return "other_pitch_error"


# ------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------

def compare_rows(etalon_rows: List[ReadableRow], detected_rows: List[ReadableRow]) -> List[dict]:
    out: List[dict] = []

    for det in detected_rows:
        et = choose_best_etalon(det, etalon_rows)

        if et is None:
            out.append(
                {
                    "det_chain_id": det.chain_id,
                    "det_segment_no": det.segment_no,
                    "det_time_start_sec": det.time_start_sec,
                    "det_time_end_sec": det.time_end_sec,
                    "det_f0_note": det.f0_note,
                    "det_octave_token": det.octave_token,
                    "det_octave_index1": det.octave_index1,
                    "et_chain_id": "",
                    "et_segment_no": "",
                    "et_time_start_sec": "",
                    "et_time_end_sec": "",
                    "et_f0_note": "",
                    "et_octave_token": "",
                    "et_octave_index1": "",
                    "time_overlap_sec": 0.0,
                    "abs_delta_steps": "",
                    "octave_delta": "",
                    "matched_harmonic_index": "",
                    "matched_harmonic_note": "",
                    "nearest_harmonic_index": "",
                    "nearest_harmonic_note": "",
                    "nearest_harmonic_delta_steps": "",
                    "classification": "no_time_match",
                }
            )
            continue

        det_abs = det.f0_abs
        et_abs = et.f0_abs
        abs_delta = det_abs - et_abs
        octave_delta = det.octave_index1 - et.octave_index1

        matched_h_idx = harmonic_match_index(det.f0_note, et)
        matched_h_note = et.harmonics.get(matched_h_idx, "") if matched_h_idx is not None else ""

        nearest_idx, nearest_note, nearest_delta = nearest_harmonic(det_abs, et)
        cls = classify_mismatch(det, et)

        out.append(
            {
                "det_chain_id": det.chain_id,
                "det_segment_no": det.segment_no,
                "det_time_start_sec": det.time_start_sec,
                "det_time_end_sec": det.time_end_sec,
                "det_f0_note": det.f0_note,
                "det_octave_token": det.octave_token,
                "det_octave_index1": det.octave_index1,
                "et_chain_id": et.chain_id,
                "et_segment_no": et.segment_no,
                "et_time_start_sec": et.time_start_sec,
                "et_time_end_sec": et.time_end_sec,
                "et_f0_note": et.f0_note,
                "et_octave_token": et.octave_token,
                "et_octave_index1": et.octave_index1,
                "time_overlap_sec": overlap_seconds(det, et),
                "abs_delta_steps": abs_delta,
                "octave_delta": octave_delta,
                "matched_harmonic_index": "" if matched_h_idx is None else matched_h_idx,
                "matched_harmonic_note": matched_h_note,
                "nearest_harmonic_index": "" if nearest_idx is None else nearest_idx,
                "nearest_harmonic_note": "" if nearest_note is None else nearest_note,
                "nearest_harmonic_delta_steps": "" if nearest_delta is None else nearest_delta,
                "classification": cls,
            }
        )

    return out


# ------------------------------------------------------------
# WRITE
# ------------------------------------------------------------

def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cnt = Counter(row["classification"] for row in rows)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["classification", "count"])
        for key, value in cnt.most_common():
            writer.writerow([key, value])


def write_meta(path: Path, *, etalon_csv: Path, detected_csv: Path, rows: List[dict]) -> None:
    cnt = Counter(row["classification"] for row in rows)
    data = {
        "inputs": {
            "etalon_readable_csv": str(etalon_csv),
            "detected_readable_csv": str(detected_csv),
        },
        "derived": {
            "compared_row_count": len(rows),
            "classification_counts": dict(cnt),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare Block001 etalon readable harmonic chains with Block002 detected readable harmonic chains."
    )
    ap.add_argument("--etalon_csv", required=True)
    ap.add_argument("--detected_csv", required=True)
    ap.add_argument("--out_compare_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    etalon_csv = Path(args.etalon_csv).resolve()
    detected_csv = Path(args.detected_csv).resolve()

    etalon_rows = load_readable_csv(etalon_csv, "etalon")
    detected_rows = load_readable_csv(detected_csv, "detected")

    compare_rows_out = compare_rows(etalon_rows, detected_rows)

    fieldnames = [
        "det_chain_id",
        "det_segment_no",
        "det_time_start_sec",
        "det_time_end_sec",
        "det_f0_note",
        "det_octave_token",
        "det_octave_index1",
        "et_chain_id",
        "et_segment_no",
        "et_time_start_sec",
        "et_time_end_sec",
        "et_f0_note",
        "et_octave_token",
        "et_octave_index1",
        "time_overlap_sec",
        "abs_delta_steps",
        "octave_delta",
        "matched_harmonic_index",
        "matched_harmonic_note",
        "nearest_harmonic_index",
        "nearest_harmonic_note",
        "nearest_harmonic_delta_steps",
        "classification",
    ]

    write_csv(Path(args.out_compare_csv).resolve(), compare_rows_out, fieldnames)
    write_summary_csv(Path(args.out_summary_csv).resolve(), compare_rows_out)
    write_meta(
        Path(args.out_meta_json).resolve(),
        etalon_csv=etalon_csv,
        detected_csv=detected_csv,
        rows=compare_rows_out,
    )

    print("compare f0 chains complete")
    print(json.dumps(
        {
            "compared_row_count": len(compare_rows_out),
            "out_compare_csv": str(Path(args.out_compare_csv).resolve()),
            "out_summary_csv": str(Path(args.out_summary_csv).resolve()),
            "out_meta_json": str(Path(args.out_meta_json).resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()