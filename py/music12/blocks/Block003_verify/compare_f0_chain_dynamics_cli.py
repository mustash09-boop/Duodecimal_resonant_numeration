from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from music12.core.notation12 import token_to_abs_semitone_index


COMPARE_NOTE = (
    "This stage compares Block001 etalon f0 with Block002 framewise rc inference. "
    "Detected rc is not true f0; it is compared as a candidate against etalon harmonic structure."
)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@dataclass(frozen=True)
class EtalonRow:
    chain_id: int
    segment_no: int
    event_index: int
    time_start_sec: float
    time_end_sec: float
    duration_sec: float
    onset_group: int
    onset_polyphony: int
    f0_note: str
    f0_hz: float
    harmonics: Dict[int, str]

    @property
    def f0_abs(self) -> int:
        return token_to_abs_semitone_index(self.f0_note)


@dataclass(frozen=True)
class DetectedFrameRow:
    frame_index: int
    time_sec: float
    candidate_count: int
    chosen_rc_note: str
    chosen_rc_hz: float
    chosen_rc_energy: float
    chain_score: float
    strongest_peak_note: str
    strongest_peak_hz: float
    strongest_peak_energy: float
    strongest_peak_rejected_reason: str
    supports: Dict[int, str]

    @property
    def has_rc(self) -> bool:
        return bool(self.chosen_rc_note)

    @property
    def rc_abs(self) -> Optional[int]:
        if not self.chosen_rc_note:
            return None
        return token_to_abs_semitone_index(self.chosen_rc_note)


# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------

def _to_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_etalon_readable_csv(path: Path) -> List[EtalonRow]:
    rows: List[EtalonRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            harmonics: Dict[int, str] = {}
            for k, v in r.items():
                if k.startswith("h") and k.endswith("_note"):
                    try:
                        idx = int(k[1:-5])
                        harmonics[idx] = (v or "").strip()
                    except Exception:
                        pass

            rows.append(
                EtalonRow(
                    chain_id=_to_int(r.get("chain_id", "")),
                    segment_no=_to_int(r.get("segment_no", "")),
                    event_index=_to_int(r.get("event_index", "")),
                    time_start_sec=_to_float(r.get("time_start_sec", "")),
                    time_end_sec=_to_float(r.get("time_end_sec", "")),
                    duration_sec=_to_float(r.get("duration_sec", "")),
                    onset_group=_to_int(r.get("onset_group", "")),
                    onset_polyphony=_to_int(r.get("onset_polyphony", "")),
                    f0_note=(r.get("f0_note", "") or "").strip(),
                    f0_hz=_to_float(r.get("f0_hz", "")),
                    harmonics=harmonics,
                )
            )

    rows.sort(key=lambda x: (x.time_start_sec, x.time_end_sec, x.chain_id, x.segment_no))
    return rows


def load_detected_framewise_readable_csv(path: Path) -> List[DetectedFrameRow]:
    rows: List[DetectedFrameRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            supports: Dict[int, str] = {}
            for k, v in r.items():
                if k.startswith("support_h") and k.endswith("_note"):
                    try:
                        idx = int(k[len("support_h"):-len("_note")])
                        supports[idx] = (v or "").strip()
                    except Exception:
                        pass

            rows.append(
                DetectedFrameRow(
                    frame_index=_to_int(r.get("frame_index", "")),
                    time_sec=_to_float(r.get("time_sec", "")),
                    candidate_count=_to_int(r.get("candidate_count", "")),
                    chosen_rc_note=(r.get("chosen_rc_note", "") or "").strip(),
                    chosen_rc_hz=_to_float(r.get("chosen_rc_hz", "")),
                    chosen_rc_energy=_to_float(r.get("chosen_rc_energy", "")),
                    chain_score=_to_float(r.get("chain_score", "")),
                    strongest_peak_note=(r.get("strongest_peak_note", "") or "").strip(),
                    strongest_peak_hz=_to_float(r.get("strongest_peak_hz", "")),
                    strongest_peak_energy=_to_float(r.get("strongest_peak_energy", "")),
                    strongest_peak_rejected_reason=(r.get("strongest_peak_rejected_reason", "") or "").strip(),
                    supports=supports,
                )
            )

    rows.sort(key=lambda x: x.frame_index)
    return rows


# ------------------------------------------------------------
# MATCHING
# ------------------------------------------------------------

def find_etalon_at_time(t: float, etalon_rows: List[EtalonRow]) -> Optional[EtalonRow]:
    overlapped = [r for r in etalon_rows if r.time_start_sec <= t <= r.time_end_sec]
    if overlapped:
        overlapped.sort(key=lambda r: (r.duration_sec, abs((r.time_start_sec + r.time_end_sec) * 0.5 - t)))
        return overlapped[0]

    if not etalon_rows:
        return None

    best = min(
        etalon_rows,
        key=lambda r: abs(((r.time_start_sec + r.time_end_sec) * 0.5) - t),
    )
    if abs(((best.time_start_sec + best.time_end_sec) * 0.5) - t) > max(0.08, best.duration_sec):
        return None
    return best


def matched_rc_harmonic_index(det_note: str, et: EtalonRow) -> Optional[int]:
    if not det_note:
        return None
    for idx, tok in sorted(et.harmonics.items()):
        if det_note == tok:
            return idx
    return None


# ------------------------------------------------------------
# DYNAMICS
# ------------------------------------------------------------

def classify_dynamic_state(
    current_error: Optional[int],
    prev_error: Optional[int],
    next_error: Optional[int],
    current_energy: float,
    prev_energy: float,
    next_energy: float,
    harmonic_idx: Optional[int],
    has_match: bool,
) -> str:
    if not has_match or current_error is None:
        return "NO_MATCH"

    if harmonic_idx == 1 and abs(current_error) == 0:
        if prev_error is None or next_error is None:
            return "STABLE"
        if abs(current_error) <= abs(prev_error) and abs(current_error) <= abs(next_error):
            return "STABLE"

    dE_prev = current_energy - prev_energy
    dE_next = next_energy - current_energy

    improving_from_prev = prev_error is not None and abs(current_error) < abs(prev_error)
    worsening_to_next = next_error is not None and abs(next_error) > abs(current_error)
    improving_to_next = next_error is not None and abs(next_error) < abs(current_error)

    if dE_prev > 0 and (current_error is not None and abs(current_error) > 0):
        return "ATTACK"

    if current_error is not None and abs(current_error) <= 1:
        if harmonic_idx in (None, 1) and dE_next <= 0.1:
            return "STABLE"

    if dE_next < 0 and (worsening_to_next or (current_error is not None and abs(current_error) <= 2)):
        return "DECAY"

    if current_error is not None and next_error is not None:
        if abs(next_error) > abs(current_error):
            return "DRIFT"

    if improving_from_prev or improving_to_next:
        return "CONVERGING"

    return "UNDEFINED"


# ------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------

def compare_dynamics(
    etalon_rows: List[EtalonRow],
    detected_rows: List[DetectedFrameRow],
) -> List[dict]:
    out: List[dict] = []

    matched_etalon: List[Optional[EtalonRow]] = []
    abs_errors: List[Optional[int]] = []
    harmonic_matches: List[Optional[int]] = []

    for det in detected_rows:
        et = find_etalon_at_time(det.time_sec, etalon_rows)
        matched_etalon.append(et)

        if et is None or not det.has_rc:
            abs_errors.append(None)
            harmonic_matches.append(None)
            continue

        err = det.rc_abs - et.f0_abs
        abs_errors.append(err)

        h_idx = matched_rc_harmonic_index(det.chosen_rc_note, et)
        harmonic_matches.append(h_idx)

    for i, det in enumerate(detected_rows):
        et = matched_etalon[i]
        err = abs_errors[i]
        h_idx = harmonic_matches[i]

        prev_err = abs_errors[i - 1] if i > 0 else None
        next_err = abs_errors[i + 1] if i + 1 < len(abs_errors) else None

        prev_energy = detected_rows[i - 1].chosen_rc_energy if i > 0 else det.chosen_rc_energy
        next_energy = detected_rows[i + 1].chosen_rc_energy if i + 1 < len(detected_rows) else det.chosen_rc_energy

        has_match = et is not None and det.has_rc

        state = classify_dynamic_state(
            current_error=err,
            prev_error=prev_err,
            next_error=next_err,
            current_energy=det.chosen_rc_energy,
            prev_energy=prev_energy,
            next_energy=next_energy,
            harmonic_idx=h_idx,
            has_match=has_match,
        )

        out.append(
            {
                "frame_index": det.frame_index,
                "time_sec": det.time_sec,
                "det_rc_note": det.chosen_rc_note,
                "det_rc_hz": det.chosen_rc_hz,
                "det_rc_energy": det.chosen_rc_energy,
                "det_chain_score": det.chain_score,
                "strongest_peak_note": det.strongest_peak_note,
                "strongest_peak_hz": det.strongest_peak_hz,
                "et_f0_note": "" if et is None else et.f0_note,
                "et_f0_hz": "" if et is None else et.f0_hz,
                "et_chain_id": "" if et is None else et.chain_id,
                "et_segment_no": "" if et is None else et.segment_no,
                "abs_error_steps": "" if err is None else err,
                "matched_rc_harmonic_index": "" if h_idx is None else h_idx,
                "dynamic_state": state,
                "strongest_peak_rejected_reason": det.strongest_peak_rejected_reason,
            }
        )

    return out


# ------------------------------------------------------------
# WRITE
# ------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cnt = Counter(r["dynamic_state"] for r in rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dynamic_state", "count"])
        for key, value in cnt.most_common():
            writer.writerow([key, value])


def write_meta_json(
    path: Path,
    *,
    etalon_csv: Path,
    detected_csv: Path,
    out_rows: List[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cnt = Counter(r["dynamic_state"] for r in out_rows)
    data = {
        "inputs": {
            "etalon_readable_csv": str(etalon_csv),
            "detected_framewise_readable_csv": str(detected_csv),
        },
        "semantic_note": COMPARE_NOTE,
        "derived": {
            "row_count": len(out_rows),
            "dynamic_state_counts": dict(cnt),
        },
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare Block001 etalon f0 with Block002 framewise rc inference "
            "and classify chain dynamics."
        )
    )
    ap.add_argument("--etalon_csv", required=True)
    ap.add_argument("--detected_csv", required=True)
    ap.add_argument("--out_compare_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    etalon_csv = Path(args.etalon_csv).resolve()
    detected_csv = Path(args.detected_csv).resolve()

    etalon_rows = load_etalon_readable_csv(etalon_csv)
    detected_rows = load_detected_framewise_readable_csv(detected_csv)

    out_rows = compare_dynamics(etalon_rows, detected_rows)

    write_csv(Path(args.out_compare_csv).resolve(), out_rows)
    write_summary_csv(Path(args.out_summary_csv).resolve(), out_rows)
    write_meta_json(
        Path(args.out_meta_json).resolve(),
        etalon_csv=etalon_csv,
        detected_csv=detected_csv,
        out_rows=out_rows,
    )

    print("compare rc chain dynamics complete")
    print(json.dumps(
        {
            "row_count": len(out_rows),
            "out_compare_csv": str(Path(args.out_compare_csv).resolve()),
            "out_summary_csv": str(Path(args.out_summary_csv).resolve()),
            "out_meta_json": str(Path(args.out_meta_json).resolve()),
            "semantic_note": COMPARE_NOTE,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()