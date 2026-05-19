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
    "This stage compares Block001 etalon f0 with Block002 stabilized representative rc. "
    "Representative rc is not true f0; it is compared as a candidate against etalon harmonic structure."
)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@dataclass(frozen=True)
class EtalonRow:
    time_start_sec: float
    time_end_sec: float
    f0_note: str
    harmonics: Dict[int, str]

    @property
    def f0_abs(self) -> int:
        return token_to_abs_semitone_index(self.f0_note)


@dataclass(frozen=True)
class StabilizedRow:
    segment_index: int
    chosen_time_sec: float
    representative_rc_note: str
    representative_rc_hz: float
    representative_rc_energy: float
    rc_chain_score: float
    support_hits: int
    strongest_peak_note: str
    strongest_peak_hz: float
    stabilization_role: str
    stabilization_reason: str
    stabilization_score: float

    @property
    def has_rc(self) -> bool:
        return bool(self.representative_rc_note)

    @property
    def rc_abs(self) -> Optional[int]:
        if not self.representative_rc_note:
            return None
        return token_to_abs_semitone_index(self.representative_rc_note)


# ------------------------------------------------------------
# SAFE CONVERTERS
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


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load_etalon(path: Path) -> List[EtalonRow]:
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
                    time_start_sec=_to_float(r.get("time_start_sec", "")),
                    time_end_sec=_to_float(r.get("time_end_sec", "")),
                    f0_note=(r.get("f0_note", "") or "").strip(),
                    harmonics=harmonics,
                )
            )

    rows.sort(key=lambda x: (x.time_start_sec, x.time_end_sec))
    return rows


def load_stabilized(path: Path) -> List[StabilizedRow]:
    rows: List[StabilizedRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                StabilizedRow(
                    segment_index=_to_int(r.get("segment_index", "")),
                    chosen_time_sec=_to_float(r.get("chosen_time_sec", "")),
                    representative_rc_note=(r.get("representative_rc_note", "") or "").strip(),
                    representative_rc_hz=_to_float(r.get("representative_rc_hz", "")),
                    representative_rc_energy=_to_float(r.get("representative_rc_energy", "")),
                    rc_chain_score=_to_float(r.get("rc_chain_score", "")),
                    support_hits=_to_int(r.get("support_hits", "")),
                    strongest_peak_note=(r.get("strongest_peak_note", "") or "").strip(),
                    strongest_peak_hz=_to_float(r.get("strongest_peak_hz", "")),
                    stabilization_role=(r.get("stabilization_role", "") or "").strip(),
                    stabilization_reason=(r.get("stabilization_reason", "") or "").strip(),
                    stabilization_score=_to_float(r.get("stabilization_score", "")),
                )
            )

    rows.sort(key=lambda x: x.segment_index)
    return rows


# ------------------------------------------------------------
# MATCH
# ------------------------------------------------------------

def find_etalon_at_time(etalon_rows: List[EtalonRow], t: float) -> Optional[EtalonRow]:
    for et in etalon_rows:
        if et.time_start_sec <= t <= et.time_end_sec:
            return et
    return None


def match_type(det_note: str, et: EtalonRow) -> str:
    if det_note == et.f0_note:
        return "F0_MATCH"

    for idx, h in sorted(et.harmonics.items()):
        if det_note == h:
            return f"H{idx}_MATCH"

    return "NO_MATCH"


# ------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------

def compare(etalon_rows: List[EtalonRow], stabilized_rows: List[StabilizedRow]) -> List[dict]:
    out: List[dict] = []

    for row in stabilized_rows:
        et = find_etalon_at_time(etalon_rows, row.chosen_time_sec)

        if et is None or not row.has_rc:
            out.append(
                {
                    "segment_index": row.segment_index,
                    "chosen_time_sec": row.chosen_time_sec,
                    "det_rc_note": row.representative_rc_note,
                    "det_rc_hz": row.representative_rc_hz,
                    "det_rc_energy": row.representative_rc_energy,
                    "et_f0_note": "",
                    "match_type": "NO_ETALON",
                    "abs_error": "",
                    "support_hits": row.support_hits,
                    "rc_chain_score": row.rc_chain_score,
                    "stabilization_role": row.stabilization_role,
                    "stabilization_reason": row.stabilization_reason,
                    "stabilization_score": row.stabilization_score,
                    "strongest_peak_note": row.strongest_peak_note,
                    "strongest_peak_hz": row.strongest_peak_hz,
                }
            )
            continue

        det_abs = row.rc_abs
        et_abs = et.f0_abs
        error = "" if det_abs is None else (det_abs - et_abs)

        mtype = match_type(row.representative_rc_note, et)

        out.append(
            {
                "segment_index": row.segment_index,
                "chosen_time_sec": row.chosen_time_sec,
                "det_rc_note": row.representative_rc_note,
                "det_rc_hz": row.representative_rc_hz,
                "det_rc_energy": row.representative_rc_energy,
                "et_f0_note": et.f0_note,
                "match_type": mtype,
                "abs_error": error,
                "support_hits": row.support_hits,
                "rc_chain_score": row.rc_chain_score,
                "stabilization_role": row.stabilization_role,
                "stabilization_reason": row.stabilization_reason,
                "stabilization_score": row.stabilization_score,
                "strongest_peak_note": row.strongest_peak_note,
                "strongest_peak_hz": row.strongest_peak_hz,
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

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    counter = Counter(r["match_type"] for r in rows)
    role_counter = Counter(r["stabilization_role"] for r in rows if (r["stabilization_role"] or "").strip())

    valid_errors = [
        abs(int(r["abs_error"]))
        for r in rows
        if str(r["abs_error"]).strip() != ""
    ]
    mean_abs_error = (sum(valid_errors) / len(valid_errors)) if valid_errors else ""

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["row_count", len(rows)])
        writer.writerow(["mean_abs_error", mean_abs_error])

        for k, v in counter.items():
            writer.writerow([f"match_{k}", v])

        for k, v in role_counter.items():
            writer.writerow([f"role_{k}", v])


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare Block001 etalon f0 with Block002 stabilized representative rc. "
            "This stage does NOT interpret representative rc as true f0."
        )
    )
    ap.add_argument("--etalon_csv", required=True)
    ap.add_argument("--stabilized_csv", required=True)
    ap.add_argument("--out_compare_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    etalon = load_etalon(Path(args.etalon_csv).resolve())
    stabilized = load_stabilized(Path(args.stabilized_csv).resolve())

    rows = compare(etalon, stabilized)

    write_csv(Path(args.out_compare_csv).resolve(), rows)
    write_summary(Path(args.out_summary_csv).resolve(), rows)

    Path(args.out_meta_json).resolve().write_text(
        json.dumps(
            {
                "rows": len(rows),
                "semantic_note": COMPARE_NOTE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("compare stabilized rc complete")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "out_compare_csv": str(Path(args.out_compare_csv).resolve()),
                "out_summary_csv": str(Path(args.out_summary_csv).resolve()),
                "out_meta_json": str(Path(args.out_meta_json).resolve()),
                "semantic_note": COMPARE_NOTE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()