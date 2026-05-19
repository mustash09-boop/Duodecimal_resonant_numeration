from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# NOTE DETECT FROM STABILIZED CHAIN CANDIDATES
# ------------------------------------------------------------
# This version does NOT detect notes directly from matrix peaks.
# It builds note events from already stabilized chain candidates.
#
# Input:
#   stabilized_chain_candidates CSV
#
# Logic:
#   rows -> grouped by note token -> merged into note events
#
# Polyphony:
#   naturally supported, because several note tokens may coexist
#   in the same time region.
# ============================================================


# ============================================================
# HELPERS
# ============================================================

def _safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _load_json(v: str):
    try:
        return json.loads(v)
    except Exception:
        return None


# ============================================================
# DATA
# ============================================================

@dataclass(frozen=True)
class StabilizedRow:
    segment_index: int
    window_start_frame: int
    window_end_frame: int
    window_start_sec: float
    window_end_sec: float

    active_note_rank: int
    active_note_count: int

    representative_rc_note: str
    representative_rc_hz: float
    representative_rc_energy: float

    best_theoretical_root_token: str
    best_theoretical_root_score: float
    best_theoretical_chain_string: str

    support_hits: int
    spiral_match_count: int
    spiral_consistency_score: float
    window_chain_match_score: float
    theoretical_chain_verdict: str

    stabilization_role: str
    stabilization_reason: str
    stabilization_score: float


@dataclass(frozen=True)
class DetectedNoteEvent:
    note_index: int

    note_token: str
    time_start: float
    time_end: float
    duration: float

    segment_start: int
    segment_end: int
    event_count: int

    representative_rc_hz_mean: float
    representative_rc_energy_mean: float

    best_theoretical_root_score_mean: float
    support_hits_mean: float
    spiral_match_count_mean: float
    spiral_consistency_score_mean: float
    window_chain_match_score_mean: float
    stabilization_score_mean: float

    theoretical_chain_verdict_mode: str
    stabilization_role_mode: str
    best_theoretical_chain_string_mode: str


# ============================================================
# LOAD
# ============================================================

def load_stabilized_rows(path: Path) -> List[StabilizedRow]:
    rows: List[StabilizedRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            note_token = _safe_str(r.get("best_theoretical_root_token", ""))
            if not note_token:
                note_token = _safe_str(r.get("representative_rc_note", ""))

            if not note_token:
                continue

            rows.append(
                StabilizedRow(
                    segment_index=_safe_int(r.get("segment_index", ""), 0),
                    window_start_frame=_safe_int(r.get("window_start_frame", ""), 0),
                    window_end_frame=_safe_int(r.get("window_end_frame", ""), 0),
                    window_start_sec=_safe_float(r.get("window_start_sec", ""), 0.0),
                    window_end_sec=_safe_float(r.get("window_end_sec", ""), 0.0),

                    active_note_rank=_safe_int(r.get("active_note_rank", ""), 0),
                    active_note_count=_safe_int(r.get("active_note_count", ""), 0),

                    representative_rc_note=_safe_str(r.get("representative_rc_note", "")),
                    representative_rc_hz=_safe_float(r.get("representative_rc_hz", ""), 0.0),
                    representative_rc_energy=_safe_float(r.get("representative_rc_energy", ""), 0.0),

                    best_theoretical_root_token=note_token,
                    best_theoretical_root_score=_safe_float(r.get("best_theoretical_root_score", ""), 0.0),
                    best_theoretical_chain_string=_safe_str(r.get("best_theoretical_chain_string", "")),

                    support_hits=_safe_int(r.get("support_hits", ""), 0),
                    spiral_match_count=_safe_int(r.get("spiral_match_count", ""), 0),
                    spiral_consistency_score=_safe_float(r.get("spiral_consistency_score", ""), 0.0),
                    window_chain_match_score=_safe_float(r.get("window_chain_match_score", ""), 0.0),
                    theoretical_chain_verdict=_safe_str(r.get("theoretical_chain_verdict", "")),

                    stabilization_role=_safe_str(r.get("stabilization_role", "")),
                    stabilization_reason=_safe_str(r.get("stabilization_reason", "")),
                    stabilization_score=_safe_float(r.get("stabilization_score", ""), 0.0),
                )
            )

    rows.sort(key=lambda x: (x.window_start_sec, x.active_note_rank, x.best_theoretical_root_token))
    return rows


# ============================================================
# EVENT BUILDING
# ============================================================

def _mode(values: List[str]) -> str:
    counts: Dict[str, int] = {}
    for v in values:
        if not v:
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda x: x[1])[0]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _can_merge(
    prev: StabilizedRow,
    current: StabilizedRow,
    *,
    max_gap_sec: float,
    require_same_verdict_family: bool,
) -> bool:
    if prev.best_theoretical_root_token != current.best_theoretical_root_token:
        return False

    gap = current.window_start_sec - prev.window_end_sec
    if gap > max_gap_sec:
        return False

    if require_same_verdict_family:
        a = prev.theoretical_chain_verdict.split("_")[0] if prev.theoretical_chain_verdict else ""
        b = current.theoretical_chain_verdict.split("_")[0] if current.theoretical_chain_verdict else ""
        if a != b:
            return False

    return True


def build_note_events(
    rows: List[StabilizedRow],
    *,
    max_gap_sec: float,
    min_event_count: int,
    require_same_verdict_family: bool,
) -> List[DetectedNoteEvent]:
    events: List[DetectedNoteEvent] = []

    if not rows:
        return events

    note_index = 0

    # group by note token first
    by_note: Dict[str, List[StabilizedRow]] = {}
    for r in rows:
        by_note.setdefault(r.best_theoretical_root_token, []).append(r)

    for note_token, note_rows in by_note.items():
        note_rows.sort(key=lambda x: x.window_start_sec)

        current_group: List[StabilizedRow] = []

        def flush_group():
            nonlocal note_index, current_group

            if not current_group:
                return

            if len(current_group) < min_event_count:
                current_group = []
                return

            time_start = current_group[0].window_start_sec
            time_end = current_group[-1].window_end_sec

            event = DetectedNoteEvent(
                note_index=note_index,
                note_token=note_token,
                time_start=time_start,
                time_end=time_end,
                duration=max(0.0, time_end - time_start),

                segment_start=current_group[0].segment_index,
                segment_end=current_group[-1].segment_index,
                event_count=len(current_group),

                representative_rc_hz_mean=_mean([x.representative_rc_hz for x in current_group]),
                representative_rc_energy_mean=_mean([x.representative_rc_energy for x in current_group]),

                best_theoretical_root_score_mean=_mean([x.best_theoretical_root_score for x in current_group]),
                support_hits_mean=_mean([float(x.support_hits) for x in current_group]),
                spiral_match_count_mean=_mean([float(x.spiral_match_count) for x in current_group]),
                spiral_consistency_score_mean=_mean([x.spiral_consistency_score for x in current_group]),
                window_chain_match_score_mean=_mean([x.window_chain_match_score for x in current_group]),
                stabilization_score_mean=_mean([x.stabilization_score for x in current_group]),

                theoretical_chain_verdict_mode=_mode([x.theoretical_chain_verdict for x in current_group]),
                stabilization_role_mode=_mode([x.stabilization_role for x in current_group]),
                best_theoretical_chain_string_mode=_mode([x.best_theoretical_chain_string for x in current_group]),
            )

            events.append(event)
            note_index += 1
            current_group = []

        for row in note_rows:
            if not current_group:
                current_group = [row]
                continue

            prev = current_group[-1]
            if _can_merge(
                prev,
                row,
                max_gap_sec=max_gap_sec,
                require_same_verdict_family=require_same_verdict_family,
            ):
                current_group.append(row)
            else:
                flush_group()
                current_group = [row]

        flush_group()

    events.sort(key=lambda e: (e.time_start, e.note_token))
    return events


# ============================================================
# WRITE
# ============================================================

def write_events_csv(path: Path, events: List[DetectedNoteEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "note_index",
        "note_token",
        "time_start",
        "time_end",
        "duration",
        "segment_start",
        "segment_end",
        "event_count",
        "representative_rc_hz_mean",
        "representative_rc_energy_mean",
        "best_theoretical_root_score_mean",
        "support_hits_mean",
        "spiral_match_count_mean",
        "spiral_consistency_score_mean",
        "window_chain_match_score_mean",
        "stabilization_score_mean",
        "theoretical_chain_verdict_mode",
        "stabilization_role_mode",
        "best_theoretical_chain_string_mode",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in events:
            writer.writerow(
                {
                    "note_index": ev.note_index,
                    "note_token": ev.note_token,
                    "time_start": ev.time_start,
                    "time_end": ev.time_end,
                    "duration": ev.duration,
                    "segment_start": ev.segment_start,
                    "segment_end": ev.segment_end,
                    "event_count": ev.event_count,
                    "representative_rc_hz_mean": ev.representative_rc_hz_mean,
                    "representative_rc_energy_mean": ev.representative_rc_energy_mean,
                    "best_theoretical_root_score_mean": ev.best_theoretical_root_score_mean,
                    "support_hits_mean": ev.support_hits_mean,
                    "spiral_match_count_mean": ev.spiral_match_count_mean,
                    "spiral_consistency_score_mean": ev.spiral_consistency_score_mean,
                    "window_chain_match_score_mean": ev.window_chain_match_score_mean,
                    "stabilization_score_mean": ev.stabilization_score_mean,
                    "theoretical_chain_verdict_mode": ev.theoretical_chain_verdict_mode,
                    "stabilization_role_mode": ev.stabilization_role_mode,
                    "best_theoretical_chain_string_mode": ev.best_theoretical_chain_string_mode,
                }
            )


def write_meta_json(
    path: Path,
    *,
    input_csv: Path,
    output_csv: Path,
    event_count: int,
    max_gap_sec: float,
    min_event_count: int,
    require_same_verdict_family: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "inputs": {
            "stabilized_chain_candidates_csv": str(input_csv),
        },
        "outputs": {
            "notes_csv": str(output_csv),
            "meta_json": str(path),
        },
        "params": {
            "max_gap_sec": max_gap_sec,
            "min_event_count": min_event_count,
            "require_same_verdict_family": require_same_verdict_family,
        },
        "event_count": event_count,
        "semantic_note": (
            "Detected note events are built from stabilized chain candidates, "
            "not from strongest matrix probes. "
            "This version supports polyphony by grouping per note token and merging over time."
        ),
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build note events from stabilized chain candidates. "
            "This version does not detect notes directly from matrix peaks."
        )
    )
    ap.add_argument("--stabilized_csv", required=True)
    ap.add_argument("--out_notes_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)

    ap.add_argument("--max_gap_sec", type=float, default=0.05)
    ap.add_argument("--min_event_count", type=int, default=1)
    ap.add_argument("--require_same_verdict_family", action="store_true")

    args = ap.parse_args()

    stabilized_csv = Path(args.stabilized_csv).resolve()
    out_notes_csv = Path(args.out_notes_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    rows = load_stabilized_rows(stabilized_csv)
    events = build_note_events(
        rows,
        max_gap_sec=args.max_gap_sec,
        min_event_count=max(1, args.min_event_count),
        require_same_verdict_family=bool(args.require_same_verdict_family),
    )

    write_events_csv(out_notes_csv, events)
    write_meta_json(
        out_meta_json,
        input_csv=stabilized_csv,
        output_csv=out_notes_csv,
        event_count=len(events),
        max_gap_sec=args.max_gap_sec,
        min_event_count=max(1, args.min_event_count),
        require_same_verdict_family=bool(args.require_same_verdict_family),
    )

    print("spiral note event build complete")
    print(json.dumps(
        {
            "event_count": len(events),
            "out_notes_csv": str(out_notes_csv),
            "out_meta_json": str(out_meta_json),
            "max_gap_sec": args.max_gap_sec,
            "min_event_count": args.min_event_count,
            "require_same_verdict_family": bool(args.require_same_verdict_family),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()