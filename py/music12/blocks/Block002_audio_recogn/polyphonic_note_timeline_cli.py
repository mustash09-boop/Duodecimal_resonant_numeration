from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


MAX_SIMULTANEOUS_NOTES = 8


# ============================================================
# HELPERS
# ============================================================

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(value)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        s = _safe_str(value)
        if s == "":
            return default
        return int(s)
    except Exception:
        return default


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


# ============================================================
# DATA
# ============================================================

@dataclass(frozen=True)
class VoiceEvent:
    voice_id: int
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

def load_voice_events_csv(path: Path) -> List[VoiceEvent]:
    rows: List[VoiceEvent] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "voice_id",
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
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for r in reader:
            rows.append(
                VoiceEvent(
                    voice_id=_safe_int(r.get("voice_id", ""), 0),
                    note_index=_safe_int(r.get("note_index", ""), 0),
                    note_token=_safe_str(r.get("note_token", "")),

                    time_start=_safe_float(r.get("time_start", ""), 0.0),
                    time_end=_safe_float(r.get("time_end", ""), 0.0),
                    duration=_safe_float(r.get("duration", ""), 0.0),

                    segment_start=_safe_int(r.get("segment_start", ""), 0),
                    segment_end=_safe_int(r.get("segment_end", ""), 0),
                    event_count=_safe_int(r.get("event_count", ""), 0),

                    representative_rc_hz_mean=_safe_float(r.get("representative_rc_hz_mean", ""), 0.0),
                    representative_rc_energy_mean=_safe_float(r.get("representative_rc_energy_mean", ""), 0.0),

                    best_theoretical_root_score_mean=_safe_float(r.get("best_theoretical_root_score_mean", ""), 0.0),
                    support_hits_mean=_safe_float(r.get("support_hits_mean", ""), 0.0),
                    spiral_match_count_mean=_safe_float(r.get("spiral_match_count_mean", ""), 0.0),
                    spiral_consistency_score_mean=_safe_float(r.get("spiral_consistency_score_mean", ""), 0.0),
                    window_chain_match_score_mean=_safe_float(r.get("window_chain_match_score_mean", ""), 0.0),
                    stabilization_score_mean=_safe_float(r.get("stabilization_score_mean", ""), 0.0),

                    theoretical_chain_verdict_mode=_safe_str(r.get("theoretical_chain_verdict_mode", "")),
                    stabilization_role_mode=_safe_str(r.get("stabilization_role_mode", "")),
                    best_theoretical_chain_string_mode=_safe_str(r.get("best_theoretical_chain_string_mode", "")),
                )
            )

    rows.sort(key=lambda r: (r.time_start, r.voice_id, r.note_index))
    return rows


# ============================================================
# TIMELINE WINDOWS
# ============================================================

def _build_time_grid(
    events: List[VoiceEvent],
    *,
    step_sec: float,
) -> List[tuple[float, float]]:
    if not events:
        return []

    t0 = min(ev.time_start for ev in events)
    t1 = max(ev.time_end for ev in events)

    if step_sec <= 0:
        raise ValueError("step_sec must be > 0")

    windows: List[tuple[float, float]] = []

    cur = t0
    while cur < t1:
        nxt = cur + step_sec
        windows.append((round(cur, 6), round(nxt, 6)))
        cur = nxt

    return windows


def _event_overlaps_window(ev: VoiceEvent, w0: float, w1: float) -> bool:
    return ev.time_start < w1 and ev.time_end > w0


def _overlap_duration(ev: VoiceEvent, w0: float, w1: float) -> float:
    left = max(ev.time_start, w0)
    right = min(ev.time_end, w1)
    return max(0.0, right - left)


def _aggregate_window(
    events: List[VoiceEvent],
    w0: float,
    w1: float,
) -> dict[str, Any]:
    active = [ev for ev in events if _event_overlaps_window(ev, w0, w1)]

    # one note per voice inside a window: keep strongest overlap/stability representative
    by_voice: Dict[int, List[VoiceEvent]] = defaultdict(list)
    for ev in active:
        by_voice[ev.voice_id].append(ev)

    selected: List[dict[str, Any]] = []

    for voice_id, voice_events in by_voice.items():
        voice_events.sort(
            key=lambda ev: (
                _overlap_duration(ev, w0, w1),
                ev.stabilization_score_mean,
                ev.best_theoretical_root_score_mean,
                ev.spiral_consistency_score_mean,
            ),
            reverse=True,
        )
        ev = voice_events[0]

        selected.append(
            {
                "voice_id": voice_id,
                "note_token": ev.note_token,
                "best_theoretical_chain_string_mode": ev.best_theoretical_chain_string_mode,
                "theoretical_chain_verdict_mode": ev.theoretical_chain_verdict_mode,
                "stabilization_role_mode": ev.stabilization_role_mode,
                "representative_rc_hz_mean": round(ev.representative_rc_hz_mean, 6),
                "representative_rc_energy_mean": round(ev.representative_rc_energy_mean, 9),
                "best_theoretical_root_score_mean": round(ev.best_theoretical_root_score_mean, 6),
                "support_hits_mean": round(ev.support_hits_mean, 6),
                "spiral_match_count_mean": round(ev.spiral_match_count_mean, 6),
                "spiral_consistency_score_mean": round(ev.spiral_consistency_score_mean, 6),
                "window_chain_match_score_mean": round(ev.window_chain_match_score_mean, 6),
                "stabilization_score_mean": round(ev.stabilization_score_mean, 6),
                "overlap_duration": round(_overlap_duration(ev, w0, w1), 6),
            }
        )

    # sort for readable top-N display only, not to redefine truth
    selected.sort(
        key=lambda x: (
            x["stabilization_score_mean"],
            x["best_theoretical_root_score_mean"],
            x["spiral_consistency_score_mean"],
            x["overlap_duration"],
            -x["voice_id"],
        ),
        reverse=True,
    )

    top_notes = selected[:MAX_SIMULTANEOUS_NOTES]

    return {
        "active_note_count_raw": len(selected),
        "active_note_count_capped": len(top_notes),
        "top_notes": top_notes,
    }


# ============================================================
# WRITE
# ============================================================

def write_csv(path: Path, rows: List[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_meta_json(
    path: Path,
    *,
    input_csv: Path,
    output_csv: Path,
    output_readable_csv: Path,
    rows_in_input: int,
    window_count: int,
    step_sec: float,
    max_simultaneous_notes_observed: int,
    histogram: dict[str, int],
    sample_windows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "input_csv": str(input_csv),
        "rows_in_input": rows_in_input,
        "window_count": window_count,
        "timeline_step_sec": step_sec,
        "max_simultaneous_notes_observed": max_simultaneous_notes_observed,
        "max_simultaneous_notes_cap": MAX_SIMULTANEOUS_NOTES,
        "active_note_count_histogram": histogram,
        "output_timeline_csv": str(output_csv),
        "output_timeline_readable_csv": str(output_readable_csv),
        "sample_windows": sample_windows[:20],
        "semantic_note": (
            "Voice-aware polyphonic timeline. "
            "This file projects already tracked voice events onto fixed time windows. "
            "It does not re-score, re-rank, or reinterpret stabilized chain logic."
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
        description="Build voice-aware polyphonic timeline from voice events"
    )
    ap.add_argument("--voice_events_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--step_sec", type=float, default=0.05)

    args = ap.parse_args()

    input_csv = Path(args.voice_events_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_readable_csv = Path(args.out_readable_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    events = load_voice_events_csv(input_csv)
    windows = _build_time_grid(events, step_sec=float(args.step_sec))

    timeline_rows: List[dict[str, Any]] = []
    readable_rows: List[dict[str, Any]] = []

    for (w0, w1) in windows:
        agg = _aggregate_window(events, w0, w1)
        top_notes = agg["top_notes"]

        timeline_rows.append(
            {
                "window_start_sec": w0,
                "window_end_sec": w1,
                "active_note_count_raw": agg["active_note_count_raw"],
                "active_note_count_capped": agg["active_note_count_capped"],
                "active_notes_json": json.dumps(top_notes, ensure_ascii=False),
            }
        )

        readable_rows.append(
            {
                "window_start_sec": w0,
                "window_end_sec": w1,
                "active_note_count": agg["active_note_count_capped"],
                "top_8_notes": " | ".join(n["note_token"] for n in top_notes),
                "top_8_voices": " | ".join(str(n["voice_id"]) for n in top_notes),
                "top_8_scores": " | ".join(
                    f'{n["note_token"]}:S={n["stabilization_score_mean"]:.2f}'
                    for n in top_notes
                ),
            }
        )

    write_csv(out_csv, timeline_rows)
    write_csv(out_readable_csv, readable_rows)

    active_counts = [int(r["active_note_count"]) for r in readable_rows]
    max_active = max(active_counts) if active_counts else 0

    histogram: Dict[str, int] = {}
    for count in active_counts:
        histogram[str(count)] = histogram.get(str(count), 0) + 1

    write_meta_json(
        out_meta_json,
        input_csv=input_csv,
        output_csv=out_csv,
        output_readable_csv=out_readable_csv,
        rows_in_input=len(events),
        window_count=len(readable_rows),
        step_sec=float(args.step_sec),
        max_simultaneous_notes_observed=max_active,
        histogram=histogram,
        sample_windows=readable_rows,
    )

    print("voice-aware polyphonic timeline complete")
    print(json.dumps(
        {
            "rows_in_input": len(events),
            "window_count": len(readable_rows),
            "max_simultaneous_notes_observed": max_active,
            "step_sec": float(args.step_sec),
            "out_csv": str(out_csv),
            "out_readable_csv": str(out_readable_csv),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()