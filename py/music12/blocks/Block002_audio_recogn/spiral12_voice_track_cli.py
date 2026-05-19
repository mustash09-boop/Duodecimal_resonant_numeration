from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# CHAIN-BASED VOICE TRACKING
# ------------------------------------------------------------
# Input:
#   note events built from stabilized chain candidates
#
# Principle:
#   voice continuity is defined primarily by chain identity,
#   not by nearest geometric distance.
#
# Priority of linking:
#   1. same best_theoretical_chain_string_mode
#   2. same note_token
#   3. similar verdict family / stabilization role
#   4. temporal continuity
#
# Spiral/geometric closeness is intentionally NOT the main driver here.
# ============================================================


# ============================================================
# DATA
# ============================================================

@dataclass(frozen=True)
class NoteEvent:
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


@dataclass
class Voice:
    voice_id: int
    events: List[NoteEvent]

    @property
    def last_event(self) -> NoteEvent:
        return self.events[-1]

    def append(self, ev: NoteEvent) -> None:
        self.events.append(ev)


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


def _verdict_family(verdict: str) -> str:
    verdict = _safe_str(verdict)
    if not verdict:
        return ""
    parts = verdict.split("_")
    if not parts:
        return verdict
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return verdict


def _chain_identity(ev: NoteEvent) -> Tuple[str, str]:
    """
    Primary chain identity:
      (chain_string_mode, note_token)

    chain string is stronger than note token.
    """
    return (
        _safe_str(ev.best_theoretical_chain_string_mode),
        _safe_str(ev.note_token),
    )


def _time_gap(prev: NoteEvent, nxt: NoteEvent) -> float:
    return float(nxt.time_start - prev.time_end)


def _time_overlap(prev: NoteEvent, nxt: NoteEvent) -> float:
    return max(0.0, min(prev.time_end, nxt.time_end) - max(prev.time_start, nxt.time_start))


def _normalized_gap_score(gap: float, max_time_gap: float) -> float:
    if gap < 0:
        return 0.0
    if max_time_gap <= 0:
        return 0.0
    x = 1.0 - (gap / max_time_gap)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ============================================================
# LOAD
# ============================================================

def load_note_events_csv(path: Path) -> List[NoteEvent]:
    events: List[NoteEvent] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
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
            events.append(
                NoteEvent(
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

    events.sort(key=lambda e: (e.time_start, e.note_token, e.note_index))
    return events


# ============================================================
# LINKING
# ============================================================

def _voice_link_score(
    prev_event: NoteEvent,
    next_event: NoteEvent,
    *,
    max_time_gap: float,
    allow_overlap: bool,
) -> Optional[float]:
    gap = _time_gap(prev_event, next_event)
    overlap = _time_overlap(prev_event, next_event)

    if gap > max_time_gap:
        return None

    if (not allow_overlap) and overlap > 0.0:
        return None

    score = 0.0

    prev_chain, prev_note = _chain_identity(prev_event)
    next_chain, next_note = _chain_identity(next_event)

    # 1. chain identity is primary
    if prev_chain and next_chain and prev_chain == next_chain:
        score += 6.0

    # 2. same note token is still important
    if prev_note and next_note and prev_note == next_note:
        score += 3.0

    # 3. verdict family continuity
    if _verdict_family(prev_event.theoretical_chain_verdict_mode) == _verdict_family(next_event.theoretical_chain_verdict_mode):
        score += 1.5

    # 4. stabilization role continuity
    if prev_event.stabilization_role_mode == next_event.stabilization_role_mode and prev_event.stabilization_role_mode:
        score += 0.8

    # 5. temporal continuity
    if gap >= 0.0:
        score += 2.0 * _normalized_gap_score(gap, max_time_gap)

    # 6. overlap is allowed but mildly penalized
    if overlap > 0.0:
        score -= min(1.5, overlap * 10.0)

    # 7. soft similarity in support/stability
    score += max(
        0.0,
        1.0 - abs(prev_event.spiral_consistency_score_mean - next_event.spiral_consistency_score_mean),
    )
    score += max(
        0.0,
        1.0 - abs(prev_event.window_chain_match_score_mean - next_event.window_chain_match_score_mean),
    ) * 0.8

    # 8. ambiguity penalty
    root_score_gap = abs(prev_event.best_theoretical_root_score_mean - next_event.best_theoretical_root_score_mean)
    score -= min(1.0, root_score_gap * 0.2)

    return score


def assign_events_to_voices(
    events: List[NoteEvent],
    *,
    max_time_gap: float,
    allow_overlap: bool,
    min_link_score: float,
    max_voices_hint: int,
) -> List[Voice]:
    voices: List[Voice] = []

    for ev in events:
        best_voice: Optional[Voice] = None
        best_score: Optional[float] = None

        for voice in voices:
            prev = voice.last_event

            score = _voice_link_score(
                prev_event=prev,
                next_event=ev,
                max_time_gap=max_time_gap,
                allow_overlap=allow_overlap,
            )

            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_voice = voice

        if best_voice is not None and best_score is not None and best_score >= min_link_score:
            best_voice.append(ev)
        else:
            voices.append(
                Voice(
                    voice_id=len(voices),
                    events=[ev],
                )
            )

    return voices


# ============================================================
# WRITE
# ============================================================

def write_voice_events_csv(path: Path, voices: List[Voice]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for v in voices:
            for ev in v.events:
                writer.writerow(
                    {
                        "voice_id": v.voice_id,
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


def write_voice_summary_csv(path: Path, voices: List[Voice]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "voice_id",
        "event_count",
        "time_start",
        "time_end",
        "duration",
        "unique_note_count",
        "dominant_note_token",
        "dominant_chain_string",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for v in voices:
            t0 = v.events[0].time_start
            t1 = v.events[-1].time_end

            note_counts: Dict[str, int] = {}
            chain_counts: Dict[str, int] = {}

            for ev in v.events:
                note_counts[ev.note_token] = note_counts.get(ev.note_token, 0) + 1
                chain = _safe_str(ev.best_theoretical_chain_string_mode)
                if chain:
                    chain_counts[chain] = chain_counts.get(chain, 0) + 1

            dominant_note = max(note_counts.items(), key=lambda x: x[1])[0] if note_counts else ""
            dominant_chain = max(chain_counts.items(), key=lambda x: x[1])[0] if chain_counts else ""

            writer.writerow(
                {
                    "voice_id": v.voice_id,
                    "event_count": len(v.events),
                    "time_start": t0,
                    "time_end": t1,
                    "duration": max(0.0, t1 - t0),
                    "unique_note_count": len(note_counts),
                    "dominant_note_token": dominant_note,
                    "dominant_chain_string": dominant_chain,
                }
            )


def write_meta_json(
    path: Path,
    *,
    input_event_count: int,
    voices: List[Voice],
    args,
) -> None:
    data = {
        "input_event_count": input_event_count,
        "voice_tracking": {
            "max_time_gap": args.max_time_gap,
            "allow_overlap": bool(args.allow_overlap),
            "min_link_score": args.min_link_score,
            "max_voices_hint": args.max_voices_hint,
        },
        "derived": {
            "voice_count": len(voices),
            "assigned_event_count": sum(len(v.events) for v in voices),
        },
        "semantic_note": (
            "Chain-based voice tracking. "
            "Voice continuity is defined primarily by chain identity and temporal continuity, "
            "not by geometric nearest-neighbor alone."
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track voices from note events using chain-based continuity"
    )

    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--out_voice_events_csv", required=True)
    ap.add_argument("--out_voice_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)

    ap.add_argument("--max_time_gap", type=float, default=0.12)
    ap.add_argument("--allow_overlap", action="store_true")
    ap.add_argument("--min_link_score", type=float, default=3.0)
    ap.add_argument("--max_voices_hint", type=int, default=8)

    args = ap.parse_args()

    events = load_note_events_csv(Path(args.events_csv))

    voices = assign_events_to_voices(
        events,
        max_time_gap=args.max_time_gap,
        allow_overlap=bool(args.allow_overlap),
        min_link_score=args.min_link_score,
        max_voices_hint=args.max_voices_hint,
    )

    write_voice_events_csv(Path(args.out_voice_events_csv), voices)
    write_voice_summary_csv(Path(args.out_voice_summary_csv), voices)
    write_meta_json(
        Path(args.out_meta_json),
        input_event_count=len(events),
        voices=voices,
        args=args,
    )

    print("chain-based voice tracking complete")
    print(json.dumps(
        {
            "input_event_count": len(events),
            "voice_count": len(voices),
            "assigned_event_count": sum(len(v.events) for v in voices),
            "out_voice_events_csv": str(Path(args.out_voice_events_csv).resolve()),
            "out_voice_summary_csv": str(Path(args.out_voice_summary_csv).resolve()),
            "out_meta_json": str(Path(args.out_meta_json).resolve()),
            "max_time_gap": args.max_time_gap,
            "allow_overlap": bool(args.allow_overlap),
            "min_link_score": args.min_link_score,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()