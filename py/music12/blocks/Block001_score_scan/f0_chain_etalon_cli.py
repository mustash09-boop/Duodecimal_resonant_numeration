from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from music12.core.notation12 import hz_to_token, token_to_abs_semitone_index


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@dataclass
class MidiEvent:
    event_index: int
    track_index: int
    midi_note: int
    velocity: int
    time_start_sec: float
    time_end_sec: float
    duration_sec: float
    start_frame60: int
    end_frame60: int
    duration_frames60: int
    note_token: str
    octave_token: str
    step_token: str
    onset_group: int
    onset_polyphony: int
    onset_notes: str

    @property
    def abs_pitch(self) -> int:
        return token_to_abs_semitone_index(self.note_token)


@dataclass
class Chain:
    chain_id: int
    events: List[MidiEvent]

    @property
    def last_event(self) -> MidiEvent:
        return self.events[-1]

    def append(self, ev: MidiEvent) -> None:
        self.events.append(ev)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load_midi_events_csv(path: Path) -> List[MidiEvent]:
    events: List[MidiEvent] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "event_index",
            "track_index",
            "midi_note",
            "velocity",
            "time_start_sec",
            "time_end_sec",
            "duration_sec",
            "start_frame60",
            "end_frame60",
            "duration_frames60",
            "note_token",
            "octave_token",
            "step_token",
            "onset_group",
            "onset_polyphony",
            "onset_notes",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for r in reader:
            events.append(
                MidiEvent(
                    event_index=int(r["event_index"]),
                    track_index=int(r["track_index"]),
                    midi_note=int(r["midi_note"]),
                    velocity=int(r["velocity"]),
                    time_start_sec=float(r["time_start_sec"]),
                    time_end_sec=float(r["time_end_sec"]),
                    duration_sec=float(r["duration_sec"]),
                    start_frame60=int(r["start_frame60"]),
                    end_frame60=int(r["end_frame60"]),
                    duration_frames60=int(r["duration_frames60"]),
                    note_token=r["note_token"],
                    octave_token=r["octave_token"],
                    step_token=r["step_token"],
                    onset_group=int(r["onset_group"]),
                    onset_polyphony=int(r["onset_polyphony"]),
                    onset_notes=r["onset_notes"],
                )
            )

    events.sort(key=lambda e: (e.time_start_sec, e.abs_pitch, e.time_end_sec))
    return events


# ------------------------------------------------------------
# CHAIN BUILDING
# ------------------------------------------------------------

def _chain_link_cost(
    prev_event: MidiEvent,
    next_event: MidiEvent,
    *,
    max_gap_frames60: int,
    overlap_tolerance_frames60: int,
    pitch_weight: float,
    time_weight: float,
    same_onset_forbidden: bool,
) -> Optional[float]:
    if same_onset_forbidden and prev_event.onset_group == next_event.onset_group:
        return None

    gap = next_event.start_frame60 - prev_event.end_frame60

    if gap > max_gap_frames60:
        return None

    if next_event.start_frame60 < prev_event.end_frame60 - overlap_tolerance_frames60:
        return None

    pitch_delta = abs(next_event.abs_pitch - prev_event.abs_pitch)
    time_cost = max(0, gap) * time_weight
    pitch_cost = pitch_delta * pitch_weight

    return pitch_cost + time_cost


def build_f0_chains(
    events: List[MidiEvent],
    *,
    max_gap_frames60: int = 6,
    overlap_tolerance_frames60: int = 1,
    pitch_weight: float = 1.0,
    time_weight: float = 0.35,
    same_onset_forbidden: bool = True,
) -> List[Chain]:
    chains: List[Chain] = []

    grouped: Dict[int, List[MidiEvent]] = {}
    for ev in events:
        grouped.setdefault(ev.onset_group, []).append(ev)

    onset_ids = sorted(grouped.keys())

    for onset_id in onset_ids:
        onset_events = sorted(grouped[onset_id], key=lambda e: e.abs_pitch)
        used_chain_ids: set[int] = set()

        for ev in onset_events:
            best_chain: Optional[Chain] = None
            best_cost: Optional[float] = None

            for ch in chains:
                if ch.chain_id in used_chain_ids:
                    continue

                cost = _chain_link_cost(
                    prev_event=ch.last_event,
                    next_event=ev,
                    max_gap_frames60=max_gap_frames60,
                    overlap_tolerance_frames60=overlap_tolerance_frames60,
                    pitch_weight=pitch_weight,
                    time_weight=time_weight,
                    same_onset_forbidden=same_onset_forbidden,
                )

                if cost is None:
                    continue

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_chain = ch

            if best_chain is not None:
                best_chain.append(ev)
                used_chain_ids.add(best_chain.chain_id)
            else:
                new_chain = Chain(chain_id=len(chains), events=[ev])
                chains.append(new_chain)
                used_chain_ids.add(new_chain.chain_id)

    return chains


# ------------------------------------------------------------
# HARMONICS
# ------------------------------------------------------------

def midi_note_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def build_readable_harmonic_rows(
    chains: List[Chain],
    *,
    harmonic_count: int = 12,
) -> List[dict]:
    rows: List[dict] = []

    for ch in chains:
        for segment_no, ev in enumerate(ch.events):
            f0_hz = midi_note_to_hz(ev.midi_note)

            row = {
                "chain_id": ch.chain_id,
                "segment_no": segment_no,
                "event_index": ev.event_index,
                "time_start_sec": ev.time_start_sec,
                "time_end_sec": ev.time_end_sec,
                "duration_sec": ev.duration_sec,
                "start_frame60": ev.start_frame60,
                "end_frame60": ev.end_frame60,
                "duration_frames60": ev.duration_frames60,
                "onset_group": ev.onset_group,
                "onset_polyphony": ev.onset_polyphony,
                "f0_note": ev.note_token,
                "f0_hz": f0_hz,
            }

            for harmonic_index in range(1, harmonic_count + 1):
                hz = f0_hz * harmonic_index
                harmonic_token = hz_to_token(
                    hz,
                    a4_hz=440.0,
                    anchor_token="9.A",
                    micro_depth=2,
                    force_micro_dash_when_exact=True,
                )
                row[f"h{harmonic_index}_note"] = harmonic_token
                row[f"h{harmonic_index}_hz"] = hz

            rows.append(row)

    return rows


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


def write_chain_summary_csv(path: Path, chains: List[Chain]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chain_id",
                "event_count",
                "time_start_sec",
                "time_end_sec",
                "duration_sec",
                "first_note",
                "last_note",
                "min_pitch_abs",
                "max_pitch_abs",
            ]
        )

        for ch in chains:
            first_ev = ch.events[0]
            last_ev = ch.events[-1]
            abs_values = [ev.abs_pitch for ev in ch.events]

            writer.writerow(
                [
                    ch.chain_id,
                    len(ch.events),
                    first_ev.time_start_sec,
                    last_ev.time_end_sec,
                    last_ev.time_end_sec - first_ev.time_start_sec,
                    first_ev.note_token,
                    last_ev.note_token,
                    min(abs_values),
                    max(abs_values),
                ]
            )


def write_meta(
    path: Path,
    *,
    input_events_csv: Path,
    chains: List[Chain],
    readable_rows: List[dict],
    args: argparse.Namespace,
) -> None:
    data = {
        "input_events_csv": str(input_events_csv),
        "chain_builder": {
            "max_gap_frames60": args.max_gap_frames60,
            "overlap_tolerance_frames60": args.overlap_tolerance_frames60,
            "pitch_weight": args.pitch_weight,
            "time_weight": args.time_weight,
            "same_onset_forbidden": True,
        },
        "harmonics": {
            "harmonic_count": args.harmonic_count,
            "anchor_note": "9.A",
            "anchor_hz": 440.0,
        },
        "derived": {
            "chain_count": len(chains),
            "readable_row_count": len(readable_rows),
        },
        "outputs": {
            "readable_csv": str(Path(args.out_readable_csv).resolve()),
            "summary_csv": str(Path(args.out_chain_summary_csv).resolve()),
            "meta_json": str(Path(args.out_meta_json).resolve()),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build readable etalon f0 chains with harmonic columns from MIDI events CSV."
    )

    ap.add_argument("--events_csv", required=True, help="Input midi_events_12.csv")
    ap.add_argument("--out_readable_csv", required=True, help="Readable CSV: one row = one time segment with h1..hN")
    ap.add_argument("--out_chain_summary_csv", required=True, help="Output CSV for chain summary")
    ap.add_argument("--out_meta_json", required=True, help="Output JSON metadata")

    ap.add_argument("--max_gap_frames60", type=int, default=6)
    ap.add_argument("--overlap_tolerance_frames60", type=int, default=1)
    ap.add_argument("--pitch_weight", type=float, default=1.0)
    ap.add_argument("--time_weight", type=float, default=0.35)
    ap.add_argument("--harmonic_count", type=int, default=12)

    args = ap.parse_args()

    events_csv = Path(args.events_csv).resolve()
    events = load_midi_events_csv(events_csv)

    chains = build_f0_chains(
        events,
        max_gap_frames60=args.max_gap_frames60,
        overlap_tolerance_frames60=args.overlap_tolerance_frames60,
        pitch_weight=args.pitch_weight,
        time_weight=args.time_weight,
        same_onset_forbidden=True,
    )

    readable_rows = build_readable_harmonic_rows(
        chains,
        harmonic_count=args.harmonic_count,
    )

    readable_fields = [
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
    ]

    for harmonic_index in range(1, args.harmonic_count + 1):
        readable_fields.append(f"h{harmonic_index}_note")
        readable_fields.append(f"h{harmonic_index}_hz")

    write_csv(
        Path(args.out_readable_csv).resolve(),
        readable_rows,
        readable_fields,
    )

    write_chain_summary_csv(
        Path(args.out_chain_summary_csv).resolve(),
        chains,
    )

    write_meta(
        Path(args.out_meta_json).resolve(),
        input_events_csv=events_csv,
        chains=chains,
        readable_rows=readable_rows,
        args=args,
    )

    print("readable f0 etalon chain build complete")
    print(json.dumps(
        {
            "chain_count": len(chains),
            "readable_row_count": len(readable_rows),
            "out_readable_csv": str(Path(args.out_readable_csv).resolve()),
            "out_chain_summary_csv": str(Path(args.out_chain_summary_csv).resolve()),
            "out_meta_json": str(Path(args.out_meta_json).resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()