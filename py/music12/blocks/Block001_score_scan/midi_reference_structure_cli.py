from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@dataclass(frozen=True)
class MidiReferenceEvent:
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


@dataclass(frozen=True)
class ReferenceVoiceEvent:
    voice_id: int
    event_index: int

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

    track_index: int
    midi_note: int
    velocity: int


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load_midi_events_csv(path: Path) -> List[MidiReferenceEvent]:
    events: List[MidiReferenceEvent] = []

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
                MidiReferenceEvent(
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

    events.sort(key=lambda e: (e.time_start_sec, e.midi_note, e.time_end_sec, e.event_index))
    return events


# ------------------------------------------------------------
# VOICE-LIKE REFERENCE STRUCTURE
# ------------------------------------------------------------

def _link_cost(
    prev_event: MidiReferenceEvent,
    next_event: MidiReferenceEvent,
    *,
    max_gap_frames60: int,
    overlap_tolerance_frames60: int,
    pitch_weight: float,
    time_weight: float,
    same_onset_forbidden: bool,
) -> float | None:
    if same_onset_forbidden and prev_event.onset_group == next_event.onset_group:
        return None

    gap = next_event.start_frame60 - prev_event.end_frame60

    if gap > max_gap_frames60:
        return None

    if next_event.start_frame60 < prev_event.end_frame60 - overlap_tolerance_frames60:
        return None

    pitch_delta = abs(next_event.midi_note - prev_event.midi_note)
    time_cost = max(0, gap) * time_weight
    pitch_cost = pitch_delta * pitch_weight

    return pitch_cost + time_cost


def build_reference_voices(
    events: List[MidiReferenceEvent],
    *,
    max_gap_frames60: int = 6,
    overlap_tolerance_frames60: int = 1,
    pitch_weight: float = 1.0,
    time_weight: float = 0.35,
    same_onset_forbidden: bool = True,
) -> List[List[MidiReferenceEvent]]:
    voices: List[List[MidiReferenceEvent]] = []

    grouped: Dict[int, List[MidiReferenceEvent]] = {}
    for ev in events:
        grouped.setdefault(ev.onset_group, []).append(ev)

    onset_ids = sorted(grouped.keys())

    for onset_id in onset_ids:
        onset_events = sorted(grouped[onset_id], key=lambda e: e.midi_note)
        used_voice_ids: set[int] = set()

        for ev in onset_events:
            best_voice_id: int | None = None
            best_cost: float | None = None

            for voice_id, voice in enumerate(voices):
                if voice_id in used_voice_ids:
                    continue

                prev_event = voice[-1]
                cost = _link_cost(
                    prev_event,
                    ev,
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
                    best_voice_id = voice_id

            if best_voice_id is None:
                voices.append([ev])
                used_voice_ids.add(len(voices) - 1)
            else:
                voices[best_voice_id].append(ev)
                used_voice_ids.add(best_voice_id)

    return voices


def flatten_reference_voices(voices: List[List[MidiReferenceEvent]]) -> List[ReferenceVoiceEvent]:
    rows: List[ReferenceVoiceEvent] = []

    for voice_id, voice in enumerate(voices):
        for ev in voice:
            rows.append(
                ReferenceVoiceEvent(
                    voice_id=voice_id,
                    event_index=ev.event_index,
                    time_start_sec=ev.time_start_sec,
                    time_end_sec=ev.time_end_sec,
                    duration_sec=ev.duration_sec,
                    start_frame60=ev.start_frame60,
                    end_frame60=ev.end_frame60,
                    duration_frames60=ev.duration_frames60,
                    note_token=ev.note_token,
                    octave_token=ev.octave_token,
                    step_token=ev.step_token,
                    onset_group=ev.onset_group,
                    onset_polyphony=ev.onset_polyphony,
                    onset_notes=ev.onset_notes,
                    track_index=ev.track_index,
                    midi_note=ev.midi_note,
                    velocity=ev.velocity,
                )
            )

    rows.sort(key=lambda r: (r.time_start_sec, r.voice_id, r.midi_note, r.event_index))
    return rows


# ------------------------------------------------------------
# ONSET SUMMARY
# ------------------------------------------------------------

def build_onset_reference_summary(events: List[MidiReferenceEvent]) -> List[dict]:
    grouped: Dict[int, List[MidiReferenceEvent]] = {}
    for ev in events:
        grouped.setdefault(ev.onset_group, []).append(ev)

    rows: List[dict] = []

    for onset_group in sorted(grouped.keys()):
        items = sorted(grouped[onset_group], key=lambda e: e.midi_note)
        rows.append(
            {
                "onset_group": onset_group,
                "time_start_sec": items[0].time_start_sec,
                "start_frame60": items[0].start_frame60,
                "polyphony": items[0].onset_polyphony,
                "note_tokens": " | ".join(ev.note_token for ev in items),
                "event_indices": " | ".join(str(ev.event_index) for ev in items),
                "track_indices": " | ".join(str(ev.track_index) for ev in items),
            }
        )

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


def write_voice_summary_csv(path: Path, voices: List[List[MidiReferenceEvent]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "voice_id",
                "event_count",
                "time_start_sec",
                "time_end_sec",
                "duration_sec",
                "first_note",
                "last_note",
                "min_midi_note",
                "max_midi_note",
            ]
        )

        for voice_id, voice in enumerate(voices):
            first_ev = voice[0]
            last_ev = voice[-1]
            midi_values = [ev.midi_note for ev in voice]

            writer.writerow(
                [
                    voice_id,
                    len(voice),
                    first_ev.time_start_sec,
                    last_ev.time_end_sec,
                    last_ev.time_end_sec - first_ev.time_start_sec,
                    first_ev.note_token,
                    last_ev.note_token,
                    min(midi_values),
                    max(midi_values),
                ]
            )


def write_meta(
    path: Path,
    *,
    input_events_csv: Path,
    events: List[MidiReferenceEvent],
    voices: List[List[MidiReferenceEvent]],
    onset_rows: List[dict],
    args: argparse.Namespace,
) -> None:
    data = {
        "input_events_csv": str(input_events_csv),
        "reference_structure": {
            "max_gap_frames60": args.max_gap_frames60,
            "overlap_tolerance_frames60": args.overlap_tolerance_frames60,
            "pitch_weight": args.pitch_weight,
            "time_weight": args.time_weight,
            "same_onset_forbidden": True,
            "semantic_role": "reference_structure_only_not_audio_inference",
        },
        "derived": {
            "event_count": len(events),
            "voice_count": len(voices),
            "onset_group_count": len(onset_rows),
            "max_polyphony": max((row["polyphony"] for row in onset_rows), default=0),
        },
        "outputs": {
            "reference_voice_events_csv": str(Path(args.out_reference_voice_events_csv).resolve()),
            "reference_voice_summary_csv": str(Path(args.out_reference_voice_summary_csv).resolve()),
            "reference_onsets_csv": str(Path(args.out_reference_onsets_csv).resolve()),
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
        description=(
            "Build reference structure from MIDI events in 12-radix notation. "
            "This stage does NOT infer f0 and does NOT generate harmonic hypotheses."
        )
    )

    ap.add_argument("--events_csv", required=True, help="Input midi_events_12.csv")
    ap.add_argument("--out_reference_voice_events_csv", required=True)
    ap.add_argument("--out_reference_voice_summary_csv", required=True)
    ap.add_argument("--out_reference_onsets_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)

    ap.add_argument("--max_gap_frames60", type=int, default=6)
    ap.add_argument("--overlap_tolerance_frames60", type=int, default=1)
    ap.add_argument("--pitch_weight", type=float, default=1.0)
    ap.add_argument("--time_weight", type=float, default=0.35)

    args = ap.parse_args()

    events_csv = Path(args.events_csv).resolve()
    events = load_midi_events_csv(events_csv)

    voices = build_reference_voices(
        events,
        max_gap_frames60=args.max_gap_frames60,
        overlap_tolerance_frames60=args.overlap_tolerance_frames60,
        pitch_weight=args.pitch_weight,
        time_weight=args.time_weight,
        same_onset_forbidden=True,
    )

    voice_rows = flatten_reference_voices(voices)
    onset_rows = build_onset_reference_summary(events)

    voice_fields = [
        "voice_id",
        "event_index",
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
        "track_index",
        "midi_note",
        "velocity",
    ]

    onset_fields = [
        "onset_group",
        "time_start_sec",
        "start_frame60",
        "polyphony",
        "note_tokens",
        "event_indices",
        "track_indices",
    ]

    write_csv(
        Path(args.out_reference_voice_events_csv).resolve(),
        [row.__dict__ for row in voice_rows],
        voice_fields,
    )

    write_voice_summary_csv(
        Path(args.out_reference_voice_summary_csv).resolve(),
        voices,
    )

    write_csv(
        Path(args.out_reference_onsets_csv).resolve(),
        onset_rows,
        onset_fields,
    )

    write_meta(
        Path(args.out_meta_json).resolve(),
        input_events_csv=events_csv,
        events=events,
        voices=voices,
        onset_rows=onset_rows,
        args=args,
    )

    print("midi reference structure build complete")
    print(json.dumps(
        {
            "event_count": len(events),
            "voice_count": len(voices),
            "onset_group_count": len(onset_rows),
            "out_reference_voice_events_csv": str(Path(args.out_reference_voice_events_csv).resolve()),
            "out_reference_voice_summary_csv": str(Path(args.out_reference_voice_summary_csv).resolve()),
            "out_reference_onsets_csv": str(Path(args.out_reference_onsets_csv).resolve()),
            "out_meta_json": str(Path(args.out_meta_json).resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()