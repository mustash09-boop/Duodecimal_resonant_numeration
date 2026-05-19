# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mido


GM_PROGRAM_NAMES = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavi",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "SynthStrings 1", "SynthStrings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "SynthBrass 1", "SynthBrass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bag pipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _program_name(program: int) -> str:
    if 0 <= program < len(GM_PROGRAM_NAMES):
        return GM_PROGRAM_NAMES[program]
    return ""


def _collect_midi_track_info(midi_path: Path) -> Dict[str, Dict[str, Any]]:
    mid = mido.MidiFile(midi_path)
    out: Dict[str, Dict[str, Any]] = {}
    for i, tr in enumerate(mid.tracks):
        names: List[str] = []
        programs: List[tuple[int, int]] = []
        channels = set()
        note_ons = 0
        for msg in tr:
            if msg.type == "track_name":
                names.append(msg.name)
            if hasattr(msg, "channel"):
                channels.add(int(msg.channel))
            if msg.type == "program_change":
                programs.append((int(msg.channel), int(msg.program)))
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                note_ons += 1
        if i == 0 and note_ons == 0 and not names and not programs:
            continue
        unique_programs = []
        seen = set()
        for ch, pr in programs:
            key = (ch, pr)
            if key not in seen:
                seen.add(key)
                unique_programs.append(key)
        primary_program = unique_programs[0][1] if unique_programs else None
        out[str(i)] = {
            "track_name": names[0] if names else "",
            "all_track_names": " | ".join(names),
            "channels": " ".join(str(x) for x in sorted(channels)),
            "primary_channel": sorted(channels)[0] if channels else "",
            "program_changes": " | ".join(f"ch{ch}:{pr}" for ch, pr in unique_programs),
            "primary_program": primary_program if primary_program is not None else "",
            "primary_program_name": _program_name(primary_program) if primary_program is not None else "",
            "midi_note_on_count": note_ons,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich exported MIDI event CSV with track names and program metadata from the source MIDI.")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    midi_path = Path(args.midi)
    events_csv = Path(args.events_csv)
    rows = _load_csv(events_csv)
    track_info = _collect_midi_track_info(midi_path)

    csv_track_counts = Counter(str(r.get("track_index", "")).strip() for r in rows)
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        track_index = str(r.get("track_index", "")).strip()
        info = track_info.get(track_index, {})
        rr = dict(r)
        rr["track_name"] = info.get("track_name", "")
        rr["all_track_names"] = info.get("all_track_names", "")
        rr["midi_channels_for_track"] = info.get("channels", "")
        rr["primary_channel_for_track"] = info.get("primary_channel", "")
        rr["program_changes_for_track"] = info.get("program_changes", "")
        rr["primary_program_for_track"] = info.get("primary_program", "")
        rr["primary_program_name_for_track"] = info.get("primary_program_name", "")
        rr["midi_note_on_count_for_track"] = info.get("midi_note_on_count", "")
        rr["csv_event_count_for_track"] = csv_track_counts.get(track_index, 0)
        rr["track_count_delta"] = csv_track_counts.get(track_index, 0) - int(info.get("midi_note_on_count", 0) or 0)
        out_rows.append(rr)

    out_rows.sort(key=lambda r: int(r.get("event_index", 0) or 0))
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    _write_csv(Path(args.out_csv), out_rows, fieldnames)

    track_lines = []
    for track_index in sorted(set(csv_track_counts) | set(track_info), key=lambda x: int(x) if str(x).isdigit() else 9999):
        info = track_info.get(track_index, {})
        csv_count = csv_track_counts.get(track_index, 0)
        midi_count = int(info.get("midi_note_on_count", 0) or 0)
        track_lines.append(
            f"  track {track_index}: "
            f"name='{info.get('track_name', '')}', "
            f"channel(s)={info.get('channels', '')}, "
            f"program={info.get('primary_program', '')} {info.get('primary_program_name', '')}, "
            f"midi_note_on={midi_count}, csv_rows={csv_count}, delta={csv_count - midi_count}"
        )

    summary_lines = [
        "MIDI EVENTS TRACK METADATA ENRICHMENT",
        "=" * 72,
        f"midi              : {midi_path}",
        f"events_csv        : {events_csv}",
        f"out_csv           : {args.out_csv}",
        "",
        f"csv_rows_total    : {len(rows)}",
        f"midi_tracks_used  : {len(track_info)}",
        f"midi_note_on_total: {sum(int(v.get('midi_note_on_count', 0) or 0) for v in track_info.values())}",
        f"track_count_mismatches: {sum(1 for t in csv_track_counts if csv_track_counts.get(t, 0) != int(track_info.get(t, {}).get('midi_note_on_count', 0) or 0))}",
        "",
        "track_breakdown:",
        *track_lines,
    ]
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {"midi": args.midi, "events_csv": args.events_csv},
                "outputs": {"out_csv": args.out_csv, "out_summary_txt": args.out_summary_txt},
                "result": {
                    "csv_rows_total": len(rows),
                    "midi_tracks_used": len(track_info),
                    "midi_note_on_total": sum(int(v.get("midi_note_on_count", 0) or 0) for v in track_info.values()),
                    "track_count_mismatches": {
                        t: {
                            "midi_note_on_count": int(track_info.get(t, {}).get("midi_note_on_count", 0) or 0),
                            "csv_rows": csv_track_counts.get(t, 0),
                            "delta": csv_track_counts.get(t, 0) - int(track_info.get(t, {}).get("midi_note_on_count", 0) or 0),
                            "track_name": track_info.get(t, {}).get("track_name", ""),
                        }
                        for t in sorted(set(csv_track_counts) | set(track_info), key=lambda x: int(x) if str(x).isdigit() else 9999)
                        if csv_track_counts.get(t, 0) != int(track_info.get(t, {}).get("midi_note_on_count", 0) or 0)
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
