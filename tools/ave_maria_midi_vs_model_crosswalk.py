# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a per-event MIDI vs model crosswalk for an Ave Maria fragment.")
    ap.add_argument("--midi-csv", required=True)
    ap.add_argument("--model-csv", required=True, help="Attack-first separation CSV")
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--end-sec", type=float, required=True)
    ap.add_argument("--frame-tolerance", type=int, default=18)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    start_frame = int(round(float(args.start_sec) * 60.0))
    end_frame = int(round(float(args.end_sec) * 60.0))

    midi = pd.read_csv(args.midi_csv)
    model = pd.read_csv(args.model_csv)

    midi_sub = midi[(midi["start_sec"] < float(args.end_sec)) & (midi["end_sec"] >= float(args.start_sec))].copy()
    midi_sub = midi_sub.sort_values(["start_frame60", "track_name", "freq_hz"]).reset_index(drop=True)
    model_sub = model[(model["birth_frame"] >= start_frame) & (model["birth_frame"] <= end_frame)].copy()
    model_sub = model_sub.sort_values(["birth_frame", "merged_event_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    exact_count = 0
    same_note_near_count = 0
    piano_backbone_count = 0
    string_like_backbone_count = 0
    no_match_count = 0

    for _, midi_row in midi_sub.iterrows():
        midi_note = str(midi_row.get("note12", "")).strip()
        midi_track = str(midi_row.get("track_name", "")).strip()
        midi_frame = _safe_int(midi_row.get("start_frame60"), 0)
        midi_start_sec = _safe_float(midi_row.get("start_sec"), 0.0)

        near = model_sub[(model_sub["birth_frame"] >= midi_frame - int(args.frame_tolerance)) & (model_sub["birth_frame"] <= midi_frame + int(args.frame_tolerance))].copy()
        same_note = near[near["candidate_note"].astype(str) == midi_note].copy()

        chosen = None
        match_kind = "NO_NEAR_MODEL_EVENT"
        if len(same_note):
            same_note["frame_delta_abs"] = (same_note["birth_frame"] - midi_frame).abs()
            same_note["same_note_bonus"] = 1
            chosen = same_note.sort_values(["frame_delta_abs", "winner_score"], ascending=[True, False]).iloc[0]
            frame_delta = int(chosen["birth_frame"]) - midi_frame
            match_kind = "EXACT_NOTE_NEAR_TIME" if abs(frame_delta) <= 6 else "SAME_NOTE_NEAR_TIME"
        elif len(near):
            near["frame_delta_abs"] = (near["birth_frame"] - midi_frame).abs()
            chosen = near.sort_values(["frame_delta_abs", "winner_score"], ascending=[True, False]).iloc[0]
            match_kind = "NEAREST_OTHER_NOTE"

        out: dict[str, Any] = {
            "midi_track_name": midi_track,
            "midi_note_token": midi_note,
            "midi_freq_hz": f"{_safe_float(midi_row.get('freq_hz'), 0.0):.6f}",
            "midi_start_sec": f"{midi_start_sec:.6f}",
            "midi_start_frame60": midi_frame,
            "match_kind": match_kind,
        }

        if chosen is not None:
            out.update(
                {
                    "model_merged_event_id": _safe_int(chosen.get("merged_event_id"), 0),
                    "model_candidate_note": str(chosen.get("candidate_note", "")).strip(),
                    "model_birth_frame60": _safe_int(chosen.get("birth_frame"), 0),
                    "model_birth_sec": f"{_safe_int(chosen.get('birth_frame'), 0) / 60.0:.6f}",
                    "frame_delta60": _safe_int(chosen.get("birth_frame"), 0) - midi_frame,
                    "model_winner_instrument": str(chosen.get("winner_instrument", "")).strip(),
                    "model_dominant_instrument": str(chosen.get("dominant_instrument", "")).strip(),
                    "model_attack_first_owner": str(chosen.get("attack_first_owner", "")).strip(),
                    "model_late_owner": str(chosen.get("late_owner_after_attack", "")).strip(),
                    "model_role_pattern": str(chosen.get("role_pattern", "")).strip(),
                    "model_window_alignment": str(chosen.get("winner_window_alignment", "")).strip(),
                    "model_support_instruments": str(chosen.get("support_instruments", "")).strip(),
                }
            )
        else:
            out.update(
                {
                    "model_merged_event_id": "",
                    "model_candidate_note": "",
                    "model_birth_frame60": "",
                    "model_birth_sec": "",
                    "frame_delta60": "",
                    "model_winner_instrument": "",
                    "model_dominant_instrument": "",
                    "model_attack_first_owner": "",
                    "model_late_owner": "",
                    "model_role_pattern": "",
                    "model_window_alignment": "",
                    "model_support_instruments": "",
                }
            )

        if match_kind == "EXACT_NOTE_NEAR_TIME":
            exact_count += 1
        elif match_kind == "SAME_NOTE_NEAR_TIME":
            same_note_near_count += 1
        elif match_kind == "NO_NEAR_MODEL_EVENT":
            no_match_count += 1

        attack_owner = out["model_attack_first_owner"]
        if attack_owner == "piano":
            piano_backbone_count += 1
        if midi_track in {"Cello", "Violin"} and attack_owner != "piano" and chosen is not None:
            string_like_backbone_count += 1

        rows.append(out)

    out_df = pd.DataFrame(rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(f"{args.out_prefix}.csv")
    out_txt = Path(f"{args.out_prefix}.txt")
    out_json = Path(f"{args.out_prefix}.json")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    by_track_match = out_df.groupby(["midi_track_name", "match_kind"]).size().reset_index(name="count")
    lines = [
        "AVE MARIA MIDI VS MODEL CROSSWALK",
        "=" * 72,
        f"window_sec: {float(args.start_sec):.3f} -> {float(args.end_sec):.3f}",
        f"window_frames60: {start_frame} -> {end_frame}",
        f"midi_events: {len(midi_sub)}",
        f"model_events: {len(model_sub)}",
        f"exact_note_near_time: {exact_count}",
        f"same_note_near_time: {same_note_near_count}",
        f"no_near_model_event: {no_match_count}",
        "",
        "by_track_and_match_kind:",
    ]
    for _, row in by_track_match.iterrows():
        lines.append(f"  {row['midi_track_name']} | {row['match_kind']}: {int(row['count'])}")
    lines.extend(
        [
            "",
            "reading_hint:",
            "  - EXACT_NOTE_NEAR_TIME means same note and close frame",
            "  - SAME_NOTE_NEAR_TIME means same note but looser timing",
            "  - NEAREST_OTHER_NOTE means the model saw another nearby note/event instead",
            "  - NO_NEAR_MODEL_EVENT means MIDI has an event but no model event was born nearby",
            "",
            f"piano_attack_owner_hits: {piano_backbone_count}",
            f"string_events_not_reduced_to_piano_attack: {string_like_backbone_count}",
        ]
    )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "window_sec": [float(args.start_sec), float(args.end_sec)],
                "window_frames60": [start_frame, end_frame],
                "midi_events": int(len(midi_sub)),
                "model_events": int(len(model_sub)),
                "exact_note_near_time": exact_count,
                "same_note_near_time": same_note_near_count,
                "no_near_model_event": no_match_count,
                "rows_csv": str(out_csv),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"WROTE {out_csv}")
    print(f"WROTE {out_txt}")
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()
