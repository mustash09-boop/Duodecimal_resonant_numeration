from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


FRAME_RATE = 60.0


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _iter_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _frame_to_sec(frame: int) -> float:
    return frame / FRAME_RATE


def _select_piano_rows(
    layered_rows: Iterable[dict[str, str]],
    mode: str,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in layered_rows:
        dominant = str(row.get("dominant_instrument", "")).strip()
        state = str(row.get("dominant_state_layered", "")).strip()
        support = str(row.get("support_combo_key", "")).strip()
        piano_window = str(row.get("piano_window", "")).strip()

        if dominant != "piano":
            continue

        if mode == "conservative":
            if state not in {"CLEAR_DOMINANT", "OWN_WINDOW_DOMINANT"}:
                continue
            if piano_window not in {"TARGET_ONLY_WINDOW", "MIXED_WINDOW"}:
                continue
            if support not in {"<NONE>", "organ"}:
                continue
        elif mode == "balanced":
            if state not in {"CLEAR_DOMINANT", "OWN_WINDOW_DOMINANT", "LEANING_DOMINANT"}:
                continue
            if piano_window == "EMPTY_WINDOW":
                continue
            if support not in {"<NONE>", "organ", "cello", "piano", "cello+organ"}:
                continue
        elif mode == "broad":
            if state == "NO_STRUCTURAL_OWNER":
                continue
        else:
            raise SystemExit(f"Unknown mode: {mode}")

        selected.append(row)
    return selected


def _index_event_rows(event_rows: Iterable[dict[str, str]]) -> dict[int, dict[str, str]]:
    idx: dict[int, dict[str, str]] = {}
    for row in event_rows:
        idx[_safe_int(row.get("merged_event_id"))] = row
    return idx


def _build_intervals(
    selected_rows: Iterable[dict[str, str]],
    event_index: dict[int, dict[str, str]],
    pre_roll_sec: float,
    post_roll_sec: float,
    merge_gap_sec: float,
) -> tuple[list[dict[str, float]], list[dict[str, str]]]:
    raw_intervals: list[tuple[float, float]] = []
    chosen_events: list[dict[str, str]] = []

    for row in selected_rows:
        event_id = _safe_int(row.get("merged_event_id"))
        ev = event_index.get(event_id)
        if not ev:
            continue
        chosen_events.append(ev | row)
        start_sec = _frame_to_sec(_safe_int(ev.get("birth_frame"))) - pre_roll_sec
        end_sec = _frame_to_sec(_safe_int(ev.get("end_frame"))) + post_roll_sec
        raw_intervals.append((max(0.0, start_sec), max(0.0, end_sec)))

    raw_intervals.sort()
    merged: list[dict[str, float]] = []
    for start_sec, end_sec in raw_intervals:
        if not merged:
            merged.append({"start_sec": start_sec, "end_sec": end_sec})
            continue
        prev = merged[-1]
        if start_sec <= prev["end_sec"] + merge_gap_sec:
            prev["end_sec"] = max(prev["end_sec"], end_sec)
        else:
            merged.append({"start_sec": start_sec, "end_sec": end_sec})
    return merged, chosen_events


def _render_masked_audio(
    wav_path: Path,
    intervals: list[dict[str, float]],
    out_wav: Path,
    fade_ms: float,
) -> dict[str, float]:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    mask = np.zeros_like(audio, dtype=np.float32)
    fade_samples = max(1, int(round(sr * fade_ms / 1000.0)))

    for interval in intervals:
        start = max(0, int(round(interval["start_sec"] * sr)))
        end = min(len(audio), int(round(interval["end_sec"] * sr)))
        if end <= start:
            continue
        mask[start:end] = np.maximum(mask[start:end], 1.0)

        fade_in_end = min(end, start + fade_samples)
        if fade_in_end > start:
            ramp = np.linspace(0.0, 1.0, fade_in_end - start, dtype=np.float32)
            mask[start:fade_in_end] = np.maximum(mask[start:fade_in_end], ramp)

        fade_out_start = max(start, end - fade_samples)
        if end > fade_out_start:
            ramp = np.linspace(1.0, 0.0, end - fade_out_start, dtype=np.float32)
            mask[fade_out_start:end] = np.maximum(mask[fade_out_start:end], ramp)

    rendered = audio * np.clip(mask, 0.0, 1.0)
    sf.write(str(out_wav), rendered, sr, subtype="PCM_16")

    total_duration_sec = len(audio) / float(sr)
    active_duration_sec = float(np.count_nonzero(mask > 0.0)) / float(sr)
    return {
        "sample_rate": float(sr),
        "source_duration_sec": total_duration_sec,
        "active_duration_sec": active_duration_sec,
        "active_ratio": active_duration_sec / total_duration_sec if total_duration_sec > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a rough piano stem prototype from layered instrument assignment."
    )
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--audio_wav", required=True)
    ap.add_argument("--out_selected_csv", required=True)
    ap.add_argument("--out_intervals_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--mode", choices=["conservative", "balanced", "broad"], default="conservative")
    ap.add_argument("--pre_roll_sec", type=float, default=0.040)
    ap.add_argument("--post_roll_sec", type=float, default=0.180)
    ap.add_argument("--merge_gap_sec", type=float, default=0.080)
    ap.add_argument("--fade_ms", type=float, default=18.0)
    args = ap.parse_args()

    layered_rows = _iter_rows(Path(args.layered_csv))
    event_rows = _iter_rows(Path(args.events_csv))
    selected_layered = _select_piano_rows(layered_rows, args.mode)
    event_index = _index_event_rows(event_rows)
    intervals, chosen_events = _build_intervals(
        selected_rows=selected_layered,
        event_index=event_index,
        pre_roll_sec=args.pre_roll_sec,
        post_roll_sec=args.post_roll_sec,
        merge_gap_sec=args.merge_gap_sec,
    )
    audio_stats = _render_masked_audio(
        wav_path=Path(args.audio_wav),
        intervals=intervals,
        out_wav=Path(args.out_wav),
        fade_ms=args.fade_ms,
    )

    selected_counter = Counter(str(r.get("dominant_state_layered", "")).strip() for r in selected_layered)
    support_counter = Counter(str(r.get("support_combo_key", "")).strip() or "<NONE>" for r in selected_layered)
    window_counter = Counter(str(r.get("piano_window", "")).strip() or "<UNKNOWN>" for r in selected_layered)

    out_selected_csv = Path(args.out_selected_csv)
    if chosen_events:
        fields = list(chosen_events[0].keys())
    else:
        fields = [
            "merged_event_id",
            "candidate_note",
            "birth_frame",
            "end_frame",
            "dominant_instrument",
            "dominant_state_layered",
        ]
    with out_selected_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen_events)

    with Path(args.out_intervals_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["interval_id", "start_sec", "end_sec", "duration_sec"])
        writer.writeheader()
        for idx, interval in enumerate(intervals, start=1):
            start_sec = float(interval["start_sec"])
            end_sec = float(interval["end_sec"])
            writer.writerow(
                {
                    "interval_id": idx,
                    "start_sec": f"{start_sec:.6f}",
                    "end_sec": f"{end_sec:.6f}",
                    "duration_sec": f"{max(0.0, end_sec - start_sec):.6f}",
                }
            )

    summary_lines = [
        "PIANO STEM PROTOTYPE",
        "=" * 72,
        f"mode: {args.mode}",
        f"input_layered_events: {len(layered_rows)}",
        f"selected_piano_events: {len(selected_layered)}",
        f"merged_audio_intervals: {len(intervals)}",
        f"active_audio_ratio: {audio_stats['active_ratio']:.6f}",
        f"active_audio_duration_sec: {audio_stats['active_duration_sec']:.3f}",
        f"source_duration_sec: {audio_stats['source_duration_sec']:.3f}",
        "",
        "selected_dominant_states:",
    ]
    for key, value in selected_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("")
    summary_lines.append("selected_support_combos:")
    for key, value in support_counter.most_common(10):
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("")
    summary_lines.append("selected_piano_windows:")
    for key, value in window_counter.most_common():
        summary_lines.append(f"  {key}: {value}")

    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "mode": args.mode,
        "input_layered_events": len(layered_rows),
        "selected_piano_events": len(selected_layered),
        "merged_audio_intervals": len(intervals),
        "pre_roll_sec": args.pre_roll_sec,
        "post_roll_sec": args.post_roll_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "fade_ms": args.fade_ms,
        "audio_stats": audio_stats,
        "selected_dominant_states": dict(selected_counter),
        "selected_support_combos": dict(support_counter),
        "selected_piano_windows": dict(window_counter),
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
