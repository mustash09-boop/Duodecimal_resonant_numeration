# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


FPS60 = 60.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_wav(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def _interpolate_series(frame_map: dict[int, float], full_frames: list[int]) -> np.ndarray:
    if not frame_map:
        return np.zeros(len(full_frames), dtype=np.float32)
    x = np.asarray(sorted(frame_map.keys()), dtype=np.float64)
    y = np.asarray([frame_map[int(idx)] for idx in x], dtype=np.float64)
    xp = np.asarray(full_frames, dtype=np.float64)
    return np.interp(xp, x, y, left=0.0, right=0.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a continuous bowed bundle preview from selected harmonic probe families."
    )
    ap.add_argument("--families-csv", required=True)
    ap.add_argument("--owner-rows-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--top-families-per-harmonic", type=int, default=2)
    ap.add_argument("--min-family-score", type=float, default=0.60)
    ap.add_argument("--max-piano-overlap", type=float, default=0.05)
    ap.add_argument("--amplitude-scale", type=float, default=0.08)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-selected-families-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    family_rows = _load_csv(Path(args.families_csv))
    owner_rows = _load_csv(Path(args.owner_rows_csv))

    selected_family_rows: list[dict[str, Any]] = []
    for harmonic_index in range(1, 8):
        rows = [
            row
            for row in family_rows
            if _safe_int(row.get("harmonic_index"), 0) == harmonic_index
            and _safe_float(row.get("extraction_score"), 0.0) >= args.min_family_score
            and _safe_float(row.get("pianoish_overlap_ratio"), 1.0) <= args.max_piano_overlap
        ]
        rows.sort(key=lambda row: _safe_float(row.get("extraction_score"), 0.0), reverse=True)
        selected_family_rows.extend(rows[: max(1, int(args.top_families_per_harmonic))])

    selected_keys = {
        (_safe_int(row.get("harmonic_index"), 0), _safe_int(row.get("probe_index"), -1))
        for row in selected_family_rows
    }
    selected_owner_rows = [
        row
        for row in owner_rows
        if (_safe_int(row.get("harmonic_index"), 0), _safe_int(row.get("probe_index"), -1)) in selected_keys
        and args.window_start_sec <= _safe_float(row.get("time_sec"), -1.0) <= args.window_end_sec
    ]

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_bundle_series",
            "selected_family_count": len(selected_family_rows),
            "selected_owner_rows": len(selected_owner_rows),
        },
    )

    frame_set = sorted({_safe_int(row.get("frame_index"), -1) for row in selected_owner_rows if _safe_int(row.get("frame_index"), -1) >= 0})
    if not frame_set:
        raise SystemExit("No selected bowed bundle rows for preview.")
    frame_time_map = {
        _safe_int(row.get("frame_index"), -1): _safe_float(row.get("time_sec"), 0.0)
        for row in selected_owner_rows
    }

    duration_sec = max(0.0, float(args.window_end_sec) - float(args.window_start_sec))
    sample_count = max(1, int(round(duration_sec * args.sample_rate)))
    time_axis = np.arange(sample_count, dtype=np.float32) / float(args.sample_rate) + float(args.window_start_sec)
    audio = np.zeros(sample_count, dtype=np.float32)

    harmonic_gain_map = {
        1: 1.00,
        2: 0.95,
        3: 0.70,
        4: 0.42,
        5: 0.56,
        6: 0.34,
        7: 0.20,
    }

    selected_series_rows: list[dict[str, Any]] = []
    for family_row in selected_family_rows:
        harmonic_index = _safe_int(family_row.get("harmonic_index"), 0)
        probe_index = _safe_int(family_row.get("probe_index"), -1)
        key_rows = [
            row
            for row in selected_owner_rows
            if _safe_int(row.get("harmonic_index"), 0) == harmonic_index
            and _safe_int(row.get("probe_index"), -1) == probe_index
        ]
        key_rows.sort(key=lambda row: _safe_int(row.get("frame_index"), 0))
        amp_by_frame: dict[int, float] = {}
        freq_by_frame: dict[int, float] = {}
        for row in key_rows:
            frame_index = _safe_int(row.get("frame_index"), -1)
            amp_by_frame[frame_index] = _safe_float(row.get("energy_over_frame_p95"), 0.0)
            freq_by_frame[frame_index] = _safe_float(row.get("frequency_hz"), 0.0)
        amp_series = _interpolate_series(amp_by_frame, frame_set)
        freq_series = _interpolate_series(freq_by_frame, frame_set)

        frame_times = np.asarray([frame_time_map[idx] for idx in frame_set], dtype=np.float64)
        amp_interp = np.interp(time_axis, frame_times, amp_series, left=0.0, right=0.0).astype(np.float32)
        freq_interp = np.interp(time_axis, frame_times, freq_series, left=0.0, right=0.0).astype(np.float32)

        family_gain = (
            float(args.amplitude_scale)
            * harmonic_gain_map.get(harmonic_index, 0.15)
            * max(0.25, min(1.0, _safe_float(family_row.get("extraction_score"), 0.0)))
        )
        amp_signal = np.sqrt(np.maximum(0.0, amp_interp)) * family_gain
        phase = 2.0 * math.pi * np.cumsum(freq_interp, dtype=np.float64) / float(args.sample_rate)
        partial = amp_signal * np.sin(phase).astype(np.float32)
        audio += partial

        selected_series_rows.append(
            {
                "harmonic_index": harmonic_index,
                "probe_index": probe_index,
                "observed_note_token": str(family_row.get("observed_note_token", "")).strip(),
                "active_frame_count": _safe_int(family_row.get("active_frame_count"), 0),
                "max_consecutive_frames": _safe_int(family_row.get("max_consecutive_frames"), 0),
                "mean_energy_over_frame_p95": _safe_float(family_row.get("mean_energy_over_frame_p95"), 0.0),
                "pianoish_overlap_ratio": _safe_float(family_row.get("pianoish_overlap_ratio"), 0.0),
                "extraction_score": _safe_float(family_row.get("extraction_score"), 0.0),
                "render_gain": family_gain,
            }
        )

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        audio = audio / peak * 0.95

    _write_wav(Path(args.out_wav), audio, args.sample_rate)

    with Path(args.out_selected_families_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "harmonic_index",
                "probe_index",
                "observed_note_token",
                "active_frame_count",
                "max_consecutive_frames",
                "mean_energy_over_frame_p95",
                "pianoish_overlap_ratio",
                "extraction_score",
                "render_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_series_rows)

    summary_lines = [
        "WINDOW BOWED BUNDLE PREVIEW",
        "=" * 72,
        f"selected_family_count              : {len(selected_family_rows)}",
        f"selected_owner_row_count           : {len(selected_owner_rows)}",
        f"window_duration_sec                : {duration_sec:.6f}",
        f"sample_rate                        : {args.sample_rate}",
        f"peak_after_render                  : {peak:.6f}",
        "",
        "selected_families:",
    ]
    for row in selected_series_rows:
        summary_lines.append(
            "  "
            f"h{row['harmonic_index']} probe={row['probe_index']} token={row['observed_note_token']} "
            f"score={row['extraction_score']:.6f} "
            f"gain={row['render_gain']:.6f} "
            f"piano_overlap={row['pianoish_overlap_ratio']:.6f}"
        )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This preview is rendered from selected bowed harmonic probe families,",
            "  not from a whole-frame time mask. It is still synthetic, but it follows",
            "  the extracted frequency-aware bundle instead of carrying the full piano frame.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_bundle_preview",
                "selected_family_count": len(selected_family_rows),
                "selected_owner_row_count": len(selected_owner_rows),
                "window_duration_sec": duration_sec,
                "sample_rate": args.sample_rate,
                "peak_after_render": peak,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    _write_progress(
        args.progress_json,
        {
            "status": "done",
            "phase": "complete",
            "selected_family_count": len(selected_family_rows),
            "selected_owner_row_count": len(selected_owner_rows),
        },
    )


if __name__ == "__main__":
    main()
