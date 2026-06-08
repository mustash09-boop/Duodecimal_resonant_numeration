# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import wave
from collections import Counter
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_wav_window(
    *,
    wav_path: Path,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, int, int]:
    with wave.open(str(wav_path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        if sampwidth != 2:
            raise RuntimeError(f"Unsupported sample width: {sampwidth}")
        start_frame = max(0, int(round(start_sec * framerate)))
        end_frame = min(wf.getnframes(), int(round(end_sec * framerate)))
        wf.setpos(start_frame)
        raw = wf.readframes(max(0, end_frame - start_frame))
    data = np.frombuffer(raw, dtype=np.int16).copy()
    if nchannels > 1:
        data = data.reshape((-1, nchannels))
    else:
        data = data.reshape((-1, 1))
    return data, framerate, nchannels


def _write_wav(path: Path, data: np.ndarray, sample_rate: int, nchannels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(data), -32768, 32767).astype(np.int16)
    interleaved = clipped.reshape((-1, nchannels))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


def _build_activity_mask(
    *,
    frame_rows: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
    sample_rate: int,
    target_key: str,
) -> np.ndarray:
    sample_count = max(0, int(round((end_sec - start_sec) * sample_rate)))
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float32)
    mask = np.zeros(sample_count, dtype=np.float32)
    times = [(_safe_float(r.get("time_sec"), 0.0), _safe_int(r.get(target_key), 0)) for r in frame_rows]
    max_value = max((value for _, value in times), default=0)
    if max_value <= 0:
        return mask
    frame_duration = 1.0 / FPS60
    for time_sec, value in times:
        frame_start = max(start_sec, time_sec)
        frame_end = min(end_sec, time_sec + frame_duration)
        if frame_end <= frame_start:
            continue
        left = int(round((frame_start - start_sec) * sample_rate))
        right = int(round((frame_end - start_sec) * sample_rate))
        right = min(sample_count, max(right, left + 1))
        mask[left:right] = max(mask[left:right].max(initial=0.0), float(value) / float(max_value))
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit a local Ave Maria window using the new main-backbone vs second-sustain logic and create preview layer WAVs."
    )
    ap.add_argument("--roles-csv", required=True)
    ap.add_argument("--backbone-lineages-csv", required=True)
    ap.add_argument("--frame-overlap-csv", required=True)
    ap.add_argument("--audio-wav", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
        },
    )

    roles_rows = _load_csv(Path(args.roles_csv))
    lineage_rows = _load_csv(Path(args.backbone_lineages_csv))
    frame_rows = _load_csv(Path(args.frame_overlap_csv))

    start_frame = int(np.floor(args.window_start_sec * FPS60))
    end_frame = int(np.ceil(args.window_end_sec * FPS60))

    selected_roles = [
        row for row in roles_rows
        if _safe_int(row.get("start_frame"), 0) <= end_frame and _safe_int(row.get("end_frame"), 0) >= start_frame
    ]
    selected_lineages = [
        row for row in lineage_rows
        if _safe_int(row.get("start_frame"), 0) <= end_frame and _safe_int(row.get("end_frame"), 0) >= start_frame
    ]
    selected_frames = [
        row for row in frame_rows
        if start_frame <= _safe_int(row.get("frame_index"), 0) <= end_frame
    ]

    main_rows = [r for r in selected_lineages if str(r.get("backbone_lineage_class", "")).strip() == "MAIN_HARMONIC_BACKBONE"]
    second_rows = [r for r in selected_lineages if str(r.get("backbone_lineage_class", "")).strip() == "POSSIBLE_SECOND_SUSTAINED_LINEAGE"]
    body_rows = [r for r in selected_lineages if str(r.get("backbone_lineage_class", "")).strip() == "BODY_CONTINUATION_LINEAGE"]

    roles_csv = out_dir / "window_roles.csv"
    with roles_csv.open("w", encoding="utf-8", newline="") as fh:
        if selected_roles:
            writer = csv.DictWriter(fh, fieldnames=list(selected_roles[0].keys()))
            writer.writeheader()
            for row in selected_roles:
                writer.writerow(row)

    lineages_csv = out_dir / "window_backbone_lineages.csv"
    with lineages_csv.open("w", encoding="utf-8", newline="") as fh:
        if selected_lineages:
            writer = csv.DictWriter(fh, fieldnames=list(selected_lineages[0].keys()))
            writer.writeheader()
            for row in selected_lineages:
                writer.writerow(row)

    frame_csv = out_dir / "window_frame_overlap.csv"
    with frame_csv.open("w", encoding="utf-8", newline="") as fh:
        if selected_frames:
            writer = csv.DictWriter(fh, fieldnames=list(selected_frames[0].keys()))
            writer.writeheader()
            for row in selected_frames:
                writer.writerow(row)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "audio_window_masks",
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
            "selected_role_rows": len(selected_roles),
            "selected_lineage_rows": len(selected_lineages),
        },
    )

    audio_data, sample_rate, nchannels = _extract_wav_window(
        wav_path=Path(args.audio_wav),
        start_sec=float(args.window_start_sec),
        end_sec=float(args.window_end_sec),
    )
    _write_wav(out_dir / "window_raw_excerpt.wav", audio_data, sample_rate, nchannels)

    main_mask = _build_activity_mask(
        frame_rows=selected_frames,
        start_sec=float(args.window_start_sec),
        end_sec=float(args.window_end_sec),
        sample_rate=sample_rate,
        target_key="main_backbone_activity",
    )
    second_mask = _build_activity_mask(
        frame_rows=selected_frames,
        start_sec=float(args.window_start_sec),
        end_sec=float(args.window_end_sec),
        sample_rate=sample_rate,
        target_key="second_sustain_activity",
    )

    if audio_data.shape[0] != len(main_mask):
        main_mask = np.resize(main_mask, audio_data.shape[0])
    if audio_data.shape[0] != len(second_mask):
        second_mask = np.resize(second_mask, audio_data.shape[0])

    main_preview = audio_data.astype(np.float32) * main_mask[:, None]
    second_preview = audio_data.astype(np.float32) * second_mask[:, None]
    _write_wav(out_dir / "window_main_backbone_preview.wav", main_preview, sample_rate, nchannels)
    _write_wav(out_dir / "window_second_sustain_preview.wav", second_preview, sample_rate, nchannels)

    summary_lines = [
        "WINDOW BACKBONE SECOND SUSTAIN AUDIT",
        "=" * 72,
        f"window_start_sec          : {args.window_start_sec:.6f}",
        f"window_end_sec            : {args.window_end_sec:.6f}",
        f"window_frame_start        : {start_frame}",
        f"window_frame_end          : {end_frame}",
        f"selected_role_rows        : {len(selected_roles)}",
        f"selected_lineage_rows     : {len(selected_lineages)}",
        f"main_backbone_rows        : {len(main_rows)}",
        f"second_sustain_rows       : {len(second_rows)}",
        f"body_continuation_rows    : {len(body_rows)}",
        "",
        f"main_mean_freq_hz         : {_mean([_safe_float(r.get('mean_frequency_hz'), 0.0) for r in main_rows]):.6f}",
        f"second_mean_freq_hz       : {_mean([_safe_float(r.get('mean_frequency_hz'), 0.0) for r in second_rows]):.6f}",
        f"main_mean_obs_frames      : {_mean([float(_safe_int(r.get('observation_frame_count'), 0)) for r in main_rows]):.6f}",
        f"second_mean_obs_frames    : {_mean([float(_safe_int(r.get('observation_frame_count'), 0)) for r in second_rows]):.6f}",
        "",
        "main_top_coarse_tokens:",
    ]
    for token, count in Counter(str(r.get("anchor_coarse_note", "")).strip() for r in main_rows).most_common(12):
        summary_lines.append(f"  {token}: {count}")
    summary_lines.extend(["", "second_top_coarse_tokens:"])
    for token, count in Counter(str(r.get("anchor_coarse_note", "")).strip() for r in second_rows).most_common(12):
        summary_lines.append(f"  {token}: {count}")
    (out_dir / "window_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    (out_dir / "window_meta.json").write_text(
        json.dumps(
            {
                "window_start_sec": args.window_start_sec,
                "window_end_sec": args.window_end_sec,
                "window_frame_start": start_frame,
                "window_frame_end": end_frame,
                "selected_role_rows": len(selected_roles),
                "selected_lineage_rows": len(selected_lineages),
                "main_backbone_rows": len(main_rows),
                "second_sustain_rows": len(second_rows),
                "body_continuation_rows": len(body_rows),
                "sample_rate": sample_rate,
                "channels": nchannels,
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
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
            "selected_role_rows": len(selected_roles),
            "selected_lineage_rows": len(selected_lineages),
        },
    )


if __name__ == "__main__":
    main()
