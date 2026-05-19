from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


FRAME_RATE = 60.0


def _read_mask_csv(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                values.append(float(str(row.get("mask", "0")).strip()))
            except Exception:
                values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _to_audio_mask(frame_mask: np.ndarray, sample_rate: int, sample_count: int, smoothing_ms: float) -> np.ndarray:
    frame_times = np.arange(len(frame_mask), dtype=np.float64) / FRAME_RATE
    sample_times = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    audio_mask = np.interp(sample_times, frame_times, frame_mask.astype(np.float64), left=0.0, right=0.0).astype(np.float32)
    window = max(1, int(round(sample_rate * smoothing_ms / 1000.0)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        audio_mask = np.convolve(audio_mask, kernel, mode="same")
    return np.clip(audio_mask, 0.0, 1.0)


def _combine_masks(masks: list[np.ndarray], combine_mode: str) -> np.ndarray:
    max_len = max(len(m) for m in masks)
    padded: list[np.ndarray] = []
    for m in masks:
        if len(m) < max_len:
            buf = np.zeros(max_len, dtype=np.float32)
            buf[: len(m)] = m
            padded.append(buf)
        else:
            padded.append(m.astype(np.float32, copy=False))
    stack = np.vstack(padded)
    if combine_mode == "max":
        return np.max(stack, axis=0)
    if combine_mode == "sum_clip":
        return np.clip(np.sum(stack, axis=0), 0.0, 1.0)
    if combine_mode == "prob_union":
        out = np.ones(max_len, dtype=np.float32)
        for m in padded:
            out *= (1.0 - np.clip(m, 0.0, 1.0))
        return 1.0 - out
    raise SystemExit(f"Unknown combine mode: {combine_mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple frame masks into one stem rendered from source audio.")
    ap.add_argument("--audio_wav", required=True)
    ap.add_argument("--mask_csv", required=True, nargs="+")
    ap.add_argument("--mask_label", nargs="*")
    ap.add_argument("--combine_mode", choices=["max", "sum_clip", "prob_union"], default="prob_union")
    ap.add_argument("--smoothing_ms", type=float, default=22.0)
    ap.add_argument("--out_frame_mask_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    mask_paths = [Path(x) for x in args.mask_csv]
    labels = list(args.mask_label or [])
    if labels and len(labels) != len(mask_paths):
        raise SystemExit("mask_label count must match mask_csv count")
    if not labels:
        labels = [p.stem for p in mask_paths]

    masks = [_read_mask_csv(p) for p in mask_paths]
    combined_frame_mask = _combine_masks(masks, args.combine_mode)

    audio, sr = sf.read(str(Path(args.audio_wav)))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    audio_mask = _to_audio_mask(
        frame_mask=combined_frame_mask,
        sample_rate=int(sr),
        sample_count=len(audio),
        smoothing_ms=args.smoothing_ms,
    )
    rendered = audio * audio_mask
    sf.write(str(Path(args.out_wav)), rendered, int(sr), subtype="PCM_16")

    with Path(args.out_frame_mask_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_index", "time_sec", "mask"])
        writer.writeheader()
        for idx, value in enumerate(combined_frame_mask):
            writer.writerow(
                {
                    "frame_index": idx,
                    "time_sec": f"{idx / FRAME_RATE:.6f}",
                    "mask": f"{float(value):.9f}",
                }
            )

    active_frames = int(np.count_nonzero(combined_frame_mask > 0.02))
    strong_frames = int(np.count_nonzero(combined_frame_mask > 0.25))
    per_mask_active = Counter()
    for label, mask in zip(labels, masks):
        per_mask_active[label] = int(np.count_nonzero(mask > 0.25))

    summary_lines = [
        "MERGED FRAME MASK STEM",
        "=" * 72,
        f"combine_mode: {args.combine_mode}",
        f"input_masks: {len(mask_paths)}",
        f"active_frames_gt_0_02: {active_frames}",
        f"strong_frames_gt_0_25: {strong_frames}",
        f"active_frame_ratio: {active_frames / max(1, len(combined_frame_mask)):.6f}",
        f"strong_frame_ratio: {strong_frames / max(1, len(combined_frame_mask)):.6f}",
        f"audio_active_ratio_gt_0_02: {float(np.count_nonzero(audio_mask > 0.02)) / max(1, len(audio_mask)):.6f}",
        f"audio_strong_ratio_gt_0_25: {float(np.count_nonzero(audio_mask > 0.25)) / max(1, len(audio_mask)):.6f}",
        "",
        "per_mask_strong_frames_gt_0_25:",
    ]
    for key, value in per_mask_active.items():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "combine_mode": args.combine_mode,
                "mask_csv": [str(p) for p in mask_paths],
                "mask_label": labels,
                "active_frames_gt_0_02": active_frames,
                "strong_frames_gt_0_25": strong_frames,
                "per_mask_strong_frames_gt_0_25": dict(per_mask_active),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
