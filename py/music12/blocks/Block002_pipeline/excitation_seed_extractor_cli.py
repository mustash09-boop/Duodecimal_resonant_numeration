from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from music12.blocks.Block002_audio_recogn.resonance_candidate_inference_core import load_matrix_csv
from music12.blocks.Block002_audio_recogn.resonance_field_builder_core import (
    load_probe_coords_delta_csv,
    load_probe_times_csv,
)


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pick_unique_top_indices(
    *,
    scores: np.ndarray,
    coarse_notes: list[str],
    top_k: int,
    max_per_coarse: int,
) -> list[int]:
    positive = np.flatnonzero(scores > 0.0)
    if positive.size == 0:
        return []

    if positive.size > top_k * 10:
        local = positive[np.argpartition(scores[positive], -top_k * 10)[-top_k * 10 :]]
    else:
        local = positive

    order = local[np.argsort(scores[local])[::-1]]
    selected: list[int] = []
    by_coarse: dict[str, int] = {}

    for idx in order.tolist():
        coarse = coarse_notes[idx]
        if not coarse:
            continue
        count = by_coarse.get(coarse, 0)
        if count >= max_per_coarse:
            continue
        selected.append(idx)
        by_coarse[coarse] = count + 1
        if len(selected) >= top_k:
            break

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract short-lived excitation seeds directly from probe field before family/bridge layers."
    )
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--probe-times-csv", required=True)
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--out-seeds-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--top-k-per-frame", type=int, default=10)
    ap.add_argument("--max-per-coarse", type=int, default=1)
    ap.add_argument("--min-seed-score", type=float, default=0.08)
    ap.add_argument("--score-std-threshold", type=float, default=1.75)
    args = ap.parse_args()

    matrix = load_matrix_csv(args.probe_matrix_csv)
    times = load_probe_times_csv(args.probe_times_csv)
    coords = load_probe_coords_delta_csv(args.probe_coords_csv)

    if matrix.ndim != 2 or matrix.size == 0:
        raise SystemExit("Empty probe matrix")

    probe_count, frame_count = int(matrix.shape[0]), int(matrix.shape[1])
    usable_probe_count = min(probe_count, len(coords))
    if usable_probe_count <= 0:
        raise SystemExit("No usable probe coordinates")

    note_tokens = [coords[i].note_token for i in range(usable_probe_count)]
    coarse_notes = [_normalize_note(note_tokens[i]) for i in range(usable_probe_count)]
    frequencies = np.asarray([coords[i].frequency_hz for i in range(usable_probe_count)], dtype=np.float32)

    out_rows: list[dict[str, Any]] = []
    frames_with_seeds = 0
    max_seed_score = 0.0

    zero = np.zeros((usable_probe_count,), dtype=np.float32)
    eps = np.float32(1e-9)

    for frame_index in range(frame_count):
        curr = np.asarray(matrix[:usable_probe_count, frame_index], dtype=np.float32)
        prev = np.asarray(matrix[:usable_probe_count, frame_index - 1], dtype=np.float32) if frame_index > 0 else zero
        prev2 = np.asarray(matrix[:usable_probe_count, frame_index - 2], dtype=np.float32) if frame_index > 1 else prev
        nxt = np.asarray(matrix[:usable_probe_count, frame_index + 1], dtype=np.float32) if frame_index + 1 < frame_count else zero

        rise = np.maximum(curr - prev, 0.0)
        window_rise = np.maximum(curr - 0.5 * (prev + prev2), 0.0)
        continuation = np.minimum(curr, nxt) / np.maximum(curr, eps)

        frame_mean = float(curr.mean())
        frame_std = float(curr.std())
        contrast = np.maximum((curr - frame_mean) / max(frame_std, 1e-6), 0.0)
        contrast_norm = np.minimum(contrast / 4.0, 1.0)

        rise_ratio = rise / np.maximum(curr, eps)
        window_rise_ratio = window_rise / np.maximum(curr, eps)

        seed_score = curr * (
            0.45 * rise_ratio
            + 0.25 * window_rise_ratio
            + 0.20 * continuation
            + 0.10 * contrast_norm
        )
        seed_score = np.maximum(seed_score, 0.0)

        dynamic_threshold = max(
            float(args.min_seed_score),
            float(seed_score.mean() + args.score_std_threshold * seed_score.std()),
        )
        masked_scores = np.where(seed_score >= dynamic_threshold, seed_score, 0.0)
        picked = _pick_unique_top_indices(
            scores=masked_scores,
            coarse_notes=coarse_notes,
            top_k=int(args.top_k_per_frame),
            max_per_coarse=int(args.max_per_coarse),
        )

        if not picked:
            continue

        frames_with_seeds += 1
        ranked = sorted(picked, key=lambda idx: float(masked_scores[idx]), reverse=True)
        time_sec = float(times[frame_index]) if frame_index < len(times) else 0.0

        for rank, probe_index in enumerate(ranked, start=1):
            score = float(masked_scores[probe_index])
            max_seed_score = max(max_seed_score, score)
            out_rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": f"{time_sec:.9f}",
                    "frame_rank": rank,
                    "probe_index": probe_index,
                    "note_token": note_tokens[probe_index],
                    "coarse_note": coarse_notes[probe_index],
                    "frequency_hz": f"{float(frequencies[probe_index]):.9f}",
                    "energy": f"{float(curr[probe_index]):.9f}",
                    "rise": f"{float(rise[probe_index]):.9f}",
                    "window_rise": f"{float(window_rise[probe_index]):.9f}",
                    "continuation": f"{float(continuation[probe_index]):.9f}",
                    "contrast_z": f"{float(contrast[probe_index]):.9f}",
                    "seed_score": f"{score:.9f}",
                }
            )

    out_csv = Path(args.out_seeds_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "frame_index",
            "time_sec",
            "frame_rank",
            "probe_index",
            "note_token",
            "coarse_note",
            "frequency_hz",
            "energy",
            "rise",
            "window_rise",
            "continuation",
            "contrast_z",
            "seed_score",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    summary = {
        "stage": "excitation_seed_extractor",
        "inputs": {
            "probe_matrix_csv": args.probe_matrix_csv,
            "probe_times_csv": args.probe_times_csv,
            "probe_coords_csv": args.probe_coords_csv,
        },
        "parameters": {
            "top_k_per_frame": int(args.top_k_per_frame),
            "max_per_coarse": int(args.max_per_coarse),
            "min_seed_score": float(args.min_seed_score),
            "score_std_threshold": float(args.score_std_threshold),
        },
        "result": {
            "probe_count": usable_probe_count,
            "frame_count": frame_count,
            "seed_rows": len(out_rows),
            "frames_with_seeds": frames_with_seeds,
            "max_seed_score": max_seed_score,
        },
    }

    lines = [
        "EXCITATION SEED EXTRACTION",
        "=" * 72,
        f"probe_count        : {usable_probe_count}",
        f"frame_count        : {frame_count}",
        f"seed_rows          : {len(out_rows)}",
        f"frames_with_seeds  : {frames_with_seeds}",
        f"max_seed_score     : {max_seed_score:.6f}",
    ]

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
