# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from resonance_candidate_inference_core import load_matrix_csv_memmap


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
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _segments(sorted_frames: list[int]) -> list[tuple[int, int]]:
    if not sorted_frames:
        return []
    out: list[tuple[int, int]] = []
    start = prev = sorted_frames[0]
    for frame in sorted_frames[1:]:
        if frame == prev + 1:
            prev = frame
        else:
            out.append((start, prev))
            start = prev = frame
    out.append((start, prev))
    return out


def _segments_json(frames: list[int]) -> str:
    return json.dumps(
        [{"start": s, "end": e, "len": e - s + 1} for s, e in _segments(frames)],
        ensure_ascii=False,
    )


def _bridge_boolean(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    active = np.flatnonzero(mask)
    if active.size < 2:
        return out
    for left, right in zip(active[:-1], active[1:]):
        gap = int(right) - int(left) - 1
        if 0 < gap <= max_gap:
            out[left + 1:right] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve local bowed continuity for a window by coupling neighboring harmonic families and holding continuity through weak frames."
    )
    ap.add_argument("--selected-families-csv", required=True)
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--window-start-frame", type=int, required=True)
    ap.add_argument("--window-end-frame", type=int, required=True)
    ap.add_argument("--min-active-harmonics", type=int, default=4)
    ap.add_argument("--min-weighted-support", type=float, default=8.0)
    ap.add_argument("--bridge-inactive-gap", type=int, default=2)
    ap.add_argument("--continuation-ratio", type=float, default=0.18)
    ap.add_argument("--candidate-min-activation", type=float, default=0.35)
    ap.add_argument("--hold-min-activation", type=float, default=0.18)
    ap.add_argument("--probe-adjacency-bonus", type=float, default=0.30)
    ap.add_argument("--same-probe-bonus", type=float, default=0.45)
    ap.add_argument("--score-weight-extraction", type=float, default=0.12)
    ap.add_argument("--out-resolved-cloud-csv", required=True)
    ap.add_argument("--out-frame-support-csv", required=True)
    ap.add_argument("--out-family-paths-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    selected_rows = _load_csv(Path(args.selected_families_csv))
    matrix, _info = load_matrix_csv_memmap(args.probe_matrix_csv)

    frame_indices = list(range(int(args.window_start_frame), int(args.window_end_frame) + 1))
    frame_slice = np.asarray(matrix[:, frame_indices], dtype=np.float32)
    frame_p95 = np.quantile(frame_slice, 0.95, axis=0).astype(np.float32)

    families_by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        probe_index = _safe_int(row.get("probe_index"), -1)
        if harmonic_index <= 0 or probe_index < 0:
            continue
        row = dict(row)
        row["_probe_index"] = probe_index
        row["_extraction_score"] = _safe_float(row.get("extraction_score"), 0.0)
        row["_mean_energy"] = max(1e-9, _safe_float(row.get("mean_energy_over_frame_p95"), 0.0))
        row["_threshold_ratio"] = row["_mean_energy"] * float(args.continuation_ratio)
        families_by_h[harmonic_index].append(row)

    harmonic_state: dict[int, dict[str, Any]] = {}
    for harmonic_index, rows in families_by_h.items():
        probe_indices = [int(row["_probe_index"]) for row in rows]
        series = np.asarray(matrix[np.ix_(probe_indices, frame_indices)], dtype=np.float32)
        ratio = np.divide(
            series,
            np.maximum(frame_p95[None, :], 1e-9),
            out=np.zeros_like(series, dtype=np.float32),
            where=frame_p95[None, :] > 0.0,
        )
        thresholds = np.asarray([float(row["_threshold_ratio"]) for row in rows], dtype=np.float32)[:, None]
        activations = np.divide(
            ratio,
            np.maximum(thresholds, 1e-9),
            out=np.zeros_like(ratio, dtype=np.float32),
            where=thresholds > 0.0,
        )
        harmonic_state[harmonic_index] = {
            "rows": rows,
            "ratio": ratio,
            "activations": activations,
        }

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "building_frame_support",
            "harmonic_count": len(harmonic_state),
            "frame_count": len(frame_indices),
        },
    )

    frame_support_rows: list[dict[str, Any]] = []
    preliminary_active = np.zeros(len(frame_indices), dtype=bool)
    strong_support = np.zeros(len(frame_indices), dtype=np.float32)
    harmonic_support_count = np.zeros(len(frame_indices), dtype=np.int32)
    top_harmonic_tokens: list[list[str]] = [[] for _ in frame_indices]

    for frame_pos, frame_index in enumerate(frame_indices):
        support_count = 0
        weighted = 0.0
        tokens = []
        for harmonic_index, state in harmonic_state.items():
            activations = state["activations"][:, frame_pos]
            if activations.size == 0:
                continue
            best_idx = int(np.argmax(activations))
            best_activation = float(activations[best_idx])
            if best_activation >= 1.0:
                support_count += 1
            weighted += min(3.0, best_activation)
            if best_activation >= 0.8:
                tokens.append(str(state["rows"][best_idx].get("observed_note_token", "")).strip())
        harmonic_support_count[frame_pos] = support_count
        strong_support[frame_pos] = weighted
        top_harmonic_tokens[frame_pos] = tokens[:8]
        if support_count >= int(args.min_active_harmonics) and weighted >= float(args.min_weighted_support):
            preliminary_active[frame_pos] = True

    active_mask = _bridge_boolean(preliminary_active, int(args.bridge_inactive_gap))

    resolved_cloud_rows: list[dict[str, Any]] = []
    family_path_rows: list[dict[str, Any]] = []
    continuity_source_counts: Counter[str] = Counter()

    for harmonic_index, state in harmonic_state.items():
        rows = state["rows"]
        activations = state["activations"]
        ratio = state["ratio"]
        prev_idx: int | None = None
        prev_active = False
        prev_probe: int | None = None
        resolved_frames: list[int] = []
        held_frames: list[int] = []
        observed_frames: list[int] = []
        handoff_count = 0
        for frame_pos, frame_index in enumerate(frame_indices):
            if not active_mask[frame_pos]:
                prev_active = False
                continue
            frame_acts = activations[:, frame_pos]
            candidate_indices = [
                idx
                for idx, act in enumerate(frame_acts.tolist())
                if act >= float(args.candidate_min_activation)
            ]
            source = ""
            chosen_idx: int | None = None
            if candidate_indices:
                best_score = -1e9
                for idx in candidate_indices:
                    row = rows[idx]
                    score = float(frame_acts[idx])
                    if prev_probe is not None:
                        probe_gap = abs(int(row["_probe_index"]) - int(prev_probe))
                        if probe_gap == 0:
                            score += float(args.same_probe_bonus)
                        elif probe_gap == 1:
                            score += float(args.probe_adjacency_bonus)
                        elif probe_gap == 2:
                            score += float(args.probe_adjacency_bonus) * 0.5
                    score += float(args.score_weight_extraction) * float(row["_extraction_score"])
                    if score > best_score:
                        best_score = score
                        chosen_idx = idx
                source = "observed"
            elif prev_idx is not None and prev_active:
                prev_activation = float(frame_acts[prev_idx])
                if prev_activation >= float(args.hold_min_activation):
                    chosen_idx = prev_idx
                    source = "held"
            if chosen_idx is None:
                prev_active = False
                continue
            row = rows[chosen_idx]
            resolved_frames.append(frame_index)
            if source == "held":
                held_frames.append(frame_index)
            else:
                observed_frames.append(frame_index)
            if prev_idx is not None and prev_idx != chosen_idx:
                handoff_count += 1
            prev_idx = chosen_idx
            prev_probe = int(row["_probe_index"])
            prev_active = True
            continuity_source_counts[source] += 1
            resolved_cloud_rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": frame_index / FPS60,
                    "harmonic_index": harmonic_index,
                    "probe_index": int(row["_probe_index"]),
                    "observed_note_token": str(row.get("observed_note_token", "")).strip(),
                    "frequency_hz": _safe_float(row.get("frequency_hz"), 0.0),
                    "activation": float(frame_acts[chosen_idx]),
                    "raw_energy": float(ratio[chosen_idx, frame_pos] * frame_p95[frame_pos]),
                    "energy_over_frame_p95": float(ratio[chosen_idx, frame_pos]),
                    "continuity_source": source,
                }
            )
        family_path_rows.append(
            {
                "harmonic_index": harmonic_index,
                "resolved_frame_count": len(resolved_frames),
                "observed_frame_count": len(observed_frames),
                "held_frame_count": len(held_frames),
                "handoff_count": handoff_count,
                "resolved_segments_json": _segments_json(resolved_frames),
                "held_segments_json": _segments_json(held_frames),
                "observed_segments_json": _segments_json(observed_frames),
            }
        )

    resolved_union_frames = sorted({_safe_int(row.get("frame_index"), -1) for row in resolved_cloud_rows if _safe_int(row.get("frame_index"), -1) >= 0})
    resolved_segments = _segments(resolved_union_frames)

    for frame_pos, frame_index in enumerate(frame_indices):
        resolved_count = sum(1 for row in resolved_cloud_rows if _safe_int(row.get("frame_index"), -1) == frame_index)
        frame_support_rows.append(
            {
                "frame_index": frame_index,
                "time_sec": frame_index / FPS60,
                "preliminary_active": int(preliminary_active[frame_pos]),
                "resolved_active": int(active_mask[frame_pos]),
                "harmonic_support_count": int(harmonic_support_count[frame_pos]),
                "weighted_support": float(strong_support[frame_pos]),
                "resolved_harmonic_count": resolved_count,
                "top_tokens_json": json.dumps(top_harmonic_tokens[frame_pos], ensure_ascii=False),
            }
        )

    _write_csv(
        Path(args.out_resolved_cloud_csv),
        resolved_cloud_rows,
        [
            "frame_index",
            "time_sec",
            "harmonic_index",
            "probe_index",
            "observed_note_token",
            "frequency_hz",
            "activation",
            "raw_energy",
            "energy_over_frame_p95",
            "continuity_source",
        ],
    )
    _write_csv(
        Path(args.out_frame_support_csv),
        frame_support_rows,
        [
            "frame_index",
            "time_sec",
            "preliminary_active",
            "resolved_active",
            "harmonic_support_count",
            "weighted_support",
            "resolved_harmonic_count",
            "top_tokens_json",
        ],
    )
    _write_csv(
        Path(args.out_family_paths_csv),
        family_path_rows,
        [
            "harmonic_index",
            "resolved_frame_count",
            "observed_frame_count",
            "held_frame_count",
            "handoff_count",
            "resolved_segments_json",
            "held_segments_json",
            "observed_segments_json",
        ],
    )

    summary_lines = [
        "WINDOW BOWED CONTINUITY RESOLVER",
        "=" * 72,
        f"selected_harmonic_count            : {len(harmonic_state)}",
        f"preliminary_active_frames          : {int(np.sum(preliminary_active))}",
        f"resolved_active_frames             : {int(np.sum(active_mask))}",
        f"resolved_union_segment_count       : {len(resolved_segments)}",
        f"resolved_cloud_row_count           : {len(resolved_cloud_rows)}",
        f"observed_rows                      : {continuity_source_counts.get('observed', 0)}",
        f"held_rows                          : {continuity_source_counts.get('held', 0)}",
        "",
        "resolved_union_segments:",
    ]
    for start, end in resolved_segments:
        summary_lines.append(f"  {start}..{end} len={end - start + 1}")
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This layer resolves bowed continuity at the family level instead of the",
            "  render level. It keeps the bowed organism alive through weak frames when",
            "  neighboring harmonic families still support the same local continuity.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_continuity_resolver",
                "selected_harmonic_count": len(harmonic_state),
                "preliminary_active_frames": int(np.sum(preliminary_active)),
                "resolved_active_frames": int(np.sum(active_mask)),
                "resolved_cloud_row_count": len(resolved_cloud_rows),
                "observed_rows": continuity_source_counts.get("observed", 0),
                "held_rows": continuity_source_counts.get("held", 0),
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
            "resolved_active_frames": int(np.sum(active_mask)),
            "resolved_cloud_row_count": len(resolved_cloud_rows),
        },
    )


if __name__ == "__main__":
    main()
