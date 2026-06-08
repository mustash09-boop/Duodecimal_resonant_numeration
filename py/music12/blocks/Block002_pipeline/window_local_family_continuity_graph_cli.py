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


def _segments(sorted_frames: list[int]) -> list[tuple[int, int]]:
    if not sorted_frames:
        return []
    out: list[tuple[int, int]] = []
    start = prev = sorted_frames[0]
    for frame in sorted_frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        out.append((start, prev))
        start = prev = frame
    out.append((start, prev))
    return out


def _segments_json(segments: list[tuple[int, int]]) -> str:
    return json.dumps([{"start": s, "end": e, "len": e - s + 1} for s, e in segments], ensure_ascii=False)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a local bowed-family continuity graph for a window, comparing baseline selected-owner continuity with relaxed raw-matrix continuity."
    )
    ap.add_argument("--selected-families-csv", required=True)
    ap.add_argument("--owner-rows-csv", required=True)
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--window-start-frame", type=int, required=True)
    ap.add_argument("--window-end-frame", type=int, required=True)
    ap.add_argument("--continuation-ratio", type=float, default=0.18)
    ap.add_argument("--frame-p95-activation-ratio", type=float, default=0.045)
    ap.add_argument("--edge-max-gap", type=int, default=2)
    ap.add_argument("--freq-rel-threshold", type=float, default=0.0025)
    ap.add_argument("--out-nodes-csv", required=True)
    ap.add_argument("--out-edges-csv", required=True)
    ap.add_argument("--out-gap-candidates-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    args = ap.parse_args()

    selected_rows = _load_csv(Path(args.selected_families_csv))
    owner_rows = _load_csv(Path(args.owner_rows_csv))
    matrix, _info = load_matrix_csv_memmap(args.probe_matrix_csv)

    selected_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in selected_rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        probe_index = _safe_int(row.get("probe_index"), -1)
        if harmonic_index <= 0 or probe_index < 0:
            continue
        selected_by_key[(harmonic_index, probe_index)] = row

    baseline_frames_by_key: dict[tuple[int, int], set[int]] = defaultdict(set)
    union_baseline_frames: set[int] = set()
    for row in owner_rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        probe_index = _safe_int(row.get("probe_index"), -1)
        key = (harmonic_index, probe_index)
        if key not in selected_by_key:
            continue
        frame_index = _safe_int(row.get("frame_index"), -1)
        if not (int(args.window_start_frame) <= frame_index <= int(args.window_end_frame)):
            continue
        baseline_frames_by_key[key].add(frame_index)
        union_baseline_frames.add(frame_index)

    frame_indices = list(range(int(args.window_start_frame), int(args.window_end_frame) + 1))
    frame_slice = np.asarray(matrix[:, frame_indices], dtype=np.float32)
    frame_p95 = np.quantile(frame_slice, 0.95, axis=0).astype(np.float32)

    nodes: list[dict[str, Any]] = []
    raw_frames_by_key: dict[tuple[int, int], set[int]] = {}
    raw_segments_by_key: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for key, row in selected_by_key.items():
        harmonic_index, probe_index = key
        mean_ratio = _safe_float(row.get("mean_energy_over_frame_p95"), 0.0)
        threshold_ratio = mean_ratio * float(args.continuation_ratio)
        probe_series = np.asarray(matrix[probe_index, frame_indices], dtype=np.float32)
        ratio_series = np.divide(
            probe_series,
            np.maximum(frame_p95, 1e-9),
            out=np.zeros_like(probe_series, dtype=np.float32),
            where=frame_p95 > 0.0,
        )
        raw_active_positions = np.flatnonzero(
            (ratio_series >= threshold_ratio) &
            (ratio_series >= frame_p95 * 0.0 + float(args.frame_p95_activation_ratio))
        )
        raw_active_frames = {frame_indices[int(pos)] for pos in raw_active_positions.tolist()}
        raw_frames_by_key[key] = raw_active_frames
        raw_segments = _segments(sorted(raw_active_frames))
        raw_segments_by_key[key] = raw_segments

        baseline_segments = _segments(sorted(baseline_frames_by_key.get(key, set())))
        nodes.append(
            {
                "family_id": f"h{harmonic_index}_p{probe_index}",
                "harmonic_index": harmonic_index,
                "probe_index": probe_index,
                "observed_note_token": str(row.get("observed_note_token", "")).strip(),
                "frequency_hz": _safe_float(row.get("frequency_hz"), 0.0),
                "extraction_score": _safe_float(row.get("extraction_score"), 0.0),
                "baseline_active_frame_count": len(baseline_frames_by_key.get(key, set())),
                "baseline_segments_json": _segments_json(baseline_segments),
                "raw_active_frame_count": len(raw_active_frames),
                "raw_segments_json": _segments_json(raw_segments),
                "raw_mean_ratio": _mean([float(ratio_series[int(pos)]) for pos in raw_active_positions.tolist()]),
                "raw_peak_ratio": _safe_float(np.max(ratio_series[raw_active_positions]), 0.0) if raw_active_positions.size else 0.0,
                "threshold_ratio": threshold_ratio,
            }
        )

    baseline_union_segments = _segments(sorted(union_baseline_frames))
    internal_gaps: list[tuple[int, int]] = []
    for left, right in zip(baseline_union_segments[:-1], baseline_union_segments[1:]):
        if right[0] > left[1] + 1:
            internal_gaps.append((left[1] + 1, right[0] - 1))

    gap_candidate_rows: list[dict[str, Any]] = []
    for gap_start, gap_end in internal_gaps:
        for key, raw_frames in raw_frames_by_key.items():
            bridge_frames = sorted(frame for frame in raw_frames if gap_start <= frame <= gap_end)
            if not bridge_frames:
                continue
            row = selected_by_key[key]
            harmonic_index, probe_index = key
            gap_candidate_rows.append(
                {
                    "gap_start_frame": gap_start,
                    "gap_end_frame": gap_end,
                    "gap_len": gap_end - gap_start + 1,
                    "family_id": f"h{harmonic_index}_p{probe_index}",
                    "harmonic_index": harmonic_index,
                    "probe_index": probe_index,
                    "observed_note_token": str(row.get("observed_note_token", "")).strip(),
                    "frequency_hz": _safe_float(row.get("frequency_hz"), 0.0),
                    "bridge_frame_count": len(bridge_frames),
                    "first_bridge_frame": bridge_frames[0],
                    "last_bridge_frame": bridge_frames[-1],
                    "covers_full_gap": int(bridge_frames[0] == gap_start and bridge_frames[-1] == gap_end),
                }
            )

    edges: list[dict[str, Any]] = []
    keys_sorted = sorted(selected_by_key)
    for idx, key_a in enumerate(keys_sorted):
        row_a = selected_by_key[key_a]
        segs_a = raw_segments_by_key.get(key_a, [])
        if not segs_a:
            continue
        freq_a = _safe_float(row_a.get("frequency_hz"), 0.0)
        for key_b in keys_sorted[idx + 1:]:
            if key_a[0] != key_b[0]:
                continue
            row_b = selected_by_key[key_b]
            freq_b = _safe_float(row_b.get("frequency_hz"), 0.0)
            rel = abs(freq_a - freq_b) / max(freq_a, 1e-9)
            if rel > float(args.freq_rel_threshold):
                continue
            segs_b = raw_segments_by_key.get(key_b, [])
            if not segs_b:
                continue
            best_kind = ""
            best_gap = 10**9
            overlap_frames = 0
            for sa, ea in segs_a:
                for sb, eb in segs_b:
                    if sb <= ea and sa <= eb:
                        overlap = min(ea, eb) - max(sa, sb) + 1
                        if overlap > overlap_frames:
                            overlap_frames = overlap
                            best_kind = "overlap"
                            best_gap = 0
                    else:
                        gap = max(sb - ea - 1, sa - eb - 1)
                        if 0 <= gap <= int(args.edge_max_gap) and gap < best_gap:
                            best_kind = "handoff"
                            best_gap = gap
            if not best_kind:
                continue
            edges.append(
                {
                    "from_family_id": f"h{key_a[0]}_p{key_a[1]}",
                    "to_family_id": f"h{key_b[0]}_p{key_b[1]}",
                    "harmonic_index": key_a[0],
                    "edge_kind": best_kind,
                    "freq_rel_diff": rel,
                    "segment_gap": best_gap if best_gap < 10**9 else "",
                    "overlap_frames": overlap_frames,
                }
            )

    _write_csv(
        Path(args.out_nodes_csv),
        nodes,
        [
            "family_id",
            "harmonic_index",
            "probe_index",
            "observed_note_token",
            "frequency_hz",
            "extraction_score",
            "baseline_active_frame_count",
            "baseline_segments_json",
            "raw_active_frame_count",
            "raw_segments_json",
            "raw_mean_ratio",
            "raw_peak_ratio",
            "threshold_ratio",
        ],
    )
    _write_csv(
        Path(args.out_edges_csv),
        edges,
        [
            "from_family_id",
            "to_family_id",
            "harmonic_index",
            "edge_kind",
            "freq_rel_diff",
            "segment_gap",
            "overlap_frames",
        ],
    )
    _write_csv(
        Path(args.out_gap_candidates_csv),
        gap_candidate_rows,
        [
            "gap_start_frame",
            "gap_end_frame",
            "gap_len",
            "family_id",
            "harmonic_index",
            "probe_index",
            "observed_note_token",
            "frequency_hz",
            "bridge_frame_count",
            "first_bridge_frame",
            "last_bridge_frame",
            "covers_full_gap",
        ],
    )

    edge_kind_counts = Counter(row["edge_kind"] for row in edges)
    gap_counts = Counter((row["gap_start_frame"], row["gap_end_frame"]) for row in gap_candidate_rows)
    top_gap_lines = []
    for (gap_start, gap_end), count in gap_counts.most_common(10):
        top_gap_lines.append(f"  {gap_start}..{gap_end} families={count}")

    summary_lines = [
        "WINDOW LOCAL FAMILY CONTINUITY GRAPH",
        "=" * 72,
        f"selected_family_count              : {len(selected_by_key)}",
        f"baseline_union_segment_count       : {len(baseline_union_segments)}",
        f"baseline_internal_gap_count        : {len(internal_gaps)}",
        f"graph_edge_count                   : {len(edges)}",
        f"overlap_edges                      : {edge_kind_counts.get('overlap', 0)}",
        f"handoff_edges                      : {edge_kind_counts.get('handoff', 0)}",
        f"gap_candidate_count                : {len(gap_candidate_rows)}",
        "",
        "baseline_union_segments:",
    ]
    for start, end in baseline_union_segments:
        summary_lines.append(f"  {start}..{end} len={end - start + 1}")
    summary_lines.extend(["", "top_gap_candidate_counts:"])
    summary_lines.extend(top_gap_lines or ["  none"])
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This is a structural diagnostic layer. It compares the sparse baseline",
            "  selected-owner continuity with relaxed raw continuity for the same chosen",
            "  bowed families, and then builds local overlap/handoff edges between nearby",
            "  same-harmonic families. Large synchronized baseline gaps indicate a failure",
            "  of global continuity logic rather than isolated missing harmonics.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
