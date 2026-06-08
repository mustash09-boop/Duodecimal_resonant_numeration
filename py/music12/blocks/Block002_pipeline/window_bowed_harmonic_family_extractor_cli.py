# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from resonance_candidate_inference_core import load_coords_csv, load_matrix_csv_memmap

FREQ_ERROR_SCALE = 0.015625


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max_run(sorted_frames: list[int]) -> int:
    if not sorted_frames:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(sorted_frames)):
        if sorted_frames[idx] == sorted_frames[idx - 1] + 1:
            current += 1
        else:
            if current > best:
                best = current
            current = 1
    if current > best:
        best = current
    return best


def _load_tolerance_windows(path: Path) -> dict[int, dict[str, float | str]]:
    rows = _load_csv(path)
    out: dict[int, dict[str, float | str]] = {}
    for row in rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        if harmonic_index <= 0 or harmonic_index in out:
            continue
        out[harmonic_index] = {
            "theoretical_hz": _safe_float(row.get("theoretical_hz"), 0.0),
            "lower_hz_tolerance": _safe_float(row.get("lower_hz_tolerance"), 0.0),
            "upper_hz_tolerance": _safe_float(row.get("upper_hz_tolerance"), 0.0),
            "theoretical_token": str(row.get("theoretical_token", "")).strip(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract expanded bowed harmonic probe families from the window and separate them from piano-ish ownership."
    )
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--probe-times-csv", required=True)
    ap.add_argument("--data-grounded-owner-csv", required=True)
    ap.add_argument("--window-observations-csv", required=True)
    ap.add_argument("--violin-dense-vs-theory-csv", required=True)
    ap.add_argument("--cello-dense-vs-theory-csv", required=True)
    ap.add_argument("--max-harmonic", type=int, default=7)
    ap.add_argument("--out-families-csv", required=True)
    ap.add_argument("--out-owner-rows-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    grounded_rows = _load_csv(Path(args.data_grounded_owner_csv))
    observation_rows = _load_csv(Path(args.window_observations_csv))
    times_rows = _load_csv(Path(args.probe_times_csv))
    selected_frames = sorted(
        {_safe_int(row.get("frame_index"), -1) for row in grounded_rows if _safe_int(row.get("frame_index"), -1) >= 0}
    )
    if not selected_frames:
        raise SystemExit("No selected frames found in data-grounded owner CSV.")

    lower_rows = [row for row in grounded_rows if str(row.get("data_grounded_support_role", "")).strip() == "LOWER_OCTAVE_SUPPORT"]
    upper_rows = [row for row in grounded_rows if str(row.get("data_grounded_support_role", "")).strip() == "UPPER_DIRECT_SUPPORT"]
    lower_root_hz = _mean([_safe_float(row.get("frequency_hz"), 0.0) for row in lower_rows])
    upper_h2_hz = _mean([_safe_float(row.get("frequency_hz"), 0.0) for row in upper_rows])
    root_hz = lower_root_hz if lower_root_hz > 0.0 else (upper_h2_hz / 2.0 if upper_h2_hz > 0.0 else 0.0)

    coords = load_coords_csv(args.probe_coords_csv)
    matrix, _info = load_matrix_csv_memmap(args.probe_matrix_csv)
    coord_freqs = np.asarray([float(coord.frequency_hz) for coord in coords], dtype=np.float64)
    frame_time_map = {
        _safe_int(row.get("frame_index"), -1): _safe_float(row.get("time_seconds"), 0.0)
        for row in times_rows
    }

    violin_windows = _load_tolerance_windows(Path(args.violin_dense_vs_theory_csv))
    cello_windows = _load_tolerance_windows(Path(args.cello_dense_vs_theory_csv))

    frame_indices_arr = np.asarray(selected_frames, dtype=np.int64)
    matrix_slice = np.asarray(matrix[:, frame_indices_arr], dtype=np.float32)
    frame_p95 = np.quantile(matrix_slice, 0.95, axis=0).astype(np.float32)

    owner_by_cell: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for row in observation_rows:
        frame_index = _safe_int(row.get("frame_index"), -1)
        probe_index = _safe_int(row.get("probe_index"), -1)
        if frame_index < 0 or probe_index < 0:
            continue
        owner_label = str(row.get("owner_label", "")).strip()
        if owner_label:
            owner_by_cell[(frame_index, probe_index)][owner_label] += 1

    harmonic_indices = list(range(1, max(1, int(args.max_harmonic)) + 1))
    threshold_by_h = {
        1: 0.30,
        2: 0.30,
        3: 0.10,
        4: 0.08,
        5: 0.10,
        6: 0.07,
        7: 0.03,
    }
    min_frames_by_h = {
        1: 8,
        2: 8,
        3: 8,
        4: 6,
        5: 8,
        6: 6,
        7: 4,
    }

    _write_progress(args.progress_json, {"status": "running", "phase": "extracting_families", "selected_frames": len(selected_frames)})

    family_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []

    for harmonic_index in harmonic_indices:
        violin_def = violin_windows.get(harmonic_index, {})
        cello_def = cello_windows.get(harmonic_index, {})
        expected_hz = root_hz * harmonic_index if root_hz > 0.0 else _safe_float(violin_def.get("theoretical_hz"), 0.0)
        low = min(
            _safe_float(violin_def.get("lower_hz_tolerance"), expected_hz * 0.985),
            _safe_float(cello_def.get("lower_hz_tolerance"), expected_hz * 0.985),
        )
        high = max(
            _safe_float(violin_def.get("upper_hz_tolerance"), expected_hz * 1.015),
            _safe_float(cello_def.get("upper_hz_tolerance"), expected_hz * 1.015),
        )
        left_idx = int(np.searchsorted(coord_freqs, low, side="left"))
        right_idx = int(np.searchsorted(coord_freqs, high, side="right"))
        if right_idx <= left_idx:
            continue

        band_slice = matrix_slice[left_idx:right_idx, :]
        band_indices = range(left_idx, right_idx)
        threshold = threshold_by_h.get(
            harmonic_index,
            max(0.006, 0.03 * (7.0 / float(max(7, harmonic_index)))),
        )
        min_frames = min_frames_by_h.get(
            harmonic_index,
            max(3, int(round(7.0 - min(3.0, 0.18 * float(harmonic_index - 7))))),
        )

        for rel_idx, coord_idx in enumerate(band_indices):
            coord = coords[coord_idx]
            energies = band_slice[rel_idx, :]
            ratios = np.divide(
                energies,
                frame_p95,
                out=np.zeros_like(energies, dtype=np.float32),
                where=frame_p95 > 0.0,
            )
            active_mask = ratios >= threshold
            active_pos = np.flatnonzero(active_mask)
            if active_pos.size < min_frames:
                continue
            active_frames = [selected_frames[int(pos)] for pos in active_pos.tolist()]
            owner_counts: Counter[str] = Counter()
            for pos in active_pos.tolist():
                frame_index = selected_frames[int(pos)]
                owner_counts.update(owner_by_cell.get((frame_index, coord.probe_index), Counter()))

            total_owner_hits = sum(owner_counts.values())
            pianoish_hits = sum(count for label, count in owner_counts.items() if "PIANOISH" in label)
            second_hits = owner_counts.get("SECOND_SUSTAIN_OWNER", 0) + owner_counts.get("SECOND_SUSTAIN_DATA_GROUNDED_OWNER", 0)
            body_hits = owner_counts.get("BODY_CONTINUATION_OWNER", 0)
            overlap_ratio = (float(pianoish_hits) / float(total_owner_hits)) if total_owner_hits > 0 else 0.0

            freq_error_ratio = abs(float(coord.frequency_hz) - expected_hz) / expected_hz if expected_hz > 0.0 else 0.0
            continuity_score = min(1.0, float(_max_run(active_frames)) / 19.0)
            coverage_ratio = float(active_pos.size) / float(len(selected_frames))
            strength_score = min(1.0, _mean([float(ratios[int(pos)]) for pos in active_pos.tolist()]) / 2.0)
            closeness_score = max(0.0, 1.0 - min(1.0, freq_error_ratio / FREQ_ERROR_SCALE))
            anti_piano_score = 1.0 - overlap_ratio
            extraction_score = (
                0.30 * continuity_score
                + 0.20 * coverage_ratio
                + 0.20 * strength_score
                + 0.15 * closeness_score
                + 0.15 * anti_piano_score
            )

            family_rows.append(
                {
                    "harmonic_index": harmonic_index,
                    "expected_hz": expected_hz,
                    "theoretical_token": str(violin_def.get("theoretical_token") or cello_def.get("theoretical_token") or ""),
                    "probe_index": coord.probe_index,
                    "observed_note_token": coord.note_token,
                    "frequency_hz": coord.frequency_hz,
                    "active_frame_count": int(active_pos.size),
                    "coverage_ratio": coverage_ratio,
                    "first_frame_index": active_frames[0],
                    "first_time_sec": frame_time_map.get(active_frames[0], 0.0),
                    "last_frame_index": active_frames[-1],
                    "last_time_sec": frame_time_map.get(active_frames[-1], 0.0),
                    "max_consecutive_frames": _max_run(active_frames),
                    "mean_energy_over_frame_p95": _mean([float(ratios[int(pos)]) for pos in active_pos.tolist()]),
                    "freq_error_ratio": freq_error_ratio,
                    "pianoish_overlap_ratio": overlap_ratio,
                    "second_owner_hits": second_hits,
                    "body_owner_hits": body_hits,
                    "owner_label_counts_json": json.dumps(dict(owner_counts), ensure_ascii=False),
                    "extraction_score": extraction_score,
                }
            )

            for pos in active_pos.tolist():
                frame_index = selected_frames[int(pos)]
                owner_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": frame_time_map.get(frame_index, 0.0),
                        "harmonic_index": harmonic_index,
                        "probe_index": coord.probe_index,
                        "observed_note_token": coord.note_token,
                        "frequency_hz": coord.frequency_hz,
                        "energy": float(energies[int(pos)]),
                        "energy_over_frame_p95": float(ratios[int(pos)]),
                        "expected_hz": expected_hz,
                        "freq_error_ratio": freq_error_ratio,
                        "pianoish_overlap_ratio": overlap_ratio,
                        "owner_label_counts_json": json.dumps(dict(owner_counts), ensure_ascii=False),
                        "expanded_owner_label": "BOWED_HARMONIC_OWNER_CANDIDATE",
                    }
                )

    family_rows.sort(key=lambda row: (_safe_int(row.get("harmonic_index"), 0), -_safe_float(row.get("extraction_score"), 0.0)))
    owner_rows.sort(key=lambda row: (_safe_int(row.get("frame_index"), 0), _safe_int(row.get("harmonic_index"), 0), -_safe_float(row.get("energy_over_frame_p95"), 0.0)))

    _write_csv(
        Path(args.out_families_csv),
        family_rows,
        [
            "harmonic_index",
            "expected_hz",
            "theoretical_token",
            "probe_index",
            "observed_note_token",
            "frequency_hz",
            "active_frame_count",
            "coverage_ratio",
            "first_frame_index",
            "first_time_sec",
            "last_frame_index",
            "last_time_sec",
            "max_consecutive_frames",
            "mean_energy_over_frame_p95",
            "freq_error_ratio",
            "pianoish_overlap_ratio",
            "second_owner_hits",
            "body_owner_hits",
            "owner_label_counts_json",
            "extraction_score",
        ],
    )
    _write_csv(
        Path(args.out_owner_rows_csv),
        owner_rows,
        [
            "frame_index",
            "time_sec",
            "harmonic_index",
            "probe_index",
            "observed_note_token",
            "frequency_hz",
            "energy",
            "energy_over_frame_p95",
            "expected_hz",
            "freq_error_ratio",
            "pianoish_overlap_ratio",
            "owner_label_counts_json",
            "expanded_owner_label",
        ],
    )

    harmonic_counts = Counter(_safe_int(row.get("harmonic_index"), 0) for row in family_rows)
    summary_lines = [
        "WINDOW BOWED HARMONIC FAMILY EXTRACTION",
        "=" * 72,
        f"selected_frames                    : {len(selected_frames)}",
        f"estimated_root_hz                  : {root_hz:.6f}",
        f"family_count                       : {len(family_rows)}",
        f"owner_rows_count                   : {len(owner_rows)}",
        "",
        "harmonic_family_counts:",
    ]
    for harmonic_index in harmonic_indices:
        summary_lines.append(f"  h{harmonic_index}: {harmonic_counts.get(harmonic_index, 0)}")
    summary_lines.extend(["", "top_families_by_harmonic:"])
    for harmonic_index in harmonic_indices:
        top_rows = [row for row in family_rows if _safe_int(row.get("harmonic_index"), 0) == harmonic_index][:3]
        for row in top_rows:
            summary_lines.append(
                "  "
                f"h{harmonic_index} probe={row['probe_index']} token={row['observed_note_token']} "
                f"score={_safe_float(row['extraction_score'], 0.0):.6f} "
                f"frames={row['active_frame_count']} "
                f"piano_overlap={_safe_float(row['pianoish_overlap_ratio'], 0.0):.6f}"
            )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This extraction expands the bowed layer from raw harmonic families h1..h7",
            "  and keeps per-probe continuity plus piano-overlap diagnostics instead of",
            "  cutting the signal only by time.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_harmonic_family_extraction",
                "selected_frames": len(selected_frames),
                "estimated_root_hz": root_hz,
                "family_count": len(family_rows),
                "owner_rows_count": len(owner_rows),
                "harmonic_family_counts": dict(harmonic_counts),
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
            "family_count": len(family_rows),
            "owner_rows_count": len(owner_rows),
        },
    )


if __name__ == "__main__":
    main()
