# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from resonance_candidate_inference_core import load_coords_csv, load_matrix_csv_memmap


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_tolerance_windows(path: Path) -> dict[int, dict[str, float]]:
    rows = _load_csv(path)
    out: dict[int, dict[str, float]] = {}
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


def _best_track_amplitudes(path: Path) -> dict[int, float]:
    data = _load_json(path)
    best = dict(data.get("best_track", {}) or {})
    hits = best.get("representative_hits", []) or []
    out: dict[int, float] = {}
    for hit in hits:
        idx = _safe_int(hit.get("harmonic_index"), 0)
        if idx <= 0:
            continue
        out[idx] = _safe_float(hit.get("matched_amplitude"), 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan raw probe matrix for hidden bowed harmonics linked to the current second-sustain window."
    )
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--probe-times-csv", required=True)
    ap.add_argument("--probe-matrix-csv", required=True)
    ap.add_argument("--data-grounded-owner-csv", required=True)
    ap.add_argument("--violin-dense-vs-theory-csv", required=True)
    ap.add_argument("--violin-chain-json", required=True)
    ap.add_argument("--cello-dense-vs-theory-csv", required=True)
    ap.add_argument("--cello-chain-json", required=True)
    ap.add_argument("--out-frame-csv", required=True)
    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    grounded_rows = _load_csv(Path(args.data_grounded_owner_csv))
    selected_frames = sorted(
        {_safe_int(row.get("frame_index"), -1) for row in grounded_rows if _safe_int(row.get("frame_index"), -1) >= 0}
    )
    lower_rows = [row for row in grounded_rows if str(row.get("data_grounded_support_role", "")).strip() == "LOWER_OCTAVE_SUPPORT"]
    upper_rows = [row for row in grounded_rows if str(row.get("data_grounded_support_role", "")).strip() == "UPPER_DIRECT_SUPPORT"]
    lower_root_hz = _mean([_safe_float(row.get("frequency_hz"), 0.0) for row in lower_rows])
    upper_h2_hz = _mean([_safe_float(row.get("frequency_hz"), 0.0) for row in upper_rows])
    root_hz = lower_root_hz if lower_root_hz > 0.0 else (upper_h2_hz / 2.0 if upper_h2_hz > 0.0 else 0.0)

    coords = load_coords_csv(args.probe_coords_csv)
    matrix, _info = load_matrix_csv_memmap(args.probe_matrix_csv)
    times_rows = _load_csv(Path(args.probe_times_csv))
    time_by_frame = {
        _safe_int(row.get("frame_index"), -1): _safe_float(row.get("time_seconds"), 0.0)
        for row in times_rows
    }
    coord_freqs = np.asarray([float(coord.frequency_hz) for coord in coords], dtype=np.float64)

    violin_windows = _load_tolerance_windows(Path(args.violin_dense_vs_theory_csv))
    cello_windows = _load_tolerance_windows(Path(args.cello_dense_vs_theory_csv))
    violin_hamp = _best_track_amplitudes(Path(args.violin_chain_json))
    cello_hamp = _best_track_amplitudes(Path(args.cello_chain_json))

    _write_progress(args.progress_json, {"status": "running", "phase": "scanning_raw_harmonics", "selected_frames": len(selected_frames)})

    frame_rows: list[dict[str, Any]] = []
    by_harmonic: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(1, 13)}

    for frame_index in selected_frames:
        frame_values = np.asarray(matrix[:, frame_index], dtype=np.float32)
        frame_p95 = float(np.quantile(frame_values, 0.95))
        frame_p99 = float(np.quantile(frame_values, 0.99))
        for harmonic_index in range(1, 13):
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
            probe_slice = frame_values[left_idx:right_idx]
            if probe_slice.size == 0:
                continue
            local_best_rel = int(np.argmax(probe_slice))
            local_best_idx = left_idx + local_best_rel
            local_best_energy = float(frame_values[local_best_idx])
            coord = coords[local_best_idx]
            violin_amp = _safe_float(violin_hamp.get(harmonic_index), 0.0)
            cello_amp = _safe_float(cello_hamp.get(harmonic_index), 0.0)
            passport_ref_amp = max(violin_amp, cello_amp)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": time_by_frame.get(frame_index, 0.0),
                    "harmonic_index": harmonic_index,
                    "estimated_root_hz": root_hz,
                    "expected_hz": expected_hz,
                    "scan_low_hz": low,
                    "scan_high_hz": high,
                    "best_probe_index": coord.probe_index,
                    "best_note_token": coord.note_token,
                    "best_frequency_hz": coord.frequency_hz,
                    "best_energy": local_best_energy,
                    "energy_over_frame_p95": (local_best_energy / frame_p95) if frame_p95 > 0.0 else 0.0,
                    "energy_over_frame_p99": (local_best_energy / frame_p99) if frame_p99 > 0.0 else 0.0,
                    "violin_passport_ref_amp": violin_amp,
                    "cello_passport_ref_amp": cello_amp,
                    "passport_ref_amp_max": passport_ref_amp,
                }
            )
            by_harmonic[harmonic_index].append(frame_rows[-1])

    summary_rows: list[dict[str, Any]] = []
    for harmonic_index in range(1, 13):
        hits = by_harmonic[harmonic_index]
        best_energies = [_safe_float(row.get("best_energy"), 0.0) for row in hits]
        over_p95 = [_safe_float(row.get("energy_over_frame_p95"), 0.0) for row in hits]
        over_p99 = [_safe_float(row.get("energy_over_frame_p99"), 0.0) for row in hits]
        active_hits = [row for row in hits if _safe_float(row.get("energy_over_frame_p95"), 0.0) >= 0.12]
        active_frames = sorted(_safe_int(row.get("frame_index"), -1) for row in active_hits if _safe_int(row.get("frame_index"), -1) >= 0)
        token_counts: dict[str, int] = {}
        for row in active_hits:
            token = str(row.get("best_note_token", "")).strip()
            token_counts[token] = token_counts.get(token, 0) + 1
        dominant_token = max(token_counts.items(), key=lambda item: item[1])[0] if token_counts else ""
        first_time = min((_safe_float(row.get("time_sec"), 0.0) for row in active_hits), default=0.0)
        summary_rows.append(
            {
                "harmonic_index": harmonic_index,
                "active_frame_count": len(active_frames),
                "coverage_ratio": (float(len(active_frames)) / float(len(selected_frames))) if selected_frames else 0.0,
                "first_time_sec": first_time if active_hits else "",
                "first_relative_sec": (first_time - time_by_frame.get(selected_frames[0], 0.0)) if active_hits and selected_frames else "",
                "max_consecutive_frames": _max_run(active_frames),
                "mean_best_energy": _mean(best_energies),
                "mean_energy_over_frame_p95": _mean(over_p95),
                "mean_energy_over_frame_p99": _mean(over_p99),
                "dominant_token": dominant_token,
                "token_counts_json": json.dumps(token_counts, ensure_ascii=False),
            }
        )

    _write_csv(
        Path(args.out_frame_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "harmonic_index",
            "estimated_root_hz",
            "expected_hz",
            "scan_low_hz",
            "scan_high_hz",
            "best_probe_index",
            "best_note_token",
            "best_frequency_hz",
            "best_energy",
            "energy_over_frame_p95",
            "energy_over_frame_p99",
            "violin_passport_ref_amp",
            "cello_passport_ref_amp",
            "passport_ref_amp_max",
        ],
    )
    _write_csv(
        Path(args.out_summary_csv),
        summary_rows,
        [
            "harmonic_index",
            "active_frame_count",
            "coverage_ratio",
            "first_time_sec",
            "first_relative_sec",
            "max_consecutive_frames",
            "mean_best_energy",
            "mean_energy_over_frame_p95",
            "mean_energy_over_frame_p99",
            "dominant_token",
            "token_counts_json",
        ],
    )

    h_focus = {row["harmonic_index"]: row for row in summary_rows}
    summary_lines = [
        "WINDOW RAW HARMONIC PROBE SCAN",
        "=" * 72,
        f"selected_frames                    : {len(selected_frames)}",
        f"estimated_root_hz                  : {root_hz:.6f}",
        f"lower_root_mean_hz                 : {lower_root_hz:.6f}",
        f"upper_h2_mean_hz                   : {upper_h2_hz:.6f}",
        "",
    ]
    for idx in (1, 2, 3, 5, 7):
        row = h_focus.get(idx)
        if not row:
            continue
        summary_lines.append(
            f"h{idx}: active_frames={row['active_frame_count']} "
            f"coverage={_safe_float(row.get('coverage_ratio'), 0.0):.6f} "
            f"first_rel={_safe_float(row.get('first_relative_sec'), -1.0):.6f} "
            f"mean_p95={_safe_float(row.get('mean_energy_over_frame_p95'), 0.0):.6f} "
            f"token={row['dominant_token']}"
        )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This scan works at raw probe level on the same frames as the bowed second layer.",
            "  It is intended to reveal hidden or masked h1/h3/h5/h7 support that earlier",
            "  owner layers may have discarded.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_raw_harmonic_probe_scan",
                "selected_frames": len(selected_frames),
                "estimated_root_hz": root_hz,
                "lower_root_mean_hz": lower_root_hz,
                "upper_h2_mean_hz": upper_h2_hz,
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
            "selected_frames": len(selected_frames),
            "estimated_root_hz": root_hz,
        },
    )


if __name__ == "__main__":
    main()
