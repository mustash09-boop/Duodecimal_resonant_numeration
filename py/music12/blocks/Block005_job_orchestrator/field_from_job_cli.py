from __future__ import annotations

import argparse
import json
from pathlib import Path

from music12.blocks.Block002_audio_recogn.resonance_field_builder_core import (
    build_resonance_field_events,
    build_resonance_trajectories,
    field_events_to_rows,
    load_probe_coords_delta_csv,
    load_probe_matrix_csv,
    load_probe_times_csv,
    trajectories_to_rows,
    write_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one distributed resonance-field job from JobSpec JSON")
    ap.add_argument("--job_json", required=True)
    args = ap.parse_args()

    job_json = Path(args.job_json).resolve()
    if not job_json.exists():
        raise SystemExit(f"job_json not found: {job_json}")

    spec = json.loads(job_json.read_text(encoding="utf-8"))

    task_kind = str(spec.get("task_kind", "")).strip()
    if task_kind != "resonance_field_segment":
        raise SystemExit(f"Unsupported task_kind: {task_kind!r}")

    extra = spec.get("extra_args", {})
    if not isinstance(extra, dict):
        raise SystemExit("extra_args must be a dict")

    matrix_csv = Path(str(extra["matrix_csv"]))
    coords_delta_csv = Path(str(extra["coords_delta_csv"]))
    times_csv = Path(str(extra["times_csv"]))

    frame_start = int(extra["frame_start"])
    frame_end = int(extra["frame_end"])  # exclusive

    energy_threshold = float(extra.get("energy_threshold", 0.0))
    top_k_per_frame = int(extra.get("top_k_per_frame", 0))
    max_time_gap_sec = float(extra.get("max_time_gap_sec", 0.05))
    max_phase_gap_deg = float(extra.get("max_phase_gap_deg", 3.0))
    max_radial_gap = float(extra.get("max_radial_gap", 0.35))

    out_dir = Path(str(spec["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_probe_matrix_csv(matrix_csv)
    coords = load_probe_coords_delta_csv(coords_delta_csv)
    times = load_probe_times_csv(times_csv)

    if frame_start < 0 or frame_end <= frame_start:
        raise SystemExit(f"Bad frame range: {frame_start}..{frame_end}")
    if frame_end > matrix.shape[1]:
        raise SystemExit(f"frame_end={frame_end} exceeds matrix frame count={matrix.shape[1]}")
    if len(times) != matrix.shape[1]:
        raise SystemExit(f"times length {len(times)} != matrix frame count {matrix.shape[1]}")

    matrix_seg = matrix[:, frame_start:frame_end]
    times_seg = times[frame_start:frame_end]

    events = build_resonance_field_events(
        matrix_seg,
        coords,
        times_seg,
        energy_threshold=energy_threshold,
        top_k_per_frame=top_k_per_frame,
    )

    trajectories = build_resonance_trajectories(
        events,
        max_time_gap_sec=max_time_gap_sec,
        max_phase_gap_deg=max_phase_gap_deg,
        max_radial_gap=max_radial_gap,
    )

    events_csv = out_dir / "field_events.csv"
    trajectories_csv = out_dir / "field_trajectories.csv"
    meta_json = out_dir / "field_meta.json"

    write_csv(events_csv, field_events_to_rows(events))
    write_csv(trajectories_csv, trajectories_to_rows(trajectories))

    meta = {
        "job_id": spec.get("job_id", ""),
        "task_kind": task_kind,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_end - frame_start,
        "event_count": len(events),
        "trajectory_count": len(trajectories),
        "outputs": {
            "field_events_csv": str(events_csv),
            "field_trajectories_csv": str(trajectories_csv),
            "field_meta_json": str(meta_json),
        },
        "params": {
            "energy_threshold": energy_threshold,
            "top_k_per_frame": top_k_per_frame,
            "max_time_gap_sec": max_time_gap_sec,
            "max_phase_gap_deg": max_phase_gap_deg,
            "max_radial_gap": max_radial_gap,
        },
    }

    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()