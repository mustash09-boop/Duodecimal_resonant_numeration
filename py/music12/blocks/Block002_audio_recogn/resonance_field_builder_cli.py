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


def write_meta_json(
    path: Path,
    *,
    matrix_csv: Path,
    coords_csv: Path,
    times_csv: Path,
    events_csv: Path,
    trajectories_csv: Path,
    event_count: int,
    trajectory_count: int,
    params: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "matrix_csv": str(matrix_csv),
            "coords_delta_csv": str(coords_csv),
            "times_csv": str(times_csv),
        },
        "outputs": {
            "field_events_csv": str(events_csv),
            "field_trajectories_csv": str(trajectories_csv),
            "meta_json": str(path),
        },
        "event_count": event_count,
        "trajectory_count": trajectory_count,
        "params": params,
        "semantic_note": (
            "Spiral-based resonance field builder. "
            "No phase/radial geometry is used. "
            "Events and trajectories are built from note tokens mapped to SpiralPosition."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build spiral-based resonance field events and trajectories from probe data"
    )

    ap.add_argument("--matrix_csv", required=True)
    ap.add_argument("--coords_delta_csv", required=True)
    ap.add_argument("--times_csv", required=True)
    ap.add_argument("--out_field_events_csv", required=True)
    ap.add_argument("--out_field_trajectories_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)

    ap.add_argument("--energy_threshold", type=float, default=0.0)
    ap.add_argument("--top_k_per_frame", type=int, default=0)

    ap.add_argument("--max_time_gap_sec", type=float, default=0.05)
    ap.add_argument("--max_arc_gap", type=float, default=0.5)
    ap.add_argument("--min_events_per_trajectory", type=int, default=2)

    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv).resolve()
    coords_csv = Path(args.coords_delta_csv).resolve()
    times_csv = Path(args.times_csv).resolve()

    out_field_events_csv = Path(args.out_field_events_csv).resolve()
    out_field_trajectories_csv = Path(args.out_field_trajectories_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    matrix = load_probe_matrix_csv(matrix_csv)
    coords = load_probe_coords_delta_csv(coords_csv)
    times = load_probe_times_csv(times_csv)

    events = build_resonance_field_events(
        matrix=matrix,
        times=times,
        coords=coords,
        energy_threshold=args.energy_threshold,
        top_k_per_frame=args.top_k_per_frame,
    )

    trajectories = build_resonance_trajectories(
        events=events,
        max_time_gap_sec=args.max_time_gap_sec,
        max_arc_gap=args.max_arc_gap,
        min_events_per_trajectory=args.min_events_per_trajectory,
    )

    event_rows = field_events_to_rows(events)
    trajectory_rows = trajectories_to_rows(trajectories)

    write_csv(out_field_events_csv, event_rows)
    write_csv(out_field_trajectories_csv, trajectory_rows)

    write_meta_json(
        out_meta_json,
        matrix_csv=matrix_csv,
        coords_csv=coords_csv,
        times_csv=times_csv,
        events_csv=out_field_events_csv,
        trajectories_csv=out_field_trajectories_csv,
        event_count=len(event_rows),
        trajectory_count=len(trajectory_rows),
        params={
            "energy_threshold": args.energy_threshold,
            "top_k_per_frame": args.top_k_per_frame,
            "max_time_gap_sec": args.max_time_gap_sec,
            "max_arc_gap": args.max_arc_gap,
            "min_events_per_trajectory": args.min_events_per_trajectory,
        },
    )

    print("spiral resonance field build complete")
    print(json.dumps(
        {
            "event_count": len(event_rows),
            "trajectory_count": len(trajectory_rows),
            "out_field_events_csv": str(out_field_events_csv),
            "out_field_trajectories_csv": str(out_field_trajectories_csv),
            "out_meta_json": str(out_meta_json),
            "energy_threshold": args.energy_threshold,
            "top_k_per_frame": args.top_k_per_frame,
            "max_time_gap_sec": args.max_time_gap_sec,
            "max_arc_gap": args.max_arc_gap,
            "min_events_per_trajectory": args.min_events_per_trajectory,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()