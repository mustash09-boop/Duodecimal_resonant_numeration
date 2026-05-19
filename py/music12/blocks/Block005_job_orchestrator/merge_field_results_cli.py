from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge resonance-field segment results")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--job_prefix", required=True)
    ap.add_argument("--out_events_csv", required=True)
    ap.add_argument("--out_trajectories_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    seg_dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(args.job_prefix)])

    all_events: list[dict] = []
    all_traj: list[dict] = []
    traj_offset = 0
    merged_segments = []

    for seg in seg_dirs:
        events = load_csv(seg / "field_events.csv")
        trajs = load_csv(seg / "field_trajectories.csv")
        meta_path = seg / "field_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        for row in trajs:
            row["trajectory_id"] = str(int(row["trajectory_id"]) + traj_offset)

        all_events.extend(events)
        all_traj.extend(trajs)
        traj_offset += len(trajs)

        merged_segments.append(
            {
                "segment_dir": str(seg),
                "event_count": len(events),
                "trajectory_count": len(trajs),
                "frame_start": meta.get("frame_start"),
                "frame_end": meta.get("frame_end"),
            }
        )

    out_events_csv = Path(args.out_events_csv).resolve()
    out_trajectories_csv = Path(args.out_trajectories_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    write_csv(out_events_csv, all_events)
    write_csv(out_trajectories_csv, all_traj)

    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.write_text(
        json.dumps(
            {
                "results_root": str(results_root),
                "job_prefix": args.job_prefix,
                "segment_count": len(seg_dirs),
                "merged_event_count": len(all_events),
                "merged_trajectory_count": len(all_traj),
                "segments": merged_segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "segment_count": len(seg_dirs),
            "merged_event_count": len(all_events),
            "merged_trajectory_count": len(all_traj),
            "out_events_csv": str(out_events_csv),
            "out_trajectories_csv": str(out_trajectories_csv),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()