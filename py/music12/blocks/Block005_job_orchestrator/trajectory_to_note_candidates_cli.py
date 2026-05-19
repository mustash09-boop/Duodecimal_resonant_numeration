from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_note_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    for r in rows:
        event_count = int(float(r.get("event_count", "0") or 0))
        duration_sec = float(r.get("duration_sec", "0") or 0.0)
        temporal_density = float(r.get("temporal_density", "0") or 0.0)
        energy_sum = float(r.get("energy_sum", "0") or 0.0)
        energy_mean = float(r.get("energy_mean", "0") or 0.0)

        # пока без схлопывания и без утрат:
        # просто превращаем траекторию в "кандидата в ноту"
        # с явными признаками, чтобы потом фильтровать/объединять по теории.
        score = (
            duration_sec * 3.0
            + temporal_density * 2.0
            + min(event_count / 20.0, 5.0)
            + min(energy_mean / 100.0, 5.0)
        )

        out.append(
            {
                "note_candidate_id": r.get("trajectory_id", ""),
                "source_trajectory_id": r.get("trajectory_id", ""),
                "time_start_sec": r.get("time_start_sec", ""),
                "time_end_sec": r.get("time_end_sec", ""),
                "duration_sec": r.get("duration_sec", ""),
                "event_count": r.get("event_count", ""),
                "temporal_density": r.get("temporal_density", ""),
                "dominant_probe_index": r.get("dominant_probe_index", ""),
                "octave_mode": r.get("octave_mode", ""),
                "degree12_mode": r.get("degree12_mode", ""),
                "subdivisions_mode": r.get("subdivisions_mode", ""),
                "delta_radius_mode": r.get("delta_radius_mode", ""),
                "delta_vector_mode": r.get("delta_vector_mode", ""),
                "mean_phase_deg": r.get("mean_phase_deg", ""),
                "mean_radial_level": r.get("mean_radial_level", ""),
                "energy_sum": r.get("energy_sum", ""),
                "energy_mean": r.get("energy_mean", ""),
                "note_score": f"{score:.6f}",
            }
        )

    out.sort(
        key=lambda x: (
            float(x["time_start_sec"] or 0.0),
            float(x["time_end_sec"] or 0.0),
            int(x["source_trajectory_id"] or 0),
        )
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert merged field trajectories into note candidates")
    ap.add_argument("--in_trajectories_csv", required=True)
    ap.add_argument("--out_note_candidates_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    in_csv = Path(args.in_trajectories_csv).resolve()
    out_csv = Path(args.out_note_candidates_csv).resolve()
    out_meta = Path(args.out_meta_json).resolve()

    rows = load_rows(in_csv)
    note_rows = build_note_candidates(rows)
    write_rows(out_csv, note_rows)

    meta = {
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "trajectory_rows": len(rows),
        "note_candidate_rows": len(note_rows),
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()