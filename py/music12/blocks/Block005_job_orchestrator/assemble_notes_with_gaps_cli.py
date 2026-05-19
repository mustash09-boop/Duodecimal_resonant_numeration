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


def same_note_family(a: dict[str, str], b: dict[str, str], phase_tol_deg: float) -> bool:
    if a.get("octave_mode", "") != b.get("octave_mode", ""):
        return False
    if a.get("degree12_mode", "") != b.get("degree12_mode", ""):
        return False
    if a.get("subdivisions_mode", "") != b.get("subdivisions_mode", ""):
        return False
    if a.get("delta_vector_mode", "") != b.get("delta_vector_mode", ""):
        return False

    pa = float(a.get("mean_phase_deg", 0.0) or 0.0)
    pb = float(b.get("mean_phase_deg", 0.0) or 0.0)
    d = abs(pa - pb) % 360.0
    d = min(d, 360.0 - d)
    return d <= phase_tol_deg


def assemble_notes(
    rows: list[dict[str, str]],
    *,
    max_join_gap_sec: float,
    phase_tol_deg: float,
    gap_marker_threshold_sec: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = sorted(
        rows,
        key=lambda r: (
            float(r.get("time_start_sec", 0.0) or 0.0),
            float(r.get("time_end_sec", 0.0) or 0.0),
            int(r.get("trajectory_id", 0) or 0),
        )
    )

    notes: list[dict[str, str]] = []
    gaps: list[dict[str, str]] = []

    active_note = None
    note_id = 1
    prev_global_end = None

    for r in rows:
        t0 = float(r.get("time_start_sec", 0.0) or 0.0)
        t1 = float(r.get("time_end_sec", 0.0) or 0.0)
        event_count = int(r.get("event_count", 0) or 0)
        energy_sum = float(r.get("energy_sum", 0.0) or 0.0)
        energy_mean = float(r.get("energy_mean", 0.0) or 0.0)

        if prev_global_end is not None:
            gap = t0 - prev_global_end
            if gap > gap_marker_threshold_sec:
                gaps.append(
                    {
                        "gap_id": str(len(gaps) + 1),
                        "time_start_sec": f"{prev_global_end:.6f}",
                        "time_end_sec": f"{t0:.6f}",
                        "gap_duration_sec": f"{gap:.6f}",
                        "kind": "GAP",
                    }
                )
        prev_global_end = max(prev_global_end, t1) if prev_global_end is not None else t1

        if active_note is None:
            active_note = {
                "note_id": str(note_id),
                "time_start_sec": t0,
                "time_end_sec": t1,
                "source_trajectory_ids": [r["trajectory_id"]],
                "trajectory_count": 1,
                "event_count_sum": event_count,
                "energy_sum": energy_sum,
                "energy_mean_acc": energy_mean,
                "octave_mode": r.get("octave_mode", ""),
                "degree12_mode": r.get("degree12_mode", ""),
                "subdivisions_mode": r.get("subdivisions_mode", ""),
                "delta_radius_mode": r.get("delta_radius_mode", ""),
                "delta_vector_mode": r.get("delta_vector_mode", ""),
                "mean_phase_deg": float(r.get("mean_phase_deg", 0.0) or 0.0),
            }
            note_id += 1
            continue

        join_gap = t0 - float(active_note["time_end_sec"])
        same_family = same_note_family(active_note, r, phase_tol_deg)

        if join_gap <= max_join_gap_sec and same_family:
            active_note["time_end_sec"] = max(float(active_note["time_end_sec"]), t1)
            active_note["source_trajectory_ids"].append(r["trajectory_id"])
            active_note["trajectory_count"] += 1
            active_note["event_count_sum"] += event_count
            active_note["energy_sum"] += energy_sum
            active_note["energy_mean_acc"] += energy_mean
        else:
            notes.append(
                {
                    "note_id": active_note["note_id"],
                    "time_start_sec": f"{float(active_note['time_start_sec']):.6f}",
                    "time_end_sec": f"{float(active_note['time_end_sec']):.6f}",
                    "duration_sec": f"{float(active_note['time_end_sec']) - float(active_note['time_start_sec']):.6f}",
                    "trajectory_count": str(active_note["trajectory_count"]),
                    "event_count_sum": str(active_note["event_count_sum"]),
                    "energy_sum": f"{float(active_note['energy_sum']):.6f}",
                    "energy_mean_rough": f"{float(active_note['energy_mean_acc']) / max(1, active_note['trajectory_count']):.6f}",
                    "octave_mode": active_note["octave_mode"],
                    "degree12_mode": active_note["degree12_mode"],
                    "subdivisions_mode": active_note["subdivisions_mode"],
                    "delta_radius_mode": active_note["delta_radius_mode"],
                    "delta_vector_mode": active_note["delta_vector_mode"],
                    "source_trajectory_ids": json.dumps(active_note["source_trajectory_ids"], ensure_ascii=False),
                    "kind": "NOTE",
                }
            )

            active_note = {
                "note_id": str(note_id),
                "time_start_sec": t0,
                "time_end_sec": t1,
                "source_trajectory_ids": [r["trajectory_id"]],
                "trajectory_count": 1,
                "event_count_sum": event_count,
                "energy_sum": energy_sum,
                "energy_mean_acc": energy_mean,
                "octave_mode": r.get("octave_mode", ""),
                "degree12_mode": r.get("degree12_mode", ""),
                "subdivisions_mode": r.get("subdivisions_mode", ""),
                "delta_radius_mode": r.get("delta_radius_mode", ""),
                "delta_vector_mode": r.get("delta_vector_mode", ""),
                "mean_phase_deg": float(r.get("mean_phase_deg", 0.0) or 0.0),
            }
            note_id += 1

    if active_note is not None:
        notes.append(
            {
                "note_id": active_note["note_id"],
                "time_start_sec": f"{float(active_note['time_start_sec']):.6f}",
                "time_end_sec": f"{float(active_note['time_end_sec']):.6f}",
                "duration_sec": f"{float(active_note['time_end_sec']) - float(active_note['time_start_sec']):.6f}",
                "trajectory_count": str(active_note["trajectory_count"]),
                "event_count_sum": str(active_note["event_count_sum"]),
                "energy_sum": f"{float(active_note['energy_sum']):.6f}",
                "energy_mean_rough": f"{float(active_note['energy_mean_acc']) / max(1, active_note['trajectory_count']):.6f}",
                "octave_mode": active_note["octave_mode"],
                "degree12_mode": active_note["degree12_mode"],
                "subdivisions_mode": active_note["subdivisions_mode"],
                "delta_radius_mode": active_note["delta_radius_mode"],
                "delta_vector_mode": active_note["delta_vector_mode"],
                "source_trajectory_ids": json.dumps(active_note["source_trajectory_ids"], ensure_ascii=False),
                "kind": "NOTE",
            }
        )

    return notes, gaps


def write_timeline(path: Path, notes: list[dict[str, str]], gaps: list[dict[str, str]]) -> None:
    timeline = []

    for n in notes:
        timeline.append(
            {
                "kind": "NOTE",
                "id": n["note_id"],
                "time_start_sec": n["time_start_sec"],
                "time_end_sec": n["time_end_sec"],
                "duration_sec": n["duration_sec"],
                "octave_mode": n["octave_mode"],
                "degree12_mode": n["degree12_mode"],
                "subdivisions_mode": n["subdivisions_mode"],
                "delta_vector_mode": n["delta_vector_mode"],
                "trajectory_count": n["trajectory_count"],
            }
        )

    for g in gaps:
        timeline.append(
            {
                "kind": "GAP",
                "id": g["gap_id"],
                "time_start_sec": g["time_start_sec"],
                "time_end_sec": g["time_end_sec"],
                "duration_sec": g["gap_duration_sec"],
                "octave_mode": "",
                "degree12_mode": "",
                "subdivisions_mode": "",
                "delta_vector_mode": "",
                "trajectory_count": "",
            }
        )

    timeline.sort(key=lambda x: (float(x["time_start_sec"]), x["kind"] != "GAP"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timeline[0].keys()) if timeline else ["kind", "id", "time_start_sec", "time_end_sec", "duration_sec", "octave_mode", "degree12_mode", "subdivisions_mode", "delta_vector_mode", "trajectory_count"])
        w.writeheader()
        if timeline:
            w.writerows(timeline)


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble notes from trajectories with explicit gap markers")
    ap.add_argument("--in_trajectories_csv", required=True)
    ap.add_argument("--out_notes_csv", required=True)
    ap.add_argument("--out_gaps_csv", required=True)
    ap.add_argument("--out_timeline_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_join_gap_sec", type=float, default=0.12)
    ap.add_argument("--phase_tol_deg", type=float, default=4.0)
    ap.add_argument("--gap_marker_threshold_sec", type=float, default=0.5)
    args = ap.parse_args()

    rows = load_rows(Path(args.in_trajectories_csv).resolve())
    notes, gaps = assemble_notes(
        rows,
        max_join_gap_sec=args.max_join_gap_sec,
        phase_tol_deg=args.phase_tol_deg,
        gap_marker_threshold_sec=args.gap_marker_threshold_sec,
    )

    write_rows(Path(args.out_notes_csv).resolve(), notes)
    write_rows(Path(args.out_gaps_csv).resolve(), gaps if gaps else [{"gap_id": "", "time_start_sec": "", "time_end_sec": "", "gap_duration_sec": "", "kind": ""}])
    write_timeline(Path(args.out_timeline_csv).resolve(), notes, gaps)

    meta = {
        "input_trajectory_rows": len(rows),
        "assembled_note_rows": len(notes),
        "gap_rows": len(gaps),
        "max_join_gap_sec": args.max_join_gap_sec,
        "phase_tol_deg": args.phase_tol_deg,
        "gap_marker_threshold_sec": args.gap_marker_threshold_sec,
    }
    Path(args.out_meta_json).resolve().write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()