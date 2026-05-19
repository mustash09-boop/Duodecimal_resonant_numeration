from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class SegmentData:
    segment_dir: Path
    segment_start_seconds: float
    probe_indices: List[int]
    matrix_rows: List[List[float]]
    frame_times_local: List[float]
    frame_times_absolute: List[float]
    coords_header: List[str]
    coords_rows: List[List[str]]


def read_matrix_csv(path: Path) -> Tuple[List[int], List[List[float]]]:
    probe_indices: List[int] = []
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or header[0] != "probe_index":
            raise ValueError(f"Invalid matrix CSV header in {path}")
        for row in r:
            probe_indices.append(int(row[0]))
            rows.append([float(x) for x in row[1:]])

    return probe_indices, rows


def read_times_csv(path: Path) -> List[float]:
    out: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(float(row["time_seconds"]))
    return out


def read_coords_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"Empty coords CSV: {path}")
    return rows[0], rows[1:]


def read_meta_start_seconds(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing meta JSON: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    time_slice = data.get("time_slice", {})
    if isinstance(time_slice, dict):
        value = time_slice.get("start_seconds")
        if value is not None:
            return float(value)

    for key in ("segment_time_start_sec", "time_start", "start_seconds"):
        if key in data and data[key] is not None:
            return float(data[key])

    return 0.0


def write_matrix_csv(path: Path, probe_indices: List[int], matrix_rows: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = len(matrix_rows[0]) if matrix_rows else 0

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_index"] + [f"frame_{i}" for i in range(num_frames)])
        for probe_idx, row in zip(probe_indices, matrix_rows):
            w.writerow([probe_idx] + row)


def write_times_csv(path: Path, times: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_seconds"])
        for i, t in enumerate(times):
            w.writerow([i, t])


def write_coords_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_segment(segment_dir: Path) -> SegmentData:
    matrix_path = segment_dir / "probe_matrix.csv"
    times_path = segment_dir / "probe_times.csv"
    coords_path = segment_dir / "probe_coords.csv"
    meta_path = segment_dir / "probe_meta.json"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing file: {matrix_path}")
    if not times_path.exists():
        raise FileNotFoundError(f"Missing file: {times_path}")
    if not coords_path.exists():
        raise FileNotFoundError(f"Missing file: {coords_path}")

    probe_indices, matrix_rows = read_matrix_csv(matrix_path)
    frame_times_local = read_times_csv(times_path)
    coords_header, coords_rows = read_coords_csv(coords_path)
    segment_start_seconds = read_meta_start_seconds(meta_path)

    if matrix_rows and len(matrix_rows[0]) != len(frame_times_local):
        raise ValueError(
            f"Frame count mismatch in {segment_dir}: "
            f"matrix has {len(matrix_rows[0])} frames, times has {len(frame_times_local)}"
        )

    if len(probe_indices) != len(coords_rows):
        raise ValueError(
            f"Probe count mismatch in {segment_dir}: "
            f"matrix has {len(probe_indices)} probes, coords has {len(coords_rows)} rows"
        )

    frame_times_absolute = [segment_start_seconds + t for t in frame_times_local]

    return SegmentData(
        segment_dir=segment_dir,
        segment_start_seconds=segment_start_seconds,
        probe_indices=probe_indices,
        matrix_rows=matrix_rows,
        frame_times_local=frame_times_local,
        frame_times_absolute=frame_times_absolute,
        coords_header=coords_header,
        coords_rows=coords_rows,
    )


def list_segment_dirs(results_root: Path, prefix: str) -> List[Path]:
    dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    dirs.sort(key=lambda p: p.name)
    return dirs


def merge_segments(
    segments: List[SegmentData],
    time_round_digits: int = 9,
) -> Tuple[List[int], List[List[float]], List[float], List[str], List[List[str]], dict]:

    if not segments:
        raise ValueError("No segments to merge")

    segments = sorted(segments, key=lambda s: (s.segment_start_seconds, s.segment_dir.name))

    base = segments[0]
    base_probe_indices = base.probe_indices
    base_coords_header = base.coords_header
    base_coords_rows = base.coords_rows

    for seg in segments[1:]:
        if seg.probe_indices != base_probe_indices:
            raise ValueError("Probe index mismatch between segments")
        if seg.coords_header != base_coords_header:
            raise ValueError("Coords header mismatch between segments")
        if seg.coords_rows != base_coords_rows:
            raise ValueError("Coords rows mismatch between segments")

    merged_times: List[float] = []
    merged_matrix: List[List[float]] = [[] for _ in base_probe_indices]
    seen_times: Dict[int, int] = {}

    total_input_frames = 0
    duplicate_frames_skipped = 0

    EPS = 10 ** (-time_round_digits)

    for seg in segments:
        total_input_frames += len(seg.frame_times_absolute)

        for frame_idx, t_abs in enumerate(seg.frame_times_absolute):

            # collapse ONLY floating point noise, not real temporal structure
            key = int(t_abs / EPS)

            if key in seen_times:
                duplicate_frames_skipped += 1
                continue

            seen_times[key] = len(merged_times)
            merged_times.append(t_abs)

            for row_idx in range(len(base_probe_indices)):
                merged_matrix[row_idx].append(seg.matrix_rows[row_idx][frame_idx])

    summary = {
        "num_segments": len(segments),
        "segment_names": [seg.segment_dir.name for seg in segments],
        "num_probes": len(base_probe_indices),
        "total_input_frames": total_input_frames,
        "merged_frames": len(merged_times),
        "duplicate_frames_skipped": duplicate_frames_skipped,
        "time_resolution_eps": EPS,
    }

    return (
        base_probe_indices,
        merged_matrix,
        merged_times,
        base_coords_header,
        base_coords_rows,
        summary,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge probe segments into one continuous field")

    ap.add_argument("--results_root", required=True)
    ap.add_argument("--segment_prefix", required=True)
    ap.add_argument("--out_matrix_csv", required=True)
    ap.add_argument("--out_times_csv", required=True)
    ap.add_argument("--out_coords_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--time_round_digits", type=int, default=9)

    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    segment_dirs = list_segment_dirs(results_root, args.segment_prefix)

    segments = [load_segment(d) for d in segment_dirs]

    probe_indices, merged_matrix, merged_times, coords_header, coords_rows, summary = merge_segments(
        segments,
        time_round_digits=args.time_round_digits,
    )

    write_matrix_csv(Path(args.out_matrix_csv), probe_indices, merged_matrix)
    write_times_csv(Path(args.out_times_csv), merged_times)
    write_coords_csv(Path(args.out_coords_csv), coords_header, coords_rows)

    Path(args.out_summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()