from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_matrix_csv(path: Path) -> np.ndarray:
    """
    Ожидаемый формат:
      header: probe_index, frame_0, frame_1, ...
      rows:   probe_idx, val0, val1, ...
    """
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")

        if len(header) < 2:
            raise ValueError(f"Matrix CSV must have at least 2 columns: {path}")

        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            values = [float(x) for x in row[1:]]
            rows.append(values)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32)

    matrix = np.asarray(rows, dtype=np.float32)
    return matrix


def _load_times_csv(path: Path) -> np.ndarray:
    """
    Ожидаемый формат:
      frame_index,time_seconds
    """
    values = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")

        for row in reader:
            if not row or len(row) < 2:
                continue
            values.append(float(row[1]))

    return np.asarray(values, dtype=np.float32)


def _load_coords_csv(path: Path) -> list[dict[str, Any]]:
    """
    Ожидаемый формат:
      probe_index,octave,degree12,subdivisions,frequency_hz,global_index
    subdivisions хранится как JSON-строка, например "[3, 7]"
    """
    coords: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subdivisions_raw = row.get("subdivisions", "[]")
            try:
                subdivisions = json.loads(subdivisions_raw)
            except Exception:
                subdivisions = []

            coords.append(
                {
                    "probe_index": int(row.get("probe_index", len(coords))),
                    "octave": int(row.get("octave", 0)),
                    "degree12": int(row.get("degree12", 0)),
                    "subdivisions": subdivisions,
                    "frequency_hz": float(row.get("frequency_hz", 0.0)),
                    "global_index": int(row.get("global_index", row.get("probe_index", len(coords)))),
                }
            )

    return coords


def _top_probes(matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []

    probe_scores = matrix.max(axis=1)
    order = np.argsort(probe_scores)[::-1]
    result = []

    for idx in order[:top_k]:
        result.append((int(idx), float(probe_scores[idx])))

    return result


def _frame_peaks(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {
            "mean_frame_peak": 0.0,
            "max_frame_peak": 0.0,
            "min_frame_peak": 0.0,
        }

    frame_peaks = matrix.max(axis=0)
    return {
        "mean_frame_peak": float(np.mean(frame_peaks)),
        "max_frame_peak": float(np.max(frame_peaks)),
        "min_frame_peak": float(np.min(frame_peaks)),
    }


def _matrix_stats(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": 0.0,
            "nonzero_ratio": 0.0,
        }

    nonzero_ratio = float(np.count_nonzero(matrix) / matrix.size)

    return {
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
        "max": float(np.max(matrix)),
        "min": float(np.min(matrix)),
        "nonzero_ratio": nonzero_ratio,
    }


def _format_coord(coord: dict[str, Any]) -> str:
    subs = coord.get("subdivisions", [])
    if subs:
        return f"{coord['octave']}.{coord['degree12']}<{'.'.join(str(x) for x in subs)}>"
    return f"{coord['octave']}.{coord['degree12']}"


def _build_report(
    *,
    matrix: np.ndarray,
    times: np.ndarray | None,
    coords: list[dict[str, Any]] | None,
    top_k: int,
    detail_depth: int | None,
    source_name: str,
) -> tuple[dict[str, Any], str]:
    stats = _matrix_stats(matrix)
    frame_stats = _frame_peaks(matrix)
    top = _top_probes(matrix, top_k=top_k)

    payload: dict[str, Any] = {
        "source_name": source_name,
        "matrix_shape": list(matrix.shape),
        "detail_depth": detail_depth,
        "matrix_stats": stats,
        "frame_stats": frame_stats,
        "top_probes": [],
        "time_range_seconds": None,
    }

    if times is not None and len(times) > 0:
        payload["time_range_seconds"] = {
            "start": float(times[0]),
            "end": float(times[-1]),
            "duration": float(times[-1] - times[0]) if len(times) >= 2 else 0.0,
            "n_frames": int(len(times)),
        }

    lines = []
    lines.append("MUSIC12 RESONANCE SCAN REPORT")
    lines.append("=" * 72)
    lines.append(f"source_name     : {source_name}")
    lines.append(f"matrix_shape    : {tuple(matrix.shape)}")
    lines.append(f"detail_depth    : {detail_depth}")
    lines.append("")

    if times is not None and len(times) > 0:
        lines.append("TIME")
        lines.append("-" * 72)
        lines.append(f"frames          : {len(times)}")
        lines.append(f"start_s         : {float(times[0]):.6f}")
        lines.append(f"end_s           : {float(times[-1]):.6f}")
        duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
        lines.append(f"duration_s      : {duration:.6f}")
        lines.append("")

    lines.append("MATRIX STATS")
    lines.append("-" * 72)
    lines.append(f"mean            : {stats['mean']:.6f}")
    lines.append(f"std             : {stats['std']:.6f}")
    lines.append(f"min             : {stats['min']:.6f}")
    lines.append(f"max             : {stats['max']:.6f}")
    lines.append(f"nonzero_ratio   : {stats['nonzero_ratio']:.6f}")
    lines.append("")

    lines.append("FRAME PEAKS")
    lines.append("-" * 72)
    lines.append(f"mean_frame_peak : {frame_stats['mean_frame_peak']:.6f}")
    lines.append(f"max_frame_peak  : {frame_stats['max_frame_peak']:.6f}")
    lines.append(f"min_frame_peak  : {frame_stats['min_frame_peak']:.6f}")
    lines.append("")

    lines.append("TOP PROBES")
    lines.append("-" * 72)

    for rank, (probe_idx, score) in enumerate(top, start=1):
        item: dict[str, Any] = {
            "rank": rank,
            "probe_index": probe_idx,
            "score": score,
        }

        if coords is not None and 0 <= probe_idx < len(coords):
            coord = coords[probe_idx]
            item["coord"] = {
                "octave": coord["octave"],
                "degree12": coord["degree12"],
                "subdivisions": coord["subdivisions"],
            }
            item["label"] = _format_coord(coord)
            item["frequency_hz"] = coord["frequency_hz"]

            lines.append(
                f"{rank:>2}. probe={probe_idx:<6} "
                f"score={score:.6f} "
                f"coord={item['label']:<18} "
                f"freq={coord['frequency_hz']:.6f}"
            )
        else:
            lines.append(f"{rank:>2}. probe={probe_idx:<6} score={score:.6f}")

        payload["top_probes"].append(item)

    if not top:
        lines.append("(no active probes above data floor)")

    lines.append("")

    return payload, "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a text/json demon report from resonance probe scan outputs"
    )
    parser.add_argument("--matrix_csv", required=True, help="CSV with response matrix")
    parser.add_argument("--out_txt", required=True, help="Output TXT report")
    parser.add_argument("--out_json", required=True, help="Output JSON summary")
    parser.add_argument("--times_csv", default="", help="Optional frame times CSV")
    parser.add_argument("--coords_csv", default="", help="Optional probe coordinates CSV")
    parser.add_argument("--detail_depth", type=int, default=-1, help="Optional detail depth hint")
    parser.add_argument("--top_k", type=int, default=12, help="How many strongest probes to show")
    parser.add_argument("--source_name", default="", help="Human-readable source name")
    args = parser.parse_args()

    matrix_csv = Path(args.matrix_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_json = Path(args.out_json).resolve()
    times_csv = Path(args.times_csv).resolve() if args.times_csv else None
    coords_csv = Path(args.coords_csv).resolve() if args.coords_csv else None

    matrix = _load_matrix_csv(matrix_csv)
    times = _load_times_csv(times_csv) if times_csv is not None else None
    coords = _load_coords_csv(coords_csv) if coords_csv is not None else None

    source_name = args.source_name or matrix_csv.stem
    detail_depth = None if args.detail_depth < 0 else int(args.detail_depth)

    payload, txt_report = _build_report(
        matrix=matrix,
        times=times,
        coords=coords,
        top_k=args.top_k,
        detail_depth=detail_depth,
        source_name=source_name,
    )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_txt.write_text(txt_report, encoding="utf-8")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report txt : {out_txt}")
    print(f"report json: {out_json}")


if __name__ == "__main__":
    main()