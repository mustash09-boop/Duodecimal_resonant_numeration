from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# LOADERS
# ============================================================

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _load_json_list(raw: str) -> list[Any]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _load_matrix_csv(path: Path) -> np.ndarray:
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")

        if len(header) < 2:
            raise ValueError(f"Matrix CSV must have at least 2 columns: {path}")

        for row in reader:
            if not row or len(row) < 2:
                continue
            rows.append([float(x) for x in row[1:]])

    if not rows:
        return np.zeros((0, 0), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def _load_times_csv(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None

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


def _degree_to_symbol(degree12: int) -> str:
    alphabet = "123456789ABC"
    if 0 <= degree12 < 12:
        return alphabet[degree12]
    return str(degree12)


def _coord_to_note_token(octave: int, degree12: int, subdivisions: tuple[int, ...]) -> str:
    base = f"{octave}.{_degree_to_symbol(degree12)}"
    if not subdivisions:
        return base

    if all(v == 0 for v in subdivisions):
        return base

    subs = "".join(_degree_to_symbol(v) if 0 <= v < 12 else str(v) for v in subdivisions)
    return f"{base}'{subs}"


def _load_coords_csv(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None

    coords: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subdivisions = tuple(_safe_int(x, 0) for x in _load_json_list(row.get("subdivisions", "[]")))
            octave = _safe_int(row.get("octave", 0), 0)
            degree12 = _safe_int(row.get("degree12", 0), 0)

            coords.append(
                {
                    "probe_index": _safe_int(row.get("probe_index", len(coords)), len(coords)),
                    "octave": octave,
                    "degree12": degree12,
                    "subdivisions": subdivisions,
                    "frequency_hz": _safe_float(row.get("frequency_hz", 0.0), 0.0),
                    "global_index": _safe_int(row.get("global_index", row.get("probe_index", len(coords))), len(coords)),
                    "note_token": _coord_to_note_token(octave, degree12, subdivisions),
                }
            )

    return coords


# ============================================================
# DISPLAY PREP
# ============================================================

def _coord_label(coord: dict[str, Any]) -> str:
    return str(coord.get("note_token", ""))


def _pick_tick_indices(n: int, max_ticks: int = 12) -> list[int]:
    if n <= 0:
        return []
    if n <= max_ticks:
        return list(range(n))

    step = max(1, n // max_ticks)
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    return ticks[: max_ticks + 1]


def _prepare_display_matrix(
    matrix: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "raw":
        return matrix

    if mode == "log":
        return np.log1p(matrix)

    if mode == "frame_norm":
        denom = np.max(matrix, axis=0, keepdims=True)
        denom[denom == 0.0] = 1.0
        return matrix / denom

    if mode == "probe_norm":
        denom = np.max(matrix, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return matrix / denom

    raise ValueError(f"Unsupported mode: {mode}")


def _select_probe_slice(
    matrix: np.ndarray,
    coords: list[dict[str, Any]] | None,
    probe_min: int | None,
    probe_max: int | None,
    top_k_probes: int | None,
) -> tuple[np.ndarray, list[int]]:
    n_probes = matrix.shape[0]
    indices = list(range(n_probes))

    if top_k_probes is not None:
        probe_scores = matrix.max(axis=1)
        order = np.argsort(probe_scores)[::-1]
        indices = sorted(int(i) for i in order[:top_k_probes].tolist())

    if probe_min is not None or probe_max is not None:
        pmin = 0 if probe_min is None else max(0, int(probe_min))
        pmax = n_probes - 1 if probe_max is None else min(n_probes - 1, int(probe_max))
        indices = [i for i in indices if pmin <= i <= pmax]

    sliced = matrix[indices, :] if indices else np.zeros((0, matrix.shape[1]), dtype=matrix.dtype)
    return sliced, indices


# ============================================================
# SPIRAL ARC HELPERS
# ============================================================

def _token_to_spiral_arc(note_token: str) -> Optional[float]:
    from music12.core.spiral12_geometry import parse_token_to_spiral

    sp = parse_token_to_spiral(note_token)
    if sp is None:
        return None
    return float(sp.absolute_arc)


def _selected_spiral_arcs(
    selected_indices: list[int],
    coords: list[dict[str, Any]] | None,
) -> list[float]:
    arcs: list[float] = []

    for probe_idx in selected_indices:
        if coords is None or probe_idx >= len(coords):
            arcs.append(float(len(arcs)))
            continue

        note_token = str(coords[probe_idx].get("note_token", "")).strip()
        arc = _token_to_spiral_arc(note_token)
        if arc is None:
            arc = float(len(arcs))
        arcs.append(arc)

    return arcs


# ============================================================
# PLOTS
# ============================================================

def _plot_matrix_mode(
    *,
    display_matrix: np.ndarray,
    matrix: np.ndarray,
    times: np.ndarray | None,
    coords: list[dict[str, Any]] | None,
    selected_indices: list[int],
    title: str,
    out_png: Path,
) -> None:
    fig_width = 16
    fig_height = 8
    plt.figure(figsize=(fig_width, fig_height))

    if times is not None and len(times) == matrix.shape[1]:
        x0 = float(times[0])
        x1 = float(times[-1]) if len(times) > 1 else float(times[0])
        extent = [x0, x1, 0, display_matrix.shape[0] - 1]
        plt.imshow(
            display_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
        )
        plt.xlabel("Time (seconds)")
    else:
        plt.imshow(
            display_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        plt.xlabel("Frame index")

    plt.ylabel("Selected probe row (technical debug)")
    plt.title(title)
    plt.colorbar(label="Response intensity")

    if coords is not None and selected_indices:
        tick_rows = _pick_tick_indices(len(selected_indices), max_ticks=12)
        tick_labels = []
        for row_idx in tick_rows:
            probe_idx = selected_indices[row_idx]
            coord = coords[probe_idx]
            tick_labels.append(_coord_label(coord))

        plt.yticks(tick_rows, tick_labels)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_spiral_mode(
    *,
    display_matrix: np.ndarray,
    matrix: np.ndarray,
    times: np.ndarray | None,
    coords: list[dict[str, Any]] | None,
    selected_indices: list[int],
    title: str,
    out_png: Path,
) -> None:
    arcs = _selected_spiral_arcs(selected_indices, coords)
    if not arcs:
        raise ValueError("No spiral arcs available for selected probe slice")

    y = np.asarray(arcs, dtype=np.float32)

    if times is not None and len(times) == matrix.shape[1]:
        x = np.asarray(times, dtype=np.float32)
    else:
        x = np.arange(matrix.shape[1], dtype=np.float32)

    # sort by spiral arc so the visual field follows physical order
    order = np.argsort(y)
    y_sorted = y[order]
    display_sorted = display_matrix[order, :]

    plt.figure(figsize=(16, 8))
    plt.pcolormesh(
        x,
        y_sorted,
        display_sorted,
        shading="auto",
    )

    plt.xlabel("Time (seconds)" if times is not None and len(times) == matrix.shape[1] else "Frame index")
    plt.ylabel("Spiral arc")
    plt.title(title)
    plt.colorbar(label="Response intensity")

    if coords is not None and selected_indices:
        tick_rows = _pick_tick_indices(len(y_sorted), max_ticks=12)
        tick_values = [float(y_sorted[i]) for i in tick_rows]
        tick_labels = []
        selected_sorted = [selected_indices[i] for i in order]
        for row_idx in tick_rows:
            probe_idx = selected_sorted[row_idx]
            tick_labels.append(_coord_label(coords[probe_idx]))
        plt.yticks(tick_values, tick_labels)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot resonance probe response field in technical matrix or spiral space"
    )
    parser.add_argument("--matrix_csv", required=True, help="Response matrix CSV")
    parser.add_argument("--out_png", required=True, help="Output PNG")
    parser.add_argument("--times_csv", default="", help="Optional frame times CSV")
    parser.add_argument("--coords_csv", default="", help="Optional probe coordinates CSV")

    parser.add_argument(
        "--display_mode",
        default="log",
        choices=["raw", "log", "frame_norm", "probe_norm"],
        help="Display transform mode",
    )
    parser.add_argument(
        "--plot_mode",
        default="spiral",
        choices=["matrix", "spiral"],
        help="matrix = technical debug heatmap, spiral = physical resonance field view",
    )
    parser.add_argument(
        "--top_k_probes",
        type=int,
        default=0,
        help="If >0, plot only strongest K probes",
    )
    parser.add_argument("--probe_min", type=int, default=-1, help="Optional lower probe index bound")
    parser.add_argument("--probe_max", type=int, default=-1, help="Optional upper probe index bound")
    parser.add_argument("--title", default="", help="Optional plot title")
    args = parser.parse_args()

    matrix_csv = Path(args.matrix_csv).resolve()
    out_png = Path(args.out_png).resolve()
    times_csv = Path(args.times_csv).resolve() if args.times_csv else None
    coords_csv = Path(args.coords_csv).resolve() if args.coords_csv else None

    matrix = _load_matrix_csv(matrix_csv)
    if matrix.size == 0:
        raise ValueError("Matrix is empty")

    times = _load_times_csv(times_csv)
    coords = _load_coords_csv(coords_csv)

    probe_min = None if args.probe_min < 0 else args.probe_min
    probe_max = None if args.probe_max < 0 else args.probe_max
    top_k_probes = None if args.top_k_probes <= 0 else args.top_k_probes

    sliced_matrix, selected_indices = _select_probe_slice(
        matrix=matrix,
        coords=coords,
        probe_min=probe_min,
        probe_max=probe_max,
        top_k_probes=top_k_probes,
    )

    if sliced_matrix.size == 0:
        raise ValueError("Selected probe slice is empty")

    display_matrix = _prepare_display_matrix(sliced_matrix, mode=args.display_mode)

    title = args.title.strip()
    if not title:
        if args.plot_mode == "spiral":
            title = f"Resonance field in spiral space ({args.display_mode})"
        else:
            title = f"Resonance field map / technical matrix ({args.display_mode})"

    if args.plot_mode == "spiral":
        _plot_spiral_mode(
            display_matrix=display_matrix,
            matrix=matrix,
            times=times,
            coords=coords,
            selected_indices=selected_indices,
            title=title,
            out_png=out_png,
        )
    else:
        _plot_matrix_mode(
            display_matrix=display_matrix,
            matrix=matrix,
            times=times,
            coords=coords,
            selected_indices=selected_indices,
            title=title,
            out_png=out_png,
        )

    print(json.dumps(
        {
            "plot_saved": str(out_png),
            "plot_mode": args.plot_mode,
            "display_mode": args.display_mode,
            "selected_probe_count": len(selected_indices),
            "matrix_shape": list(matrix.shape),
            "semantic_note": (
                "Probe response visualization. "
                "Spiral mode is the primary physical view; matrix mode is a technical debug view."
            ),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()