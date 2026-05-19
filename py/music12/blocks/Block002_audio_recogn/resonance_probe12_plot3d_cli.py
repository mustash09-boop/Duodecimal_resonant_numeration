from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def _prepare_display_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
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


def _downsample_matrix(
    matrix: np.ndarray,
    x_max_points: int,
    y_max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols = matrix.shape

    x_idx = np.arange(n_cols)
    y_idx = np.arange(n_rows)

    if n_cols > x_max_points:
        x_idx = np.linspace(0, n_cols - 1, x_max_points, dtype=int)

    if n_rows > y_max_points:
        y_idx = np.linspace(0, n_rows - 1, y_max_points, dtype=int)

    reduced = matrix[np.ix_(y_idx, x_idx)]
    return reduced, x_idx, y_idx


def _pick_tick_indices(n: int, max_ticks: int = 10) -> list[int]:
    if n <= 0:
        return []
    if n <= max_ticks:
        return list(range(n))
    step = max(1, n // max_ticks)
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    return ticks[: max_ticks + 1]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot resonance probe field as a 3D surface in spiral space"
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
    parser.add_argument("--top_k_probes", type=int, default=0, help="If >0, keep only strongest K probes")
    parser.add_argument("--probe_min", type=int, default=-1, help="Optional lower probe index bound")
    parser.add_argument("--probe_max", type=int, default=-1, help="Optional upper probe index bound")
    parser.add_argument("--x_max_points", type=int, default=300, help="Max time points for 3D surface")
    parser.add_argument("--y_max_points", type=int, default=120, help="Max probe rows for 3D surface")
    parser.add_argument("--elev", type=float, default=35.0, help="3D elevation angle")
    parser.add_argument("--azim", type=float, default=-65.0, help="3D azimuth angle")
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
        probe_min=probe_min,
        probe_max=probe_max,
        top_k_probes=top_k_probes,
    )

    if sliced_matrix.size == 0:
        raise ValueError("Selected probe slice is empty")

    display_matrix = _prepare_display_matrix(sliced_matrix, mode=args.display_mode)
    reduced, x_idx, y_idx = _downsample_matrix(
        display_matrix,
        x_max_points=max(10, args.x_max_points),
        y_max_points=max(5, args.y_max_points),
    )

    if times is not None and len(times) == matrix.shape[1]:
        x_values = times[x_idx]
        x_label = "Time (seconds)"
    else:
        x_values = x_idx.astype(np.float32)
        x_label = "Frame index"

    arcs = _selected_spiral_arcs(selected_indices, coords)
    if not arcs:
        raise ValueError("No spiral arcs available for selected probe slice")

    arc_values = np.asarray(arcs, dtype=np.float32)

    # first select the rows that survived downsampling
    y_arc = arc_values[y_idx]
    reduced_y_labels_indices = y_idx.copy()

    # then sort by physical spiral order
    order = np.argsort(y_arc)
    y_arc_sorted = y_arc[order]
    reduced_sorted = reduced[order, :]
    reduced_y_labels_indices_sorted = reduced_y_labels_indices[order]

    X, Y = np.meshgrid(x_values, y_arc_sorted)
    Z = reduced_sorted

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Spiral arc")
    ax.set_zlabel("Response intensity")

    title = args.title.strip()
    if not title:
        title = f"3D resonance field in spiral space ({args.display_mode})"
    ax.set_title(title)

    if coords is not None and selected_indices:
        tick_rows = _pick_tick_indices(len(y_arc_sorted), max_ticks=10)
        tick_values = [float(y_arc_sorted[i]) for i in tick_rows]
        tick_labels = []

        for row in tick_rows:
            original_row_in_selected = reduced_y_labels_indices_sorted[row]
            probe_idx = selected_indices[original_row_in_selected]
            tick_labels.append(_coord_label(coords[probe_idx]))

        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_labels)

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.08, label="Response intensity")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(json.dumps(
        {
            "plot_saved": str(out_png),
            "display_mode": args.display_mode,
            "selected_probe_count": len(selected_indices),
            "matrix_shape": list(matrix.shape),
            "semantic_note": (
                "3D probe response visualization in spiral space. "
                "Axes are time × spiral_arc × intensity, not time × probe_row × intensity."
            ),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()