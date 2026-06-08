# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _apply_preset(preset: str, args: argparse.Namespace) -> None:
    if preset == "none":
        return
    if preset == "note_compare_scientific":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 1.0
        return
    if preset == "note_compare_cinematic":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 0.12
        return
    if preset == "note_compare_cinematic_z025":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 0.25
        return
    if preset == "note_compare_cinematic_z035":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 0.35
        return
    if preset == "note_compare_cinematic_z050":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 0.50
        return
    if preset == "note_compare_cinematic_z300":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 3.00
        return
    if preset == "note_compare_cinematic_z500":
        args.x_scale = 1.0
        args.y_scale = 1.0
        args.z_scale = 5.00
        return
    if preset == "harmonic_compare_scientific":
        args.x_scale = 1.0
        args.y_scale = 12.0
        args.z_scale = 18.0
        return
    if preset == "harmonic_compare_cinematic":
        args.x_scale = 1.0
        args.y_scale = 7.0
        args.z_scale = 10.0
        return
    raise ValueError(f"Unknown preset: {preset}")


def _extract_plotly_traces(html_path: Path) -> list[dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"const\s+traces\s*=\s*(\[.*?\]);", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find Plotly trace block in {html_path}")
    payload = match.group(1)
    return json.loads(payload)


def _parse_hex_color(value: str) -> tuple[float, float, float]:
    s = str(value or "").strip()
    if s.startswith("#") and len(s) == 7:
        return (
            int(s[1:3], 16) / 255.0,
            int(s[3:5], 16) / 255.0,
            int(s[5:7], 16) / 255.0,
        )
    if s.startswith("rgb(") and s.endswith(")"):
        nums = [x.strip() for x in s[4:-1].split(",")]
        if len(nums) == 3:
            return (
                max(0.0, min(255.0, float(nums[0]))) / 255.0,
                max(0.0, min(255.0, float(nums[1]))) / 255.0,
                max(0.0, min(255.0, float(nums[2]))) / 255.0,
            )
    named = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
    }
    return named.get(s.lower(), (0.68, 0.68, 0.68))


def _material_name(label: str) -> str:
    out = []
    for ch in str(label):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "material"


def _split_trace_name(name: str) -> tuple[str, str]:
    text = str(name or "")
    if " / " in text:
        instrument, layer = text.split(" / ", 1)
        return instrument.strip(), layer.strip()
    return text.strip(), ""


def _instrument_base_rgb(instrument: str) -> tuple[float, float, float] | None:
    key = str(instrument or "").strip().lower()
    palette = {
        "cello": (0.86, 0.26, 0.22),
        "cello2": (0.86, 0.26, 0.22),
        "violin": (0.22, 0.47, 0.95),
        "violin2": (0.22, 0.47, 0.95),
        "banjo": (0.18, 0.68, 0.37),
        "guitar": (0.90, 0.62, 0.15),
        "bass_guitar": (0.18, 0.58, 0.56),
        "realpiano_1_1": (0.56, 0.40, 0.78),
    }
    return palette.get(key)


def _semantic_layer_rgb(base_rgb: tuple[float, float, float], layer: str) -> tuple[float, float, float]:
    r, g, b = base_rgb
    key = str(layer or "").strip().lower()
    if key == "chain":
        return (r, g, b)
    if key == "note_box":
        return (
            min(1.0, r * 0.55 + 0.45),
            min(1.0, g * 0.55 + 0.35),
            min(1.0, b * 0.55 + 0.20),
        )
    if key == "dense_other":
        return (
            min(1.0, r * 0.45 + 0.30),
            min(1.0, g * 0.45 + 0.26),
            min(1.0, b * 0.45 + 0.38),
        )
    return base_rgb


def _get_trace_rgb(trace: dict[str, Any], color_mode: str = "plotly") -> tuple[float, float, float]:
    if color_mode == "note_compare_semantic":
        instrument, layer = _split_trace_name(str(trace.get("name", "")))
        base = _instrument_base_rgb(instrument)
        if base is not None:
            return _semantic_layer_rgb(base, layer)
    marker = trace.get("marker") or {}
    color = marker.get("color", "#999999")
    if isinstance(color, list):
        if color:
            color = color[0]
        else:
            color = "#999999"
    return _parse_hex_color(str(color))


def _get_sizes(trace: dict[str, Any], point_count: int) -> list[float]:
    marker = trace.get("marker") or {}
    size = marker.get("size", 4.0)
    if isinstance(size, list):
        values = [_safe_float(x, 4.0) for x in size[:point_count]]
        if len(values) < point_count:
            values.extend([4.0] * (point_count - len(values)))
        return values
    return [_safe_float(size, 4.0)] * point_count


def _octahedron(center: tuple[float, float, float], radius: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    cx, cy, cz = center
    r = max(radius, 1e-6)
    verts = [
        (cx + r, cy, cz),
        (cx - r, cy, cz),
        (cx, cy + r, cz),
        (cx, cy - r, cz),
        (cx, cy, cz + r),
        (cx, cy, cz - r),
    ]
    faces = [
        (1, 3, 5),
        (3, 2, 5),
        (2, 4, 5),
        (4, 1, 5),
        (3, 1, 6),
        (2, 3, 6),
        (4, 2, 6),
        (1, 4, 6),
    ]
    return verts, faces


def _box_mesh(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    return verts, faces


def _write_mtl(path: Path, materials: dict[str, tuple[float, float, float]]) -> None:
    lines = []
    for name, (r, g, b) in materials.items():
        lines.extend(
            [
                f"newmtl {name}",
                f"Kd {r:.6f} {g:.6f} {b:.6f}",
                "Ka 0.000000 0.000000 0.000000",
                "Ks 0.050000 0.050000 0.050000",
                "Ns 16.000000",
                "d 1.000000",
                "illum 2",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Plotly 3D HTML into an OBJ/MTL point sculpture that can later be converted to USDZ for Apple Motion."
    )
    ap.add_argument("--html", required=True, help="Input Plotly 3D HTML with const traces = [...]")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--preset",
        default="none",
        choices=[
            "none",
            "note_compare_scientific",
            "note_compare_cinematic",
            "note_compare_cinematic_z025",
            "note_compare_cinematic_z035",
            "note_compare_cinematic_z050",
            "note_compare_cinematic_z300",
            "note_compare_cinematic_z500",
            "harmonic_compare_scientific",
            "harmonic_compare_cinematic",
        ],
        help="Apply a scene scaling preset before export.",
    )
    ap.add_argument("--x-scale", type=float, default=1.0)
    ap.add_argument("--y-scale", type=float, default=1.0)
    ap.add_argument("--z-scale", type=float, default=1.0)
    ap.add_argument("--point-radius", type=float, default=0.09)
    ap.add_argument("--size-influence", type=float, default=0.35)
    ap.add_argument(
        "--color-mode",
        default="plotly",
        choices=["plotly", "note_compare_semantic"],
        help="How to choose material colors for exported traces.",
    )
    ap.add_argument("--center-origin", action="store_true", help="Center the sculpture around the origin")
    ap.add_argument("--with-axes", action="store_true", help="Add simple axis rods to the sculpture")
    args = ap.parse_args()
    _apply_preset(str(args.preset), args)

    html_path = Path(args.html)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    traces = _extract_plotly_traces(html_path)
    point_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    materials: dict[str, tuple[float, float, float]] = {}

    for trace_idx, trace in enumerate(traces):
        name = str(trace.get("name", f"trace_{trace_idx}"))
        xs = trace.get("x", []) or []
        ys = trace.get("y", []) or []
        zs = trace.get("z", []) or []
        rgb = _get_trace_rgb(trace, str(args.color_mode))
        material = _material_name(name)
        materials[material] = rgb
        sizes = _get_sizes(trace, min(len(xs), len(ys), len(zs)))

        trace_rows.append(
            {
                "trace_index": trace_idx,
                "trace_name": name,
                "material_name": material,
                "rgb_r": rgb[0],
                "rgb_g": rgb[1],
                "rgb_b": rgb[2],
                "point_count": min(len(xs), len(ys), len(zs)),
            }
        )

        for point_idx, (x, y, z) in enumerate(zip(xs, ys, zs)):
            point_rows.append(
                {
                    "trace_index": trace_idx,
                    "trace_name": name,
                    "material_name": material,
                    "point_index": point_idx,
                    "x_raw": _safe_float(x),
                    "y_raw": _safe_float(y),
                    "z_raw": _safe_float(z),
                    "marker_size_raw": sizes[point_idx],
                }
            )

    if not point_rows:
        raise SystemExit("No points extracted from Plotly HTML.")

    x_vals = [row["x_raw"] for row in point_rows]
    y_vals = [row["y_raw"] for row in point_rows]
    z_vals = [row["z_raw"] for row in point_rows]

    x_mid = (min(x_vals) + max(x_vals)) * 0.5
    y_mid = (min(y_vals) + max(y_vals)) * 0.5
    z_mid = (min(z_vals) + max(z_vals)) * 0.5

    global_sizes = [row["marker_size_raw"] for row in point_rows]
    size_median = sorted(global_sizes)[len(global_sizes) // 2] if global_sizes else 4.0
    size_median = max(float(size_median), 1e-6)

    for row in point_rows:
        x = row["x_raw"] * float(args.x_scale)
        y = row["y_raw"] * float(args.y_scale)
        z = row["z_raw"] * float(args.z_scale)
        if args.center_origin:
            x -= x_mid * float(args.x_scale)
            y -= y_mid * float(args.y_scale)
            z -= z_mid * float(args.z_scale)
        row["x_scene"] = x
        row["y_scene"] = y
        row["z_scene"] = z
        size_ratio = max(row["marker_size_raw"], 0.0) / size_median
        row["point_radius_scene"] = float(args.point_radius) * (1.0 + float(args.size_influence) * (math.sqrt(max(size_ratio, 0.0)) - 1.0))

    points_csv = outdir / "plotly_points.csv"
    with points_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trace_index",
                "trace_name",
                "material_name",
                "point_index",
                "x_raw",
                "y_raw",
                "z_raw",
                "marker_size_raw",
                "x_scene",
                "y_scene",
                "z_scene",
                "point_radius_scene",
            ],
        )
        writer.writeheader()
        writer.writerows(point_rows)

    traces_csv = outdir / "plotly_traces.csv"
    with traces_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trace_index",
                "trace_name",
                "material_name",
                "rgb_r",
                "rgb_g",
                "rgb_b",
                "point_count",
            ],
        )
        writer.writeheader()
        writer.writerows(trace_rows)

    obj_path = outdir / "plotly_point_sculpture.obj"
    mtl_path = outdir / "plotly_point_sculpture.mtl"
    _write_mtl(mtl_path, materials | {"axes_gray": (0.55, 0.55, 0.55)})

    scene_rows = list(point_rows)
    xs_scene = [row["x_scene"] for row in scene_rows]
    ys_scene = [row["y_scene"] for row in scene_rows]
    zs_scene = [row["z_scene"] for row in scene_rows]

    obj_lines = [f"mtllib {mtl_path.name}"]
    vertex_offset = 1
    current_material = None
    current_trace_key = None

    for row in scene_rows:
        material = row["material_name"]
        trace_key = f"trace_{row['trace_index']}__{_material_name(row['trace_name'])}"
        if trace_key != current_trace_key:
            obj_lines.append(f"o {trace_key}")
            obj_lines.append(f"g {trace_key}")
            current_trace_key = trace_key
        if material != current_material:
            obj_lines.append(f"usemtl {material}")
            current_material = material
        verts, faces = _octahedron(
            (row["x_scene"], row["y_scene"], row["z_scene"]),
            row["point_radius_scene"],
        )
        for vx, vy, vz in verts:
            obj_lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
        for a, b, c in faces:
            obj_lines.append(f"f {vertex_offset + a - 1} {vertex_offset + b - 1} {vertex_offset + c - 1}")
        vertex_offset += len(verts)

    if args.with_axes:
        current_material = "axes_gray"
        obj_lines.append(f"usemtl {current_material}")
        min_x, max_x = min(xs_scene), max(xs_scene)
        min_y, max_y = min(ys_scene), max(ys_scene)
        min_z, max_z = min(zs_scene), max(zs_scene)
        pad = max(float(args.point_radius) * 8.0, 0.25)
        thickness = max(float(args.point_radius) * 0.7, 0.03)
        axis_specs = [
            _box_mesh(min_x - pad, max_x + pad, min_y - pad - thickness, min_y - pad + thickness, min_z - pad - thickness, min_z - pad + thickness),
            _box_mesh(min_x - pad - thickness, min_x - pad + thickness, min_y - pad, max_y + pad, min_z - pad - thickness, min_z - pad + thickness),
            _box_mesh(min_x - pad - thickness, min_x - pad + thickness, min_y - pad - thickness, min_y - pad + thickness, min_z - pad, max_z + pad),
        ]
        for verts, faces in axis_specs:
            for vx, vy, vz in verts:
                obj_lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
            for a, b, c in faces:
                obj_lines.append(f"f {vertex_offset + a - 1} {vertex_offset + b - 1} {vertex_offset + c - 1}")
            vertex_offset += len(verts)

    obj_path.write_text("\n".join(obj_lines) + "\n", encoding="ascii")

    scene_json = outdir / "plotly_point_sculpture_scene.json"
    scene_json.write_text(
        json.dumps(
            {
                "source_html": str(html_path),
                "obj": str(obj_path),
                "mtl": str(mtl_path),
                "point_count": len(scene_rows),
                "trace_count": len(trace_rows),
                "preset": str(args.preset),
                "center_origin": bool(args.center_origin),
                "with_axes": bool(args.with_axes),
                "x_scale": float(args.x_scale),
                "y_scale": float(args.y_scale),
                "z_scale": float(args.z_scale),
                "point_radius": float(args.point_radius),
                "size_influence": float(args.size_influence),
                "color_mode": str(args.color_mode),
                "next_motion_step": "Convert OBJ/MTL to USDZ on macOS with Apple tools, then import USDZ into Motion.",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    summary_txt = outdir / "plotly_point_sculpture_summary.txt"
    summary_lines = [
        "PLOTLY HTML TO MOTION OBJ",
        "=" * 72,
        f"source_html              : {html_path}",
        f"point_count              : {len(scene_rows)}",
        f"trace_count              : {len(trace_rows)}",
        f"preset                   : {args.preset}",
        f"obj                      : {obj_path.name}",
        f"mtl                      : {mtl_path.name}",
        f"points_csv               : {points_csv.name}",
        f"scene_json               : {scene_json.name}",
        f"center_origin            : {bool(args.center_origin)}",
        f"with_axes                : {bool(args.with_axes)}",
        f"color_mode               : {args.color_mode}",
        "",
        "trace_summary:",
    ]
    for row in trace_rows:
        summary_lines.append(
            f"  trace#{row['trace_index']} {row['trace_name']} -> {row['point_count']} points, material={row['material_name']}"
        )
    summary_lines.extend(
        [
            "",
            "motion_workflow:",
            "  1. Inspect the OBJ/MTL in Blender or another 3D viewer.",
            "  2. On macOS, convert the OBJ to USDZ using Apple tooling.",
            "  3. Import the USDZ into Apple Motion and rotate/scale the sculpture there.",
            "",
            "important:",
            "  This exporter builds a data sculpture, not an interactive Plotly graph.",
            "  Motion will rotate the object as a 3D asset, but it will not keep Plotly interactivity.",
        ]
    )
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("PLOTLY HTML TO MOTION OBJ DONE")
    print(obj_path)
    print(mtl_path)
    print(points_csv)
    print(scene_json)


if __name__ == "__main__":
    main()
