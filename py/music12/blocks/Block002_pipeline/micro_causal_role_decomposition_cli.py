# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _bridge_symmetry(out_w: float, in_w: float, total_w: float) -> float:
    if total_w <= 0.0:
        return 0.0
    return 1.0 - abs(out_w - in_w) / total_w


def _role_label(
    out_w: float,
    in_w: float,
    total_w: float,
    out_edges: int,
    in_edges: int,
) -> str:
    if total_w <= 0:
        return "silent"

    out_ratio = out_w / total_w
    in_ratio = in_w / total_w
    symmetry = _bridge_symmetry(out_w, in_w, total_w)

    if out_ratio >= 0.70 and symmetry <= 0.42:
        return "dominant_exciter"

    if in_ratio >= 0.70 and symmetry <= 0.42:
        return "response_sink"

    if symmetry >= 0.82 and (out_edges + in_edges) >= 3:
        asymmetry = 1.0 - symmetry
        if out_w > in_w and asymmetry >= 0.035:
            return "bridge_exciter_like"
        if in_w > out_w and asymmetry >= 0.035:
            return "bridge_response_like"
        return "bridge_resonator"

    if out_ratio > in_ratio:
        return "exciter_like"

    if in_ratio > out_ratio:
        return "response_like"

    return "balanced"


def _degree(token: str) -> str:
    try:
        return token.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _octave(token: str) -> str:
    try:
        return token.split(".", 1)[0]
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose directed resonance graph into causal roles."
    )

    ap.add_argument("--directed_edges_csv", required=True)

    ap.add_argument("--out_roles_csv", required=True)
    ap.add_argument("--out_note_centers_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_center_score", type=float, default=0.010)
    ap.add_argument("--bridge_center_min_score", type=float, default=0.020)
    ap.add_argument("--bridge_center_min_asymmetry", type=float, default=0.10)

    args = ap.parse_args()

    edges = _load_csv(Path(args.directed_edges_csv))
    nodes: Dict[str, Dict[str, Any]] = {}

    def ensure(node: str) -> Dict[str, Any]:
        if node not in nodes:
            nodes[node] = {
                "node": node,
                "out_weight": 0.0,
                "in_weight": 0.0,
                "cross_degree_out_weight": 0.0,
                "cross_degree_in_weight": 0.0,
                "same_degree_out_weight": 0.0,
                "same_degree_in_weight": 0.0,
                "out_edges": 0,
                "in_edges": 0,
                "targets": set(),
                "sources": set(),
                "first_out_frame": None,
                "first_in_frame": None,
                "birth_delta_sum": 0.0,
                "birth_delta_count": 0,
                "out_lag_sum": 0.0,
                "out_lag_count": 0,
                "short_lag_out_edges": 0,
            }
        return nodes[node]

    for e in edges:
        src = str(e.get("source_node", "")).strip()
        dst = str(e.get("response_node", "")).strip()
        if not src or not dst:
            continue

        weight = _safe_float(e.get("causal_weight"), 0.0)
        transition_kind = str(e.get("transition_kind", "")).strip()
        is_same_degree = transition_kind == "same_degree_sustain"
        first_source_frame = _safe_int(e.get("first_source_frame"), 0)
        first_response_frame = _safe_int(e.get("first_response_frame"), first_source_frame)
        birth_delta = _safe_int(e.get("birth_delta_frames"), first_response_frame - first_source_frame)
        mean_lag_frames = _safe_float(e.get("mean_lag_frames"), 0.0)

        s = ensure(src)
        d = ensure(dst)

        s["out_weight"] += weight
        s["out_edges"] += 1
        s["targets"].add(dst)
        s["birth_delta_sum"] += birth_delta
        s["birth_delta_count"] += 1
        s["out_lag_sum"] += mean_lag_frames
        s["out_lag_count"] += 1
        if mean_lag_frames > 0.0 and mean_lag_frames <= 2.5:
            s["short_lag_out_edges"] += 1
        if s["first_out_frame"] is None or first_source_frame < s["first_out_frame"]:
            s["first_out_frame"] = first_source_frame
        if is_same_degree:
            s["same_degree_out_weight"] += weight
        else:
            s["cross_degree_out_weight"] += weight

        d["in_weight"] += weight
        d["in_edges"] += 1
        d["sources"].add(src)
        if d["first_in_frame"] is None or first_response_frame < d["first_in_frame"]:
            d["first_in_frame"] = first_response_frame
        if is_same_degree:
            d["same_degree_in_weight"] += weight
        else:
            d["cross_degree_in_weight"] += weight

    role_rows: List[Dict[str, Any]] = []
    for node, data in nodes.items():
        out_w = data["out_weight"]
        in_w = data["in_weight"]
        total_w = out_w + in_w
        symmetry = _bridge_symmetry(out_w, in_w, total_w)
        asymmetry = 1.0 - symmetry if total_w > 0.0 else 0.0

        role = _role_label(
            out_w,
            in_w,
            total_w,
            data["out_edges"],
            data["in_edges"],
        )

        source_count = len(data["sources"])
        target_count = len(data["targets"])
        first_out_frame = data["first_out_frame"]
        first_in_frame = data["first_in_frame"]
        mean_birth_delta = data["birth_delta_sum"] / max(data["birth_delta_count"], 1)
        mean_out_lag = data["out_lag_sum"] / max(data["out_lag_count"], 1)
        birth_priority = 0.0
        if first_out_frame is not None and first_in_frame is not None:
            birth_priority += max(0.0, float(first_in_frame - first_out_frame)) * 0.0025
        birth_priority += max(0.0, mean_birth_delta) * 0.0015
        short_lag_bonus = min(data["short_lag_out_edges"], 6) * 0.0035
        sharpness_bonus = 0.0
        if mean_out_lag > 0.0:
            sharpness_bonus += max(0.0, 3.0 - mean_out_lag) * 0.004
            sharpness_bonus -= max(0.0, mean_out_lag - 4.5) * 0.0015

        cross_balance = (
            data["cross_degree_out_weight"]
            - data["cross_degree_in_weight"]
        )
        cross_bias = (
            data["cross_degree_out_weight"]
            - data["cross_degree_in_weight"] * 0.40
        )
        outwardness = max(0.0, out_w - in_w)
        source_target_asymmetry = max(0, target_count - source_count)
        same_degree_penalty = (
            data["same_degree_out_weight"] + data["same_degree_in_weight"]
        ) * 0.35
        bridge_penalty = 0.0
        if role == "bridge_resonator":
            bridge_penalty = symmetry * in_w * 0.45
        elif role == "bridge_exciter_like":
            bridge_penalty = symmetry * in_w * 0.18
        elif role == "bridge_response_like":
            bridge_penalty = symmetry * in_w * 0.28

        center_score = max(
            0.0,
            cross_balance * 0.90
            + cross_bias * 0.08
            + outwardness * 0.60
            + birth_priority
            + short_lag_bonus
            + sharpness_bonus
            + source_target_asymmetry * 0.004
            + target_count * 0.002
            - same_degree_penalty
            - bridge_penalty,
        )

        role_rows.append({
            "node": node,
            "degree": _degree(node),
            "octave": _octave(node),
            "causal_role": role,
            "out_weight": f"{out_w:.9f}",
            "in_weight": f"{in_w:.9f}",
            "total_weight": f"{total_w:.9f}",
            "cross_degree_out_weight": f"{data['cross_degree_out_weight']:.9f}",
            "cross_degree_in_weight": f"{data['cross_degree_in_weight']:.9f}",
            "same_degree_out_weight": f"{data['same_degree_out_weight']:.9f}",
            "same_degree_in_weight": f"{data['same_degree_in_weight']:.9f}",
            "out_edges": data["out_edges"],
            "in_edges": data["in_edges"],
            "target_count": target_count,
            "source_count": source_count,
            "first_out_frame": first_out_frame if first_out_frame is not None else "",
            "first_in_frame": first_in_frame if first_in_frame is not None else "",
            "mean_birth_delta": f"{mean_birth_delta:.6f}",
            "birth_priority": f"{birth_priority:.9f}",
            "mean_out_lag": f"{mean_out_lag:.6f}",
            "short_lag_out_edges": data["short_lag_out_edges"],
            "birth_sharpness_bonus": f"{(short_lag_bonus + sharpness_bonus):.9f}",
            "bridge_symmetry": f"{symmetry:.9f}",
            "asymmetry": f"{asymmetry:.9f}",
            "center_score": f"{center_score:.9f}",
        })

    role_rows.sort(
        key=lambda r: (
            -_safe_float(r["center_score"]),
            -_safe_float(r["cross_degree_out_weight"]),
            -_safe_float(r["out_weight"]),
        )
    )

    centers: List[Dict[str, Any]] = []
    for row in role_rows:
        role = str(row["causal_role"])
        center_score = _safe_float(row["center_score"])
        asymmetry = _safe_float(row["asymmetry"])

        if role in {"dominant_exciter", "exciter_like"}:
            if center_score >= args.min_center_score:
                centers.append(row)
            continue

        if role == "bridge_exciter_like":
            if center_score >= args.min_center_score:
                centers.append(row)
            continue

        if role == "bridge_resonator":
            if (
                center_score >= args.bridge_center_min_score
                and asymmetry >= args.bridge_center_min_asymmetry
            ):
                centers.append(row)

    centers.sort(
        key=lambda r: (
            -_safe_float(r["center_score"]),
            -_safe_float(r["cross_degree_out_weight"]),
            -_safe_float(r["out_weight"]),
        )
    )

    out_roles = Path(args.out_roles_csv)
    out_centers = Path(args.out_note_centers_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)
    out_roles.parent.mkdir(parents=True, exist_ok=True)

    role_fields = [
        "node",
        "degree",
        "octave",
        "causal_role",
        "out_weight",
        "in_weight",
        "total_weight",
        "cross_degree_out_weight",
        "cross_degree_in_weight",
        "same_degree_out_weight",
        "same_degree_in_weight",
        "out_edges",
        "in_edges",
        "target_count",
        "source_count",
        "first_out_frame",
        "first_in_frame",
        "mean_birth_delta",
        "birth_priority",
        "mean_out_lag",
        "short_lag_out_edges",
        "birth_sharpness_bonus",
        "bridge_symmetry",
        "asymmetry",
        "center_score",
    ]

    with out_roles.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=role_fields)
        w.writeheader()
        w.writerows(role_rows)

    with out_centers.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=role_fields)
        w.writeheader()
        w.writerows(centers)

    role_counts: Dict[str, int] = {}
    for row in role_rows:
        role_counts[row["causal_role"]] = role_counts.get(row["causal_role"], 0) + 1

    meta = {
        "stage": "micro_causal_role_decomposition",
        "inputs": {
            "directed_edges_csv": args.directed_edges_csv,
        },
        "outputs": {
            "roles_csv": args.out_roles_csv,
            "note_centers_csv": args.out_note_centers_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_center_score": args.min_center_score,
            "bridge_center_min_score": args.bridge_center_min_score,
            "bridge_center_min_asymmetry": args.bridge_center_min_asymmetry,
        },
        "result": {
            "input_edges": len(edges),
            "nodes": len(role_rows),
            "note_centers": len(centers),
            "role_counts": role_counts,
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "MICRO CAUSAL ROLE DECOMPOSITION",
        "=" * 72,
        f"directed_edges_csv : {args.directed_edges_csv}",
        "",
        f"input_edges        : {len(edges)}",
        f"nodes              : {len(role_rows)}",
        f"note_centers       : {len(centers)}",
        "",
        "Role counts:",
    ]

    for key in sorted(role_counts):
        txt.append(f"  {key}: {role_counts[key]}")

    txt.extend([
        "",
        "Principle:",
        "  True note centers should keep outward causal pressure.",
        "  Earlier births receive a modest prior over later response clusters.",
        "  Short early outgoing edges are weighted above slow loop-like propagation.",
        "  Bridge-like resonators are kept visible.",
        "  Only outward-leaning bridges may graduate into centers;",
        "  fully symmetric bridges remain visible but non-central.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro causal role decomposition complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
