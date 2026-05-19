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


def _role_label(out_w: float, in_w: float, total_w: float) -> str:
    if total_w <= 0:
        return "silent"

    out_ratio = out_w / total_w
    in_ratio = in_w / total_w

    if out_ratio >= 0.68:
        return "dominant_exciter"

    if in_ratio >= 0.68:
        return "response_sink"

    if out_ratio >= 0.45 and in_ratio >= 0.45:
        return "feedback_bridge"

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

    ap.add_argument("--min_center_score", type=float, default=0.015)

    args = ap.parse_args()

    edges = _load_csv(Path(args.directed_edges_csv))

    nodes: Dict[str, Dict[str, Any]] = {}

    def ensure(node: str) -> Dict[str, Any]:
        if node not in nodes:
            nodes[node] = {
                "node": node,
                "out_weight": 0.0,
                "in_weight": 0.0,
                "out_edges": 0,
                "in_edges": 0,
                "targets": set(),
                "sources": set(),
            }
        return nodes[node]

    for e in edges:
        src = str(e.get("source_node", "")).strip()
        dst = str(e.get("response_node", "")).strip()

        if not src or not dst:
            continue

        w = _safe_float(e.get("causal_weight"), 0.0)

        s = ensure(src)
        d = ensure(dst)

        s["out_weight"] += w
        s["out_edges"] += 1
        s["targets"].add(dst)

        d["in_weight"] += w
        d["in_edges"] += 1
        d["sources"].add(src)

    role_rows = []

    for node, data in nodes.items():
        out_w = data["out_weight"]
        in_w = data["in_weight"]
        total_w = out_w + in_w

        role = _role_label(out_w, in_w, total_w)

        center_score = max(0.0, out_w - in_w * 0.35)

        role_rows.append({
            "node": node,
            "degree": _degree(node),
            "octave": _octave(node),
            "causal_role": role,
            "out_weight": f"{out_w:.9f}",
            "in_weight": f"{in_w:.9f}",
            "total_weight": f"{total_w:.9f}",
            "out_edges": data["out_edges"],
            "in_edges": data["in_edges"],
            "target_count": len(data["targets"]),
            "source_count": len(data["sources"]),
            "center_score": f"{center_score:.9f}",
        })

    role_rows.sort(
        key=lambda r: (
            -_safe_float(r["center_score"]),
            -_safe_float(r["out_weight"]),
        )
    )

    centers = [
        r for r in role_rows
        if _safe_float(r["center_score"], 0.0) >= args.min_center_score
        and r["causal_role"] in {
            "dominant_exciter",
            "exciter_like",
            "feedback_bridge",
        }
    ]

    centers.sort(
        key=lambda r: (
            -_safe_float(r["center_score"]),
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
        "out_edges",
        "in_edges",
        "target_count",
        "source_count",
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

    for r in role_rows:
        role_counts[r["causal_role"]] = role_counts.get(r["causal_role"], 0) + 1

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

    for k in sorted(role_counts):
        txt.append(f"  {k}: {role_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Simultaneous notes should emerge as independent",
        "  causal excitation centers, not merely as spectral peaks.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro causal role decomposition complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()