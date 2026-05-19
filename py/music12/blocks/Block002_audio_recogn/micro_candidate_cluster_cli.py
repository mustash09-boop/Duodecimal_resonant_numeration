# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Safe helpers
# ============================================================

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


# ============================================================
# Token helpers
# ============================================================

def _normalize_letters(s: str) -> str:
    s = str(s or "").strip()
    s = s.replace("А", "A").replace("В", "B").replace("С", "C")
    s = s.replace("а", "A").replace("в", "B").replace("с", "C")
    return s


def _split_token_micro(token: str) -> tuple[str, str]:
    """
    Return (coarse_token_without_apostrophe, micro_suffix_without_apostrophe).

    Examples:
        9.A'i27  -> ("9.A", "i27")
        9.A'a55  -> ("9.A", "a55")
        9.A'-    -> ("9.A", "-")
        9.A      -> ("9.A", "")
    """
    token = _normalize_letters(token)

    if "'" not in token:
        return token, ""

    left, right = token.split("'", 1)
    return left, right or "-"


def _token_coarse(token: str) -> str:
    coarse, _micro = _split_token_micro(token)
    return coarse


def _token_micro_or_dash(token: str) -> str:
    _coarse, micro = _split_token_micro(token)
    return micro or "-"


def _micro_center_token(token: str) -> str:
    """
    Return the center token for the same coarse pitch.

    This is NOT a replacement for the real micro token.
    It is only the cluster anchor center.
    """
    coarse = _token_coarse(token)
    return coarse + "'-"


def _extract_micro_shift(token: str) -> str:
    return _token_micro_or_dash(token)


def _parse_candidates(raw: str) -> List[Dict[str, Any]]:
    raw = str(raw or "").strip()

    if not raw:
        return []

    try:
        data = json.loads(raw)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def _candidate_energy(c: Dict[str, Any]) -> float:
    """
    Read energy from several possible legacy/new field names.
    """
    for key in (
        "energy",
        "amplitude",
        "matched_amplitude",
        "score",
        "chain_score",
        "weighted_support_score",
    ):
        if key in c:
            return _safe_float(c.get(key), 0.0)
    return 0.0


def _pick_representative(items: List[Dict[str, Any]]) -> tuple[str, float]:
    """
    Pick strongest micro token inside a coarse cluster.
    """
    best_token = ""
    best_energy = -1.0

    for item in items:
        token = str(item.get("note_token", "")).strip()
        energy = _candidate_energy(item)

        if token and energy > best_energy:
            best_token = token
            best_energy = energy

    if best_energy < 0:
        best_energy = 0.0

    return best_token, best_energy


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Group framewise micro candidates into coarse ownership clusters "
            "while preserving micro individuality."
        )
    )

    ap.add_argument("--framewise_csv", required=True)

    ap.add_argument("--out_cluster_csv", required=True)
    ap.add_argument("--out_cluster_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    framewise_csv = Path(args.framewise_csv)

    out_cluster_csv = Path(args.out_cluster_csv)
    out_cluster_readable_csv = Path(args.out_cluster_readable_csv)
    out_meta_json = Path(args.out_meta_json)
    out_summary_txt = Path(args.out_summary_txt)

    rows = _load_csv(framewise_csv)

    cluster_rows: list[dict[str, Any]] = []
    readable_rows: list[dict[str, Any]] = []

    total_clusters = 0
    total_micro_members = 0
    total_micro_diversity = 0

    for r in rows:
        frame_index = _safe_int(r.get("frame_index", 0), 0)
        time_sec = _safe_float(r.get("time_sec", 0.0), 0.0)

        candidates = _parse_candidates(
            r.get("selected_candidates_json", "")
        )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for c in candidates:
            token = str(c.get("note_token", "")).strip()

            if not token:
                continue

            # Grouping by coarse ownership anchor is allowed,
            # but micro-token is preserved inside the cluster.
            anchor_coarse = _token_coarse(token)

            grouped[anchor_coarse].append(c)

        frame_clusters: list[dict[str, Any]] = []

        for anchor_coarse, items in grouped.items():
            cluster_energy = 0.0
            max_energy = 0.0

            members_micro: list[str] = []
            micro_types: set[str] = set()

            source_items: list[dict[str, Any]] = []

            for item in items:
                token_micro = str(item.get("note_token", "")).strip()

                if not token_micro:
                    continue

                energy = _candidate_energy(item)

                cluster_energy += energy

                if energy > max_energy:
                    max_energy = energy

                members_micro.append(token_micro)

                micro_types.add(
                    _extract_micro_shift(token_micro)
                )

                source_items.append(item)

            representative_token_micro, representative_energy = _pick_representative(items)
            representative_token_coarse = _token_coarse(representative_token_micro) if representative_token_micro else anchor_coarse

            members_unique = sorted(set(members_micro))

            frame_clusters.append(
                {
                    "frame_index": frame_index,
                    "time_sec": time_sec,

                    # Cluster identity fields
                    "anchor_token_coarse": anchor_coarse,
                    "anchor_token_micro_center": anchor_coarse + "'-",

                    # Representative micro identity
                    "representative_token_micro": representative_token_micro,
                    "representative_token_coarse": representative_token_coarse,
                    "representative_energy": representative_energy,

                    # Backward-compatible alias for older downstream modules.
                    # IMPORTANT: this is center anchor only, not a replacement
                    # for representative_token_micro or micro_members.
                    "anchor_token": anchor_coarse + "'-",

                    # Energetics
                    "cluster_energy": cluster_energy,
                    "max_energy": max_energy,

                    # Micro identity preservation
                    "micro_count": len(members_micro),
                    "micro_diversity": len(micro_types),
                    "micro_types": " ".join(sorted(micro_types)),
                    "micro_members": " ".join(members_unique),
                    "micro_members_json": json.dumps(members_unique, ensure_ascii=False),

                    # Raw source candidates for deeper audit/debug.
                    "source_candidates_json": json.dumps(source_items, ensure_ascii=False),
                }
            )

        frame_clusters.sort(
            key=lambda x: (
                -_safe_float(x["cluster_energy"], 0.0),
                -_safe_int(x["micro_count"], 0),
                str(x["anchor_token_coarse"]),
            )
        )

        total_clusters += len(frame_clusters)

        readable_items = []

        for fc in frame_clusters:
            total_micro_members += _safe_int(fc["micro_count"], 0)
            total_micro_diversity += _safe_int(fc["micro_diversity"], 0)

            cluster_rows.append(fc)

            readable_items.append(
                f"{fc['representative_token_micro']} "
                f"[coarse={fc['anchor_token_coarse']}, "
                f"E={fc['cluster_energy']:.3f}, "
                f"micro={fc['micro_diversity']}]"
            )

        readable_rows.append(
            {
                "frame_index": frame_index,
                "time_sec": time_sec,
                "top_clusters": " | ".join(readable_items[:12]),
            }
        )

    fields = list(cluster_rows[0].keys()) if cluster_rows else [
        "frame_index",
        "time_sec",
        "anchor_token_coarse",
        "anchor_token_micro_center",
        "representative_token_micro",
        "representative_token_coarse",
        "representative_energy",
        "anchor_token",
        "cluster_energy",
        "max_energy",
        "micro_count",
        "micro_diversity",
        "micro_types",
        "micro_members",
        "micro_members_json",
        "source_candidates_json",
    ]

    out_cluster_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_cluster_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cluster_rows)

    with out_cluster_readable_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "time_sec",
                "top_clusters",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "micro_candidate_cluster",
        "semantic_version": "micro_preserved_v2",
        "inputs": {
            "framewise_csv": str(framewise_csv),
        },
        "outputs": {
            "cluster_csv": str(out_cluster_csv),
            "cluster_readable_csv": str(out_cluster_readable_csv),
            "meta_json": str(out_meta_json),
            "summary_txt": str(out_summary_txt),
        },
        "result": {
            "input_rows": len(rows),
            "total_clusters": total_clusters,
            "total_micro_members": total_micro_members,
            "total_micro_diversity_sum": total_micro_diversity,
        },
        "ontology_note": (
            "This version groups candidates by coarse ownership anchor but does not "
            "discard micro identities. The real strongest micro identity is stored as "
            "representative_token_micro; all members are preserved in micro_members_json."
        ),
    }

    out_meta_json.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = []
    txt.append("MICRO CANDIDATE CLUSTER")
    txt.append("=" * 72)
    txt.append(f"framewise_csv       : {framewise_csv}")
    txt.append("")
    txt.append(f"input_rows          : {len(rows)}")
    txt.append(f"total_clusters      : {total_clusters}")
    txt.append(f"total_micro_members : {total_micro_members}")
    txt.append(f"micro_diversity_sum : {total_micro_diversity}")
    txt.append("")
    txt.append("Principle:")
    txt.append("  Group micro-shift resonance clouds by coarse ownership anchor,")
    txt.append("  but preserve every micro identity inside the cluster.")
    txt.append("  Do not replace representative_token_micro with anchor_token_micro_center.")
    txt.append("")

    out_summary_txt.write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print("micro candidate clustering complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
