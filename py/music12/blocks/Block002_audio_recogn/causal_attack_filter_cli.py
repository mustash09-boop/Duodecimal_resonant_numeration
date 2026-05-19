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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _duration(r: Dict[str, Any]) -> float:
    return _safe_float(r.get("duration_sec"), 0.0)


def _mean_score(r: Dict[str, Any]) -> float:
    return _safe_float(r.get("mean_score"), 0.0)


def _max_score(r: Dict[str, Any]) -> float:
    return _safe_float(r.get("max_score"), 0.0)


def _attack_strength(r: Dict[str, Any]) -> float:
    mean_s = _mean_score(r)
    max_s = _max_score(r)

    if mean_s <= 0:
        return 0.0

    # attack dominance
    return max(0.0, (max_s - mean_s) / mean_s)


def _causal_score(r: Dict[str, Any]) -> float:
    dur = _duration(r)

    mean_s = _mean_score(r)
    max_s = _max_score(r)

    atk = _attack_strength(r)

    score = mean_s

    # strong initial birth
    score += atk * 0.35

    # long resonance tails are suspicious
    if dur > 0.80:
        score -= (dur - 0.80) * 0.25

    # short impulses without stability are suspicious too
    if dur < 0.10:
        score -= 0.20

    # max dominance bonus
    score += max(0.0, max_s - mean_s) * 0.15

    return max(score, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter resonance continuations and preserve causal note attacks."
    )

    ap.add_argument("--persistent_csv", required=True)

    ap.add_argument("--out_filtered_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_causal_score", type=float, default=0.90)

    args = ap.parse_args()

    rows = _load_csv(Path(args.persistent_csv))

    out_rows = []

    kept = 0
    rejected = 0

    for r in rows:
        causal = _causal_score(r)

        rr = dict(r)

        rr["causal_attack_score"] = f"{causal:.9f}"

        if causal >= args.min_causal_score:
            rr["causal_status"] = "KEEP"
            kept += 1
            out_rows.append(rr)
        else:
            rr["causal_status"] = "REJECT"
            rejected += 1

    out_rows.sort(
        key=lambda r: (
            _safe_float(r.get("time_start_sec"), 0.0),
            -_safe_float(r.get("causal_attack_score"), 0.0),
        )
    )

    out_csv = Path(args.out_filtered_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys()) if out_rows else []

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "stage": "causal_attack_filter",
        "inputs": {
            "persistent_csv": args.persistent_csv,
        },
        "outputs": {
            "filtered_csv": args.out_filtered_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_causal_score": args.min_causal_score,
        },
        "result": {
            "input_events": len(rows),
            "kept_events": kept,
            "rejected_events": rejected,
        },
    }

    Path(args.out_meta_json).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "CAUSAL ATTACK FILTER",
        "=" * 72,
        f"persistent_csv   : {args.persistent_csv}",
        "",
        f"input_events     : {len(rows)}",
        f"kept_events      : {kept}",
        f"rejected_events  : {rejected}",
        "",
        "Principle:",
        "  Separate causal note births from resonance afterlife.",
        "  Strong attack dominance is treated as causal excitation.",
        "",
    ]

    Path(args.out_summary_txt).write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print("causal attack filter complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()