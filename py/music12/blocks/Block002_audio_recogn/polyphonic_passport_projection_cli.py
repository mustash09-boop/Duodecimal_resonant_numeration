# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _tokens(raw: Any) -> Set[str]:
    return {
        x.strip()
        for x in str(raw or "").replace("|", " ").replace(",", " ").split()
        if x.strip()
    }


def _parse_distribution(raw: Any) -> Counter:
    c = Counter()
    for part in str(raw or "").split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            c[k.strip()] += float(v.strip())
        except Exception:
            pass
    return c


def _counter_text(c: Counter) -> str:
    return " | ".join(f"{k}:{v:g}" for k, v in c.most_common())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Project isolated Block004 body fingerprint into expected polyphonic ecology fingerprint."
    )

    ap.add_argument("--passport_fingerprint_csv", required=True)
    ap.add_argument("--observed_fingerprint_csv", required=True)

    ap.add_argument("--out_projected_fingerprint_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--polyphony_gain", type=float, default=1.65)
    ap.add_argument("--delayed_return_gain", type=float, default=1.35)
    ap.add_argument("--fragment_growth", type=float, default=1.80)

    args = ap.parse_args()

    passport = _load_csv(Path(args.passport_fingerprint_csv))[0]
    observed = _load_csv(Path(args.observed_fingerprint_csv))[0]

    p_ranges = _parse_distribution(passport.get("range_distribution", ""))
    o_ranges = _parse_distribution(observed.get("range_distribution", ""))

    p_beh = _parse_distribution(passport.get("field_behavior_distribution", ""))
    o_beh = _parse_distribution(observed.get("field_behavior_distribution", ""))

    p_clusters = _parse_distribution(passport.get("cluster_kind_distribution", ""))
    o_clusters = _parse_distribution(observed.get("cluster_kind_distribution", ""))

    body_tokens = _tokens(passport.get("body_tokens", ""))
    secondary_tokens = _tokens(passport.get("secondary_tokens", ""))

    observed_cluster_count = _safe_float(observed.get("cluster_count"), 1.0)
    passport_cluster_count = _safe_float(passport.get("cluster_count"), 1.0)

    cluster_scale = observed_cluster_count / max(passport_cluster_count, 1.0)
    cluster_scale = max(0.75, min(cluster_scale, 4.0))

    projected_ranges = Counter()
    for k, v in p_ranges.items():
        projected_ranges[k] = v * args.polyphony_gain

    # Смешиваем с наблюдаемой плотностью диапазонов, но не копируем её полностью.
    for k, v in o_ranges.items():
        projected_ranges[k] += v * 0.35

    projected_beh = Counter()
    for k, v in p_beh.items():
        projected_beh[k] = v

    projected_beh["DELAYED_RETURNING_BODY_FIELD"] += (
        p_beh.get("DELAYED_RETURNING_BODY_FIELD", 0.0) * (args.delayed_return_gain - 1.0)
    )

    if "FAST_DECAY_BODY" in projected_beh:
        projected_beh["FAST_DECAY_BODY"] *= 0.55
        projected_beh["DELAYED_RETURNING_BODY_FIELD"] += p_beh["FAST_DECAY_BODY"] * 0.45

    projected_beh["LONG_BODY_PERSISTENCE"] += (
        len(secondary_tokens) * 0.35
    )

    # Полифония увеличивает локальные фрагменты и возможные body clusters.
    projected_clusters = Counter()
    projected_clusters["POSSIBLE_INSTRUMENT_BODY_CLUSTER"] = (
        p_clusters.get("POSSIBLE_INSTRUMENT_BODY_CLUSTER", 0.0) * cluster_scale
    )
    projected_clusters["LOCAL_RESONANCE_FRAGMENT"] = (
        p_clusters.get("LOCAL_RESONANCE_FRAGMENT", 0.0) * args.fragment_growth * cluster_scale
    )

    projected_strength = _safe_float(passport.get("fingerprint_strength"), 0.0)
    projected_strength += min(len(secondary_tokens) / 80.0, 1.0) * 0.07
    projected_strength += min(cluster_scale / 3.0, 1.0) * 0.05
    projected_strength = max(0.0, min(projected_strength, 1.0))

    if projected_strength >= 0.68:
        status = "STRONG_BODY_FINGERPRINT"
    elif projected_strength >= 0.44:
        status = "PARTIAL_BODY_FINGERPRINT"
    else:
        status = "WEAK_BODY_FINGERPRINT"

    projected = dict(passport)
    projected["fingerprint_status"] = status
    projected["fingerprint_strength"] = f"{projected_strength:.9f}"
    projected["cluster_count"] = f"{observed_cluster_count:.0f}"
    projected["total_identity_count"] = observed.get("total_identity_count", passport.get("total_identity_count", ""))
    projected["range_distribution"] = _counter_text(projected_ranges)
    projected["field_behavior_distribution"] = _counter_text(projected_beh)
    projected["cluster_kind_distribution"] = _counter_text(projected_clusters)

    _write_csv(
        Path(args.out_projected_fingerprint_csv),
        [projected],
        list(projected.keys()),
    )

    summary = {
        "input_passport": passport.get("passport_name", ""),
        "projection": "isolated_to_polyphonic",
        "cluster_scale": cluster_scale,
        "projected_status": status,
        "projected_strength": projected_strength,
        "projected_range_distribution": projected["range_distribution"],
        "projected_field_behavior_distribution": projected["field_behavior_distribution"],
        "projected_cluster_kind_distribution": projected["cluster_kind_distribution"],
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()