# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


DIGITS12 = "123456789ABC"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).replace(",", ".")))
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


def _normalize_note(token: Any) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _octave_value(note: str) -> int:
    try:
        raw = _normalize_note(note).split(".", 1)[0]
        value = 0
        for ch in raw:
            ch = ch.upper()
            if ch in DIGITS12:
                value = value * 12 + (DIGITS12.index(ch) + 1)
        return value
    except Exception:
        return 0


def _range_band(note: str) -> str:
    ov = _octave_value(note)
    if ov <= 7:
        return "LOW"
    if ov >= 10:
        return "HIGH"
    return "MID"


def _counter_text(c: Counter, limit: int = 24) -> str:
    return " | ".join(f"{k}:{v}" for k, v in c.most_common(limit))


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build observed instrument-body fingerprint from attractor ecology clusters."
    )

    ap.add_argument("--instrument_ecology_clusters_csv", required=True)
    ap.add_argument("--identity_to_cluster_csv", required=True)
    ap.add_argument("--attractor_events_csv", required=True)
    ap.add_argument("--field_persistence_csv", required=True)

    ap.add_argument("--out_fingerprint_csv", required=True)
    ap.add_argument("--out_range_profile_csv", required=True)
    ap.add_argument("--out_behavior_profile_csv", required=True)
    ap.add_argument("--out_cluster_fingerprint_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    clusters = _load_csv(Path(args.instrument_ecology_clusters_csv))
    mappings = _load_csv(Path(args.identity_to_cluster_csv))
    attractors = _load_csv(Path(args.attractor_events_csv))
    fields = _load_csv(Path(args.field_persistence_csv))

    cluster_map = {str(r.get("cluster_id", "")).strip(): r for r in clusters}
    attractor_map = {str(r.get("identity_id", "")).strip(): r for r in attractors}
    field_map = {str(r.get("identity_id", "")).strip(): r for r in fields}

    by_cluster: Dict[str, List[str]] = defaultdict(list)
    for m in mappings:
        cid = str(m.get("cluster_id", "")).strip()
        iid = str(m.get("identity_id", "")).strip()
        if cid and iid:
            by_cluster[cid].append(iid)

    cluster_rows = []

    global_ranges = Counter()
    global_behaviors = Counter()
    global_cluster_kinds = Counter()

    all_body_tokens = set()
    all_secondary_tokens = set()
    all_core_tokens = set()

    cluster_energy_values = []
    persistence_values = []
    delayed_values = []
    density_values = []

    for cid, ids in sorted(by_cluster.items(), key=lambda x: _safe_int(x[0], 0)):
        cl = cluster_map.get(cid, {})

        range_counts = Counter()
        behavior_counts = Counter()
        status_counts = Counter()

        body_tokens = set()
        secondary_tokens = set()
        core_tokens = set()

        ecology_energy = _safe_float(cl.get("ecology_energy_sum"), 0.0)
        ecology_mean = _safe_float(cl.get("ecology_energy_mean"), 0.0)

        persistence = []
        delayed = []
        field_conf = []

        for iid in ids:
            a = attractor_map.get(iid, {})
            f = field_map.get(iid, {})

            note = _normalize_note(a.get("attractor_note", f.get("resolved_note", "")))
            if note:
                range_counts[_range_band(note)] += 1
                global_ranges[_range_band(note)] += 1

            behavior = str(f.get("field_behavior", a.get("field_behavior", ""))).strip()
            if behavior:
                behavior_counts[behavior] += 1
                global_behaviors[behavior] += 1

            status = str(a.get("attractor_status", "")).strip()
            if status:
                status_counts[status] += 1

            body_tokens |= _tokens(a.get("instrument_body_tokens", ""))
            secondary_tokens |= _tokens(a.get("secondary_field_tokens", ""))
            core_tokens |= _tokens(a.get("excitation_core_tokens", ""))

            persistence.append(_safe_float(f.get("secondary_persistence_ratio"), 0.0))
            delayed.append(_safe_float(f.get("delayed_secondary_count"), 0.0))
            field_conf.append(_safe_float(f.get("field_confidence"), 0.0))

        all_body_tokens |= body_tokens
        all_secondary_tokens |= secondary_tokens
        all_core_tokens |= core_tokens

        cluster_kind = str(cl.get("cluster_kind", "")).strip()
        if cluster_kind:
            global_cluster_kinds[cluster_kind] += 1

        mean_persistence = _mean(persistence)
        mean_delayed = _mean(delayed)
        mean_field_conf = _mean(field_conf)

        cluster_energy_values.append(ecology_energy)
        persistence_values.append(mean_persistence)
        delayed_values.append(mean_delayed)
        density_values.append(ecology_mean)

        if mean_persistence >= 2.5 and mean_delayed >= 3:
            body_profile = "RETURNING_RANGE_BODY"
        elif mean_persistence >= 2.5:
            body_profile = "LONG_SUSTAIN_BODY"
        elif ecology_mean >= 3.5:
            body_profile = "DENSE_BODY_CORRIDOR"
        elif len(ids) >= 4:
            body_profile = "LOCAL_BODY_REGION"
        else:
            body_profile = "BODY_FRAGMENT"

        cluster_rows.append({
            "cluster_id": cid,
            "cluster_kind": cluster_kind,
            "body_profile": body_profile,
            "identity_count": len(ids),
            "range_distribution": _counter_text(range_counts),
            "field_behavior_distribution": _counter_text(behavior_counts),
            "attractor_status_distribution": _counter_text(status_counts),
            "ecology_energy_sum": f"{ecology_energy:.9f}",
            "ecology_energy_mean": f"{ecology_mean:.9f}",
            "mean_secondary_persistence_ratio": f"{mean_persistence:.9f}",
            "mean_delayed_secondary_count": f"{mean_delayed:.9f}",
            "mean_field_confidence": f"{mean_field_conf:.9f}",
            "body_token_count": len(body_tokens),
            "secondary_token_count": len(secondary_tokens),
            "core_token_count": len(core_tokens),
            "body_tokens": " ".join(sorted(body_tokens)[:220]),
            "secondary_tokens": " ".join(sorted(secondary_tokens)[:220]),
            "core_tokens": " ".join(sorted(core_tokens)[:160]),
        })

    range_rows = []
    for band in ["LOW", "MID", "HIGH"]:
        count = global_ranges.get(band, 0)
        range_rows.append({
            "range_band": band,
            "identity_count": count,
            "identity_ratio": f"{count / max(sum(global_ranges.values()), 1):.9f}",
        })

    behavior_rows = []
    total_behavior = sum(global_behaviors.values())
    for behavior, count in global_behaviors.most_common():
        behavior_rows.append({
            "field_behavior": behavior,
            "count": count,
            "ratio": f"{count / max(total_behavior, 1):.9f}",
        })

    fingerprint_strength = 0.0
    fingerprint_strength += min(len(by_cluster) / 16.0, 1.0) * 0.18
    fingerprint_strength += min(len(all_body_tokens) / 80.0, 1.0) * 0.22
    fingerprint_strength += min(len(all_secondary_tokens) / 80.0, 1.0) * 0.14
    fingerprint_strength += min(_mean(persistence_values) / 3.0, 1.0) * 0.18
    fingerprint_strength += min(_mean(delayed_values) / 5.0, 1.0) * 0.14
    fingerprint_strength += min(_mean(density_values) / 4.0, 1.0) * 0.14
    fingerprint_strength = max(0.0, min(fingerprint_strength, 1.0))

    if fingerprint_strength >= 0.68:
        fingerprint_status = "STRONG_BODY_FINGERPRINT"
    elif fingerprint_strength >= 0.44:
        fingerprint_status = "PARTIAL_BODY_FINGERPRINT"
    else:
        fingerprint_status = "WEAK_BODY_FINGERPRINT"

    fingerprint_rows = [{
        "fingerprint_status": fingerprint_status,
        "fingerprint_strength": f"{fingerprint_strength:.9f}",
        "cluster_count": len(by_cluster),
        "total_identity_count": sum(len(v) for v in by_cluster.values()),
        "body_token_count": len(all_body_tokens),
        "secondary_token_count": len(all_secondary_tokens),
        "core_token_count": len(all_core_tokens),
        "mean_cluster_energy": f"{_mean(cluster_energy_values):.9f}",
        "mean_cluster_density": f"{_mean(density_values):.9f}",
        "mean_secondary_persistence_ratio": f"{_mean(persistence_values):.9f}",
        "mean_delayed_secondary_count": f"{_mean(delayed_values):.9f}",
        "range_distribution": _counter_text(global_ranges),
        "field_behavior_distribution": _counter_text(global_behaviors),
        "cluster_kind_distribution": _counter_text(global_cluster_kinds),
        "body_tokens": " ".join(sorted(all_body_tokens)[:300]),
        "secondary_tokens": " ".join(sorted(all_secondary_tokens)[:300]),
    }]

    _write_csv(
        Path(args.out_fingerprint_csv),
        fingerprint_rows,
        [
            "fingerprint_status",
            "fingerprint_strength",
            "cluster_count",
            "total_identity_count",
            "body_token_count",
            "secondary_token_count",
            "core_token_count",
            "mean_cluster_energy",
            "mean_cluster_density",
            "mean_secondary_persistence_ratio",
            "mean_delayed_secondary_count",
            "range_distribution",
            "field_behavior_distribution",
            "cluster_kind_distribution",
            "body_tokens",
            "secondary_tokens",
        ],
    )

    _write_csv(Path(args.out_range_profile_csv), range_rows, ["range_band", "identity_count", "identity_ratio"])
    _write_csv(Path(args.out_behavior_profile_csv), behavior_rows, ["field_behavior", "count", "ratio"])

    _write_csv(
        Path(args.out_cluster_fingerprint_csv),
        cluster_rows,
        [
            "cluster_id",
            "cluster_kind",
            "body_profile",
            "identity_count",
            "range_distribution",
            "field_behavior_distribution",
            "attractor_status_distribution",
            "ecology_energy_sum",
            "ecology_energy_mean",
            "mean_secondary_persistence_ratio",
            "mean_delayed_secondary_count",
            "mean_field_confidence",
            "body_token_count",
            "secondary_token_count",
            "core_token_count",
            "body_tokens",
            "secondary_tokens",
            "core_tokens",
        ],
    )

    summary = {
        "fingerprint_status": fingerprint_status,
        "fingerprint_strength": fingerprint_strength,
        "cluster_count": len(by_cluster),
        "total_identity_count": sum(len(v) for v in by_cluster.values()),
        "range_distribution": dict(global_ranges),
        "field_behavior_distribution": dict(global_behaviors),
        "cluster_kind_distribution": dict(global_cluster_kinds),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()