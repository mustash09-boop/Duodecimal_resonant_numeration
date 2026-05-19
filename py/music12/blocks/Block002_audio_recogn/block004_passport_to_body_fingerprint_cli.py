# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set


DIGITS12 = "123456789ABC"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _tokens_from_components(items: List[Dict[str, Any]]) -> Set[str]:
    out = set()
    for r in items:
        t = str(r.get("cluster_token", r.get("token", ""))).strip()
        if t:
            out.add(t)
    return out


def _octave_value(note: str) -> int:
    try:
        raw = str(note).split(".", 1)[0]
        value = 0
        for ch in raw.upper():
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


def _counter_text(c: Counter) -> str:
    return " | ".join(f"{k}:{v}" for k, v in c.most_common())


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Block004 instrument passport JSON into body fingerprint CSV compatible with Block002 observed fingerprints."
    )

    ap.add_argument("--passport_json", required=True)
    ap.add_argument("--passport_name", required=True)

    ap.add_argument("--out_fingerprint_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    data = json.loads(Path(args.passport_json).read_text(encoding="utf-8"))

    summary = data.get("summary", {})
    box_all = data.get("box_all_top", [])
    box_breath = data.get("box_breath_top", data.get("breath_top", []))
    box_resonance = data.get("box_resonance_top", data.get("resonance_body_top", []))
    harmonic_relation = data.get("box_harmonic_relation_top", data.get("harmonic_relation_top", []))

    body_tokens = _tokens_from_components(box_resonance or box_all)
    secondary_tokens = _tokens_from_components(box_breath)
    harmonic_tokens = _tokens_from_components(harmonic_relation)

    range_counter = Counter()
    for r in box_resonance or box_all:
        examples = str(r.get("examples", ""))
        for part in examples.replace("|", " ").split():
            note = part.strip().replace("piano_real_", "").replace("piano_midi_", "")
            if "." in note:
                range_counter[_range_band(note)] += 1

    # Старый паспорт не содержит временного поведения, поэтому создаём
    # приближенную fingerprint-проекцию по плотности тела.
    breath_n = _safe_float(summary.get("box_breath_components", len(box_breath)))
    resonance_n = _safe_float(summary.get("box_resonance_components", len(box_resonance)))
    all_n = _safe_float(summary.get("box_all_components", len(box_all)))

    behavior_counter = Counter()
    behavior_counter["DELAYED_RETURNING_BODY_FIELD"] = max(int(resonance_n), 1)
    if breath_n >= 40:
        behavior_counter["LONG_BODY_PERSISTENCE"] = int(breath_n)
    else:
        behavior_counter["FAST_DECAY_BODY"] = max(int(breath_n), 1)

    cluster_counter = Counter()
    if resonance_n >= 100:
        cluster_counter["POSSIBLE_INSTRUMENT_BODY_CLUSTER"] = int(resonance_n // 12)
    cluster_counter["LOCAL_RESONANCE_FRAGMENT"] = max(int(all_n // 12), 1)

    fingerprint_strength = 0.0
    fingerprint_strength += min(len(body_tokens) / 120.0, 1.0) * 0.34
    fingerprint_strength += min(len(secondary_tokens) / 80.0, 1.0) * 0.18
    fingerprint_strength += min(resonance_n / 120.0, 1.0) * 0.24
    fingerprint_strength += min(breath_n / 64.0, 1.0) * 0.12
    fingerprint_strength += min(len(harmonic_tokens) / 120.0, 1.0) * 0.12

    if fingerprint_strength >= 0.68:
        status = "STRONG_BODY_FINGERPRINT"
    elif fingerprint_strength >= 0.44:
        status = "PARTIAL_BODY_FINGERPRINT"
    else:
        status = "WEAK_BODY_FINGERPRINT"

    row = {
        "passport_name": args.passport_name,
        "fingerprint_status": status,
        "fingerprint_strength": f"{fingerprint_strength:.9f}",
        "cluster_count": cluster_counter["POSSIBLE_INSTRUMENT_BODY_CLUSTER"],
        "total_identity_count": int(resonance_n),
        "body_token_count": len(body_tokens),
        "secondary_token_count": len(secondary_tokens),
        "core_token_count": len(harmonic_tokens),
        "mean_cluster_energy": f"{resonance_n:.9f}",
        "mean_cluster_density": f"{(resonance_n / max(all_n, 1.0)):.9f}",
        "mean_secondary_persistence_ratio": f"{(breath_n / max(resonance_n, 1.0)):.9f}",
        "mean_delayed_secondary_count": f"{breath_n:.9f}",
        "range_distribution": _counter_text(range_counter),
        "field_behavior_distribution": _counter_text(behavior_counter),
        "cluster_kind_distribution": _counter_text(cluster_counter),
        "body_tokens": " ".join(sorted(body_tokens)[:300]),
        "secondary_tokens": " ".join(sorted(secondary_tokens)[:300]),
    }

    _write_csv(
        Path(args.out_fingerprint_csv),
        [row],
        [
            "passport_name",
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

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()