# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


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


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _variance(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _stdev(xs: List[float]) -> float:
    return math.sqrt(_variance(xs))


def _cv(xs: List[float]) -> float:
    return _stdev(xs) / max(abs(_mean(xs)), 1e-9)


def _entropy(labels: List[str]) -> float:
    c = Counter(x for x in labels if x)
    total = sum(c.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in c.values():
        p = v / total
        if p > 0:
            h -= p * math.log(p, 2)
    max_h = math.log(max(len(c), 1), 2) if c else 1.0
    return h / max(max_h, 1e-9)


def _band(note: str) -> str:
    s = str(note or "").strip()
    if "." not in s:
        return "UNKNOWN"
    octv = s.split(".", 1)[0]
    digits = "123456789ABC"
    value = 0
    for ch in octv.upper():
        if ch in digits:
            value = value * 12 + digits.index(ch) + 1
    if value <= 7:
        return "LOW"
    if value >= 10:
        return "HIGH"
    return "MID"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure ecology asymmetry: non-uniform living body behavior vs synthetic uniform turbulence."
    )

    ap.add_argument("--breath_variability_csv", required=True)
    ap.add_argument("--instrument_ecology_clusters_csv", required=True)
    ap.add_argument("--identity_to_cluster_csv", required=True)
    ap.add_argument("--attractor_events_csv", required=True)

    ap.add_argument("--out_asymmetry_csv", required=True)
    ap.add_argument("--out_cluster_asymmetry_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    breath_rows = _load_csv(Path(args.breath_variability_csv))
    clusters = _load_csv(Path(args.instrument_ecology_clusters_csv))
    mappings = _load_csv(Path(args.identity_to_cluster_csv))
    attractors = _load_csv(Path(args.attractor_events_csv))

    breath_map = {str(r.get("identity_id", "")).strip(): r for r in breath_rows}
    attractor_map = {str(r.get("identity_id", "")).strip(): r for r in attractors}
    cluster_map = {str(r.get("cluster_id", "")).strip(): r for r in clusters}

    ids_by_cluster: Dict[str, List[str]] = defaultdict(list)
    for m in mappings:
        cid = str(m.get("cluster_id", "")).strip()
        iid = str(m.get("identity_id", "")).strip()
        if cid and iid:
            ids_by_cluster[cid].append(iid)

    cluster_rows = []

    all_cluster_turbulence = []
    all_cluster_living = []
    all_cluster_mechanical = []
    all_cluster_entropy = []
    all_cluster_band_cv = []
    all_cluster_irregularity = []

    regional_counts = Counter()
    cluster_kind_counts = Counter()

    for cid, ids in sorted(ids_by_cluster.items(), key=lambda x: _safe_int(x[0], 0)):
        cl = cluster_map.get(cid, {})

        turbulences = []
        livings = []
        mechanicals = []
        irregularities = []
        entropies = []
        bands = []
        breath_classes = []
        notes = []

        for iid in ids:
            b = breath_map.get(iid, {})
            a = attractor_map.get(iid, {})

            turbulences.append(_safe_float(b.get("ecology_turbulence"), 0.0))
            livings.append(_safe_float(b.get("living_breath_score"), 0.0))
            mechanicals.append(_safe_float(b.get("mechanical_sustain_score"), 0.0))
            irregularities.append(_safe_float(b.get("rebirth_irregularity"), 0.0))
            entropies.append(_safe_float(b.get("phase_entropy"), 0.0))

            note = str(a.get("attractor_note", b.get("note_token", ""))).strip()
            notes.append(note)
            bands.append(_band(note))
            breath_classes.append(str(b.get("breath_class", "")).strip())

        turbulence_cv = _cv(turbulences)
        living_cv = _cv(livings)
        mechanical_cv = _cv(mechanicals)
        irregularity_cv = _cv(irregularities)
        entropy_cv = _cv(entropies)
        band_entropy = _entropy(bands)
        class_entropy = _entropy(breath_classes)

        asymmetry_score = 0.0
        asymmetry_score += min(turbulence_cv / 0.60, 1.0) * 0.24
        asymmetry_score += min(living_cv / 0.45, 1.0) * 0.16
        asymmetry_score += min(irregularity_cv / 0.45, 1.0) * 0.18
        asymmetry_score += band_entropy * 0.16
        asymmetry_score += class_entropy * 0.14
        asymmetry_score += min(len(set(notes)) / 8.0, 1.0) * 0.12
        asymmetry_score = max(0.0, min(asymmetry_score, 1.0))

        uniformity_score = 0.0
        uniformity_score += max(0.0, 1.0 - min(turbulence_cv / 0.40, 1.0)) * 0.30
        uniformity_score += max(0.0, 1.0 - min(living_cv / 0.35, 1.0)) * 0.20
        uniformity_score += max(0.0, 1.0 - min(irregularity_cv / 0.35, 1.0)) * 0.18
        uniformity_score += max(0.0, 1.0 - class_entropy) * 0.18
        uniformity_score += max(0.0, 1.0 - band_entropy) * 0.14
        uniformity_score = max(0.0, min(uniformity_score, 1.0))

        if asymmetry_score >= 0.58 and asymmetry_score > uniformity_score:
            regional_type = "LIVING_ASYMMETRIC_ECOLOGY"
        elif uniformity_score >= 0.58:
            regional_type = "SYNTHETIC_UNIFORM_ECOLOGY"
        elif asymmetry_score >= 0.38:
            regional_type = "PARTIAL_ASYMMETRIC_ECOLOGY"
        else:
            regional_type = "LOW_STRUCTURE_ECOLOGY"

        regional_counts[regional_type] += 1
        cluster_kind_counts[str(cl.get("cluster_kind", "")).strip()] += 1

        all_cluster_turbulence.append(_mean(turbulences))
        all_cluster_living.append(_mean(livings))
        all_cluster_mechanical.append(_mean(mechanicals))
        all_cluster_entropy.append(class_entropy)
        all_cluster_band_cv.append(band_entropy)
        all_cluster_irregularity.append(_mean(irregularities))

        cluster_rows.append({
            "cluster_id": cid,
            "cluster_kind": cl.get("cluster_kind", ""),
            "regional_type": regional_type,
            "identity_count": len(ids),
            "asymmetry_score": f"{asymmetry_score:.9f}",
            "uniformity_score": f"{uniformity_score:.9f}",
            "mean_turbulence": f"{_mean(turbulences):.9f}",
            "turbulence_cv": f"{turbulence_cv:.9f}",
            "mean_living_breath": f"{_mean(livings):.9f}",
            "living_breath_cv": f"{living_cv:.9f}",
            "mean_mechanical_sustain": f"{_mean(mechanicals):.9f}",
            "mechanical_sustain_cv": f"{mechanical_cv:.9f}",
            "mean_rebirth_irregularity": f"{_mean(irregularities):.9f}",
            "rebirth_irregularity_cv": f"{irregularity_cv:.9f}",
            "phase_entropy_cv": f"{entropy_cv:.9f}",
            "range_band_entropy": f"{band_entropy:.9f}",
            "breath_class_entropy": f"{class_entropy:.9f}",
            "range_distribution": " | ".join(f"{k}:{v}" for k, v in Counter(bands).most_common()),
            "breath_class_distribution": " | ".join(f"{k}:{v}" for k, v in Counter(breath_classes).most_common()),
            "notes": " ".join(sorted(set(notes))[:80]),
        })

    global_asymmetry = 0.0
    global_asymmetry += min(_cv(all_cluster_turbulence) / 0.45, 1.0) * 0.28
    global_asymmetry += min(_cv(all_cluster_living) / 0.35, 1.0) * 0.20
    global_asymmetry += min(_cv(all_cluster_irregularity) / 0.35, 1.0) * 0.18
    global_asymmetry += _entropy([r["regional_type"] for r in cluster_rows]) * 0.18
    global_asymmetry += _entropy([r["cluster_kind"] for r in cluster_rows]) * 0.16
    global_asymmetry = max(0.0, min(global_asymmetry, 1.0))

    global_uniformity = 0.0
    global_uniformity += max(0.0, 1.0 - min(_cv(all_cluster_turbulence) / 0.35, 1.0)) * 0.30
    global_uniformity += max(0.0, 1.0 - min(_cv(all_cluster_living) / 0.30, 1.0)) * 0.22
    global_uniformity += max(0.0, 1.0 - _entropy([r["regional_type"] for r in cluster_rows])) * 0.24
    global_uniformity += max(0.0, 1.0 - _entropy([r["cluster_kind"] for r in cluster_rows])) * 0.24
    global_uniformity = max(0.0, min(global_uniformity, 1.0))

    if global_asymmetry >= 0.56 and global_asymmetry > global_uniformity:
        global_type = "LIVING_ASYMMETRIC_RESONANCE_BODY"
    elif global_uniformity >= 0.56:
        global_type = "SYNTHETIC_UNIFORM_RESONANCE_BODY"
    else:
        global_type = "MIXED_OR_UNRESOLVED_RESONANCE_BODY"

    asymmetry_rows = [{
        "global_type": global_type,
        "global_asymmetry": f"{global_asymmetry:.9f}",
        "global_uniformity": f"{global_uniformity:.9f}",
        "cluster_count": len(cluster_rows),
        "regional_distribution": " | ".join(f"{k}:{v}" for k, v in regional_counts.most_common()),
        "cluster_kind_distribution": " | ".join(f"{k}:{v}" for k, v in cluster_kind_counts.most_common()),
        "mean_cluster_turbulence": f"{_mean(all_cluster_turbulence):.9f}",
        "cv_cluster_turbulence": f"{_cv(all_cluster_turbulence):.9f}",
        "mean_cluster_living_breath": f"{_mean(all_cluster_living):.9f}",
        "cv_cluster_living_breath": f"{_cv(all_cluster_living):.9f}",
        "mean_cluster_mechanical_sustain": f"{_mean(all_cluster_mechanical):.9f}",
        "mean_cluster_irregularity": f"{_mean(all_cluster_irregularity):.9f}",
    }]

    _write_csv(
        Path(args.out_asymmetry_csv),
        asymmetry_rows,
        [
            "global_type",
            "global_asymmetry",
            "global_uniformity",
            "cluster_count",
            "regional_distribution",
            "cluster_kind_distribution",
            "mean_cluster_turbulence",
            "cv_cluster_turbulence",
            "mean_cluster_living_breath",
            "cv_cluster_living_breath",
            "mean_cluster_mechanical_sustain",
            "mean_cluster_irregularity",
        ],
    )

    _write_csv(
        Path(args.out_cluster_asymmetry_csv),
        cluster_rows,
        [
            "cluster_id",
            "cluster_kind",
            "regional_type",
            "identity_count",
            "asymmetry_score",
            "uniformity_score",
            "mean_turbulence",
            "turbulence_cv",
            "mean_living_breath",
            "living_breath_cv",
            "mean_mechanical_sustain",
            "mechanical_sustain_cv",
            "mean_rebirth_irregularity",
            "rebirth_irregularity_cv",
            "phase_entropy_cv",
            "range_band_entropy",
            "breath_class_entropy",
            "range_distribution",
            "breath_class_distribution",
            "notes",
        ],
    )

    summary = {
        "global_type": global_type,
        "global_asymmetry": global_asymmetry,
        "global_uniformity": global_uniformity,
        "cluster_count": len(cluster_rows),
        "regional_distribution": dict(regional_counts),
        "cluster_kind_distribution": dict(cluster_kind_counts),
        "mean_cluster_turbulence": _mean(all_cluster_turbulence),
        "cv_cluster_turbulence": _cv(all_cluster_turbulence),
        "mean_cluster_living_breath": _mean(all_cluster_living),
        "cv_cluster_living_breath": _cv(all_cluster_living),
        "mean_cluster_mechanical_sustain": _mean(all_cluster_mechanical),
        "mean_cluster_irregularity": _mean(all_cluster_irregularity),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()