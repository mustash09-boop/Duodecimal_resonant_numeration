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


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    h = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            h -= p * math.log(p, 2)

    max_h = math.log(max(len(counter), 1), 2) if counter else 1.0
    return h / max(max_h, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure resonance breath variability: living turbulence vs mechanical sustain."
    )

    ap.add_argument("--life_cycle_csv", required=True)
    ap.add_argument("--life_curve_csv", required=True)
    ap.add_argument("--field_persistence_csv", required=True)
    ap.add_argument("--attractor_events_csv", required=True)

    ap.add_argument("--out_breath_variability_csv", required=True)
    ap.add_argument("--out_breath_variability_frame_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    life_rows = _load_csv(Path(args.life_cycle_csv))
    curve_rows = _load_csv(Path(args.life_curve_csv))
    field_rows = _load_csv(Path(args.field_persistence_csv))
    attractor_rows = _load_csv(Path(args.attractor_events_csv))

    field_map = {str(r.get("identity_id", "")).strip(): r for r in field_rows}
    attractor_map = {str(r.get("identity_id", "")).strip(): r for r in attractor_rows}

    curve_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in curve_rows:
        iid = str(r.get("identity_id", "")).strip()
        if iid:
            curve_by_id[iid].append(r)

    out_rows = []
    frame_rows = []

    class_counts = Counter()
    global_variability = []
    global_turbulence = []
    global_entropy = []
    global_irregularity = []

    for life in life_rows:
        iid = str(life.get("identity_id", "")).strip()
        note = str(life.get("note_token", "")).strip()

        field = field_map.get(iid, {})
        attractor = attractor_map.get(iid, {})
        curves = curve_by_id.get(iid, [])

        phase_counter = Counter()
        phase_energy = defaultdict(list)

        for c in curves:
            phase = str(c.get("phase", "")).strip()
            energy = _safe_float(c.get("phase_energy"), 0.0)
            if phase:
                phase_counter[phase] += 1
                phase_energy[phase].append(energy)

        all_energy = [_safe_float(c.get("phase_energy"), 0.0) for c in curves]

        energy_variance = _variance(all_energy)
        energy_mean = _mean(all_energy)
        norm_energy_variability = energy_variance / max(energy_mean * energy_mean, 1e-9)

        phase_entropy = _entropy(phase_counter)

        delayed_return = _safe_float(life.get("delayed_return_energy"), 0.0)
        rebirth = _safe_float(life.get("resonance_rebirth"), 0.0)
        breathing = _safe_float(life.get("body_breathing"), 0.0)
        persistence = _safe_float(life.get("secondary_persistence_ratio"), 0.0)

        delayed_count = _safe_float(field.get("delayed_secondary_count"), 0.0)
        linked_count = _safe_float(field.get("linked_secondary_count"), 0.0)
        field_conf = _safe_float(field.get("field_confidence"), 0.0)
        attractor_conf = _safe_float(attractor.get("attractor_confidence"), 0.0)

        # Неровность повторного возвращения поля.
        rebirth_irregularity = (
            abs(delayed_return - rebirth) / max(abs(delayed_return) + abs(rebirth), 1e-9)
        )

        # Турбулентность: много вторичных связей + задержки + неравномерная энергия.
        ecology_turbulence = 0.0
        ecology_turbulence += min(delayed_count / 8.0, 1.0) * 0.22
        ecology_turbulence += min(linked_count / 12.0, 1.0) * 0.20
        ecology_turbulence += min(norm_energy_variability / 4.0, 1.0) * 0.22
        ecology_turbulence += phase_entropy * 0.16
        ecology_turbulence += min(persistence / 3.0, 1.0) * 0.20
        ecology_turbulence = max(0.0, min(ecology_turbulence, 1.0))

        # Mechanical sustain: высокая длительность/постоянство при малой вариативности.
        mechanical_sustain_score = 0.0
        mechanical_sustain_score += min(persistence / 3.0, 1.0) * 0.34
        mechanical_sustain_score += max(0.0, 1.0 - min(norm_energy_variability / 2.0, 1.0)) * 0.30
        mechanical_sustain_score += max(0.0, 1.0 - min(rebirth_irregularity / 0.8, 1.0)) * 0.18
        mechanical_sustain_score += field_conf * 0.10
        mechanical_sustain_score += attractor_conf * 0.08
        mechanical_sustain_score = max(0.0, min(mechanical_sustain_score, 1.0))

        living_breath_score = 0.0
        living_breath_score += ecology_turbulence * 0.34
        living_breath_score += min(rebirth_irregularity / 0.8, 1.0) * 0.22
        living_breath_score += phase_entropy * 0.16
        living_breath_score += min(breathing / 0.45, 1.0) * 0.16
        living_breath_score += min(delayed_count / 8.0, 1.0) * 0.12
        living_breath_score = max(0.0, min(living_breath_score, 1.0))

        if living_breath_score >= 0.62 and living_breath_score > mechanical_sustain_score:
            breath_class = "LIVING_TURBULENT_BREATH"
        elif mechanical_sustain_score >= 0.62:
            breath_class = "MECHANICAL_SUSTAIN_BREATH"
        elif living_breath_score >= 0.42:
            breath_class = "PARTIAL_LIVING_BREATH"
        else:
            breath_class = "LOW_BREATH_VARIABILITY"

        class_counts[breath_class] += 1

        global_variability.append(norm_energy_variability)
        global_turbulence.append(ecology_turbulence)
        global_entropy.append(phase_entropy)
        global_irregularity.append(rebirth_irregularity)

        out_rows.append({
            "identity_id": iid,
            "note_token": note,
            "breath_class": breath_class,
            "living_breath_score": f"{living_breath_score:.9f}",
            "mechanical_sustain_score": f"{mechanical_sustain_score:.9f}",
            "ecology_turbulence": f"{ecology_turbulence:.9f}",
            "phase_entropy": f"{phase_entropy:.9f}",
            "phase_energy_variability": f"{norm_energy_variability:.9f}",
            "rebirth_irregularity": f"{rebirth_irregularity:.9f}",
            "body_breathing": f"{breathing:.9f}",
            "secondary_persistence_ratio": f"{persistence:.9f}",
            "delayed_secondary_count": f"{delayed_count:.9f}",
            "linked_secondary_count": f"{linked_count:.9f}",
            "field_behavior": field.get("field_behavior", life.get("field_behavior", "")),
            "physiology": life.get("physiology", ""),
            "birth_frame": life.get("birth_frame", ""),
            "end_frame": life.get("end_frame", ""),
            "duration_frames": life.get("duration_frames", ""),
        })

        for c in curves:
            frame_rows.append({
                "frame_index": c.get("frame_index", ""),
                "time_sec": c.get("time_sec", ""),
                "identity_id": iid,
                "note_token": note,
                "phase": c.get("phase", ""),
                "phase_energy": c.get("phase_energy", ""),
                "breath_class": breath_class,
                "living_breath_score": f"{living_breath_score:.9f}",
                "mechanical_sustain_score": f"{mechanical_sustain_score:.9f}",
                "ecology_turbulence": f"{ecology_turbulence:.9f}",
            })

    _write_csv(
        Path(args.out_breath_variability_csv),
        out_rows,
        [
            "identity_id",
            "note_token",
            "breath_class",
            "living_breath_score",
            "mechanical_sustain_score",
            "ecology_turbulence",
            "phase_entropy",
            "phase_energy_variability",
            "rebirth_irregularity",
            "body_breathing",
            "secondary_persistence_ratio",
            "delayed_secondary_count",
            "linked_secondary_count",
            "field_behavior",
            "physiology",
            "birth_frame",
            "end_frame",
            "duration_frames",
        ],
    )

    _write_csv(
        Path(args.out_breath_variability_frame_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "identity_id",
            "note_token",
            "phase",
            "phase_energy",
            "breath_class",
            "living_breath_score",
            "mechanical_sustain_score",
            "ecology_turbulence",
        ],
    )

    mean_living = _mean([_safe_float(r["living_breath_score"]) for r in out_rows])
    mean_mechanical = _mean([_safe_float(r["mechanical_sustain_score"]) for r in out_rows])

    if mean_living > mean_mechanical and mean_living >= 0.48:
        global_breath_type = "LIVING_RESONANCE_BREATH_SYSTEM"
    elif mean_mechanical >= 0.48:
        global_breath_type = "MECHANICAL_SUSTAIN_BREATH_SYSTEM"
    else:
        global_breath_type = "WEAK_OR_MIXED_BREATH_SYSTEM"

    summary = {
        "global_breath_type": global_breath_type,
        "life_events": len(out_rows),
        "frame_rows": len(frame_rows),
        "mean_living_breath_score": mean_living,
        "mean_mechanical_sustain_score": mean_mechanical,
        "mean_ecology_turbulence": _mean(global_turbulence),
        "mean_phase_entropy": _mean(global_entropy),
        "mean_phase_energy_variability": _mean(global_variability),
        "mean_rebirth_irregularity": _mean(global_irregularity),
        "breath_class_counts": dict(class_counts),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()