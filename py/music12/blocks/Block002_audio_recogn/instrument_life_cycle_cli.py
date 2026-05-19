# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
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


def _phase(duration_ratio: float) -> str:
    if duration_ratio <= 0.18:
        return "BIRTH"

    if duration_ratio <= 0.52:
        return "SUSTAIN"

    if duration_ratio <= 0.82:
        return "DELAYED_RETURN"

    return "DECAY"


def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Build temporal resonance physiology "
            "from attractor ecology."
        )
    )

    ap.add_argument(
        "--attractor_events_csv",
        required=True,
    )

    ap.add_argument(
        "--field_persistence_csv",
        required=True,
    )

    ap.add_argument(
        "--out_life_cycle_csv",
        required=True,
    )

    ap.add_argument(
        "--out_life_curve_csv",
        required=True,
    )

    ap.add_argument(
        "--out_summary_txt",
        required=True,
    )

    ap.add_argument(
        "--fps",
        type=float,
        default=60.0,
    )

    args = ap.parse_args()

    attractors = _load_csv(
        Path(args.attractor_events_csv)
    )

    fields = _load_csv(
        Path(args.field_persistence_csv)
    )

    field_map = {
        str(
            r.get("identity_id", "")
        ).strip(): r
        for r in fields
    }

    lifecycle_rows = []
    curve_rows = []

    phase_counter = Counter()

    total_birth_energy = []
    total_sustain_energy = []
    total_delayed_energy = []
    total_decay_energy = []

    total_lifetimes = []
    total_rebirth = []
    total_breathing = []

    for row in attractors:

        iid = str(
            row.get(
                "identity_id",
                ""
            )
        ).strip()

        field = field_map.get(iid, {})

        birth = _safe_int(
            row.get(
                "birth_frame",
                0
            )
        )

        end = _safe_int(
            row.get(
                "end_frame",
                birth
            )
        )

        duration = max(
            end - birth + 1,
            1
        )

        note = str(
            row.get(
                "attractor_note",
                ""
            )
        ).strip()

        field_behavior = str(
            field.get(
                "field_behavior",
                row.get(
                    "field_behavior",
                    ""
                )
            )
        ).strip()

        conf = _safe_float(
            row.get(
                "attractor_confidence",
                0.0
            )
        )

        persistence = _safe_float(
            field.get(
                "secondary_persistence_ratio",
                0.0
            )
        )

        delayed_count = _safe_float(
            field.get(
                "delayed_secondary_count",
                0.0
            )
        )

        linked_count = _safe_float(
            field.get(
                "linked_secondary_count",
                0.0
            )
        )

        body_strength = _safe_float(
            row.get(
                "body_strength",
                0.0
            )
        )

        secondary_strength = _safe_float(
            row.get(
                "secondary_strength",
                0.0
            )
        )

        birth_energy = conf * 1.0
        sustain_energy = (
            conf * 0.65 +
            body_strength * 0.35
        )

        delayed_energy = (
            persistence * 0.45 +
            delayed_count * 0.06 +
            linked_count * 0.04 +
            secondary_strength * 0.45
        )

        decay_energy = max(
            sustain_energy - delayed_energy * 0.55,
            0.0
        )

        total_birth_energy.append(
            birth_energy
        )

        total_sustain_energy.append(
            sustain_energy
        )

        total_delayed_energy.append(
            delayed_energy
        )

        total_decay_energy.append(
            decay_energy
        )

        ecology_lifetime = (
            duration / max(args.fps, 1e-9)
        )

        resonance_rebirth = (
            delayed_energy *
            persistence
        )

        body_breathing = (
            (
                linked_count +
                delayed_count
            ) / max(duration, 1.0)
        )

        total_lifetimes.append(
            ecology_lifetime
        )

        total_rebirth.append(
            resonance_rebirth
        )

        total_breathing.append(
            body_breathing
        )

        if resonance_rebirth >= 0.85:
            physiology = "RESONANCE_REBIRTH"

        elif body_breathing >= 0.09:
            physiology = "BREATHING_BODY"

        elif persistence >= 2.5:
            physiology = "LONG_RESONANCE_SURVIVAL"

        else:
            physiology = "FAST_DECAY_STRUCTURE"

        lifecycle_rows.append({
            "identity_id":
                iid,

            "note_token":
                note,

            "field_behavior":
                field_behavior,

            "physiology":
                physiology,

            "birth_frame":
                birth,

            "end_frame":
                end,

            "duration_frames":
                duration,

            "ecology_lifetime_sec":
                f"{ecology_lifetime:.9f}",

            "birth_energy":
                f"{birth_energy:.9f}",

            "sustain_energy":
                f"{sustain_energy:.9f}",

            "delayed_return_energy":
                f"{delayed_energy:.9f}",

            "decay_energy":
                f"{decay_energy:.9f}",

            "resonance_rebirth":
                f"{resonance_rebirth:.9f}",

            "body_breathing":
                f"{body_breathing:.9f}",

            "secondary_persistence_ratio":
                f"{persistence:.9f}",
        })

        for frame in range(
            birth,
            end + 1,
        ):

            ratio = (
                (
                    frame - birth
                ) / max(
                    duration - 1,
                    1
                )
            )

            phase = _phase(ratio)

            phase_counter[phase] += 1

            if phase == "BIRTH":
                energy = birth_energy

            elif phase == "SUSTAIN":
                energy = sustain_energy

            elif phase == "DELAYED_RETURN":
                energy = delayed_energy

            else:
                energy = decay_energy

            curve_rows.append({
                "frame_index":
                    frame,

                "time_sec":
                    f"{frame / max(args.fps, 1e-9):.9f}",

                "identity_id":
                    iid,

                "note_token":
                    note,

                "phase":
                    phase,

                "phase_energy":
                    f"{energy:.9f}",

                "physiology":
                    physiology,
            })

    mean_birth = sum(total_birth_energy) / max(len(total_birth_energy), 1)
    mean_sustain = sum(total_sustain_energy) / max(len(total_sustain_energy), 1)
    mean_delayed = sum(total_delayed_energy) / max(len(total_delayed_energy), 1)
    mean_decay = sum(total_decay_energy) / max(len(total_decay_energy), 1)

    mean_lifetime = sum(total_lifetimes) / max(len(total_lifetimes), 1)
    mean_rebirth = sum(total_rebirth) / max(len(total_rebirth), 1)
    mean_breathing = sum(total_breathing) / max(len(total_breathing), 1)

    if mean_rebirth >= 0.85:
        global_physiology = "RESONANCE_REBIRTH_SYSTEM"

    elif mean_breathing >= 0.09:
        global_physiology = "BREATHING_RESONANCE_SYSTEM"

    elif mean_delayed >= mean_decay:
        global_physiology = "LONG_FIELD_SURVIVAL_SYSTEM"

    else:
        global_physiology = "FAST_DECAY_RESONANCE_SYSTEM"

    _write_csv(
        Path(args.out_life_cycle_csv),
        lifecycle_rows,
        [
            "identity_id",
            "note_token",
            "field_behavior",
            "physiology",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "ecology_lifetime_sec",
            "birth_energy",
            "sustain_energy",
            "delayed_return_energy",
            "decay_energy",
            "resonance_rebirth",
            "body_breathing",
            "secondary_persistence_ratio",
        ]
    )

    _write_csv(
        Path(args.out_life_curve_csv),
        curve_rows,
        [
            "frame_index",
            "time_sec",
            "identity_id",
            "note_token",
            "phase",
            "phase_energy",
            "physiology",
        ]
    )

    summary = {
        "global_physiology":
            global_physiology,

        "mean_birth_energy":
            mean_birth,

        "mean_sustain_energy":
            mean_sustain,

        "mean_delayed_return_energy":
            mean_delayed,

        "mean_decay_energy":
            mean_decay,

        "mean_ecology_lifetime_sec":
            mean_lifetime,

        "mean_resonance_rebirth":
            mean_rebirth,

        "mean_body_breathing":
            mean_breathing,

        "phase_distribution":
            dict(phase_counter),

        "life_events":
            len(lifecycle_rows),

        "curve_rows":
            len(curve_rows),
    }

    Path(args.out_summary_txt).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.out_summary_txt).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()