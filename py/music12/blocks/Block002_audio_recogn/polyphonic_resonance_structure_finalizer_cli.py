# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Set


# =========================================================
# helpers
# =========================================================

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
    s = str(raw or "")
    s = s.replace("|", " ").replace(",", " ")
    return {x.strip() for x in s.split() if x.strip()}


def _entity_id(row: Dict[str, Any]) -> str:
    return str(
        row.get(
            "ecology_entity_id",
            row.get(
                "trajectory_entity_id",
                row.get(
                    "stable_entity_id",
                    row.get("entity_id", "")
                )
            )
        )
    ).strip()


def _normalize_note(note: str) -> str:
    note = str(note or "").strip()

    if not note:
        return ""

    if "'" in note:
        return note.split("'", 1)[0] + "'-"

    if note.endswith("-"):
        return note[:-1] + "'-"

    return note + "'-"


# =========================================================
# loading
# =========================================================

def _load_ecology_entities(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)

    out = {}

    for r in rows:
        eid = _entity_id(r)

        if not eid:
            continue

        out[eid] = {
            "entity_id": eid,
            "birth_frame": _safe_int(r.get("birth_frame"), 0),
            "end_frame": _safe_int(r.get("end_frame"), 0),
            "duration_frames": _safe_int(r.get("duration_frames"), 0),

            "token_union": _tokens(r.get("token_union", "")),
            "topology_signatures": _tokens(r.get("topology_signatures", "")),
            "observed_roots": _tokens(r.get("observed_roots", "")),

            "root_hint": str(
                r.get("root_hint_not_identity", "")
            ).strip(),

            "mean_score": _safe_float(
                r.get("mean_family_score"), 0.0
            ),

            "coherence": _safe_float(
                r.get("mean_topology_coherence"), 0.0
            ),
        }

    return out


def _load_ownership(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)

    out = {}

    for r in rows:
        eid = str(r.get("entity_id", "")).strip()

        if not eid:
            continue

        out[eid] = {
            "ownership_role": str(
                r.get("ownership_role", "")
            ).strip(),

            "ownership_strength": _safe_float(
                r.get("ownership_strength"), 0.0
            ),

            "feeding_strength": _safe_float(
                r.get("feeding_strength"), 0.0
            ),

            "carrying_strength": _safe_float(
                r.get("carrying_strength"), 0.0
            ),

            "masking_strength": _safe_float(
                r.get("masking_strength"), 0.0
            ),
        }

    return out


def _load_flow_edges(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows = _load_csv(path)

    out = defaultdict(list)

    for r in rows:
        src = str(r.get("source_entity", "")).strip()

        if not src:
            continue

        out[src].append(r)

    return out


def _load_field_windows(path: Path) -> List[Dict[str, Any]]:
    return _load_csv(path)


def _load_micro_summary(path: Path) -> List[Dict[str, Any]]:
    return _load_csv(path)


# =========================================================
# structure extraction
# =========================================================

def _classify_tokens(
    entity: Dict[str, Any],
    ownership: Dict[str, Any],
) -> Dict[str, Set[str]]:

    tokens = set(entity["token_union"])

    core_chain = set()
    box_tokens = set()
    echo_tokens = set()

    coherence = entity["coherence"]

    for t in tokens:

        # simplistic but interpretable split
        # will evolve later

        if coherence >= 0.65:
            core_chain.add(t)
            continue

        if ownership["carrying_strength"] >= ownership["masking_strength"]:
            box_tokens.add(t)
        else:
            echo_tokens.add(t)

    if not core_chain:
        core_chain = set(sorted(tokens)[:min(6, len(tokens))])

    return {
        "core_chain": core_chain,
        "box_tokens": box_tokens,
        "echo_tokens": echo_tokens,
    }


def _candidate_note(entity: Dict[str, Any]) -> str:

    roots = list(entity["observed_roots"])

    if roots:
        counts = Counter(roots)
        return _normalize_note(
            counts.most_common(1)[0][0]
        )

    return _normalize_note(entity["root_hint"])


def _field_state_for_entity(
    entity: Dict[str, Any],
    field_windows: List[Dict[str, Any]],
) -> str:

    birth = entity["birth_frame"]

    for fw in field_windows:

        start = _safe_int(
            fw.get("window_start_frame"), 0
        )

        end = _safe_int(
            fw.get("window_end_frame"), 0
        )

        if start <= birth < end:
            return str(
                fw.get("field_state", "")
            ).strip()

    return "UNKNOWN_FIELD"


# =========================================================
# main
# =========================================================

def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Build final polyphonic resonance structures "
            "before final note/instrument interpretation."
        )
    )

    ap.add_argument("--ecology_entities_csv", required=True)
    ap.add_argument("--ownership_roles_csv", required=True)
    ap.add_argument("--causality_flow_edges_csv", required=True)
    ap.add_argument("--field_windows_csv", required=True)
    ap.add_argument("--micro_family_frame_summary_csv", required=True)

    ap.add_argument("--out_resonance_structure_csv", required=True)
    ap.add_argument("--out_frame_structure_csv", required=True)
    ap.add_argument("--out_unresolved_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    args = ap.parse_args()

    ecology = _load_ecology_entities(
        Path(args.ecology_entities_csv)
    )

    ownership = _load_ownership(
        Path(args.ownership_roles_csv)
    )

    flow_edges = _load_flow_edges(
        Path(args.causality_flow_edges_csv)
    )

    field_windows = _load_field_windows(
        Path(args.field_windows_csv)
    )

    micro_summary = _load_micro_summary(
        Path(args.micro_family_frame_summary_csv)
    )

    structure_rows = []
    frame_rows = []
    unresolved_rows = []

    structure_type_counts = defaultdict(int)

    for eid, ent in ecology.items():

        own = ownership.get(eid)

        if not own:

            unresolved_rows.append({
                "entity_id": eid,
                "reason": "NO_OWNERSHIP",
            })

            continue

        token_split = _classify_tokens(
            ent,
            own,
        )

        candidate_note = _candidate_note(ent)

        field_state = _field_state_for_entity(
            ent,
            field_windows,
        )

        causal_edges = flow_edges.get(eid, [])

        causal_kinds = Counter(
            str(x.get("flow_kind", "")).strip()
            for x in causal_edges
        )

        structure_confidence = 0.0

        structure_confidence += ent["coherence"] * 0.30
        structure_confidence += ent["mean_score"] * 0.25
        structure_confidence += own["ownership_strength"] * 0.20
        structure_confidence += own["carrying_strength"] * 0.15
        structure_confidence -= own["masking_strength"] * 0.10

        structure_confidence = max(
            min(structure_confidence, 1.0),
            0.0,
        )

        if own["ownership_strength"] >= 0.45:
            structure_type = "PRIMARY_RESONANCE_STRUCTURE"

        elif own["carrying_strength"] >= 0.40:
            structure_type = "CARRIER_STRUCTURE"

        elif own["masking_strength"] >= 0.35:
            structure_type = "MASKING_STRUCTURE"

        else:
            structure_type = "SECONDARY_RESONANCE_STRUCTURE"

        structure_type_counts[structure_type] += 1

        row = {
            "entity_id": eid,

            "birth_frame": ent["birth_frame"],
            "end_frame": ent["end_frame"],
            "duration_frames": ent["duration_frames"],

            "structure_type": structure_type,
            "field_state": field_state,

            "candidate_note_not_final": candidate_note,

            "core_chain_tokens":
                " ".join(
                    sorted(token_split["core_chain"])
                ),

            "box_resonance_tokens":
                " ".join(
                    sorted(token_split["box_tokens"])
                ),

            "secondary_echo_tokens":
                " ".join(
                    sorted(token_split["echo_tokens"])
                ),

            "carrier_strength":
                f"{own['carrying_strength']:.9f}",

            "feeding_strength":
                f"{own['feeding_strength']:.9f}",

            "masking_strength":
                f"{own['masking_strength']:.9f}",

            "ownership_strength":
                f"{own['ownership_strength']:.9f}",

            "topology_coherence":
                f"{ent['coherence']:.9f}",

            "mean_family_score":
                f"{ent['mean_score']:.9f}",

            "causal_seeding_count":
                causal_kinds.get(
                    "CAUSAL_SEEDING", 0
                ),

            "delayed_feeding_count":
                causal_kinds.get(
                    "DELAYED_FEEDING", 0
                ),

            "masking_flow_count":
                causal_kinds.get(
                    "MASKING_OR_ABSORPTION", 0
                ),

            "structure_confidence":
                f"{structure_confidence:.9f}",
        }

        structure_rows.append(row)

        for frame in range(
            ent["birth_frame"],
            ent["end_frame"] + 1
        ):

            frame_rows.append({
                "frame_index": frame,
                "entity_id": eid,
                "structure_type": structure_type,
                "candidate_note_not_final":
                    candidate_note,
                "structure_confidence":
                    f"{structure_confidence:.9f}",
            })

    structure_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            -_safe_float(
                r.get("structure_confidence"), 0.0
            ),
        )
    )

    _write_csv(
        Path(args.out_resonance_structure_csv),
        structure_rows,
        [
            "entity_id",

            "birth_frame",
            "end_frame",
            "duration_frames",

            "structure_type",
            "field_state",

            "candidate_note_not_final",

            "core_chain_tokens",
            "box_resonance_tokens",
            "secondary_echo_tokens",

            "carrier_strength",
            "feeding_strength",
            "masking_strength",
            "ownership_strength",

            "topology_coherence",
            "mean_family_score",

            "causal_seeding_count",
            "delayed_feeding_count",
            "masking_flow_count",

            "structure_confidence",
        ]
    )

    _write_csv(
        Path(args.out_frame_structure_csv),
        frame_rows,
        [
            "frame_index",
            "entity_id",
            "structure_type",
            "candidate_note_not_final",
            "structure_confidence",
        ]
    )

    _write_csv(
        Path(args.out_unresolved_csv),
        unresolved_rows,
        [
            "entity_id",
            "reason",
        ]
    )

    summary = {
        "ecology_entities": len(ecology),
        "ownership_entities": len(ownership),
        "resonance_structures": len(structure_rows),
        "frame_rows": len(frame_rows),
        "unresolved": len(unresolved_rows),
        "structure_type_counts":
            dict(structure_type_counts),
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