# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _degree(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _time_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    s = max(a0, b0)
    e = min(a1, b1)
    return max(0, e - s + 1)


def _token_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0

    return len(a & b) / max(len(a | b), 1)


def _note_relation(a: str, b: str) -> str:
    da = _degree(a)
    db = _degree(b)

    if not da or not db:
        return "UNKNOWN"

    if da == db:
        return "OCTAVE_OR_UNISON"

    return "POLYPHONIC_NEIGHBOR"


def _field_kind(row: Dict[str, Any]) -> str:
    return str(row.get("field_behavior", "")).strip()


def _interaction_score(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, Any]:

    a_birth = _safe_int(a.get("birth_frame"), 0)
    a_end = _safe_int(a.get("end_frame"), 0)

    b_birth = _safe_int(b.get("birth_frame"), 0)
    b_end = _safe_int(b.get("end_frame"), 0)

    overlap = _time_overlap(
        a_birth,
        a_end,
        b_birth,
        b_end,
    )

    if overlap <= 0:
        return {
            "score": 0.0,
            "kind": "NO_OVERLAP",
        }

    a_core = _tokens(a.get("excitation_core_tokens", ""))
    b_core = _tokens(b.get("excitation_core_tokens", ""))

    a_body = _tokens(a.get("instrument_body_tokens", ""))
    b_body = _tokens(b.get("instrument_body_tokens", ""))

    a_echo = _tokens(a.get("secondary_field_tokens", ""))
    b_echo = _tokens(b.get("secondary_field_tokens", ""))

    core_similarity = _token_similarity(a_core, b_core)
    body_similarity = _token_similarity(a_body, b_body)
    echo_similarity = _token_similarity(a_echo, b_echo)

    note_a = _normalize_note(a.get("attractor_note", ""))
    note_b = _normalize_note(b.get("attractor_note", ""))

    relation = _note_relation(note_a, note_b)

    a_conf = _safe_float(a.get("attractor_confidence"), 0.0)
    b_conf = _safe_float(b.get("attractor_confidence"), 0.0)

    a_field = _field_kind(a)
    b_field = _field_kind(b)

    score = 0.0

    score += min(overlap / 90.0, 1.0) * 0.24
    score += core_similarity * 0.18
    score += body_similarity * 0.24
    score += echo_similarity * 0.12

    score += ((a_conf + b_conf) / 2.0) * 0.16

    if relation == "OCTAVE_OR_UNISON":
        score += 0.12

    if (
        a_field == "DELAYED_RETURNING_BODY_FIELD"
        and
        b_field == "DELAYED_RETURNING_BODY_FIELD"
    ):
        score += 0.08

    score = max(0.0, min(score, 1.0))

    if score >= 0.62:
        kind = "STRONG_RESONANCE_COUPLING"

    elif score >= 0.42:
        kind = "MODERATE_RESONANCE_COUPLING"

    elif score >= 0.22:
        kind = "WEAK_RESONANCE_COUPLING"

    else:
        kind = "MINIMAL_INTERACTION"

    return {
        "score": score,
        "kind": kind,
        "relation": relation,
        "overlap_frames": overlap,
        "core_similarity": core_similarity,
        "body_similarity": body_similarity,
        "echo_similarity": echo_similarity,
    }


def main() -> None:

    ap = argparse.ArgumentParser(
        description="Build polyphonic resonance ecology between attractors."
    )

    ap.add_argument("--attractor_events_csv", required=True)

    ap.add_argument("--out_ecology_links_csv", required=True)
    ap.add_argument("--out_ecology_nodes_csv", required=True)
    ap.add_argument("--out_ecology_frame_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument(
        "--min_interaction_score",
        type=float,
        default=0.22,
    )

    ap.add_argument(
        "--fps",
        type=float,
        default=60.0,
    )

    args = ap.parse_args()

    rows = _load_csv(
        Path(args.attractor_events_csv)
    )

    rows.sort(
        key=lambda r: (
            _safe_int(
                r.get("birth_frame"), 0
            ),
            _safe_int(
                r.get("end_frame"), 0
            ),
        )
    )

    links = []
    frame_rows = []

    coupling_counts = defaultdict(int)

    ecology_energy = defaultdict(float)

    for i in range(len(rows)):

        a = rows[i]

        a_id = str(
            a.get("identity_id", "")
        ).strip()

        a_birth = _safe_int(
            a.get("birth_frame"), 0
        )

        a_end = _safe_int(
            a.get("end_frame"), 0
        )

        for j in range(i + 1, len(rows)):

            b = rows[j]

            b_id = str(
                b.get("identity_id", "")
            ).strip()

            b_birth = _safe_int(
                b.get("birth_frame"), 0
            )

            if b_birth - a_end > 120:
                break

            inter = _interaction_score(
                a,
                b,
            )

            score = inter["score"]

            if score < args.min_interaction_score:
                continue

            kind = inter["kind"]

            coupling_counts[kind] += 1

            ecology_energy[a_id] += score
            ecology_energy[b_id] += score

            links.append({
                "source_identity_id": a_id,
                "target_identity_id": b_id,

                "source_note":
                    a.get(
                        "attractor_note",
                        ""
                    ),

                "target_note":
                    b.get(
                        "attractor_note",
                        ""
                    ),

                "interaction_kind":
                    kind,

                "note_relation":
                    inter["relation"],

                "interaction_score":
                    f"{score:.9f}",

                "overlap_frames":
                    inter["overlap_frames"],

                "core_similarity":
                    f"{inter['core_similarity']:.9f}",

                "body_similarity":
                    f"{inter['body_similarity']:.9f}",

                "echo_similarity":
                    f"{inter['echo_similarity']:.9f}",
            })

    node_rows = []

    for r in rows:

        iid = str(
            r.get("identity_id", "")
        ).strip()

        energy = ecology_energy[iid]

        if energy >= 8.0:
            ecology_status = "HIGH_ECOLOGY_DENSITY"

        elif energy >= 3.5:
            ecology_status = "MEDIUM_ECOLOGY_DENSITY"

        else:
            ecology_status = "LOW_ECOLOGY_DENSITY"

        node_rows.append({
            "identity_id": iid,

            "attractor_note":
                r.get(
                    "attractor_note",
                    ""
                ),

            "attractor_status":
                r.get(
                    "attractor_status",
                    ""
                ),

            "field_behavior":
                r.get(
                    "field_behavior",
                    ""
                ),

            "ecology_energy":
                f"{energy:.9f}",

            "ecology_status":
                ecology_status,

            "birth_frame":
                r.get(
                    "birth_frame",
                    ""
                ),

            "end_frame":
                r.get(
                    "end_frame",
                    ""
                ),

            "duration_frames":
                r.get(
                    "duration_frames",
                    ""
                ),
        })

        birth = _safe_int(
            r.get("birth_frame"), 0
        )

        end = _safe_int(
            r.get("end_frame"), birth
        )

        for frame in range(
            birth,
            end + 1,
        ):

            frame_rows.append({
                "frame_index": frame,

                "time_sec":
                    f"{frame / max(args.fps, 1e-9):.9f}",

                "identity_id": iid,

                "note_token":
                    r.get(
                        "attractor_note",
                        ""
                    ),

                "ecology_energy":
                    f"{energy:.9f}",

                "ecology_status":
                    ecology_status,
            })

    _write_csv(
        Path(args.out_ecology_links_csv),
        links,
        [
            "source_identity_id",
            "target_identity_id",

            "source_note",
            "target_note",

            "interaction_kind",
            "note_relation",

            "interaction_score",

            "overlap_frames",

            "core_similarity",
            "body_similarity",
            "echo_similarity",
        ]
    )

    _write_csv(
        Path(args.out_ecology_nodes_csv),
        node_rows,
        [
            "identity_id",

            "attractor_note",
            "attractor_status",

            "field_behavior",

            "ecology_energy",
            "ecology_status",

            "birth_frame",
            "end_frame",
            "duration_frames",
        ]
    )

    _write_csv(
        Path(args.out_ecology_frame_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",

            "identity_id",

            "note_token",

            "ecology_energy",
            "ecology_status",
        ]
    )

    summary = {
        "input_attractors":
            len(rows),

        "ecology_links":
            len(links),

        "ecology_nodes":
            len(node_rows),

        "frame_rows":
            len(frame_rows),

        "coupling_counts":
            dict(coupling_counts),
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