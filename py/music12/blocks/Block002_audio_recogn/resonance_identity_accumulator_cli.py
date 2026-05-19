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
        for x in str(raw or "")
        .replace("|", " ")
        .replace(",", " ")
        .split()
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


def _note_candidates(tokens: Set[str]) -> Counter:
    c = Counter()

    for t in tokens:
        n = _normalize_note(t)
        if n:
            c[n] += 1

    return c


def _similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0

    return len(a & b) / max(len(a | b), 1)


def _identity_vector(row: Dict[str, Any]) -> Dict[str, Any]:

    core = _tokens(row.get("core_chain_tokens", ""))
    box = _tokens(row.get("box_resonance_tokens", ""))
    echo = _tokens(row.get("secondary_echo_tokens", ""))
    carrier = _tokens(row.get("carrier_tokens", ""))
    masking = _tokens(row.get("masking_tokens", ""))

    all_tokens = core | box | echo | carrier | masking

    note_counter = _note_candidates(all_tokens)

    degree_counter = Counter()

    for n, v in note_counter.items():
        degree_counter[_degree(n)] += v

    return {
        "core": core,
        "box": box,
        "echo": echo,
        "carrier": carrier,
        "masking": masking,
        "all": all_tokens,
        "notes": note_counter,
        "degrees": degree_counter,
    }


def _temporal_identity_score(
    current: Dict[str, Any],
    previous: Dict[str, Any],
) -> float:

    score = 0.0

    score += _similarity(
        current["core"],
        previous["core"],
    ) * 0.30

    score += _similarity(
        current["box"],
        previous["box"],
    ) * 0.15

    score += _similarity(
        current["carrier"],
        previous["carrier"],
    ) * 0.15

    score += _similarity(
        current["all"],
        previous["all"],
    ) * 0.20

    degree_overlap = (
        len(
            set(current["degrees"])
            &
            set(previous["degrees"])
        )
        /
        max(
            len(
                set(current["degrees"])
                |
                set(previous["degrees"])
            ),
            1,
        )
    )

    score += degree_overlap * 0.20

    return max(
        min(score, 1.0),
        0.0,
    )


def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Accumulate persistent resonance identity "
            "across assembled polyphonic structures."
        )
    )

    ap.add_argument("--final_note_events_csv", required=True)
    ap.add_argument("--assembled_notes_csv", required=True)

    ap.add_argument("--out_identity_events_csv", required=True)
    ap.add_argument("--out_identity_frame_csv", required=True)
    ap.add_argument("--out_identity_links_csv", required=True)
    ap.add_argument("--out_identity_summary_txt", required=True)

    ap.add_argument(
        "--identity_link_threshold",
        type=float,
        default=0.32,
    )

    ap.add_argument(
        "--fps",
        type=float,
        default=60.0,
    )

    args = ap.parse_args()

    final_rows = _load_csv(
        Path(args.final_note_events_csv)
    )

    assembled_rows = _load_csv(
        Path(args.assembled_notes_csv)
    )

    assembled_map = {
        str(r.get("assembled_id", "")).strip(): r
        for r in assembled_rows
    }

    events = []

    for r in final_rows:

        aid = str(
            r.get("assembled_id", "")
        ).strip()

        assembled = assembled_map.get(aid)

        if not assembled:
            continue

        merged = dict(r)
        merged.update(assembled)

        events.append(merged)

    events.sort(
        key=lambda r: (
            _safe_int(
                r.get("birth_frame"), 0
            ),
            _safe_int(
                r.get("end_frame"), 0
            ),
        )
    )

    identity_groups = []
    identity_links = []

    consumed = set()

    for i, row in enumerate(events):

        aid = str(row.get("assembled_id"))

        if aid in consumed:
            continue

        base_vec = _identity_vector(row)

        group = [row]

        consumed.add(aid)

        for j in range(i + 1, len(events)):

            other = events[j]

            oid = str(
                other.get("assembled_id")
            )

            if oid in consumed:
                continue

            birth_a = _safe_int(
                row.get("birth_frame"), 0
            )

            end_a = _safe_int(
                row.get("end_frame"), 0
            )

            birth_b = _safe_int(
                other.get("birth_frame"), 0
            )

            if birth_b - end_a > 90:
                break

            other_vec = _identity_vector(other)

            score = _temporal_identity_score(
                base_vec,
                other_vec,
            )

            if score < args.identity_link_threshold:
                continue

            consumed.add(oid)

            group.append(other)

            identity_links.append({
                "source_assembled_id": aid,
                "target_assembled_id": oid,
                "identity_score":
                    f"{score:.9f}",
            })

        identity_groups.append(group)

    identity_rows = []
    frame_rows = []

    status_counts = defaultdict(int)

    for idx, group in enumerate(
        identity_groups,
        start=1,
    ):

        note_counter = Counter()
        degree_counter = Counter()

        all_core = set()
        all_box = set()
        all_echo = set()

        confidences = []

        births = []
        ends = []

        for g in group:

            conf = _safe_float(
                g.get(
                    "final_note_confidence",
                    0.0,
                )
            )

            confidences.append(conf)

            births.append(
                _safe_int(
                    g.get("birth_frame"), 0
                )
            )

            ends.append(
                _safe_int(
                    g.get("end_frame"), 0
                )
            )

            final_note = _normalize_note(
                g.get("final_note", "")
            )

            if final_note:
                note_counter[final_note] += 1
                degree_counter[
                    _degree(final_note)
                ] += 1

            all_core |= _tokens(
                g.get(
                    "core_chain_tokens",
                    ""
                )
            )

            all_box |= _tokens(
                g.get(
                    "box_resonance_tokens",
                    ""
                )
            )

            all_echo |= _tokens(
                g.get(
                    "secondary_echo_tokens",
                    ""
                )
            )

        if note_counter:
            best_note = (
                note_counter
                .most_common(1)[0][0]
            )
        else:
            best_note = ""

        identity_strength = (
            sum(confidences)
            /
            max(len(confidences), 1)
        )

        temporal_span = (
            max(ends) - min(births) + 1
        )

        persistence_bonus = min(
            temporal_span / 240.0,
            1.0,
        ) * 0.18

        recurrence_bonus = min(
            len(group) / 6.0,
            1.0,
        ) * 0.16

        core_density_bonus = min(
            len(all_core) / 20.0,
            1.0,
        ) * 0.14

        identity_strength += (
            persistence_bonus
            +
            recurrence_bonus
            +
            core_density_bonus
        )

        identity_strength = max(
            min(identity_strength, 1.0),
            0.0,
        )

        if identity_strength >= 0.62:
            status = "PERSISTENT_RESONANCE_IDENTITY"

        elif identity_strength >= 0.42:
            status = "PARTIAL_RESONANCE_IDENTITY"

        else:
            status = "FRAGMENTED_RESONANCE_IDENTITY"

        status_counts[status] += 1

        row = {
            "identity_id": idx,

            "resolved_note":
                best_note,

            "identity_status":
                status,

            "identity_strength":
                f"{identity_strength:.9f}",

            "group_size":
                len(group),

            "temporal_span_frames":
                temporal_span,

            "birth_frame":
                min(births),

            "end_frame":
                max(ends),

            "core_chain_tokens":
                " ".join(
                    sorted(all_core)
                ),

            "box_resonance_tokens":
                " ".join(
                    sorted(all_box)
                ),

            "secondary_echo_tokens":
                " ".join(
                    sorted(all_echo)
                ),

            "degree_distribution":
                " | ".join(
                    f"{k}:{v}"
                    for k, v
                    in degree_counter
                    .most_common()
                ),

            "assembled_members":
                " ".join(
                    sorted(
                        str(
                            x.get(
                                "assembled_id",
                                ""
                            )
                        )
                        for x in group
                    )
                ),
        }

        identity_rows.append(row)

        for frame in range(
            min(births),
            max(ends) + 1,
        ):

            frame_rows.append({
                "frame_index": frame,
                "time_sec":
                    f"{frame / max(args.fps, 1e-9):.9f}",

                "identity_id": idx,

                "resolved_note":
                    best_note,

                "identity_status":
                    status,

                "identity_strength":
                    f"{identity_strength:.9f}",
            })

    _write_csv(
        Path(args.out_identity_events_csv),
        identity_rows,
        [
            "identity_id",

            "resolved_note",

            "identity_status",
            "identity_strength",

            "group_size",
            "temporal_span_frames",

            "birth_frame",
            "end_frame",

            "core_chain_tokens",
            "box_resonance_tokens",
            "secondary_echo_tokens",

            "degree_distribution",

            "assembled_members",
        ]
    )

    _write_csv(
        Path(args.out_identity_frame_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",

            "identity_id",

            "resolved_note",

            "identity_status",
            "identity_strength",
        ]
    )

    _write_csv(
        Path(args.out_identity_links_csv),
        identity_links,
        [
            "source_assembled_id",
            "target_assembled_id",
            "identity_score",
        ]
    )

    summary = {
        "input_events":
            len(events),

        "identity_groups":
            len(identity_groups),

        "identity_links":
            len(identity_links),

        "frame_rows":
            len(frame_rows),

        "status_counts":
            dict(status_counts),
    }

    Path(args.out_identity_summary_txt).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.out_identity_summary_txt).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()