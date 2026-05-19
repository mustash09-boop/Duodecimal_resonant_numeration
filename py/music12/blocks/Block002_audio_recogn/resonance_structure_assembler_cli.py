# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
    return {x.strip() for x in str(raw or "").replace("|", " ").replace(",", " ").split() if x.strip()}


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    s = max(_safe_int(a.get("birth_frame")), _safe_int(b.get("birth_frame")))
    e = min(_safe_int(a.get("end_frame")), _safe_int(b.get("end_frame")))
    return max(0, e - s + 1)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _load_flow_links(path: Path) -> Dict[str, Set[str]]:
    links: Dict[str, Set[str]] = defaultdict(set)
    for r in _load_csv(path):
        src = str(r.get("source_entity", "")).strip()
        dst = str(r.get("target_entity", "")).strip()
        if src and dst:
            links[src].add(dst)
            links[dst].add(src)
    return links


def _structure_score(primary: Dict[str, Any], secondary: Dict[str, Any], flow_links: Dict[str, Set[str]]) -> float:
    pid = str(primary.get("entity_id", "")).strip()
    sid = str(secondary.get("entity_id", "")).strip()

    overlap_frames = _overlap(primary, secondary)
    if overlap_frames <= 0:
        return 0.0

    core = _tokens(primary.get("core_chain_tokens", ""))
    box = _tokens(secondary.get("box_resonance_tokens", ""))
    echo = _tokens(secondary.get("secondary_echo_tokens", ""))
    sec_core = _tokens(secondary.get("core_chain_tokens", ""))

    token_sim = max(
        _jaccard(core, box),
        _jaccard(core, echo),
        _jaccard(core, sec_core),
    )

    linked = 1.0 if sid in flow_links.get(pid, set()) else 0.0

    sec_conf = _safe_float(secondary.get("structure_confidence"), 0.0)
    carrying = _safe_float(secondary.get("carrier_strength"), 0.0)
    feeding = _safe_float(secondary.get("feeding_strength"), 0.0)
    masking = _safe_float(secondary.get("masking_strength"), 0.0)

    score = 0.0
    score += min(overlap_frames / 60.0, 1.0) * 0.28
    score += token_sim * 0.22
    score += linked * 0.18
    score += sec_conf * 0.14
    score += carrying * 0.10
    score += feeding * 0.08
    score -= masking * 0.08

    return max(0.0, min(score, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble primary and secondary resonance structures into note-forming resonance entities."
    )

    ap.add_argument("--resonance_structure_csv", required=True)
    ap.add_argument("--frame_structure_csv", required=True)
    ap.add_argument("--causality_flow_edges_csv", required=True)

    ap.add_argument("--out_assembled_notes_csv", required=True)
    ap.add_argument("--out_assembled_frame_notes_csv", required=True)
    ap.add_argument("--out_structure_links_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_link_score", type=float, default=0.18)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    structures = _load_csv(Path(args.resonance_structure_csv))
    flow_links = _load_flow_links(Path(args.causality_flow_edges_csv))

    primaries = [r for r in structures if r.get("structure_type") == "PRIMARY_RESONANCE_STRUCTURE"]
    secondaries = [r for r in structures if r.get("structure_type") != "PRIMARY_RESONANCE_STRUCTURE"]

    assembled_rows = []
    frame_rows = []
    link_rows = []

    for idx, p in enumerate(primaries, start=1):
        pid = str(p.get("entity_id", "")).strip()
        note = str(p.get("candidate_note_not_final", "")).strip()

        linked_secondaries = []

        for s in secondaries:
            sid = str(s.get("entity_id", "")).strip()
            score = _structure_score(p, s, flow_links)

            if score < args.min_link_score:
                continue

            linked_secondaries.append((s, score))

            link_rows.append({
                "assembled_id": idx,
                "primary_entity_id": pid,
                "secondary_entity_id": sid,
                "secondary_type": s.get("structure_type", ""),
                "link_score": f"{score:.9f}",
                "overlap_frames": _overlap(p, s),
                "secondary_candidate_note": s.get("candidate_note_not_final", ""),
            })

        linked_secondaries.sort(key=lambda x: -x[1])

        core_tokens = set(_tokens(p.get("core_chain_tokens", "")))
        box_tokens = set()
        echo_tokens = set()
        carrier_tokens = set()
        masking_tokens = set()

        total_link_score = 0.0

        for s, score in linked_secondaries:
            total_link_score += score
            box_tokens |= _tokens(s.get("box_resonance_tokens", ""))
            echo_tokens |= _tokens(s.get("secondary_echo_tokens", ""))

            if _safe_float(s.get("carrier_strength"), 0.0) >= 0.20:
                carrier_tokens |= _tokens(s.get("core_chain_tokens", ""))

            if _safe_float(s.get("masking_strength"), 0.0) >= 0.20:
                masking_tokens |= _tokens(s.get("core_chain_tokens", ""))

        confidence = _safe_float(p.get("structure_confidence"), 0.0) * 0.55
        confidence += min(total_link_score / 3.0, 1.0) * 0.35
        confidence += min(len(linked_secondaries) / 8.0, 1.0) * 0.10
        confidence = max(0.0, min(confidence, 1.0))

        birth = _safe_int(p.get("birth_frame"), 0)
        end = _safe_int(p.get("end_frame"), birth)

        assembled_rows.append({
            "assembled_id": idx,
            "primary_entity_id": pid,
            "candidate_note_not_final": note,
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": end - birth + 1,
            "linked_secondary_count": len(linked_secondaries),
            "core_chain_tokens": " ".join(sorted(core_tokens)),
            "box_resonance_tokens": " ".join(sorted(box_tokens)),
            "secondary_echo_tokens": " ".join(sorted(echo_tokens)),
            "carrier_tokens": " ".join(sorted(carrier_tokens)),
            "masking_tokens": " ".join(sorted(masking_tokens)),
            "assembly_confidence": f"{confidence:.9f}",
        })

        for frame in range(birth, end + 1):
            frame_rows.append({
                "frame_index": frame,
                "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                "assembled_id": idx,
                "primary_entity_id": pid,
                "candidate_note_not_final": note,
                "assembly_confidence": f"{confidence:.9f}",
                "linked_secondary_count": len(linked_secondaries),
            })

    _write_csv(
        Path(args.out_assembled_notes_csv),
        assembled_rows,
        [
            "assembled_id",
            "primary_entity_id",
            "candidate_note_not_final",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "linked_secondary_count",
            "core_chain_tokens",
            "box_resonance_tokens",
            "secondary_echo_tokens",
            "carrier_tokens",
            "masking_tokens",
            "assembly_confidence",
        ],
    )

    _write_csv(
        Path(args.out_assembled_frame_notes_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "assembled_id",
            "primary_entity_id",
            "candidate_note_not_final",
            "assembly_confidence",
            "linked_secondary_count",
        ],
    )

    _write_csv(
        Path(args.out_structure_links_csv),
        link_rows,
        [
            "assembled_id",
            "primary_entity_id",
            "secondary_entity_id",
            "secondary_type",
            "link_score",
            "overlap_frames",
            "secondary_candidate_note",
        ],
    )

    summary = {
        "input_structures": len(structures),
        "primary_structures": len(primaries),
        "secondary_structures": len(secondaries),
        "assembled_notes": len(assembled_rows),
        "structure_links": len(link_rows),
        "frame_rows": len(frame_rows),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()