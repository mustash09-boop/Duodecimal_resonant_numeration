# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def _tokens(raw: Any) -> set[str]:
    return {x.strip() for x in str(raw or "").replace("|", " ").replace(",", " ").split() if x.strip()}


def _load_structures(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)
    return {str(r.get("entity_id", "")).strip(): r for r in rows if str(r.get("entity_id", "")).strip()}


def _load_links(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows = _load_csv(path)
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        aid = str(r.get("assembled_id", "")).strip()
        if aid:
            out[aid].append(r)
    return out


def _classify_behavior(
    *,
    primary_duration: int,
    secondary_count: int,
    total_secondary_duration: int,
    delayed_count: int,
    carrier_energy: float,
    masking_energy: float,
    body_token_count: int,
    secondary_token_count: int,
) -> str:
    if secondary_count <= 0:
        return "DRY_EXCITATION_DOMINANT"

    persistence_ratio = total_secondary_duration / max(primary_duration, 1)

    if masking_energy > carrier_energy * 1.25:
        return "MASKING_DOMINANT_FIELD"

    if delayed_count >= 3 and persistence_ratio >= 2.0:
        return "DELAYED_RETURNING_BODY_FIELD"

    if persistence_ratio >= 2.5:
        return "LONG_BODY_PERSISTENCE"

    if persistence_ratio <= 0.75 and body_token_count <= 4:
        return "FAST_DECAY_BODY"

    if carrier_energy >= masking_energy and secondary_token_count >= body_token_count:
        return "CARRIER_ECHO_FIELD"

    return "MIXED_BODY_FIELD"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Map temporal persistence behavior of instrument body and secondary resonance field."
    )

    ap.add_argument("--body_anatomy_csv", required=True)
    ap.add_argument("--assembled_notes_csv", required=True)
    ap.add_argument("--structure_links_csv", required=True)
    ap.add_argument("--resonance_structure_csv", required=True)

    ap.add_argument("--out_field_persistence_csv", required=True)
    ap.add_argument("--out_field_frame_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    body_rows = _load_csv(Path(args.body_anatomy_csv))
    assembled_rows = _load_csv(Path(args.assembled_notes_csv))
    structures = _load_structures(Path(args.resonance_structure_csv))
    links_by_assembled = _load_links(Path(args.structure_links_csv))

    assembled_by_id = {
        str(r.get("assembled_id", "")).strip(): r
        for r in assembled_rows
        if str(r.get("assembled_id", "")).strip()
    }

    out_rows = []
    frame_rows = []
    behavior_counts = defaultdict(int)

    for body in body_rows:
        identity_id = str(body.get("identity_id", "")).strip()
        members = _tokens(body.get("assembled_members", ""))

        if not members:
            continue

        primary_duration = _safe_int(body.get("duration_frames"), 0)
        body_tokens = _tokens(body.get("instrument_body_tokens", ""))
        secondary_tokens = _tokens(body.get("secondary_field_tokens", ""))

        linked_secondary_ids = set()
        total_secondary_duration = 0
        delayed_count = 0
        carrier_energy = 0.0
        feeding_energy = 0.0
        masking_energy = 0.0
        link_score_sum = 0.0

        for aid in members:
            assembled = assembled_by_id.get(aid, {})
            primary_birth = _safe_int(assembled.get("birth_frame"), _safe_int(body.get("birth_frame"), 0))

            for link in links_by_assembled.get(aid, []):
                sid = str(link.get("secondary_entity_id", "")).strip()
                if not sid or sid in linked_secondary_ids:
                    continue

                linked_secondary_ids.add(sid)

                sec = structures.get(sid, {})
                sb = _safe_int(sec.get("birth_frame"), primary_birth)
                se = _safe_int(sec.get("end_frame"), sb)
                sd = max(se - sb + 1, 0)

                total_secondary_duration += sd

                if sb > primary_birth + 6:
                    delayed_count += 1

                carrier_energy += _safe_float(sec.get("carrier_strength"), 0.0)
                feeding_energy += _safe_float(sec.get("feeding_strength"), 0.0)
                masking_energy += _safe_float(sec.get("masking_strength"), 0.0)
                link_score_sum += _safe_float(link.get("link_score"), 0.0)

        secondary_count = len(linked_secondary_ids)
        persistence_ratio = total_secondary_duration / max(primary_duration, 1)
        mean_link_score = link_score_sum / max(secondary_count, 1)

        behavior = _classify_behavior(
            primary_duration=primary_duration,
            secondary_count=secondary_count,
            total_secondary_duration=total_secondary_duration,
            delayed_count=delayed_count,
            carrier_energy=carrier_energy,
            masking_energy=masking_energy,
            body_token_count=len(body_tokens),
            secondary_token_count=len(secondary_tokens),
        )

        behavior_counts[behavior] += 1

        field_confidence = 0.0
        field_confidence += _safe_float(body.get("anatomy_confidence"), 0.0) * 0.28
        field_confidence += min(persistence_ratio / 2.5, 1.0) * 0.24
        field_confidence += min(secondary_count / 10.0, 1.0) * 0.18
        field_confidence += mean_link_score * 0.18
        field_confidence += min((carrier_energy + feeding_energy) / max(secondary_count, 1), 1.0) * 0.12
        field_confidence = max(0.0, min(field_confidence, 1.0))

        birth = _safe_int(body.get("birth_frame"), 0)
        end = _safe_int(body.get("end_frame"), birth)

        out_rows.append({
            "identity_id": identity_id,
            "resolved_note": body.get("resolved_note", ""),
            "identity_status": body.get("identity_status", ""),
            "anatomy_status": body.get("anatomy_status", ""),
            "field_behavior": behavior,
            "field_confidence": f"{field_confidence:.9f}",
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": end - birth + 1,
            "primary_duration_frames": primary_duration,
            "linked_secondary_count": secondary_count,
            "total_secondary_duration_frames": total_secondary_duration,
            "secondary_persistence_ratio": f"{persistence_ratio:.9f}",
            "delayed_secondary_count": delayed_count,
            "mean_link_score": f"{mean_link_score:.9f}",
            "carrier_energy": f"{carrier_energy:.9f}",
            "feeding_energy": f"{feeding_energy:.9f}",
            "masking_energy": f"{masking_energy:.9f}",
            "body_token_count": len(body_tokens),
            "secondary_token_count": len(secondary_tokens),
            "instrument_body_tokens": body.get("instrument_body_tokens", ""),
            "secondary_field_tokens": body.get("secondary_field_tokens", ""),
            "assembled_members": body.get("assembled_members", ""),
        })

        for frame in range(birth, end + 1):
            frame_rows.append({
                "frame_index": frame,
                "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                "identity_id": identity_id,
                "resolved_note": body.get("resolved_note", ""),
                "field_behavior": behavior,
                "field_confidence": f"{field_confidence:.9f}",
                "secondary_persistence_ratio": f"{persistence_ratio:.9f}",
            })

    _write_csv(
        Path(args.out_field_persistence_csv),
        out_rows,
        [
            "identity_id",
            "resolved_note",
            "identity_status",
            "anatomy_status",
            "field_behavior",
            "field_confidence",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "primary_duration_frames",
            "linked_secondary_count",
            "total_secondary_duration_frames",
            "secondary_persistence_ratio",
            "delayed_secondary_count",
            "mean_link_score",
            "carrier_energy",
            "feeding_energy",
            "masking_energy",
            "body_token_count",
            "secondary_token_count",
            "instrument_body_tokens",
            "secondary_field_tokens",
            "assembled_members",
        ],
    )

    _write_csv(
        Path(args.out_field_frame_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "identity_id",
            "resolved_note",
            "field_behavior",
            "field_confidence",
            "secondary_persistence_ratio",
        ],
    )

    summary = {
        "input_body_identities": len(body_rows),
        "field_persistence_rows": len(out_rows),
        "frame_rows": len(frame_rows),
        "behavior_counts": dict(behavior_counts),
    }

    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()