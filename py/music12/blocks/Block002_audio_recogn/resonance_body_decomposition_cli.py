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


def _token_note_counter(tokens: Set[str]) -> Counter:
    c = Counter()
    for t in tokens:
        n = _normalize_note(t)
        if n:
            c[n] += 1
    return c


def _degree_counter(tokens: Set[str]) -> Counter:
    c = Counter()
    for t in tokens:
        d = _degree(_normalize_note(t))
        if d:
            c[d] += 1
    return c


def _load_identity_members(identity_rows: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for r in identity_rows:
        iid = str(r.get("identity_id", "")).strip()
        members = _tokens(r.get("assembled_members", ""))
        if iid:
            out[iid] = members
    return out


def _classify_body(row: Dict[str, Any]) -> Dict[str, Any]:
    core = _tokens(row.get("core_chain_tokens", ""))
    box = _tokens(row.get("box_resonance_tokens", ""))
    echo = _tokens(row.get("secondary_echo_tokens", ""))

    resolved_note = _normalize_note(row.get("resolved_note", ""))
    resolved_degree = _degree(resolved_note)

    core_degrees = _degree_counter(core)
    box_degrees = _degree_counter(box)
    echo_degrees = _degree_counter(echo)

    excitation_core = set()
    instrument_body = set()
    secondary_field = set()

    for t in core:
        if _degree(_normalize_note(t)) == resolved_degree:
            excitation_core.add(t)
        else:
            instrument_body.add(t)

    for t in box:
        if _degree(_normalize_note(t)) == resolved_degree:
            instrument_body.add(t)
        else:
            instrument_body.add(t)

    for t in echo:
        secondary_field.add(t)

    core_note_hits = _token_note_counter(excitation_core)
    box_note_hits = _token_note_counter(instrument_body)
    echo_note_hits = _token_note_counter(secondary_field)

    core_strength = min(len(excitation_core) / 12.0, 1.0)
    body_strength = min(len(instrument_body) / 24.0, 1.0)
    secondary_strength = min(len(secondary_field) / 24.0, 1.0)

    identity_strength = _safe_float(row.get("identity_strength"), 0.0)

    anatomy_confidence = (
        identity_strength * 0.42
        + core_strength * 0.26
        + body_strength * 0.22
        + secondary_strength * 0.10
    )

    anatomy_confidence = max(0.0, min(anatomy_confidence, 1.0))

    if anatomy_confidence >= 0.62 and core_strength >= 0.25:
        anatomy_status = "RESONANCE_BODY_RESOLVED"
    elif anatomy_confidence >= 0.42:
        anatomy_status = "RESONANCE_BODY_PARTIAL"
    else:
        anatomy_status = "RESONANCE_BODY_FRAGMENTED"

    return {
        "excitation_core_tokens": " ".join(sorted(excitation_core)),
        "instrument_body_tokens": " ".join(sorted(instrument_body)),
        "secondary_field_tokens": " ".join(sorted(secondary_field)),
        "core_token_count": len(excitation_core),
        "body_token_count": len(instrument_body),
        "secondary_token_count": len(secondary_field),
        "core_strength": core_strength,
        "body_strength": body_strength,
        "secondary_strength": secondary_strength,
        "anatomy_confidence": anatomy_confidence,
        "anatomy_status": anatomy_status,
        "core_note_distribution": " | ".join(f"{k}:{v}" for k, v in core_note_hits.most_common(12)),
        "body_note_distribution": " | ".join(f"{k}:{v}" for k, v in box_note_hits.most_common(12)),
        "echo_note_distribution": " | ".join(f"{k}:{v}" for k, v in echo_note_hits.most_common(12)),
        "core_degree_distribution": " | ".join(f"{k}:{v}" for k, v in core_degrees.most_common(12)),
        "body_degree_distribution": " | ".join(f"{k}:{v}" for k, v in box_degrees.most_common(12)),
        "echo_degree_distribution": " | ".join(f"{k}:{v}" for k, v in echo_degrees.most_common(12)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose persistent resonance identities into excitation core, instrument body, and secondary resonance field."
    )

    ap.add_argument("--identity_events_csv", required=True)
    ap.add_argument("--assembled_notes_csv", required=True)

    ap.add_argument("--out_body_anatomy_csv", required=True)
    ap.add_argument("--out_body_frame_csv", required=True)
    ap.add_argument("--out_body_summary_txt", required=True)

    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    identity_rows = _load_csv(Path(args.identity_events_csv))
    assembled_rows = _load_csv(Path(args.assembled_notes_csv))

    member_map = _load_identity_members(identity_rows)
    assembled_map = {
        str(r.get("assembled_id", "")).strip(): r
        for r in assembled_rows
    }

    anatomy_rows = []
    frame_rows = []

    status_counts = defaultdict(int)

    for ident in identity_rows:
        iid = str(ident.get("identity_id", "")).strip()
        members = member_map.get(iid, set())

        merged_core = set()
        merged_box = set()
        merged_echo = set()

        for aid in members:
            a = assembled_map.get(aid)
            if not a:
                continue

            merged_core |= _tokens(a.get("core_chain_tokens", ""))
            merged_box |= _tokens(a.get("box_resonance_tokens", ""))
            merged_echo |= _tokens(a.get("secondary_echo_tokens", ""))

        row = dict(ident)
        row["core_chain_tokens"] = " ".join(sorted(merged_core))
        row["box_resonance_tokens"] = " ".join(sorted(merged_box))
        row["secondary_echo_tokens"] = " ".join(sorted(merged_echo))

        anatomy = _classify_body(row)
        status_counts[anatomy["anatomy_status"]] += 1

        birth = _safe_int(row.get("birth_frame"), 0)
        end = _safe_int(row.get("end_frame"), birth)

        anatomy_rows.append({
            "identity_id": iid,
            "resolved_note": row.get("resolved_note", ""),
            "identity_status": row.get("identity_status", ""),
            "identity_strength": row.get("identity_strength", ""),
            "birth_frame": birth,
            "end_frame": end,
            "duration_frames": end - birth + 1,
            "group_size": row.get("group_size", ""),
            "anatomy_status": anatomy["anatomy_status"],
            "anatomy_confidence": f"{anatomy['anatomy_confidence']:.9f}",
            "core_strength": f"{anatomy['core_strength']:.9f}",
            "body_strength": f"{anatomy['body_strength']:.9f}",
            "secondary_strength": f"{anatomy['secondary_strength']:.9f}",
            "core_token_count": anatomy["core_token_count"],
            "body_token_count": anatomy["body_token_count"],
            "secondary_token_count": anatomy["secondary_token_count"],
            "excitation_core_tokens": anatomy["excitation_core_tokens"],
            "instrument_body_tokens": anatomy["instrument_body_tokens"],
            "secondary_field_tokens": anatomy["secondary_field_tokens"],
            "core_note_distribution": anatomy["core_note_distribution"],
            "body_note_distribution": anatomy["body_note_distribution"],
            "echo_note_distribution": anatomy["echo_note_distribution"],
            "core_degree_distribution": anatomy["core_degree_distribution"],
            "body_degree_distribution": anatomy["body_degree_distribution"],
            "echo_degree_distribution": anatomy["echo_degree_distribution"],
            "assembled_members": row.get("assembled_members", ""),
        })

        for frame in range(birth, end + 1):
            frame_rows.append({
                "frame_index": frame,
                "time_sec": f"{frame / max(args.fps, 1e-9):.9f}",
                "identity_id": iid,
                "resolved_note": row.get("resolved_note", ""),
                "anatomy_status": anatomy["anatomy_status"],
                "anatomy_confidence": f"{anatomy['anatomy_confidence']:.9f}",
                "core_strength": f"{anatomy['core_strength']:.9f}",
                "body_strength": f"{anatomy['body_strength']:.9f}",
                "secondary_strength": f"{anatomy['secondary_strength']:.9f}",
            })

    _write_csv(
        Path(args.out_body_anatomy_csv),
        anatomy_rows,
        [
            "identity_id",
            "resolved_note",
            "identity_status",
            "identity_strength",
            "birth_frame",
            "end_frame",
            "duration_frames",
            "group_size",
            "anatomy_status",
            "anatomy_confidence",
            "core_strength",
            "body_strength",
            "secondary_strength",
            "core_token_count",
            "body_token_count",
            "secondary_token_count",
            "excitation_core_tokens",
            "instrument_body_tokens",
            "secondary_field_tokens",
            "core_note_distribution",
            "body_note_distribution",
            "echo_note_distribution",
            "core_degree_distribution",
            "body_degree_distribution",
            "echo_degree_distribution",
            "assembled_members",
        ],
    )

    _write_csv(
        Path(args.out_body_frame_csv),
        frame_rows,
        [
            "frame_index",
            "time_sec",
            "identity_id",
            "resolved_note",
            "anatomy_status",
            "anatomy_confidence",
            "core_strength",
            "body_strength",
            "secondary_strength",
        ],
    )

    summary = {
        "input_identities": len(identity_rows),
        "body_anatomy_rows": len(anatomy_rows),
        "frame_rows": len(frame_rows),
        "status_counts": dict(status_counts),
    }

    Path(args.out_body_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_body_summary_txt).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()