from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitch_class(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _build_proto_by_start_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("start_frame"), 0)].append(row)
    return out


def _find_previous_tail_conflict(
    *,
    proto_by_start_frame: dict[int, list[dict[str, Any]]],
    current_proto_id: str,
    start_frame: int,
    note_token: str,
    rhythm_window_frames: int,
    min_confidence: float,
) -> tuple[bool, list[str]]:
    proto_pc = _pitch_class(note_token)
    neighbors: list[str] = []
    for frame in range(start_frame - rhythm_window_frames, start_frame):
        for row in proto_by_start_frame.get(frame, []):
            proto_id = str(row.get("proto_exciter_id", "")).strip()
            if proto_id == current_proto_id:
                continue
            other_note = str(row.get("rescue_group_dominant_note", "")).strip() or str(row.get("coarse_note", "")).strip()
            other_note = _normalize_note(other_note)
            if not other_note or _pitch_class(other_note) == proto_pc:
                continue
            if _safe_float(row.get("exciter_confidence"), 0.0) < min_confidence:
                continue
            if _safe_int(row.get("end_frame"), 0) < start_frame - rhythm_window_frames:
                continue
            neighbors.append(other_note)
    return bool(neighbors), sorted(set(neighbors))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit suspicious birth cases to separate false starts, weak useful starts, and previous-tail conflicts."
    )
    ap.add_argument("--primary-note-chains-csv", required=True)
    ap.add_argument("--chain-identity-audit-csv", required=True)
    ap.add_argument("--midi-identity-audit-csv", required=True)
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--branch-analysis-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--rhythm-window-frames", type=int, default=2)
    ap.add_argument("--min-competing-confidence", type=float, default=0.58)
    args = ap.parse_args()

    chains = _load_csv(Path(args.primary_note_chains_csv))
    chain_audit = _load_csv(Path(args.chain_identity_audit_csv))
    midi_audit = _load_csv(Path(args.midi_identity_audit_csv))
    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    branch_rows = _load_csv(Path(args.branch_analysis_csv))

    chain_audit_by_proto = {str(r.get("proto_exciter_id", "")).strip(): r for r in chain_audit}
    branch_by_proto = {str(r.get("proto_exciter_id", "")).strip(): r for r in branch_rows}
    midi_status_by_proto = {
        str(r.get("best_chain_proto_exciter_id", "")).strip(): r
        for r in midi_audit
        if str(r.get("best_chain_proto_exciter_id", "")).strip()
    }
    proto_by_start_frame = _build_proto_by_start_frame(proto_rows)

    audit_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    chain_status_counter: Counter[str] = Counter()
    midi_status_counter: Counter[str] = Counter()

    for chain in chains:
        if _safe_int(chain.get("suspicious_birth"), 0) != 1:
            continue

        proto_id = str(chain.get("proto_exciter_id", "")).strip()
        note_token = str(chain.get("coarse_note", "")).strip()
        start_frame = _safe_int(chain.get("chain_start_frame"), 0)
        chain_status = str(chain_audit_by_proto.get(proto_id, {}).get("status", "")).strip()
        midi_status = str(midi_status_by_proto.get(proto_id, {}).get("status", "")).strip()
        branch = branch_by_proto.get(proto_id, {})
        branch_label = str(branch.get("branch_label", "")).strip()
        route_label = str(branch.get("route_label", "")).strip()

        has_tail_conflict, neighbor_notes = _find_previous_tail_conflict(
            proto_by_start_frame=proto_by_start_frame,
            current_proto_id=proto_id,
            start_frame=start_frame,
            note_token=note_token,
            rhythm_window_frames=int(args.rhythm_window_frames),
            min_confidence=float(args.min_competing_confidence),
        )

        if midi_status == "MIDI_EXACT_CHAIN":
            audit_class = "WEAK_BUT_USEFUL_BIRTH"
        elif has_tail_conflict:
            audit_class = "PREVIOUS_TAIL_CONFLICT"
        elif chain_status in {"CHAIN_FOREIGN_CAPTURE", "CHAIN_STEP_DRIFT", "CHAIN_OCTAVE_CONFUSION"}:
            audit_class = "FALSE_OR_SHIFTED_BIRTH"
        else:
            audit_class = "WEAK_UNRESOLVED_BIRTH"

        class_counter[audit_class] += 1
        chain_status_counter[chain_status] += 1
        midi_status_counter[midi_status] += 1

        audit_rows.append(
            {
                "proto_exciter_id": proto_id,
                "coarse_note": note_token,
                "chain_start_frame": start_frame,
                "chain_frame_count": chain.get("chain_frame_count", ""),
                "birth_trimmed_frames": chain.get("birth_trimmed_frames", ""),
                "retro_trimmed_frames": chain.get("retro_trimmed_frames", ""),
                "handoff_to_new_note": chain.get("handoff_to_new_note", ""),
                "branch_label": branch_label,
                "route_label": route_label,
                "chain_status": chain_status,
                "midi_status": midi_status,
                "previous_tail_conflict": int(has_tail_conflict),
                "previous_tail_notes_json": json.dumps(neighbor_notes, ensure_ascii=False),
                "audit_class": audit_class,
            }
        )

    out_csv = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "SUSPICIOUS BIRTH AUDIT",
        "=" * 72,
        f"suspicious_birth_count : {len(audit_rows)}",
        "",
        "AUDIT CLASS COUNTS",
        "-" * 72,
    ]
    for key in sorted(class_counter):
        lines.append(f"{key:24s}: {class_counter[key]}")
    lines.extend(["", "CHAIN STATUS COUNTS", "-" * 72])
    for key in sorted(chain_status_counter):
        lines.append(f"{key:24s}: {chain_status_counter[key]}")
    lines.extend(["", "MIDI STATUS COUNTS", "-" * 72])
    for key in sorted(midi_status_counter):
        lines.append(f"{key:24s}: {midi_status_counter[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "inputs": {
            "primary_note_chains_csv": args.primary_note_chains_csv,
            "chain_identity_audit_csv": args.chain_identity_audit_csv,
            "midi_identity_audit_csv": args.midi_identity_audit_csv,
            "proto_exciters_csv": args.proto_exciters_csv,
            "branch_analysis_csv": args.branch_analysis_csv,
        },
        "parameters": {
            "rhythm_window_frames": int(args.rhythm_window_frames),
            "min_competing_confidence": float(args.min_competing_confidence),
        },
        "result": {
            "suspicious_birth_count": len(audit_rows),
            "audit_class_counts": dict(class_counter),
            "chain_status_counts": dict(chain_status_counter),
            "midi_status_counts": dict(midi_status_counter),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
