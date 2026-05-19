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


def _coarse(note: str) -> str:
    return _normalize_note(note)


def _build_families_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: _safe_int(r.get("family_rank"), 999999))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit whether suspicious note births are supported jointly by the previous tail and the new note in the same early frames."
    )
    ap.add_argument("--suspicious-birth-audit-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--inspect-frames", type=int, default=2)
    ap.add_argument("--top-family-rank", type=int, default=8)
    args = ap.parse_args()

    suspicious_rows = _load_csv(Path(args.suspicious_birth_audit_csv))
    family_rows = _load_csv(Path(args.micro_families_csv))
    families_by_frame = _build_families_by_frame(family_rows)

    audit_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()

    for row in suspicious_rows:
        if str(row.get("audit_class", "")).strip() != "PREVIOUS_TAIL_CONFLICT":
            continue

        current_note = _coarse(row.get("coarse_note", ""))
        start_frame = _safe_int(row.get("chain_start_frame"), 0)
        previous_tail_notes = [
            _coarse(x)
            for x in json.loads(str(row.get("previous_tail_notes_json", "[]")) or "[]")
            if _coarse(x)
        ]

        overlap_frames = 0
        current_support_frames = 0
        tail_support_frames = 0
        current_scores: list[float] = []
        tail_scores: list[float] = []

        for frame in range(start_frame, start_frame + int(args.inspect_frames)):
            frame_rows = [
                fr for fr in families_by_frame.get(frame, [])
                if _safe_int(fr.get("family_rank"), 999999) <= int(args.top_family_rank)
            ]
            current_here = False
            tail_here = False
            for fr in frame_rows:
                note = _coarse(fr.get("family_root_note_micro", ""))
                score = _safe_float(fr.get("family_score"), 0.0)
                if note == current_note:
                    current_here = True
                    current_scores.append(score)
                if note in previous_tail_notes:
                    tail_here = True
                    tail_scores.append(score)
            if current_here:
                current_support_frames += 1
            if tail_here:
                tail_support_frames += 1
            if current_here and tail_here:
                overlap_frames += 1

        if overlap_frames >= 1:
            audit_class = "SHARED_SUPPORT_CONFIRMED"
        elif current_support_frames >= 1 and tail_support_frames == 0:
            audit_class = "CURRENT_ONLY_SUPPORT"
        elif tail_support_frames >= 1 and current_support_frames == 0:
            audit_class = "TAIL_ONLY_SUPPORT"
        else:
            audit_class = "INTERLEAVED_OR_WEAK_SUPPORT"
        class_counter[audit_class] += 1

        audit_rows.append(
            {
                "proto_exciter_id": row.get("proto_exciter_id", ""),
                "coarse_note": current_note,
                "chain_start_frame": start_frame,
                "previous_tail_notes_json": json.dumps(previous_tail_notes, ensure_ascii=False),
                "current_support_frames": current_support_frames,
                "tail_support_frames": tail_support_frames,
                "overlap_frames": overlap_frames,
                "mean_current_score": f"{(sum(current_scores) / max(len(current_scores), 1)):.9f}" if current_scores else "",
                "mean_tail_score": f"{(sum(tail_scores) / max(len(tail_scores), 1)):.9f}" if tail_scores else "",
                "ownership_class": audit_class,
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
        "SHARED OWNERSHIP TRANSITION AUDIT",
        "=" * 72,
        f"tail_conflict_cases : {len(audit_rows)}",
        "",
        "OWNERSHIP CLASS COUNTS",
        "-" * 72,
    ]
    for key in sorted(class_counter):
        lines.append(f"{key:28s}: {class_counter[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "inputs": {
            "suspicious_birth_audit_csv": args.suspicious_birth_audit_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "inspect_frames": int(args.inspect_frames),
            "top_family_rank": int(args.top_family_rank),
        },
        "result": {
            "tail_conflict_cases": len(audit_rows),
            "ownership_class_counts": dict(class_counter),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
