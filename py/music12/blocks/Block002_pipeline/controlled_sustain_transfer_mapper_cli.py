from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
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


def _octave(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[0]
    except Exception:
        return ""


def _build_families_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: _safe_int(r.get("family_rank"), 999999))
    return out


def _build_chain_frames_by_proto(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("proto_exciter_id"), 0)].append(row)
    for proto_rows in out.values():
        proto_rows.sort(key=lambda r: _safe_int(r.get("frame_index"), 0))
    return out


def _score_extension_row(
    *,
    row: dict[str, Any],
    anchor_note: str,
    anchor_pc: str,
    anchor_oct: str,
    prev_note: str,
    transfer_mode: bool,
) -> tuple[float, str]:
    family_note = _normalize_note(row.get("family_root_note_micro", ""))
    family_pc = _pitch_class(family_note)
    family_oct = _octave(family_note)
    if family_pc != anchor_pc:
        return -1.0e9, "foreign_pc"

    family_score = _safe_float(row.get("family_score"), 0.0)
    root_micro_count = _safe_int(row.get("root_micro_count"), 0)
    root_micro_diversity = _safe_int(row.get("root_micro_diversity"), 0)

    reasons: list[str] = []
    score = 0.0

    if family_note == anchor_note:
        score += 1.10
        reasons.append("exact_anchor")
    elif family_oct == anchor_oct:
        score += 0.75
        reasons.append("same_octave")
    elif transfer_mode:
        score += 0.35
        reasons.append("transfer_octave")
    else:
        return -1.0e9, "premature_transfer"

    if prev_note:
        if family_note == prev_note:
            score += 0.55
            reasons.append("same_as_prev")
        elif _pitch_class(prev_note) == family_pc:
            score += 0.15
            reasons.append("pc_continuity")

    score += min(family_score / 8.0, 1.05)
    score += min(root_micro_count / 52.0, 0.40)
    score += min(root_micro_diversity / 40.0, 0.25)

    return score, ("|".join(reasons) if reasons else "weak")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extend primary note chains with controlled sustain and explicit box-transfer phase."
    )
    ap.add_argument("--primary-chains-csv", required=True)
    ap.add_argument("--primary-chain-frames-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-extended-frames-csv", required=True)
    ap.add_argument("--out-extended-chains-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--lookahead-frames", type=int, default=10)
    ap.add_argument("--max-gap-frames", type=int, default=2)
    ap.add_argument("--min-extend-score", type=float, default=1.15)
    ap.add_argument("--min-stable-anchor-frames", type=int, default=4)
    args = ap.parse_args()

    chain_rows = _load_csv(Path(args.primary_chains_csv))
    chain_frame_rows = _load_csv(Path(args.primary_chain_frames_csv))
    family_rows = _load_csv(Path(args.micro_families_csv))

    chain_frames_by_proto = _build_chain_frames_by_proto(chain_frame_rows)
    families_by_frame = _build_families_by_frame(family_rows)

    extended_frame_rows: list[dict[str, Any]] = []
    extended_chain_rows: list[dict[str, Any]] = []

    for chain in chain_rows:
        proto_id = _safe_int(chain.get("proto_exciter_id"), 0)
        base_rows = list(chain_frames_by_proto.get(proto_id, []))
        if not base_rows:
            continue

        anchor_note = _normalize_note(chain.get("dominant_note_token", "") or chain.get("coarse_note", ""))
        anchor_pc = _pitch_class(anchor_note)
        anchor_oct = _octave(anchor_note)
        exact_anchor_frames = _safe_int(chain.get("exact_coarse_frames"), 0)
        transfer_allowed = exact_anchor_frames >= int(args.min_stable_anchor_frames)

        # Base phase rows.
        for idx, row in enumerate(base_rows):
            extended_frame_rows.append(
                {
                    "proto_exciter_id": proto_id,
                    "frame_index": _safe_int(row.get("frame_index"), 0),
                    "selected_note_token": _normalize_note(row.get("selected_note_token", "")),
                    "coarse_note": str(row.get("coarse_note", "")).strip(),
                    "phase": "PRIMARY_CHAIN" if idx > 0 else "STABILIZATION",
                    "selection_reason": str(row.get("selection_reason", "")).strip(),
                    "phase_score": row.get("chain_score", "0"),
                }
            )

        prev_note = _normalize_note(base_rows[-1].get("selected_note_token", ""))
        last_frame = _safe_int(base_rows[-1].get("frame_index"), 0)
        gaps = 0
        added_rows = 0
        transfer_rows = 0
        sustained_exact = sum(1 for row in base_rows if _normalize_note(row.get("selected_note_token", "")) == anchor_note)
        note_counter = Counter(_normalize_note(row.get("selected_note_token", "")) for row in base_rows)

        for frame_index in range(last_frame + 1, last_frame + int(args.lookahead_frames) + 1):
            rows = families_by_frame.get(frame_index, [])
            if not rows:
                gaps += 1
                if gaps > int(args.max_gap_frames):
                    break
                continue

            best_row: dict[str, Any] | None = None
            best_score = -1.0
            best_reason = ""

            for row in rows[:8]:
                score, reason = _score_extension_row(
                    row=row,
                    anchor_note=anchor_note,
                    anchor_pc=anchor_pc,
                    anchor_oct=anchor_oct,
                    prev_note=prev_note,
                    transfer_mode=transfer_allowed,
                )
                if score > best_score:
                    best_score = score
                    best_row = row
                    best_reason = reason

            if best_row is None or best_score < float(args.min_extend_score):
                gaps += 1
                if gaps > int(args.max_gap_frames):
                    break
                continue

            gaps = 0
            note = _normalize_note(best_row.get("family_root_note_micro", ""))
            phase = "CONTROLLED_SUSTAIN"
            if note != anchor_note or _octave(note) != anchor_oct:
                phase = "BOX_TRANSFER"
                transfer_rows += 1

            if note == anchor_note:
                sustained_exact += 1
            note_counter[note] += 1
            prev_note = note
            added_rows += 1

            extended_frame_rows.append(
                {
                    "proto_exciter_id": proto_id,
                    "frame_index": frame_index,
                    "selected_note_token": note,
                    "coarse_note": str(chain.get("coarse_note", "")).strip(),
                    "phase": phase,
                    "selection_reason": best_reason,
                    "phase_score": f"{best_score:.9f}",
                }
            )

        all_rows = [row for row in extended_frame_rows if _safe_int(row.get("proto_exciter_id"), 0) == proto_id]
        all_rows.sort(key=lambda r: _safe_int(r.get("frame_index"), 0))
        extended_chain_rows.append(
            {
                "proto_exciter_id": proto_id,
                "coarse_note": str(chain.get("coarse_note", "")).strip(),
                "anchor_note_token": anchor_note,
                "start_frame": _safe_int(all_rows[0].get("frame_index"), 0),
                "end_frame": _safe_int(all_rows[-1].get("frame_index"), 0),
                "frame_count": len(all_rows),
                "added_rows": added_rows,
                "transfer_rows": transfer_rows,
                "dominant_note_token": note_counter.most_common(1)[0][0] if note_counter else "",
                "exact_anchor_frames": sustained_exact,
                "phase_counts_json": json.dumps(dict(Counter(str(r.get("phase", "")).strip() for r in all_rows)), ensure_ascii=False, sort_keys=True),
                "selected_notes_json": json.dumps(dict(note_counter), ensure_ascii=False, sort_keys=True),
            }
        )

    extended_frame_rows.sort(key=lambda r: (_safe_int(r.get("frame_index"), 0), _safe_int(r.get("proto_exciter_id"), 0)))
    extended_chain_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))

    out_frames = Path(args.out_extended_frames_csv)
    out_chains = Path(args.out_extended_chains_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_frames.parent.mkdir(parents=True, exist_ok=True)

    frame_fields = [
        "proto_exciter_id",
        "frame_index",
        "selected_note_token",
        "coarse_note",
        "phase",
        "selection_reason",
        "phase_score",
    ]
    with out_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(extended_frame_rows)

    chain_fields = [
        "proto_exciter_id",
        "coarse_note",
        "anchor_note_token",
        "start_frame",
        "end_frame",
        "frame_count",
        "added_rows",
        "transfer_rows",
        "dominant_note_token",
        "exact_anchor_frames",
        "phase_counts_json",
        "selected_notes_json",
    ]
    with out_chains.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=chain_fields)
        w.writeheader()
        w.writerows(extended_chain_rows)

    total_added = sum(_safe_int(r.get("added_rows"), 0) for r in extended_chain_rows)
    total_transfer = sum(_safe_int(r.get("transfer_rows"), 0) for r in extended_chain_rows)
    summary = {
        "stage": "controlled_sustain_transfer_mapper",
        "inputs": {
            "primary_chains_csv": args.primary_chains_csv,
            "primary_chain_frames_csv": args.primary_chain_frames_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "lookahead_frames": int(args.lookahead_frames),
            "max_gap_frames": int(args.max_gap_frames),
            "min_extend_score": float(args.min_extend_score),
            "min_stable_anchor_frames": int(args.min_stable_anchor_frames),
        },
        "result": {
            "extended_chain_count": len(extended_chain_rows),
            "extended_frame_rows": len(extended_frame_rows),
            "added_rows": total_added,
            "transfer_rows": total_transfer,
        },
    }

    lines = [
        "CONTROLLED SUSTAIN / BOX TRANSFER",
        "=" * 72,
        f"extended_chain_count : {len(extended_chain_rows)}",
        f"extended_frame_rows  : {len(extended_frame_rows)}",
        f"added_rows           : {total_added}",
        f"transfer_rows        : {total_transfer}",
    ]
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
