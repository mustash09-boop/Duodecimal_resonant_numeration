# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _group_by_note(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        note = _normalize_note(r.get("note_token", ""))
        if not note:
            continue

        rr = dict(r)
        rr["note_token"] = note
        rr["frame_index"] = _safe_int(rr.get("frame_index"), 0)
        rr["ownership_score"] = _safe_float(
            rr.get("passport_ownership_score", rr.get("score", 0.0)),
            0.0,
        )

        out.setdefault(note, []).append(rr)

    for note in out:
        out[note].sort(key=lambda x: _safe_int(x.get("frame_index"), 0))

    return out


def _classify_state(
    *,
    score: float,
    prev_score: float,
    next_score: float,
    local_age: int,
    role: str,
    echo_overlap: int,
    strong_overlap: int,
    evidence_count: int,
    min_attack_delta: float,
) -> Tuple[str, float]:
    rise = score - prev_score
    fall = score - next_score

    confidence = score

    if role in {"response_sink", "response_like"}:
        confidence -= 0.45

    if echo_overlap > 0:
        confidence -= echo_overlap * 0.20

    if strong_overlap > 0:
        confidence += strong_overlap * 0.08

    if evidence_count >= 2:
        confidence += 0.10

    if local_age == 0 and rise >= min_attack_delta:
        return "NEW_CAUSAL_EXCITATION", max(confidence + 0.35, 0.0)

    if rise >= min_attack_delta and score >= 1.20:
        return "RE_EXCITATION", max(confidence + 0.20, 0.0)

    if score >= 1.05 and local_age <= 18:
        return "ACTIVE_SUSTAIN", max(confidence, 0.0)

    if fall < 0.05 and local_age > 18 and score < 1.15:
        return "LINGERING_DECAY", max(confidence - 0.35, 0.0)

    if role == "feedback_bridge" and score >= 0.95:
        return "BOX_OR_SHARED_RESONANCE", max(confidence - 0.15, 0.0)

    if score < 0.95:
        return "WEAK_RESONANCE_TRACE", max(confidence - 0.45, 0.0)

    return "ACTIVE_SUSTAIN", max(confidence, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Discriminate temporal resonance ownership states: attack, sustain, box/shared resonance, lingering decay."
    )

    ap.add_argument("--passport_filtered_frame_notes_csv", required=True)

    ap.add_argument("--out_state_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_attack_delta", type=float, default=0.18)
    ap.add_argument("--min_final_confidence", type=float, default=0.95)
    ap.add_argument("--keep_states", default="NEW_CAUSAL_EXCITATION,RE_EXCITATION,ACTIVE_SUSTAIN")

    args = ap.parse_args()

    rows = _load_csv(Path(args.passport_filtered_frame_notes_csv))
    by_note = _group_by_note(rows)

    keep_states = {
        x.strip()
        for x in args.keep_states.split(",")
        if x.strip()
    }

    out_rows = []
    readable_by_frame: Dict[int, List[str]] = {}

    state_counts: Dict[str, int] = {}
    kept_counts: Dict[str, int] = {}
    rejected_counts: Dict[str, int] = {}

    for note, note_rows in by_note.items():
        previous_frame = None
        local_age = 0

        for i, r in enumerate(note_rows):
            frame = _safe_int(r.get("frame_index"), 0)
            score = _safe_float(r.get("ownership_score"), 0.0)

            prev_score = (
                _safe_float(note_rows[i - 1].get("ownership_score"), 0.0)
                if i > 0 and frame - _safe_int(note_rows[i - 1].get("frame_index"), 0) <= 3
                else 0.0
            )

            next_score = (
                _safe_float(note_rows[i + 1].get("ownership_score"), 0.0)
                if i + 1 < len(note_rows) and _safe_int(note_rows[i + 1].get("frame_index"), 0) - frame <= 3
                else 0.0
            )

            if previous_frame is None or frame - previous_frame > 3:
                local_age = 0
            else:
                local_age += frame - previous_frame

            previous_frame = frame

            state, confidence = _classify_state(
                score=score,
                prev_score=prev_score,
                next_score=next_score,
                local_age=local_age,
                role=str(r.get("causal_role", "")),
                echo_overlap=_safe_int(r.get("echo_overlap"), 0),
                strong_overlap=_safe_int(r.get("strong_overlap"), 0),
                evidence_count=_safe_int(r.get("evidence_count"), 0),
                min_attack_delta=args.min_attack_delta,
            )

            state_counts[state] = state_counts.get(state, 0) + 1

            keep = state in keep_states and confidence >= args.min_final_confidence

            if keep:
                kept_counts[state] = kept_counts.get(state, 0) + 1
            else:
                rejected_counts[state] = rejected_counts.get(state, 0) + 1

            rr = dict(r)
            rr["temporal_state"] = state
            rr["temporal_confidence"] = f"{confidence:.9f}"
            rr["local_age_frames"] = local_age
            rr["prev_ownership_score"] = f"{prev_score:.9f}"
            rr["next_ownership_score"] = f"{next_score:.9f}"
            rr["ownership_rise"] = f"{(score - prev_score):.9f}"
            rr["ownership_fall"] = f"{(score - next_score):.9f}"
            rr["temporal_keep"] = "YES" if keep else "NO"

            if keep:
                out_rows.append(rr)
                readable_by_frame.setdefault(frame, []).append(
                    f"{note}:{confidence:.3f}[{state}]"
                )

    out_rows.sort(
        key=lambda r: (
            _safe_int(r.get("frame_index"), 0),
            -_safe_float(r.get("temporal_confidence"), 0.0),
        )
    )

    readable_rows = []
    all_frames = sorted({_safe_int(r.get("frame_index"), 0) for r in rows})

    for frame in all_frames:
        items = readable_by_frame.get(frame, [])
        readable_rows.append({
            "frame_index": frame,
            "active_note_count": len(items),
            "notes": " | ".join(items),
        })

    out_state = Path(args.out_state_frame_notes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_state.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys()) if out_rows else []

    with out_state.open("w", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        else:
            f.write("")

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "active_note_count", "notes"])
        w.writeheader()
        w.writerows(readable_rows)

    active_distribution: Dict[int, int] = {}
    for r in readable_rows:
        n = _safe_int(r.get("active_note_count"), 0)
        active_distribution[n] = active_distribution.get(n, 0) + 1

    meta = {
        "stage": "temporal_resonance_ownership_discriminator",
        "inputs": {
            "passport_filtered_frame_notes_csv": args.passport_filtered_frame_notes_csv,
        },
        "outputs": {
            "state_frame_notes_csv": args.out_state_frame_notes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_attack_delta": args.min_attack_delta,
            "min_final_confidence": args.min_final_confidence,
            "keep_states": sorted(keep_states),
        },
        "result": {
            "input_rows": len(rows),
            "output_rows": len(out_rows),
            "frames": len(readable_rows),
            "state_counts": state_counts,
            "kept_counts": kept_counts,
            "rejected_counts": rejected_counts,
            "active_distribution": active_distribution,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "TEMPORAL RESONANCE OWNERSHIP DISCRIMINATOR",
        "=" * 72,
        f"input_csv        : {args.passport_filtered_frame_notes_csv}",
        "",
        f"input_rows       : {len(rows)}",
        f"output_rows      : {len(out_rows)}",
        f"frames           : {len(readable_rows)}",
        "",
        "State counts:",
    ]

    for k in sorted(state_counts):
        txt.append(f"  {k}: {state_counts[k]}")

    txt.append("")
    txt.append("Kept counts:")
    for k in sorted(kept_counts):
        txt.append(f"  {k}: {kept_counts[k]}")

    txt.append("")
    txt.append("Rejected counts:")
    for k in sorted(rejected_counts):
        txt.append(f"  {k}: {rejected_counts[k]}")

    txt.append("")
    txt.append("Active note distribution:")
    for k in sorted(active_distribution):
        txt.append(f"  {k}: {active_distribution[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Do not treat every resonance as a note.",
        "  Separate new causal excitation, active sustain, box/shared resonance,",
        "  lingering decay and weak resonance trace by temporal ownership behavior.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("temporal resonance ownership discriminator complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()