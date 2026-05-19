# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


ALPHABET12 = "123456789ABC"


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


def _token_to_abs_degree(token: str) -> Optional[int]:
    try:
        token = _normalize_note(token).upper()
        octave_raw, rest = token.split(".", 1)
        degree_raw = rest.split("'", 1)[0]

        octave = 0
        for ch in octave_raw:
            if ch not in ALPHABET12:
                return None
            octave = octave * 12 + (ALPHABET12.index(ch) + 1)

        if degree_raw not in ALPHABET12:
            return None

        return octave * 12 + ALPHABET12.index(degree_raw)
    except Exception:
        return None


def _pitch_distance(a: str, b: str) -> float:
    aa = _token_to_abs_degree(a)
    bb = _token_to_abs_degree(b)
    if aa is None or bb is None:
        return 9999.0
    return abs(float(aa - bb))


def _group_frame_notes(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        note = _normalize_note(r.get("note_token", ""))

        if not note:
            continue

        rr = dict(r)
        rr["frame_index"] = frame
        rr["note_token"] = note
        rr["score"] = _safe_float(
            rr.get("temporal_confidence", rr.get("passport_ownership_score", rr.get("score", 0.0))),
            0.0,
        )

        out.setdefault(frame, []).append(rr)

    for frame in out:
        out[frame].sort(key=lambda x: -_safe_float(x.get("score"), 0.0))

    return out


def _can_continue_same_event(
    event: Dict[str, Any],
    note: str,
    frame: int,
    max_gap_frames: int,
    max_same_event_pitch_drift: float,
) -> bool:
    gap = frame - _safe_int(event.get("last_frame"), 0)

    if gap < 0 or gap > max_gap_frames:
        return False

    primary = str(event.get("primary_note", ""))
    last_note = str(event.get("last_note", ""))

    if _pitch_distance(primary, note) > max_same_event_pitch_drift:
        return False

    if _pitch_distance(last_note, note) > max_same_event_pitch_drift:
        return False

    return True


def _event_match_cost(event: Dict[str, Any], candidate: Dict[str, Any], frame: int) -> float:
    note = candidate["note_token"]
    score = _safe_float(candidate.get("score"), 0.0)

    gap = frame - _safe_int(event.get("last_frame"), 0)
    primary_dist = _pitch_distance(str(event.get("primary_note", "")), note)
    last_dist = _pitch_distance(str(event.get("last_note", "")), note)

    return primary_dist * 2.0 + last_dist * 1.5 + gap * 0.75 - score * 0.03


def _classify_event_state(
    *,
    age: int,
    score: float,
    prev_score: float,
    recent_min_score: float,
    temporal_state: str,
    attack_delta: float,
    reexcite_drop: float,
) -> str:
    if age == 0:
        return "EXCITATION_BIRTH"

    if temporal_state in {"BOX_OR_SHARED_RESONANCE", "WEAK_RESONANCE_TRACE", "LINGERING_DECAY"}:
        return "SECONDARY_RESONANCE_TAIL"

    if (
        score - recent_min_score >= reexcite_drop
        and score - prev_score >= attack_delta
        and score >= 1.10
    ):
        return "EXCITATION_REBIRTH"

    if temporal_state == "NEW_CAUSAL_EXCITATION":
        return "ROOT_HYPOTHESIS"

    if score >= 1.05:
        return "CHAIN_STABILIZATION"

    if score >= 0.80:
        return "OWNERSHIP_SPLIT"

    return "BOX_TRANSFER"


def _summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    scores = event["scores"]
    notes = event["notes"]
    states = event["states"]
    frames = event["frames"]

    mean_score = sum(scores) / max(len(scores), 1)
    max_score = max(scores) if scores else 0.0
    birth_score = scores[0] if scores else 0.0
    final_score = scores[-1] if scores else 0.0
    attack_peak_frame = frames[scores.index(max_score)] if scores else event["birth_frame"]

    state_counts: Dict[str, int] = {}
    for s in states:
        state_counts[s] = state_counts.get(s, 0) + 1

    if state_counts.get("EXCITATION_REBIRTH", 0) > 0:
        lifecycle_kind = "re_excited_event"
    elif state_counts.get("CHAIN_STABILIZATION", 0) + state_counts.get("OWNERSHIP_SPLIT", 0) >= 3:
        lifecycle_kind = "chain_stabilized_event"
    elif state_counts.get("SECONDARY_RESONANCE_TAIL", 0) > state_counts.get("CHAIN_STABILIZATION", 0):
        lifecycle_kind = "secondary_tail_event"
    else:
        lifecycle_kind = "short_hypothesis_event"

    return {
        "event_id": event["event_id"],
        "candidate_note": event["primary_note"],
        "birth_frame": min(frames),
        "end_frame": max(frames),
        "duration_frames": max(frames) - min(frames) + 1,
        "frame_count": len(frames),
        "mean_score": f"{mean_score:.9f}",
        "max_score": f"{max_score:.9f}",
        "birth_score": f"{birth_score:.9f}",
        "final_score": f"{final_score:.9f}",
        "attack_peak_frame": attack_peak_frame,
        "unique_note_count": len(set(notes)),
        "note_path": " ".join(notes),
        "state_path": " ".join(states),
        "excitation_birth_count": state_counts.get("EXCITATION_BIRTH", 0),
        "root_hypothesis_count": state_counts.get("ROOT_HYPOTHESIS", 0),
        "excitation_rebirth_count": state_counts.get("EXCITATION_REBIRTH", 0),
        "chain_stabilization_count": state_counts.get("CHAIN_STABILIZATION", 0),
        "ownership_split_count": state_counts.get("OWNERSHIP_SPLIT", 0),
        "box_transfer_count": state_counts.get("BOX_TRANSFER", 0),
        "secondary_tail_count": state_counts.get("SECONDARY_RESONANCE_TAIL", 0),
        "lifecycle_kind": lifecycle_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "V2 lifecycle tracker for the canonical Block002 pipeline: "
            "strict event identity plus state-machine labels for excitation, "
            "hypothesis, stabilization, box transfer and secondary tail."
        )
    )

    ap.add_argument("--frame_notes_csv", required=True)
    ap.add_argument("--out_events_csv", required=True)
    ap.add_argument("--out_event_frames_csv", required=True)
    ap.add_argument("--out_overlap_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--max_gap_frames", type=int, default=4)
    ap.add_argument("--max_same_event_pitch_drift", type=float, default=0.0)
    ap.add_argument("--min_event_frames", type=int, default=4)
    ap.add_argument("--min_birth_score", type=float, default=0.95)
    ap.add_argument("--attack_delta", type=float, default=0.18)
    ap.add_argument("--reexcite_drop", type=float, default=0.35)

    args = ap.parse_args()

    rows = _load_csv(Path(args.frame_notes_csv))
    frame_notes = _group_frame_notes(rows)

    active_events: Dict[int, Dict[str, Any]] = {}
    finished_events: List[Dict[str, Any]] = []
    event_frame_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    readable_rows: List[Dict[str, Any]] = []
    next_event_id = 1
    all_frames = sorted(frame_notes.keys())

    for frame in all_frames:
        candidates = frame_notes.get(frame, [])
        updated_event_ids: Set[int] = set()

        for cand in candidates:
            note = cand["note_token"]
            score = _safe_float(cand.get("score"), 0.0)
            temporal_state = str(cand.get("temporal_state", "")).strip()

            best_event_id = None
            best_cost = 999999.0

            for eid, ev in active_events.items():
                if eid in updated_event_ids:
                    continue

                if not _can_continue_same_event(
                    ev,
                    note,
                    frame,
                    args.max_gap_frames,
                    args.max_same_event_pitch_drift,
                ):
                    continue

                cost = _event_match_cost(ev, cand, frame)

                if cost < best_cost:
                    best_cost = cost
                    best_event_id = eid

            if best_event_id is None:
                if score < args.min_birth_score:
                    continue

                if temporal_state in {"BOX_OR_SHARED_RESONANCE", "WEAK_RESONANCE_TRACE", "LINGERING_DECAY"}:
                    continue

                eid = next_event_id
                next_event_id += 1

                ev = {
                    "event_id": eid,
                    "primary_note": note,
                    "birth_frame": frame,
                    "last_frame": frame,
                    "last_note": note,
                    "frames": [frame],
                    "notes": [note],
                    "scores": [score],
                    "states": ["EXCITATION_BIRTH"],
                    "temporal_states": [temporal_state],
                }

                active_events[eid] = ev
                updated_event_ids.add(eid)

                event_frame_rows.append({
                    "event_id": eid,
                    "frame_index": frame,
                    "note_token": note,
                    "event_state": "EXCITATION_BIRTH",
                    "input_temporal_state": temporal_state,
                    "score": f"{score:.9f}",
                })
            else:
                eid = best_event_id
                ev = active_events[eid]

                prev_score = ev["scores"][-1] if ev["scores"] else 0.0
                recent_scores = ev["scores"][-8:] if ev["scores"] else [0.0]
                recent_min_score = min(recent_scores)
                age = frame - _safe_int(ev.get("birth_frame"), frame)

                state = _classify_event_state(
                    age=age,
                    score=score,
                    prev_score=prev_score,
                    recent_min_score=recent_min_score,
                    temporal_state=temporal_state,
                    attack_delta=args.attack_delta,
                    reexcite_drop=args.reexcite_drop,
                )

                ev["last_frame"] = frame
                ev["last_note"] = note
                ev["frames"].append(frame)
                ev["notes"].append(note)
                ev["scores"].append(score)
                ev["states"].append(state)
                ev["temporal_states"].append(temporal_state)

                updated_event_ids.add(eid)

                event_frame_rows.append({
                    "event_id": eid,
                    "frame_index": frame,
                    "note_token": note,
                    "event_state": state,
                    "input_temporal_state": temporal_state,
                    "score": f"{score:.9f}",
                })

        current_active_ids = sorted(updated_event_ids)

        for i, a in enumerate(current_active_ids):
            for b in current_active_ids[i + 1:]:
                overlap_rows.append({
                    "frame_index": frame,
                    "event_a": a,
                    "event_b": b,
                    "note_a": active_events[a].get("last_note", ""),
                    "note_b": active_events[b].get("last_note", ""),
                    "pitch_distance": f"{_pitch_distance(active_events[a].get('last_note', ''), active_events[b].get('last_note', '')):.6f}",
                })

        readable_rows.append({
            "frame_index": frame,
            "active_event_count": len(current_active_ids),
            "active_events": " | ".join(
                f"E{eid}:{active_events[eid].get('last_note')}:{active_events[eid]['scores'][-1]:.3f}:{active_events[eid]['states'][-1]}"
                for eid in current_active_ids
            ),
        })

        to_close = []
        for eid, ev in active_events.items():
            if frame - _safe_int(ev.get("last_frame"), frame) > args.max_gap_frames:
                to_close.append(eid)

        for eid in to_close:
            finished_events.append(active_events.pop(eid))

    finished_events.extend(active_events.values())

    event_rows = []
    for ev in finished_events:
        if len(ev["frames"]) < args.min_event_frames:
            continue
        event_rows.append(_summarize_event(ev))

    event_rows.sort(key=lambda r: (_safe_int(r["birth_frame"]), _safe_int(r["event_id"])))

    out_events = Path(args.out_events_csv)
    out_event_frames = Path(args.out_event_frames_csv)
    out_overlap = Path(args.out_overlap_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "event_id", "candidate_note", "birth_frame", "end_frame", "duration_frames",
        "frame_count", "mean_score", "max_score", "birth_score", "final_score",
        "attack_peak_frame", "unique_note_count", "note_path", "state_path",
        "excitation_birth_count", "root_hypothesis_count", "excitation_rebirth_count",
        "chain_stabilization_count", "ownership_split_count", "box_transfer_count",
        "secondary_tail_count", "lifecycle_kind",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(event_rows)

    frame_fields = ["event_id", "frame_index", "note_token", "event_state", "input_temporal_state", "score"]

    with out_event_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(event_frame_rows)

    overlap_fields = ["frame_index", "event_a", "event_b", "note_a", "note_b", "pitch_distance"]

    with out_overlap.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=overlap_fields)
        w.writeheader()
        w.writerows(overlap_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "active_event_count", "active_events"])
        w.writeheader()
        w.writerows(readable_rows)

    lifecycle_counts: Dict[str, int] = {}
    for r in event_rows:
        k = str(r.get("lifecycle_kind", ""))
        lifecycle_counts[k] = lifecycle_counts.get(k, 0) + 1

    active_distribution: Dict[int, int] = {}
    for r in readable_rows:
        n = _safe_int(r.get("active_event_count"), 0)
        active_distribution[n] = active_distribution.get(n, 0) + 1

    meta = {
        "stage": "resonance_event_lifecycle_tracker_v2",
        "state_machine_version": "block002_pipeline_state_machine_v1",
        "states": [
            "EXCITATION_BIRTH",
            "ROOT_HYPOTHESIS",
            "EXCITATION_REBIRTH",
            "CHAIN_STABILIZATION",
            "OWNERSHIP_SPLIT",
            "BOX_TRANSFER",
            "SECONDARY_RESONANCE_TAIL",
        ],
        "inputs": {"frame_notes_csv": args.frame_notes_csv},
        "outputs": {
            "events_csv": args.out_events_csv,
            "event_frames_csv": args.out_event_frames_csv,
            "overlap_csv": args.out_overlap_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "max_gap_frames": args.max_gap_frames,
            "max_same_event_pitch_drift": args.max_same_event_pitch_drift,
            "min_event_frames": args.min_event_frames,
            "min_birth_score": args.min_birth_score,
            "attack_delta": args.attack_delta,
            "reexcite_drop": args.reexcite_drop,
        },
        "result": {
            "input_frame_rows": len(rows),
            "event_frame_rows": len(event_frame_rows),
            "events_kept": len(event_rows),
            "overlap_rows": len(overlap_rows),
            "readable_frames": len(readable_rows),
            "lifecycle_counts": lifecycle_counts,
            "active_event_distribution": active_distribution,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "RESONANCE EVENT LIFECYCLE TRACKER V2",
        "=" * 72,
        f"frame_notes_csv  : {args.frame_notes_csv}",
        "",
        f"input_frame_rows : {len(rows)}",
        f"event_frame_rows : {len(event_frame_rows)}",
        f"events_kept      : {len(event_rows)}",
        f"overlap_rows     : {len(overlap_rows)}",
        "",
        "Lifecycle counts:",
    ]

    for k in sorted(lifecycle_counts):
        txt.append(f"  {k}: {lifecycle_counts[k]}")

    txt.append("")
    txt.append("Active event distribution:")
    for k in sorted(active_distribution):
        txt.append(f"  {k}: {active_distribution[k]}")

    txt.extend([
        "",
        "Principle:",
        "  V2 counts only events updated in the current frame as active.",
        "  A resonance event keeps strict note identity instead of drifting across neighbors.",
        "  State labels now align with the Block002 causal note machine.",
        "  Secondary resonance and box-driven tails are not mislabeled as fresh note birth.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance event lifecycle tracker v2 complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
