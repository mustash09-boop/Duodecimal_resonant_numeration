# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


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


def _load_roles(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_csv(path)
    out: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        node = _normalize_note(r.get("node", ""))
        if not node:
            continue

        out[node] = {
            "causal_role": str(r.get("causal_role", "")).strip(),
            "out_weight": _safe_float(r.get("out_weight"), 0.0),
            "in_weight": _safe_float(r.get("in_weight"), 0.0),
            "center_score": _safe_float(r.get("center_score"), 0.0),
        }

    return out


def _load_family_features(path: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    rows = _load_csv(path)
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        root = _normalize_note(r.get("family_root_note", ""))
        if not root:
            continue

        out[(frame, root)] = {
            "family_score": _safe_float(r.get("family_score"), 0.0),
            "evidence_count": _safe_int(r.get("evidence_count"), 0),
            "root_micro_count": _safe_int(r.get("root_micro_count"), 0),
            "root_micro_diversity": _safe_int(r.get("root_micro_diversity"), 0),
            "family_members": str(r.get("family_members", "")).strip(),
        }

    return out


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
        rr["score"] = _safe_float(rr.get("passport_ownership_score", rr.get("score", 0.0)), 0.0)

        out.setdefault(frame, []).append(rr)

    for frame in out:
        out[frame].sort(key=lambda x: -_safe_float(x.get("score"), 0.0))

    return out


def _event_state_from_score(
    *,
    score: float,
    prev_score: float,
    age_frames: int,
    role: str,
    attack_delta: float,
) -> str:
    rise = score - prev_score

    if age_frames == 0:
        return "BIRTH"

    if rise >= attack_delta:
        return "RE_EXCITATION"

    if role in {"response_sink", "response_like"}:
        return "RESPONSE_PHASE"

    if score >= 1.10:
        return "ACTIVE_BODY"

    if score >= 0.75:
        return "SUSTAIN_BODY"

    return "DECAY_OR_TRACE"


def _can_continue_event(
    event: Dict[str, Any],
    note: str,
    frame: int,
    max_gap_frames: int,
    max_pitch_drift: float,
) -> bool:
    gap = frame - _safe_int(event.get("last_frame"), 0)
    if gap < 0 or gap > max_gap_frames:
        return False

    dist = _pitch_distance(str(event.get("last_note", "")), note)
    if dist > max_pitch_drift:
        return False

    return True


def _event_match_cost(event: Dict[str, Any], candidate: Dict[str, Any], frame: int) -> float:
    note = candidate["note_token"]
    score = _safe_float(candidate.get("score"), 0.0)
    gap = frame - _safe_int(event.get("last_frame"), 0)
    dist = _pitch_distance(str(event.get("last_note", "")), note)

    # Smaller is better.
    return dist * 1.25 + gap * 0.65 - score * 0.04


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

    if state_counts.get("BIRTH", 0) + state_counts.get("RE_EXCITATION", 0) >= 2:
        lifecycle_kind = "re_excited_event"
    elif state_counts.get("ACTIVE_BODY", 0) >= state_counts.get("DECAY_OR_TRACE", 0):
        lifecycle_kind = "active_sustain_event"
    elif state_counts.get("RESPONSE_PHASE", 0) > 0:
        lifecycle_kind = "response_dominated_event"
    else:
        lifecycle_kind = "decay_trace_event"

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
        "birth_count": state_counts.get("BIRTH", 0),
        "re_excitation_count": state_counts.get("RE_EXCITATION", 0),
        "active_body_count": state_counts.get("ACTIVE_BODY", 0),
        "sustain_body_count": state_counts.get("SUSTAIN_BODY", 0),
        "response_phase_count": state_counts.get("RESPONSE_PHASE", 0),
        "decay_trace_count": state_counts.get("DECAY_OR_TRACE", 0),
        "lifecycle_kind": lifecycle_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track resonance events as lifecycles: birth, active body, sustain, response, decay, overlap."
    )

    ap.add_argument("--frame_notes_csv", required=True)
    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--causal_roles_csv", required=True)

    ap.add_argument("--out_events_csv", required=True)
    ap.add_argument("--out_event_frames_csv", required=True)
    ap.add_argument("--out_overlap_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--max_gap_frames", type=int, default=4)
    ap.add_argument("--max_pitch_drift", type=float, default=3.0)
    ap.add_argument("--min_event_frames", type=int, default=4)
    ap.add_argument("--min_birth_score", type=float, default=0.90)
    ap.add_argument("--attack_delta", type=float, default=0.18)

    args = ap.parse_args()

    frame_note_rows = _load_csv(Path(args.frame_notes_csv))
    frame_notes = _group_frame_notes(frame_note_rows)
    roles = _load_roles(Path(args.causal_roles_csv))
    families = _load_family_features(Path(args.micro_family_csv))

    active_events: Dict[int, Dict[str, Any]] = {}
    finished_events: List[Dict[str, Any]] = []
    event_frame_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    readable_rows: List[Dict[str, Any]] = []

    next_event_id = 1

    all_frames = sorted(frame_notes.keys())

    for frame in all_frames:
        candidates = frame_notes.get(frame, [])
        used_events: Set[int] = set()

        active_ids_at_frame_before = sorted(active_events.keys())

        for cand in candidates:
            note = cand["note_token"]
            score = _safe_float(cand.get("score"), 0.0)

            best_event_id = None
            best_cost = 999999.0

            for eid, ev in active_events.items():
                if eid in used_events:
                    continue

                if not _can_continue_event(
                    ev,
                    note,
                    frame,
                    args.max_gap_frames,
                    args.max_pitch_drift,
                ):
                    continue

                cost = _event_match_cost(ev, cand, frame)

                if cost < best_cost:
                    best_cost = cost
                    best_event_id = eid

            if best_event_id is None:
                if score < args.min_birth_score:
                    continue

                eid = next_event_id
                next_event_id += 1

                role = roles.get(note, {})
                family = families.get((frame, note), {})

                ev = {
                    "event_id": eid,
                    "primary_note": note,
                    "birth_frame": frame,
                    "last_frame": frame,
                    "last_note": note,
                    "frames": [frame],
                    "notes": [note],
                    "scores": [score],
                    "states": ["BIRTH"],
                    "roles": [role.get("causal_role", "")],
                    "family_scores": [_safe_float(family.get("family_score"), 0.0)],
                }

                active_events[eid] = ev
                used_events.add(eid)

                event_frame_rows.append({
                    "event_id": eid,
                    "frame_index": frame,
                    "note_token": note,
                    "event_state": "BIRTH",
                    "score": f"{score:.9f}",
                    "causal_role": role.get("causal_role", ""),
                    "family_score": f"{_safe_float(family.get('family_score'), 0.0):.9f}",
                    "evidence_count": family.get("evidence_count", ""),
                    "root_micro_count": family.get("root_micro_count", ""),
                    "root_micro_diversity": family.get("root_micro_diversity", ""),
                })

            else:
                eid = best_event_id
                ev = active_events[eid]
                prev_score = ev["scores"][-1] if ev["scores"] else 0.0
                age = frame - _safe_int(ev.get("birth_frame"), frame)

                role = roles.get(note, {})
                family = families.get((frame, note), {})

                state = _event_state_from_score(
                    score=score,
                    prev_score=prev_score,
                    age_frames=age,
                    role=str(role.get("causal_role", "")),
                    attack_delta=args.attack_delta,
                )

                ev["last_frame"] = frame
                ev["last_note"] = note
                ev["frames"].append(frame)
                ev["notes"].append(note)
                ev["scores"].append(score)
                ev["states"].append(state)
                ev["roles"].append(role.get("causal_role", ""))
                ev["family_scores"].append(_safe_float(family.get("family_score"), 0.0))

                used_events.add(eid)

                event_frame_rows.append({
                    "event_id": eid,
                    "frame_index": frame,
                    "note_token": note,
                    "event_state": state,
                    "score": f"{score:.9f}",
                    "causal_role": role.get("causal_role", ""),
                    "family_score": f"{_safe_float(family.get('family_score'), 0.0):.9f}",
                    "evidence_count": family.get("evidence_count", ""),
                    "root_micro_count": family.get("root_micro_count", ""),
                    "root_micro_diversity": family.get("root_micro_diversity", ""),
                })

        # Overlap graph: events alive at the same frame.
        active_ids_now = sorted(active_events.keys())
        for i, a in enumerate(active_ids_now):
            for b in active_ids_now[i + 1:]:
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
            "active_event_count": len(active_ids_now),
            "active_events": " | ".join(
                f"E{eid}:{active_events[eid].get('last_note')}:{active_events[eid]['scores'][-1]:.3f}:{active_events[eid]['states'][-1]}"
                for eid in active_ids_now
            ),
        })

        # Close stale events.
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

    event_rows.sort(
        key=lambda r: (
            _safe_int(r["birth_frame"]),
            _safe_int(r["event_id"]),
        )
    )

    out_events = Path(args.out_events_csv)
    out_event_frames = Path(args.out_event_frames_csv)
    out_overlap = Path(args.out_overlap_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_events.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "event_id",
        "candidate_note",
        "birth_frame",
        "end_frame",
        "duration_frames",
        "frame_count",
        "mean_score",
        "max_score",
        "birth_score",
        "final_score",
        "attack_peak_frame",
        "unique_note_count",
        "note_path",
        "state_path",
        "birth_count",
        "re_excitation_count",
        "active_body_count",
        "sustain_body_count",
        "response_phase_count",
        "decay_trace_count",
        "lifecycle_kind",
    ]

    with out_events.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        w.writerows(event_rows)

    frame_fields = [
        "event_id",
        "frame_index",
        "note_token",
        "event_state",
        "score",
        "causal_role",
        "family_score",
        "evidence_count",
        "root_micro_count",
        "root_micro_diversity",
    ]

    with out_event_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(event_frame_rows)

    overlap_fields = [
        "frame_index",
        "event_a",
        "event_b",
        "note_a",
        "note_b",
        "pitch_distance",
    ]

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
        "stage": "resonance_event_lifecycle_tracker",
        "inputs": {
            "frame_notes_csv": args.frame_notes_csv,
            "micro_family_csv": args.micro_family_csv,
            "causal_roles_csv": args.causal_roles_csv,
        },
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
            "max_pitch_drift": args.max_pitch_drift,
            "min_event_frames": args.min_event_frames,
            "min_birth_score": args.min_birth_score,
            "attack_delta": args.attack_delta,
        },
        "result": {
            "input_frame_rows": len(frame_note_rows),
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
        "RESONANCE EVENT LIFECYCLE TRACKER",
        "=" * 72,
        f"frame_notes_csv  : {args.frame_notes_csv}",
        f"micro_family_csv : {args.micro_family_csv}",
        f"causal_roles_csv : {args.causal_roles_csv}",
        "",
        f"input_frame_rows : {len(frame_note_rows)}",
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
        "  Track resonance as events with birth, body, sustain, response and decay.",
        "  Overlapping sound is not flattened into framewise top-N notes;",
        "  each event keeps its own lifecycle and can later be compared to passports.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance event lifecycle tracker complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()