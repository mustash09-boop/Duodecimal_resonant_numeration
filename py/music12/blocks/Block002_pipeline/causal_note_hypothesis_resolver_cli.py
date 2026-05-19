# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set


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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_note(s: str) -> str:
    s = str(s or "").strip()

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


def _tokens(raw: str) -> List[str]:
    return [_normalize_note(x) for x in str(raw or "").split() if x.strip()]


def _longest_run(notes: Sequence[str], target: str) -> int:
    best = 0
    current = 0

    for note in notes:
        if note == target:
            current += 1
            if current > best:
                best = current
        else:
            current = 0

    return best


def _build_frame_note_index(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        frame = _safe_int(row.get("frame_index"), 0)
        out[frame].append(row)

    return out


def _build_family_index(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        frame = _safe_int(row.get("frame_index"), 0)
        note = _normalize_note(
            row.get("family_root_note_micro")
            or row.get("family_root_note")
            or row.get("family_root_note_coarse")
        )

        if not note:
            continue

        out[frame][note].append(row)

    return out


def _role_bonus(role: str) -> float:
    return {
        "dominant_exciter": 0.34,
        "exciter_like": 0.26,
        "bridge_exciter_like": 0.17,
        "bridge_response_like": 0.04,
        "response_like": -0.06,
        "bridge_resonator": -0.10,
        "response_sink": -0.18,
        "untyped": -0.02,
    }.get(str(role or "").strip(), -0.02)


def _score_candidate(
    *,
    candidate: str,
    note_path: Sequence[str],
    start_frame: int,
    end_frame: int,
    frame_index: Dict[int, List[Dict[str, Any]]],
    family_index: Dict[int, Dict[str, List[Dict[str, Any]]]],
    role_map: Dict[str, Dict[str, Any]],
    center_map: Dict[str, Dict[str, Any]],
    top_center_note: str,
) -> Dict[str, Any]:
    total_len = max(len(note_path), 1)
    note_counter = Counter(note_path)

    occ = note_counter.get(candidate, 0)
    occ_ratio = occ / total_len
    longest = _longest_run(note_path, candidate)
    longest_ratio = longest / total_len

    first_idx = note_path.index(candidate) if candidate in note_path else total_len
    last_idx = (total_len - 1 - list(reversed(note_path)).index(candidate)) if candidate in note_path else -1

    candidate_frame_hits = 0
    exciter_hits = 0
    companion_hits = 0
    center_support_hits = 0
    same_degree_hits = 0

    family_scores: List[float] = []
    family_root_counts: List[float] = []
    family_diversities: List[float] = []

    for frame in range(start_frame, end_frame + 1):
        frame_rows = frame_index.get(frame, [])

        for row in frame_rows:
            note = _normalize_note(row.get("note_token", ""))
            if not note:
                continue

            if note == candidate:
                candidate_frame_hits += 1
                kind = str(row.get("candidate_kind", "")).strip()
                if kind == "STRUCTURAL_COMPANION":
                    companion_hits += 1
                else:
                    exciter_hits += 1
                if _safe_float(row.get("center_score"), 0.0) > 0.0:
                    center_support_hits += 1
            elif _degree(note) and _degree(note) == _degree(candidate):
                same_degree_hits += 1

        for fam in family_index.get(frame, {}).get(candidate, []):
            family_scores.append(_safe_float(fam.get("family_score"), 0.0))
            family_root_counts.append(_safe_float(fam.get("root_micro_count"), 0.0))
            family_diversities.append(_safe_float(fam.get("root_micro_diversity"), 0.0))

    frame_hit_ratio = candidate_frame_hits / max(end_frame - start_frame + 1, 1)
    exciter_ratio = exciter_hits / max(candidate_frame_hits, 1)
    companion_ratio = companion_hits / max(candidate_frame_hits, 1)

    role_row = role_map.get(candidate, {})
    center_row = center_map.get(candidate, {})
    center_score = _safe_float(center_row.get("center_score"), 0.0)
    role = str(role_row.get("causal_role") or center_row.get("causal_role") or "untyped").strip()

    mean_family_score = sum(family_scores) / max(len(family_scores), 1)
    mean_root_count = sum(family_root_counts) / max(len(family_root_counts), 1)
    mean_diversity = sum(family_diversities) / max(len(family_diversities), 1)

    birth_bonus = 0.0
    if first_idx <= max(2, total_len // 4) and exciter_hits > 0:
        birth_bonus += 0.14
    if first_idx <= max(1, total_len // 6) and center_support_hits > 0:
        birth_bonus += 0.06

    late_penalty = 0.0
    if first_idx > (total_len // 2) and role in {
        "bridge_resonator",
        "bridge_response_like",
        "response_like",
        "response_sink",
    }:
        late_penalty += 0.14

    unstable_penalty = 0.0
    if longest <= 2 and occ_ratio < 0.20 and len(set(note_path)) >= 4:
        unstable_penalty += 0.16

    center_bias_bonus = 0.0
    if candidate == top_center_note:
        center_bias_bonus += 0.08

    same_degree_bleed_penalty = min(same_degree_hits / max(end_frame - start_frame + 1, 1), 1.0) * 0.08

    score = 0.0
    score += occ_ratio * 0.58
    score += longest_ratio * 0.32
    score += min(frame_hit_ratio, 1.0) * 0.28
    score += min(exciter_ratio, 1.0) * 0.34
    score -= min(companion_ratio, 1.0) * 0.18

    score += min(center_score / 0.20, 1.0) * 0.24
    score += _role_bonus(role)

    score += min(mean_family_score / 12.0, 1.0) * 0.18
    score += min(mean_root_count / 24.0, 1.0) * 0.10
    score += min(mean_diversity / 18.0, 1.0) * 0.08

    if note_path and candidate == note_path[0]:
        score += 0.06
    if note_path and candidate == note_path[-1]:
        score += 0.03

    score += birth_bonus
    score += center_bias_bonus
    score -= late_penalty
    score -= unstable_penalty
    score -= same_degree_bleed_penalty

    return {
        "candidate_note": candidate,
        "hypothesis_score": max(score, 0.0),
        "occ_ratio": occ_ratio,
        "longest_ratio": longest_ratio,
        "frame_hit_ratio": frame_hit_ratio,
        "exciter_ratio": exciter_ratio,
        "companion_ratio": companion_ratio,
        "center_score": center_score,
        "causal_role": role,
        "mean_family_score": mean_family_score,
        "mean_root_count": mean_root_count,
        "mean_diversity": mean_diversity,
        "birth_bonus": birth_bonus,
        "late_penalty": late_penalty,
        "unstable_penalty": unstable_penalty,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve longer-lived causal note hypotheses from stabilized micro voices."
    )

    ap.add_argument("--stable_voices_csv", required=True)
    ap.add_argument("--frame_notes_csv", required=True)
    ap.add_argument("--causal_roles_csv", required=True)
    ap.add_argument("--causal_centers_csv", required=True)
    ap.add_argument("--micro_family_csv", required=True)

    ap.add_argument("--out_hypotheses_csv", required=True)
    ap.add_argument("--out_resolved_events_csv", required=True)
    ap.add_argument("--out_frame_notes_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_hypothesis_score", type=float, default=0.55)
    ap.add_argument("--min_confidence", type=float, default=0.44)
    ap.add_argument("--min_local_candidate_probability", type=float, default=0.24)
    ap.add_argument("--min_proto_probability", type=float, default=0.22)
    ap.add_argument("--min_proto_score", type=float, default=0.68)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    stable_rows = _load_csv(Path(args.stable_voices_csv))
    frame_rows = _load_csv(Path(args.frame_notes_csv))
    role_rows = _load_csv(Path(args.causal_roles_csv))
    center_rows = _load_csv(Path(args.causal_centers_csv))
    family_rows = _load_csv(Path(args.micro_family_csv))

    frame_index = _build_frame_note_index(frame_rows)
    family_index = _build_family_index(family_rows)

    role_map = {_normalize_note(r.get("node", "")): r for r in role_rows}
    center_map = {_normalize_note(r.get("node", "")): r for r in center_rows}

    hypothesis_rows: List[Dict[str, Any]] = []
    resolved_rows: List[Dict[str, Any]] = []
    resolved_frame_rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = defaultdict(int)

    for stable in stable_rows:
        entity_id = str(stable.get("stable_voice_id", "")).strip()
        start_frame = _safe_int(stable.get("start_frame"), 0)
        end_frame = _safe_int(stable.get("end_frame"), 0)
        note_path = _tokens(stable.get("note_path", ""))

        if not note_path:
            note_path = [
                _normalize_note(stable.get("start_note", "")),
                _normalize_note(stable.get("end_note", "")),
            ]
            note_path = [x for x in note_path if x]

        candidates: Set[str] = {x for x in note_path if x}

        if not candidates:
            status_counts["NO_CANDIDATES"] += 1
            continue

        top_center_note = ""
        top_center_score = -1.0
        for note in candidates:
            center_score = _safe_float(center_map.get(note, {}).get("center_score"), 0.0)
            if center_score > top_center_score:
                top_center_score = center_score
                top_center_note = note

        scored = [
            _score_candidate(
                candidate=note,
                note_path=note_path,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_index=frame_index,
                family_index=family_index,
                role_map=role_map,
                center_map=center_map,
                top_center_note=top_center_note,
            )
            for note in sorted(candidates)
        ]

        scored = [row for row in scored if row["hypothesis_score"] >= args.min_hypothesis_score]
        scored.sort(key=lambda x: (-x["hypothesis_score"], x["candidate_note"]))
        scored = scored[: args.top_k]

        total_score = sum(row["hypothesis_score"] for row in scored)
        best = scored[0] if scored else None
        confidence = (
            best["hypothesis_score"] / max(total_score, 1e-9)
            if best is not None else 0.0
        )

        if not scored:
            status = "NO_HYPOTHESIS"
        elif confidence >= args.min_confidence:
            status = "RESOLVED"
        else:
            status = "AMBIGUOUS"

        status_counts[status] += 1

        for rank, row in enumerate(scored, start=1):
            probability = row["hypothesis_score"] / max(total_score, 1e-9)
            hypothesis_tier = "PRIMARY" if rank == 1 else "PROTO" if (
                probability >= args.min_proto_probability
                and row["hypothesis_score"] >= args.min_proto_score
                and (
                    row["exciter_ratio"] >= 0.20
                    or row["center_score"] >= 0.01
                    or row["birth_bonus"] > 0.0
                )
            ) else "ALTERNATIVE"
            hypothesis_rows.append({
                "entity_id": entity_id,
                "rank": rank,
                "candidate_note": row["candidate_note"],
                "hypothesis_score": f"{row['hypothesis_score']:.9f}",
                "hypothesis_probability": f"{probability:.9f}",
                "hypothesis_tier": hypothesis_tier,
                "causal_role": row["causal_role"],
                "occ_ratio": f"{row['occ_ratio']:.9f}",
                "longest_ratio": f"{row['longest_ratio']:.9f}",
                "frame_hit_ratio": f"{row['frame_hit_ratio']:.9f}",
                "exciter_ratio": f"{row['exciter_ratio']:.9f}",
                "companion_ratio": f"{row['companion_ratio']:.9f}",
                "center_score": f"{row['center_score']:.9f}",
                "birth_bonus": f"{row['birth_bonus']:.9f}",
                "late_penalty": f"{row['late_penalty']:.9f}",
                "unstable_penalty": f"{row['unstable_penalty']:.9f}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": max(end_frame - start_frame + 1, 0),
            })

        best_note = best["candidate_note"] if best else ""
        best_score = best["hypothesis_score"] if best else 0.0
        candidate_probability_map = {
            row["candidate_note"]: (row["hypothesis_score"] / max(total_score, 1e-9))
            for row in scored
        }
        scored_by_note = {row["candidate_note"]: row for row in scored}
        proto_candidates = [
            row["candidate_note"]
            for row in scored[1:]
            if candidate_probability_map.get(row["candidate_note"], 0.0) >= args.min_proto_probability
            and row["hypothesis_score"] >= args.min_proto_score
            and (
                row["exciter_ratio"] >= 0.20
                or row["center_score"] >= 0.01
                or row["birth_bonus"] > 0.0
            )
            and row["causal_role"] not in {"bridge_resonator", "response_sink"}
        ]
        alternatives = " | ".join(
            f"{row['candidate_note']}:{row['hypothesis_score']:.3f}"
            for row in scored[: args.top_k]
        )

        resolved_rows.append({
            "entity_id": entity_id,
            "resolved_note": best_note,
            "resolution_status": status,
            "resolution_confidence": f"{confidence:.9f}",
            "best_hypothesis_score": f"{best_score:.9f}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": max(end_frame - start_frame + 1, 0),
            "unique_note_count": len(set(note_path)),
            "alternatives": alternatives,
        })

        if status != "RESOLVED" or not best_note:
            continue

        for frame in range(start_frame, end_frame + 1):
            path_idx = frame - start_frame
            local_token = note_path[path_idx] if 0 <= path_idx < len(note_path) else ""
            frame_candidates = frame_index.get(frame, [])

            emitted_note = ""
            frame_confidence = 0.0

            frame_top_candidates = []
            for row in frame_candidates:
                note = _normalize_note(row.get("note_token", ""))
                if note and note in candidate_probability_map:
                    frame_top_candidates.append(note)

            if frame_top_candidates:
                emitted_note = max(
                    sorted(set(frame_top_candidates)),
                    key=lambda note: candidate_probability_map.get(note, 0.0),
                )
                local_probability = candidate_probability_map.get(emitted_note, 0.0)
                frame_confidence = min(1.0, confidence * 0.50 + local_probability * 0.70)

            elif (
                local_token
                and local_token in candidate_probability_map
                and candidate_probability_map.get(local_token, 0.0) >= args.min_local_candidate_probability
            ):
                emitted_note = local_token
                local_probability = candidate_probability_map.get(local_token, 0.0)
                frame_confidence = min(1.0, confidence * 0.45 + local_probability * 0.75)

            else:
                exact_local_match = local_token == best_note
                exact_frame_match = any(
                    _normalize_note(row.get("note_token", "")) == best_note
                    for row in frame_candidates
                )
                same_degree_local = (
                    bool(local_token)
                    and _degree(local_token)
                    and _degree(local_token) == _degree(best_note)
                )

                if exact_local_match or exact_frame_match:
                    emitted_note = best_note
                    frame_confidence = confidence
                elif same_degree_local and confidence >= (args.min_confidence + 0.08):
                    emitted_note = local_token or best_note
                    frame_confidence = confidence * 0.85
                else:
                    continue

            resolved_frame_rows.append({
                "frame_index": frame,
                "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
                "entity_id": entity_id,
                "note_token": emitted_note,
                "note_confidence": f"{frame_confidence:.9f}",
                "resolution_status": status,
                "emission_kind": "PRIMARY",
                "alternatives": alternatives,
            })

            for proto_note in proto_candidates:
                if proto_note == emitted_note:
                    continue

                proto_probability = candidate_probability_map.get(proto_note, 0.0)
                proto_row = scored_by_note.get(proto_note, {})
                frame_support = False

                if proto_note == local_token:
                    frame_support = True
                elif proto_note in frame_top_candidates:
                    frame_support = True
                elif (
                    _degree(local_token)
                    and _degree(proto_note)
                    and _degree(local_token) == _degree(proto_note)
                    and proto_probability >= (args.min_proto_probability + 0.08)
                ):
                    frame_support = True

                if not frame_support:
                    continue

                proto_confidence = min(
                    1.0,
                    proto_probability * 0.70
                    + confidence * 0.20
                    + min(_safe_float(proto_row.get("exciter_ratio"), 0.0), 1.0) * 0.18,
                )

                resolved_frame_rows.append({
                    "frame_index": frame,
                    "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
                    "entity_id": entity_id,
                    "note_token": proto_note,
                    "note_confidence": f"{proto_confidence:.9f}",
                    "resolution_status": "PROTO_NOTE",
                    "emission_kind": "PROTO",
                    "alternatives": alternatives,
                })

    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in resolved_frame_rows:
        by_frame[_safe_int(row["frame_index"])].append(row)

    readable_rows = []
    active_distribution: Dict[int, int] = defaultdict(int)

    for frame in sorted(by_frame):
        items = sorted(
            by_frame[frame],
            key=lambda x: -_safe_float(x.get("note_confidence"), 0.0),
        )
        readable_rows.append({
            "frame_index": frame,
            "time_sec": f"{(frame / max(args.fps, 1e-9)):.9f}",
            "active_note_count": len(items),
            "notes": " | ".join(
                f"{row['note_token']}:{_safe_float(row['note_confidence']):.3f}[SV{row['entity_id']}]"
                for row in items[:12]
            ),
        })
        active_distribution[len(items)] += 1

    out_hyp = Path(args.out_hypotheses_csv)
    out_res = Path(args.out_resolved_events_csv)
    out_frame = Path(args.out_frame_notes_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_hyp.parent.mkdir(parents=True, exist_ok=True)

    with out_hyp.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "entity_id",
            "rank",
            "candidate_note",
            "hypothesis_score",
            "hypothesis_probability",
            "hypothesis_tier",
            "causal_role",
            "occ_ratio",
            "longest_ratio",
            "frame_hit_ratio",
            "exciter_ratio",
            "companion_ratio",
            "center_score",
            "birth_bonus",
            "late_penalty",
            "unstable_penalty",
            "start_frame",
            "end_frame",
            "duration_frames",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(hypothesis_rows)

    with out_res.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "entity_id",
            "resolved_note",
            "resolution_status",
            "resolution_confidence",
            "best_hypothesis_score",
            "start_frame",
            "end_frame",
            "duration_frames",
            "unique_note_count",
            "alternatives",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(resolved_rows)

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "frame_index",
            "time_sec",
            "entity_id",
            "note_token",
            "note_confidence",
            "resolution_status",
            "emission_kind",
            "alternatives",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(resolved_frame_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        fields = ["frame_index", "time_sec", "active_note_count", "notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "causal_note_hypothesis_resolver_micro_voices",
        "inputs": {
            "stable_voices_csv": args.stable_voices_csv,
            "frame_notes_csv": args.frame_notes_csv,
            "causal_roles_csv": args.causal_roles_csv,
            "causal_centers_csv": args.causal_centers_csv,
            "micro_family_csv": args.micro_family_csv,
        },
        "outputs": {
            "hypotheses_csv": args.out_hypotheses_csv,
            "resolved_events_csv": args.out_resolved_events_csv,
            "frame_notes_csv": args.out_frame_notes_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "result": {
            "stable_voice_entities": len(stable_rows),
            "hypotheses": len(hypothesis_rows),
            "resolved_events": len(resolved_rows),
            "resolved_frame_rows": len(resolved_frame_rows),
            "proto_frame_rows": sum(
                1
                for row in resolved_frame_rows
                if str(row.get("emission_kind", "")).strip() == "PROTO"
            ),
            "readable_frames": len(readable_rows),
            "status_counts": dict(status_counts),
            "active_distribution": dict(active_distribution),
        },
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_lines = [
        "CAUSAL NOTE HYPOTHESIS RESOLVER (MICRO VOICES)",
        "=" * 72,
        f"stable_voice_entities : {len(stable_rows)}",
        f"hypotheses            : {len(hypothesis_rows)}",
        f"resolved_events       : {len(resolved_rows)}",
        f"resolved_frame_rows   : {len(resolved_frame_rows)}",
        f"proto_frame_rows      : {sum(1 for row in resolved_frame_rows if str(row.get('emission_kind', '')).strip() == 'PROTO')}",
        "",
        "Principle:",
        "  A note is accepted as a longer-lived causal hypothesis, not only as a framewise selection.",
        "  Early exciter support and sustained local runs outrank late bridge-like resonance.",
        "  Proto-note rows preserve weaker secondary births without promoting them to full note identity.",
        "",
        "Status counts:",
    ]

    for key in sorted(status_counts):
        summary_lines.append(f"  {key}: {status_counts[key]}")

    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("causal note hypothesis resolver for micro voices complete")
    print(json.dumps(meta["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
