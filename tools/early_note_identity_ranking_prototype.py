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


def _json_object(value: Any) -> dict[str, Any]:
    try:
        raw = json.loads(str(value or "{}"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        return {}
    return {}


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []
    return []


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


def _group_truth(midi_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in midi_rows:
        gid = str(row.get("onset_group", "")).strip()
        note = _normalize_note(row.get("expected_note_token", row.get("note_token", "")))
        if gid and note:
            out[gid].append(note)
    return out


def _event_level_counts(
    truth_by_group: dict[str, list[str]],
    predicted_by_group: dict[str, list[str]],
    topk: int,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for gid, truth in truth_by_group.items():
        pred = predicted_by_group.get(gid, [])[:topk]
        pred_set = set(pred)
        pred_pc = {_pitch_class(x) for x in pred}
        for note in truth:
            if note in pred_set:
                counter["EXACT"] += 1
            elif _pitch_class(note) in pred_pc:
                counter["PITCHCLASS"] += 1
            elif pred:
                counter["WRONG"] += 1
            else:
                counter["EMPTY"] += 1
    return counter


def _group_level_counts(
    truth_by_group: dict[str, list[str]],
    predicted_by_group: dict[str, list[str]],
    topk: int,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for gid, truth in truth_by_group.items():
        pred = predicted_by_group.get(gid, [])[:topk]
        truth_set = set(truth)
        pred_set = set(pred)
        truth_pc = {_pitch_class(x) for x in truth}
        pred_pc = {_pitch_class(x) for x in pred}
        if truth_set == pred_set and truth_set:
            counter["EXACT_GROUP"] += 1
        elif truth_pc == pred_pc and truth_pc:
            counter["PITCHCLASS_GROUP"] += 1
        elif pred:
            counter["WRONG_GROUP"] += 1
        else:
            counter["EMPTY_GROUP"] += 1
    return counter


def _dsp_by_group(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        gid = str(row.get("onset_group", "")).strip()
        out[gid] = [_normalize_note(x) for x in _json_list(row.get("dsp_notes_json", ""))]
    return out


def _role_score(role_counts: dict[str, int]) -> float:
    score = 0.0
    score += 0.90 * int(role_counts.get("birth_backbone", 0))
    score += 0.75 * int(role_counts.get("local_birth", 0))
    score += 0.60 * int(role_counts.get("exact_support", 0))
    score += 0.18 * int(role_counts.get("pitchclass_support", 0))
    score += 0.22 * int(role_counts.get("field_birth", 0))
    score -= 0.08 * int(role_counts.get("previous_tail", 0))
    score -= 0.12 * int(role_counts.get("future_birth", 0))
    score -= 0.30 * int(role_counts.get("ambient_residue", 0))

    role_kinds = sum(1 for value in role_counts.values() if int(value) > 0)
    if role_kinds > 1:
        score += min(0.05 * (role_kinds - 1), 0.15)

    if int(role_counts.get("local_birth", 0)) > 0 and int(role_counts.get("previous_tail", 0)) > 0:
        score += 0.12

    pure_tail_like = (
        int(role_counts.get("birth_backbone", 0)) == 0
        and int(role_counts.get("local_birth", 0)) == 0
        and int(role_counts.get("exact_support", 0)) == 0
        and int(role_counts.get("field_birth", 0)) == 0
        and (
            int(role_counts.get("previous_tail", 0)) > 0
            or int(role_counts.get("future_birth", 0)) > 0
        )
    )
    if pure_tail_like:
        score -= 0.20

    return score


def _source_score(source_counts: dict[str, int]) -> float:
    score = 0.0
    score += 0.35 * int(source_counts.get("fused_notechain", 0))
    score += 0.10 * int(source_counts.get("field_support", 0))
    score += 0.08 * int(source_counts.get("fused_field", 0))
    score -= 0.10 * int(source_counts.get("fused_residue", 0))
    return score


def _dsp_score(note: str, dsp_notes: list[str], topk: int) -> float:
    norm_dsp = [_normalize_note(x) for x in dsp_notes[:topk]]
    if note in norm_dsp:
        return 0.85 / (norm_dsp.index(note) + 1)

    note_pc = _pitch_class(note)
    for idx, cand in enumerate(norm_dsp, start=1):
        if _pitch_class(cand) == note_pc:
            return 0.35 / idx
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "External prototype of the rewritten early note-identity logic: "
            "keep the shared ownership window, then rerank candidates by role, "
            "stability hints and DSP confirmation."
        )
    )
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--dsp-group-audit-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument(
        "--profile",
        choices=["custom", "main-note", "candidate-cloud"],
        default="custom",
    )
    ap.add_argument("--dsp-topk", type=int, default=6)
    ap.add_argument("--support-weight-scale", type=float, default=0.45)
    ap.add_argument("--role-score-scale", type=float, default=1.0)
    ap.add_argument("--source-score-scale", type=float, default=1.0)
    ap.add_argument("--dsp-score-scale", type=float, default=1.0)
    args = ap.parse_args()

    if args.profile == "main-note":
        args.support_weight_scale = 0.35
        args.dsp_topk = 4
        args.role_score_scale = 0.8
        args.source_score_scale = 1.0
        args.dsp_score_scale = 1.4
    elif args.profile == "candidate-cloud":
        args.support_weight_scale = 0.25
        args.dsp_topk = 4
        args.role_score_scale = 0.8
        args.source_score_scale = 1.0
        args.dsp_score_scale = 1.4

    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    dsp_rows = _load_csv(Path(args.dsp_group_audit_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))

    truth_by_group = _group_truth(midi_rows)
    dsp_by_group = _dsp_by_group(dsp_rows)

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    audit_rows: list[dict[str, Any]] = []
    predicted_by_group: dict[str, list[str]] = {}
    top1_changed = 0
    candidate_count_counter: Counter[str] = Counter()

    for gid, rows in own_by_group.items():
        dsp_notes = dsp_by_group.get(gid, [])
        scored_rows: list[dict[str, Any]] = []
        for row in rows:
            note = _normalize_note(row.get("note_token", ""))
            role_counts = {str(k): _safe_int(v, 0) for k, v in _json_object(row.get("role_counts_json", "")).items()}
            source_counts = {str(k): _safe_int(v, 0) for k, v in _json_object(row.get("source_counts_json", "")).items()}
            base_support = _safe_float(row.get("support_weight"), 0.0) * float(args.support_weight_scale)
            role_part = _role_score(role_counts) * float(args.role_score_scale)
            source_part = _source_score(source_counts) * float(args.source_score_scale)
            dsp_part = _dsp_score(note, dsp_notes, int(args.dsp_topk)) * float(args.dsp_score_scale)
            total = base_support + role_part + source_part + dsp_part
            scored_rows.append(
                {
                    "note_token": note,
                    "old_rank": _safe_int(row.get("candidate_rank"), 0),
                    "base_support": base_support,
                    "role_score": role_part,
                    "source_score": source_part,
                    "dsp_score": dsp_part,
                    "total_score": total,
                    "role_counts": role_counts,
                    "source_counts": source_counts,
                }
            )

        scored_rows.sort(key=lambda r: (-_safe_float(r["total_score"]), r["old_rank"], r["note_token"]))
        predicted = [str(row["note_token"]) for row in scored_rows if str(row["note_token"]).strip()]
        predicted_by_group[gid] = predicted
        candidate_count_counter[str(len(predicted))] += 1

        old_top = _normalize_note(rows[0].get("note_token", "")) if rows else ""
        new_top = predicted[0] if predicted else ""
        if old_top != new_top:
            top1_changed += 1

        for new_rank, row in enumerate(scored_rows, start=1):
            audit_rows.append(
                {
                    "onset_group": gid,
                    "new_rank": new_rank,
                    "old_rank": row["old_rank"],
                    "note_token": row["note_token"],
                    "total_score": f"{_safe_float(row['total_score']):.9f}",
                    "base_support": f"{_safe_float(row['base_support']):.9f}",
                    "role_score": f"{_safe_float(row['role_score']):.9f}",
                    "source_score": f"{_safe_float(row['source_score']):.9f}",
                    "dsp_score": f"{_safe_float(row['dsp_score']):.9f}",
                    "role_counts_json": json.dumps(row["role_counts"], ensure_ascii=False, sort_keys=True),
                    "source_counts_json": json.dumps(row["source_counts"], ensure_ascii=False, sort_keys=True),
                    "dsp_notes_json": json.dumps(dsp_notes[: int(args.dsp_topk)], ensure_ascii=False),
                }
            )

    event_top1 = _event_level_counts(truth_by_group, predicted_by_group, 1)
    event_top3 = _event_level_counts(truth_by_group, predicted_by_group, 3)
    event_top5 = _event_level_counts(truth_by_group, predicted_by_group, 5)
    group_top3 = _group_level_counts(truth_by_group, predicted_by_group, 3)
    group_top5 = _group_level_counts(truth_by_group, predicted_by_group, 5)

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
        "EARLY NOTE IDENTITY RANKING PROTOTYPE",
        "=" * 72,
        f"onset_groups                 : {len(predicted_by_group)}",
        f"groups_with_top1_change      : {top1_changed}",
        f"profile                      : {args.profile}",
        f"dsp_topk                     : {args.dsp_topk}",
        f"support_weight_scale         : {args.support_weight_scale}",
        f"role_score_scale             : {args.role_score_scale}",
        f"source_score_scale           : {args.source_score_scale}",
        f"dsp_score_scale              : {args.dsp_score_scale}",
        "",
        "CANDIDATE COUNT PER GROUP",
        "-" * 72,
    ]
    for key in sorted(candidate_count_counter, key=lambda x: int(x)):
        lines.append(f"{key:>3} candidates : {candidate_count_counter[key]}")
    lines.extend(["", "EVENT TOP-1", "-" * 72])
    for key in sorted(event_top1):
        lines.append(f"{key:28s}: {event_top1[key]}")
    lines.extend(["", "EVENT TOP-3", "-" * 72])
    for key in sorted(event_top3):
        lines.append(f"{key:28s}: {event_top3[key]}")
    lines.extend(["", "EVENT TOP-5", "-" * 72])
    for key in sorted(event_top5):
        lines.append(f"{key:28s}: {event_top5[key]}")
    lines.extend(["", "GROUP TOP-3", "-" * 72])
    for key in sorted(group_top3):
        lines.append(f"{key:28s}: {group_top3[key]}")
    lines.extend(["", "GROUP TOP-5", "-" * 72])
    for key in sorted(group_top5):
        lines.append(f"{key:28s}: {group_top5[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "early_note_identity_ranking_prototype",
        "inputs": {
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "dsp_group_audit_csv": args.dsp_group_audit_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "profile": args.profile,
            "dsp_topk": int(args.dsp_topk),
            "support_weight_scale": float(args.support_weight_scale),
            "role_score_scale": float(args.role_score_scale),
            "source_score_scale": float(args.source_score_scale),
            "dsp_score_scale": float(args.dsp_score_scale),
        },
        "result": {
            "groups_with_top1_change": top1_changed,
            "candidate_count_counter": dict(candidate_count_counter),
            "event_top1": dict(event_top1),
            "event_top3": dict(event_top3),
            "event_top5": dict(event_top5),
            "group_top3": dict(group_top3),
            "group_top5": dict(group_top5),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
