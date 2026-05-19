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


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []
    return []


def _json_object(value: Any) -> dict[str, Any]:
    try:
        raw = json.loads(str(value or "{}"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        return {}
    return {}


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


def _role_bonus(role_counts: dict[str, Any]) -> float:
    return (
        1.2 * _safe_float(role_counts.get("birth_backbone"), 0.0)
        + 1.0 * _safe_float(role_counts.get("exact_support"), 0.0)
        + 0.6 * _safe_float(role_counts.get("local_birth"), 0.0)
        - 0.15 * _safe_float(role_counts.get("future_birth"), 0.0)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Apply a local anti-capture rescue to the lower part of the two-stage cloud. "
            "It weakens known pitch-class magnets when ownership support is poor, "
            "and replaces them with stronger ownership-born candidates."
        )
    )
    ap.add_argument("--two-stage-csv", required=True)
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--ownership-depth", type=int, default=14)
    ap.add_argument("--cloud-topk", type=int, default=5)
    ap.add_argument("--magnet-rank-limit", type=int, default=14)
    ap.add_argument("--magnet-keep-ratio", type=float, default=0.26)
    ap.add_argument("--candidate-min-ratio", type=float, default=0.34)
    ap.add_argument("--candidate-role-min", type=float, default=1.2)
    ap.add_argument("--replace-max", type=int, default=1)
    ap.add_argument(
        "--magnet-classes",
        default="3,6,8,A,1,C",
        help="Comma-separated pitch classes that most often capture чужие события.",
    )
    args = ap.parse_args()

    base_rows = _load_csv(Path(args.two_stage_csv))
    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    truth_by_group = _group_truth(midi_rows)

    magnet_classes = {
        str(x).strip()
        for x in str(args.magnet_classes).split(",")
        if str(x).strip()
    }

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    predicted_by_group: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    changed_groups = 0
    replaced_notes = 0
    replacement_counter: Counter[str] = Counter()

    for row in base_rows:
        gid = str(row.get("onset_group", "")).strip()
        base_notes = [_normalize_note(x) for x in _json_list(row.get("final_notes_json", "")) if _normalize_note(x)]
        if not base_notes:
            base_notes = [_normalize_note(x) for x in _json_list(row.get("merged_topk_json", "")) if _normalize_note(x)]
        final_notes = list(base_notes[: int(args.cloud_topk)])
        main_note = _normalize_note(row.get("main_note", ""))

        own_rows = own_by_group.get(gid, [])[: int(args.ownership_depth)]
        top_support = max((_safe_float(r.get("support_weight"), 0.0) for r in own_rows), default=1.0)

        class_sum: dict[str, float] = defaultdict(float)
        class_best_rank: dict[str, int] = {}
        class_best_note: dict[str, str] = {}
        class_best_role_bonus: dict[str, float] = {}
        class_best_support: dict[str, float] = {}

        for crow in own_rows:
            note = _normalize_note(crow.get("note_token", ""))
            if not note:
                continue
            pc = _pitch_class(note)
            rank = _safe_int(crow.get("candidate_rank"), 0)
            support = _safe_float(crow.get("support_weight"), 0.0)
            roles = _json_object(crow.get("role_counts_json", "{}"))
            role_score = _role_bonus(roles)
            class_sum[pc] += support
            if pc not in class_best_rank or rank < class_best_rank[pc]:
                class_best_rank[pc] = rank
                class_best_note[pc] = note
                class_best_role_bonus[pc] = role_score
                class_best_support[pc] = support

        base_classes = {_pitch_class(x) for x in final_notes}

        weak_slots: list[int] = []
        for idx in range(1, len(final_notes)):
            note = final_notes[idx]
            pc = _pitch_class(note)
            if pc not in magnet_classes:
                continue
            ratio = class_sum.get(pc, 0.0) / max(top_support, 1e-9)
            best_rank = class_best_rank.get(pc, 10**9)
            role_score = class_best_role_bonus.get(pc, 0.0)
            if best_rank > int(args.magnet_rank_limit) or (
                ratio < float(args.magnet_keep_ratio) and role_score < float(args.candidate_role_min)
            ):
                weak_slots.append(idx)

        rescue_pool: list[tuple[float, str]] = []
        for pc, note in class_best_note.items():
            if pc in base_classes:
                continue
            ratio = class_sum.get(pc, 0.0) / max(top_support, 1e-9)
            role_score = class_best_role_bonus.get(pc, 0.0)
            best_rank = class_best_rank.get(pc, 10**9)
            if best_rank > int(args.magnet_rank_limit):
                continue
            if ratio < float(args.candidate_min_ratio):
                continue
            if role_score < float(args.candidate_role_min):
                continue
            composite = ratio + 0.12 * role_score + 0.06 * (1.0 / max(best_rank, 1))
            rescue_pool.append((composite, note))
        rescue_pool.sort(key=lambda item: (-item[0], item[1]))

        used_rescues: list[str] = []
        if weak_slots and rescue_pool:
            for idx, (_score, note) in zip(weak_slots[: int(args.replace_max)], rescue_pool[: int(args.replace_max)]):
                note = _normalize_note(note)
                if note and note not in final_notes and idx < len(final_notes):
                    final_notes[idx] = note
                    used_rescues.append(note)
                    replaced_notes += 1
                    replacement_counter[f"{_pitch_class(base_notes[idx])}->{_pitch_class(note)}"] += 1

        if used_rescues:
            changed_groups += 1

        predicted_by_group[gid] = final_notes
        audit_rows.append(
            {
                "onset_group": gid,
                "main_note": main_note,
                "base_notes_json": json.dumps(base_notes[: int(args.cloud_topk)], ensure_ascii=False),
                "final_notes_json": json.dumps(final_notes, ensure_ascii=False),
                "weak_slots_json": json.dumps(weak_slots, ensure_ascii=False),
                "rescued_notes_json": json.dumps(used_rescues, ensure_ascii=False),
                "changed": int(bool(used_rescues)),
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
        "TWO STAGE ANTI-CAPTURE RESCUE",
        "=" * 72,
        f"groups                       : {len(predicted_by_group)}",
        f"changed_groups               : {changed_groups}",
        f"replaced_notes               : {replaced_notes}",
        f"ownership_depth              : {args.ownership_depth}",
        f"cloud_topk                   : {args.cloud_topk}",
        f"magnet_rank_limit            : {args.magnet_rank_limit}",
        f"magnet_keep_ratio            : {args.magnet_keep_ratio}",
        f"candidate_min_ratio          : {args.candidate_min_ratio}",
        f"candidate_role_min           : {args.candidate_role_min}",
        f"replace_max                  : {args.replace_max}",
        f"magnet_classes               : {','.join(sorted(magnet_classes))}",
        "",
        "TOP REPLACEMENTS",
        "-" * 72,
    ]
    for key, count in replacement_counter.most_common(12):
        lines.append(f"{key:28s}: {count}")
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
        "stage": "two_stage_anticapture_rescue",
        "inputs": {
            "two_stage_csv": args.two_stage_csv,
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "ownership_depth": int(args.ownership_depth),
            "cloud_topk": int(args.cloud_topk),
            "magnet_rank_limit": int(args.magnet_rank_limit),
            "magnet_keep_ratio": float(args.magnet_keep_ratio),
            "candidate_min_ratio": float(args.candidate_min_ratio),
            "candidate_role_min": float(args.candidate_role_min),
            "replace_max": int(args.replace_max),
            "magnet_classes": sorted(magnet_classes),
        },
        "result": {
            "changed_groups": changed_groups,
            "replaced_notes": replaced_notes,
            "replacement_counter": dict(replacement_counter),
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
