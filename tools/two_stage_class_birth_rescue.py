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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rescue missing pitch classes in the lower part of the two-stage cloud, "
            "using ownership-window pitch-class aggregates without touching the main note."
        )
    )
    ap.add_argument("--two-stage-csv", required=True)
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--ownership-depth", type=int, default=12)
    ap.add_argument("--cloud-topk", type=int, default=5)
    ap.add_argument("--rescue-max", type=int, default=2)
    ap.add_argument("--class-ratio-threshold", type=float, default=0.38)
    ap.add_argument("--best-rank-threshold", type=int, default=10)
    args = ap.parse_args()

    two_stage_rows = _load_csv(Path(args.two_stage_csv))
    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    truth_by_group = _group_truth(midi_rows)

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    predicted_by_group: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    rescued_groups = 0
    rescued_classes = 0

    for row in two_stage_rows:
        gid = str(row.get("onset_group", "")).strip()
        merged = [_normalize_note(x) for x in _json_list(row.get("merged_notes_json", "")) if _normalize_note(x)]
        topk_notes = merged[: int(args.cloud_topk)]
        topk_classes = {_pitch_class(x) for x in topk_notes}

        own_rows = own_by_group.get(gid, [])[: int(args.ownership_depth)]
        class_sum: dict[str, float] = defaultdict(float)
        class_best_rank: dict[str, int] = {}
        class_best_note: dict[str, str] = {}
        class_best_roles: dict[str, dict[str, Any]] = {}
        top_support = max((_safe_float(r.get("support_weight"), 0.0) for r in own_rows), default=1.0)

        for crow in own_rows:
            note = _normalize_note(crow.get("note_token", ""))
            if not note:
                continue
            pc = _pitch_class(note)
            rank = _safe_int(crow.get("candidate_rank"), 0)
            support = _safe_float(crow.get("support_weight"), 0.0)
            class_sum[pc] += support
            if pc not in class_best_rank or rank < class_best_rank[pc]:
                class_best_rank[pc] = rank
                class_best_note[pc] = note
                class_best_roles[pc] = _json_object(crow.get("role_counts_json", ""))

        rescue_pool: list[tuple[float, str, str]] = []
        for pc, score in class_sum.items():
            if pc in topk_classes:
                continue
            best_rank = class_best_rank.get(pc, 10**9)
            if best_rank > int(args.best_rank_threshold):
                continue
            if score / max(top_support, 1e-9) < float(args.class_ratio_threshold):
                continue
            rescue_pool.append((score, pc, class_best_note.get(pc, "")))

        rescue_pool.sort(key=lambda item: (-item[0], item[1], item[2]))
        rescue_notes = [note for _score, _pc, note in rescue_pool[: int(args.rescue_max)] if note]

        rescued = False
        if rescue_notes:
            rescued = True
            rescued_groups += 1
            rescued_classes += len(rescue_notes)

        final_notes = list(topk_notes)
        for note in rescue_notes:
            note = _normalize_note(note)
            if not note or note in final_notes:
                continue
            replaced = False
            for idx in range(len(final_notes) - 1, 0, -1):
                current = _normalize_note(final_notes[idx])
                if current == note:
                    replaced = True
                    break
                if _pitch_class(current) != _pitch_class(note):
                    final_notes[idx] = note
                    replaced = True
                    break
            if not replaced and len(final_notes) < int(args.cloud_topk):
                final_notes.append(note)
        predicted_by_group[gid] = final_notes

        audit_rows.append(
            {
                "onset_group": gid,
                "main_note": _normalize_note(row.get("main_note", "")),
                "merged_topk_json": json.dumps(topk_notes, ensure_ascii=False),
                "rescued_notes_json": json.dumps(rescue_notes, ensure_ascii=False),
                "final_notes_json": json.dumps(final_notes, ensure_ascii=False),
                "rescued": int(rescued),
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
        "TWO STAGE CLASS BIRTH RESCUE",
        "=" * 72,
        f"groups                       : {len(predicted_by_group)}",
        f"rescued_groups               : {rescued_groups}",
        f"rescued_classes              : {rescued_classes}",
        f"ownership_depth              : {args.ownership_depth}",
        f"cloud_topk                   : {args.cloud_topk}",
        f"rescue_max                   : {args.rescue_max}",
        f"class_ratio_threshold        : {args.class_ratio_threshold}",
        f"best_rank_threshold          : {args.best_rank_threshold}",
        "",
        "EVENT TOP-1",
        "-" * 72,
    ]
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
        "stage": "two_stage_class_birth_rescue",
        "inputs": {
            "two_stage_csv": args.two_stage_csv,
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "ownership_depth": int(args.ownership_depth),
            "cloud_topk": int(args.cloud_topk),
            "rescue_max": int(args.rescue_max),
            "class_ratio_threshold": float(args.class_ratio_threshold),
            "best_rank_threshold": int(args.best_rank_threshold),
        },
        "result": {
            "rescued_groups": rescued_groups,
            "rescued_classes": rescued_classes,
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
