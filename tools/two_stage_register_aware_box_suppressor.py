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


def _register_band(note: str) -> str:
    octave = str(_normalize_note(note).split(".", 1)[0])
    if octave in {"6", "7"}:
        return "low_zone"
    if octave in {"8", "9"}:
        return "mid_zone"
    if octave in {"A", "B", "C", "D", "E", "F"}:
        return "high_zone"
    return "other_zone"


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


def _role_bonus(role_counts: dict[str, Any]) -> float:
    return (
        1.2 * _safe_float(role_counts.get("birth_backbone"), 0.0)
        + 1.0 * _safe_float(role_counts.get("exact_support"), 0.0)
        + 0.6 * _safe_float(role_counts.get("local_birth"), 0.0)
        - 0.15 * _safe_float(role_counts.get("future_birth"), 0.0)
    )


def _profile_params(profile: str) -> dict[str, dict[str, float]]:
    profile = str(profile).strip().lower()
    if profile == "string_like":
        return {
            "keep_ratio": {"low_zone": 0.22, "mid_zone": 0.18, "high_zone": 0.26, "other_zone": 0.20},
            "min_ratio": {"low_zone": 0.12, "mid_zone": 0.10, "high_zone": 0.14, "other_zone": 0.11},
            "role_min": {"low_zone": 0.45, "mid_zone": 0.45, "high_zone": 0.55, "other_zone": 0.45},
            "replace_penalty": {"low_zone": 0.02, "mid_zone": 0.00, "high_zone": 0.04, "other_zone": 0.01},
        }
    if profile == "generic":
        return {
            "keep_ratio": {"low_zone": 0.30, "mid_zone": 0.22, "high_zone": 0.30, "other_zone": 0.24},
            "min_ratio": {"low_zone": 0.13, "mid_zone": 0.11, "high_zone": 0.13, "other_zone": 0.12},
            "role_min": {"low_zone": 0.50, "mid_zone": 0.50, "high_zone": 0.60, "other_zone": 0.50},
            "replace_penalty": {"low_zone": 0.03, "mid_zone": 0.00, "high_zone": 0.03, "other_zone": 0.01},
        }
    return {
        "keep_ratio": {"low_zone": 0.34, "mid_zone": 0.20, "high_zone": 0.34, "other_zone": 0.24},
        "min_ratio": {"low_zone": 0.14, "mid_zone": 0.10, "high_zone": 0.14, "other_zone": 0.12},
        "role_min": {"low_zone": 0.55, "mid_zone": 0.45, "high_zone": 0.60, "other_zone": 0.50},
        "replace_penalty": {"low_zone": 0.04, "mid_zone": 0.00, "high_zone": 0.05, "other_zone": 0.02},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "External register-aware background suppressor. "
            "It weakens box/body-dominated candidate classes differently in low, mid, and high zones "
            "without touching the core note-recognition pipeline."
        )
    )
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--box-audit-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--profile", default="piano_like", choices=["piano_like", "string_like", "generic"])
    ap.add_argument("--ownership-depth", type=int, default=14)
    ap.add_argument("--cloud-topk", type=int, default=5)
    ap.add_argument("--replace-max", type=int, default=1)
    ap.add_argument("--box-ratio-threshold", type=float, default=1.4)
    ap.add_argument("--box-joint-min", type=int, default=20)
    ap.add_argument("--exclude-classes", default="C")
    args = ap.parse_args()

    params = _profile_params(args.profile)

    base_rows = _load_csv(Path(args.base_csv))
    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    box_rows = _load_csv(Path(args.box_audit_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    truth_by_group = _group_truth(midi_rows)

    excluded = {x.strip() for x in str(args.exclude_classes).split(",") if x.strip()}

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    box_by_class: dict[str, dict[str, Any]] = {}
    for row in box_rows:
        pc = str(row.get("pitch_class", "")).strip()
        if not pc or pc in excluded:
            continue
        ratio = _safe_float(row.get("background_to_capture_ratio"), 0.0)
        joint = _safe_int(row.get("joint_background_groups"), 0)
        if ratio >= float(args.box_ratio_threshold) and joint >= int(args.box_joint_min):
            box_by_class[pc] = row

    predicted_by_group: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    changed_groups = 0
    replaced_notes = 0
    replacement_counter: Counter[str] = Counter()

    for row in base_rows:
        gid = str(row.get("onset_group", "")).strip()
        base_notes = [_normalize_note(x) for x in _json_list(row.get("final_notes_json", "")) if _normalize_note(x)]
        if not base_notes:
            base_notes = [_normalize_note(x) for x in _json_list(row.get("base_notes_json", "")) if _normalize_note(x)]
        final_notes = list(base_notes[: int(args.cloud_topk)])
        main_note = _normalize_note(row.get("main_note", ""))

        own_rows = own_by_group.get(gid, [])[: int(args.ownership_depth)]
        top_support = max((_safe_float(r.get("support_weight"), 0.0) for r in own_rows), default=1.0)

        class_sum: dict[str, float] = defaultdict(float)
        class_best_rank: dict[str, int] = {}
        class_best_note: dict[str, str] = {}
        class_best_role_bonus: dict[str, float] = {}
        class_best_support: dict[str, float] = {}
        class_best_band: dict[str, str] = {}

        for crow in own_rows:
            note = _normalize_note(crow.get("note_token", ""))
            if not note:
                continue
            pc = _pitch_class(note)
            band = _register_band(note)
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
                class_best_band[pc] = band

        base_classes = {_pitch_class(x) for x in final_notes}
        weak_slots: list[int] = []
        weak_reasons: dict[int, str] = {}

        for idx in range(1, len(final_notes)):
            note = final_notes[idx]
            pc = _pitch_class(note)
            if pc not in box_by_class:
                continue
            band = _register_band(note)
            ratio = class_sum.get(pc, 0.0) / max(top_support, 1e-9)
            role_score = class_best_role_bonus.get(pc, 0.0)
            keep_ratio = params["keep_ratio"].get(band, params["keep_ratio"]["other_zone"])
            role_min = params["role_min"].get(band, params["role_min"]["other_zone"])
            if ratio < keep_ratio or role_score < role_min:
                weak_slots.append(idx)
                weak_reasons[idx] = f"box_{band}"

        rescue_pool: list[tuple[float, str]] = []
        for pc, note in class_best_note.items():
            if pc in base_classes or pc in box_by_class:
                continue
            band = class_best_band.get(pc, "other_zone")
            ratio = class_sum.get(pc, 0.0) / max(top_support, 1e-9)
            role_score = class_best_role_bonus.get(pc, 0.0)
            best_rank = class_best_rank.get(pc, 10**9)
            min_ratio = params["min_ratio"].get(band, params["min_ratio"]["other_zone"])
            role_min = params["role_min"].get(band, params["role_min"]["other_zone"])
            penalty = params["replace_penalty"].get(band, params["replace_penalty"]["other_zone"])
            if ratio < min_ratio or role_score < role_min:
                continue
            composite = ratio + 0.12 * role_score + 0.06 * (1.0 / max(best_rank, 1)) - penalty
            rescue_pool.append((composite, note))
        rescue_pool.sort(key=lambda item: (-item[0], item[1]))

        used_rescues: list[str] = []
        for idx, (_score, note) in zip(weak_slots[: int(args.replace_max)], rescue_pool[: int(args.replace_max)]):
            note = _normalize_note(note)
            if note and note not in final_notes and idx < len(final_notes):
                old_note = final_notes[idx]
                final_notes[idx] = note
                used_rescues.append(note)
                replaced_notes += 1
                replacement_counter[f"{_pitch_class(old_note)}->{_pitch_class(note)}"] += 1

        if used_rescues:
            changed_groups += 1

        if main_note and main_note not in final_notes:
            final_notes = [main_note] + [x for x in final_notes if x != main_note]
            final_notes = final_notes[: int(args.cloud_topk)]

        predicted_by_group[gid] = final_notes
        audit_rows.append(
            {
                "onset_group": gid,
                "main_note": main_note,
                "base_notes_json": json.dumps(base_notes, ensure_ascii=False),
                "final_notes_json": json.dumps(final_notes, ensure_ascii=False),
                "weak_slots_json": json.dumps(weak_slots, ensure_ascii=False),
                "weak_reasons_json": json.dumps(weak_reasons, ensure_ascii=False),
                "rescued_notes_json": json.dumps(used_rescues, ensure_ascii=False),
                "changed": "1" if used_rescues else "0",
            }
        )

    top1 = _event_level_counts(truth_by_group, predicted_by_group, topk=1)
    top3 = _event_level_counts(truth_by_group, predicted_by_group, topk=3)
    top5 = _event_level_counts(truth_by_group, predicted_by_group, topk=5)

    out_audit = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_audit.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_audit.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "TWO-STAGE REGISTER-AWARE BOX SUPPRESSOR",
        "=" * 72,
        f"profile                       : {args.profile}",
        f"cloud_topk                     : {args.cloud_topk}",
        f"ownership_depth                : {args.ownership_depth}",
        f"box_ratio_threshold            : {args.box_ratio_threshold}",
        f"box_joint_min                  : {args.box_joint_min}",
        f"excluded_classes               : {','.join(sorted(excluded)) if excluded else '-'}",
        f"box_like_classes               : {','.join(sorted(box_by_class)) if box_by_class else '-'}",
        f"changed_groups                 : {changed_groups}",
        f"replaced_notes                 : {replaced_notes}",
        "",
        "EVENT-LEVEL COUNTS",
        "-" * 72,
        f"top-1  exact={top1['EXACT']:3d}  pitchclass={top1['PITCHCLASS']:3d}  wrong={top1['WRONG']:3d}  empty={top1['EMPTY']:3d}",
        f"top-3  exact={top3['EXACT']:3d}  pitchclass={top3['PITCHCLASS']:3d}  wrong={top3['WRONG']:3d}  empty={top3['EMPTY']:3d}",
        f"top-5  exact={top5['EXACT']:3d}  pitchclass={top5['PITCHCLASS']:3d}  wrong={top5['WRONG']:3d}  empty={top5['EMPTY']:3d}",
        "",
        "TOP REPLACEMENTS",
        "-" * 72,
    ]
    for pair, count in replacement_counter.most_common(12):
        lines.append(f"{pair:>8s} : {count}")
    if not replacement_counter:
        lines.append("-")

    with out_summary.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    meta = {
        "profile": args.profile,
        "cloud_topk": args.cloud_topk,
        "ownership_depth": args.ownership_depth,
        "box_ratio_threshold": args.box_ratio_threshold,
        "box_joint_min": args.box_joint_min,
        "excluded_classes": sorted(excluded),
        "box_like_classes": sorted(box_by_class),
        "changed_groups": changed_groups,
        "replaced_notes": replaced_notes,
        "top1": dict(top1),
        "top3": dict(top3),
        "top5": dict(top5),
        "replacement_counter": dict(replacement_counter),
        "profile_params": params,
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
