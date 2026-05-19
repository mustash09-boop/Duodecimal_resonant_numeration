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


def _birth_strength(role_counts: dict[str, Any]) -> float:
    return (
        1.5 * _safe_float(role_counts.get("birth_backbone"), 0.0)
        + 1.1 * _safe_float(role_counts.get("exact_support"), 0.0)
        + 0.8 * _safe_float(role_counts.get("local_birth"), 0.0)
    )


def _secondary_strength(role_counts: dict[str, Any], source_counts: dict[str, Any]) -> float:
    return (
        0.8 * _safe_float(role_counts.get("future_birth"), 0.0)
        + 1.0 * _safe_float(source_counts.get("field_support"), 0.0)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "External suppressor for high-register secondary responses. "
            "It weakens upper candidates whose early life looks more like field/secondary return "
            "than like a true note birth."
        )
    )
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--ownership-depth", type=int, default=14)
    ap.add_argument("--cloud-topk", type=int, default=5)
    ap.add_argument("--replace-max", type=int, default=1)
    ap.add_argument("--high-support-ratio-max", type=float, default=0.22)
    ap.add_argument("--high-birth-max", type=float, default=1.1)
    ap.add_argument("--high-secondary-min", type=float, default=1.0)
    ap.add_argument("--candidate-ratio-min", type=float, default=0.12)
    ap.add_argument("--candidate-birth-min", type=float, default=1.0)
    args = ap.parse_args()

    base_rows = _load_csv(Path(args.base_csv))
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

        note_info: dict[str, dict[str, Any]] = {}
        for crow in own_rows:
            note = _normalize_note(crow.get("note_token", ""))
            if not note:
                continue
            if note in note_info:
                continue
            support = _safe_float(crow.get("support_weight"), 0.0)
            roles = _json_object(crow.get("role_counts_json", "{}"))
            sources = _json_object(crow.get("source_counts_json", "{}"))
            note_info[note] = {
                "rank": _safe_int(crow.get("candidate_rank"), 0),
                "support": support,
                "ratio": support / max(top_support, 1e-9),
                "band": _register_band(note),
                "birth": _birth_strength(roles),
                "secondary": _secondary_strength(roles, sources),
            }

        base_note_set = set(final_notes)
        weak_slots: list[int] = []
        weak_reasons: dict[int, str] = {}
        for idx in range(1, len(final_notes)):
            note = final_notes[idx]
            info = note_info.get(note)
            if not info:
                continue
            if info["band"] != "high_zone":
                continue
            if (
                info["ratio"] <= float(args.high_support_ratio_max)
                and info["birth"] <= float(args.high_birth_max)
                and info["secondary"] >= float(args.high_secondary_min)
            ):
                weak_slots.append(idx)
                weak_reasons[idx] = "high_secondary_response"

        rescue_pool: list[tuple[float, str]] = []
        for note, info in note_info.items():
            if note in base_note_set:
                continue
            if info["ratio"] < float(args.candidate_ratio_min):
                continue
            if info["birth"] < float(args.candidate_birth_min):
                continue
            composite = info["ratio"] + 0.18 * info["birth"] - 0.08 * info["secondary"] + 0.04 * (1.0 / max(info["rank"], 1))
            rescue_pool.append((composite, note))
        rescue_pool.sort(key=lambda x: (-x[0], x[1]))

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
        "TWO-STAGE HIGH SECONDARY RESPONSE SUPPRESSOR",
        "=" * 72,
        f"cloud_topk                     : {args.cloud_topk}",
        f"ownership_depth                : {args.ownership_depth}",
        f"changed_groups                 : {changed_groups}",
        f"replaced_notes                 : {replaced_notes}",
        f"high_support_ratio_max         : {args.high_support_ratio_max}",
        f"high_birth_max                 : {args.high_birth_max}",
        f"high_secondary_min            : {args.high_secondary_min}",
        f"candidate_ratio_min            : {args.candidate_ratio_min}",
        f"candidate_birth_min            : {args.candidate_birth_min}",
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
        "cloud_topk": args.cloud_topk,
        "ownership_depth": args.ownership_depth,
        "changed_groups": changed_groups,
        "replaced_notes": replaced_notes,
        "thresholds": {
            "high_support_ratio_max": args.high_support_ratio_max,
            "high_birth_max": args.high_birth_max,
            "high_secondary_min": args.high_secondary_min,
            "candidate_ratio_min": args.candidate_ratio_min,
            "candidate_birth_min": args.candidate_birth_min,
        },
        "top1": dict(top1),
        "top3": dict(top3),
        "top5": dict(top5),
        "replacement_counter": dict(replacement_counter),
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
