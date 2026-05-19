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
        out[gid].append(_normalize_note(row.get("expected_note_token", row.get("note_token", ""))))
    return out


def _event_level_counts(truth_by_group: dict[str, list[str]], predicted_by_group: dict[str, list[str]], topk: int) -> Counter[str]:
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


def _group_level_counts(truth_by_group: dict[str, list[str]], predicted_by_group: dict[str, list[str]], topk: int) -> Counter[str]:
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
        description="Use DSP as a selective second opinion only for suspicious ownership windows, without changing the main note-recognition pipeline."
    )
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--dsp-group-audit-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--gap-threshold", type=float, default=0.20)
    ap.add_argument("--candidate-count-threshold", type=int, default=10)
    ap.add_argument("--ownership-topk", type=int, default=5)
    ap.add_argument("--dsp-topk", type=int, default=5)
    ap.add_argument("--ownership-rank-weight", type=float, default=1.10)
    ap.add_argument("--dsp-rank-weight", type=float, default=1.00)
    args = ap.parse_args()

    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    dsp_rows = _load_csv(Path(args.dsp_group_audit_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))

    truth_by_group = _group_truth(midi_rows)

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    dsp_by_group = {
        str(row.get("onset_group", "")).strip(): [_normalize_note(x) for x in _json_list(row.get("dsp_notes_json", ""))]
        for row in dsp_rows
    }

    predicted_by_group: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    suspicious_count = 0
    switched_count = 0

    for gid, own_rows in own_by_group.items():
        own_notes = [_normalize_note(row.get("note_token", "")) for row in own_rows]
        own_notes = [x for x in own_notes if x]
        dsp_notes = [x for x in dsp_by_group.get(gid, []) if x]

        w1 = _safe_float(own_rows[0].get("support_weight"), 0.0) if own_rows else 0.0
        w2 = _safe_float(own_rows[1].get("support_weight"), 0.0) if len(own_rows) > 1 else 0.0
        gap = w1 - w2
        candidate_count = len(own_rows)
        suspicious = candidate_count >= int(args.candidate_count_threshold) or gap < float(args.gap_threshold)

        if suspicious:
            suspicious_count += 1
            scores: dict[str, float] = defaultdict(float)
            for idx, note in enumerate(own_notes[: int(args.ownership_topk)], start=1):
                scores[note] += float(args.ownership_rank_weight) / idx
            for idx, note in enumerate(dsp_notes[: int(args.dsp_topk)], start=1):
                scores[note] += float(args.dsp_rank_weight) / idx
            reranked = [note for note, _score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        else:
            reranked = list(own_notes)

        if reranked[: int(args.ownership_topk)] != own_notes[: int(args.ownership_topk)]:
            switched_count += 1

        predicted_by_group[gid] = reranked

        audit_rows.append(
            {
                "onset_group": gid,
                "candidate_count": candidate_count,
                "top_gap": f"{gap:.9f}",
                "suspicious": int(suspicious),
                "ownership_notes_json": json.dumps(own_notes[: int(args.ownership_topk)], ensure_ascii=False),
                "dsp_notes_json": json.dumps(dsp_notes[: int(args.dsp_topk)], ensure_ascii=False),
                "reranked_notes_json": json.dumps(reranked[: max(int(args.ownership_topk), int(args.dsp_topk))], ensure_ascii=False),
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
        "DSP SELECTIVE SECOND OPINION",
        "=" * 72,
        f"onset_groups                 : {len(predicted_by_group)}",
        f"suspicious_groups            : {suspicious_count}",
        f"groups_with_changed_ranking  : {switched_count}",
        f"gap_threshold                : {args.gap_threshold}",
        f"candidate_count_threshold    : {args.candidate_count_threshold}",
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
        "stage": "dsp_selective_second_opinion",
        "inputs": {
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "dsp_group_audit_csv": args.dsp_group_audit_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "gap_threshold": args.gap_threshold,
            "candidate_count_threshold": args.candidate_count_threshold,
            "ownership_topk": args.ownership_topk,
            "dsp_topk": args.dsp_topk,
            "ownership_rank_weight": args.ownership_rank_weight,
            "dsp_rank_weight": args.dsp_rank_weight,
        },
        "result": {
            "suspicious_groups": suspicious_count,
            "groups_with_changed_ranking": switched_count,
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
