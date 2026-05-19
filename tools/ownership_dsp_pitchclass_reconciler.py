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


def _json_list(value: Any) -> list[Any]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return raw
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


def _profile_weights(profile: str) -> dict[str, float]:
    if profile == "main-note":
        return {
            "ownership_note": 0.95,
            "ownership_pc": 0.55,
            "dsp_note": 0.90,
            "dsp_pc": 0.40,
            "harmonic_bonus": 0.12,
            "exact_consensus": 0.42,
            "pc_consensus": 0.28,
            "pc_rank_bonus": 0.00,
        }
    if profile == "candidate-cloud":
        return {
            "ownership_note": 0.65,
            "ownership_pc": 0.85,
            "dsp_note": 0.65,
            "dsp_pc": 0.72,
            "harmonic_bonus": 0.10,
            "exact_consensus": 0.30,
            "pc_consensus": 0.34,
            "pc_rank_bonus": 0.22,
        }
    return {
        "ownership_note": 0.80,
        "ownership_pc": 0.70,
        "dsp_note": 0.80,
        "dsp_pc": 0.55,
        "harmonic_bonus": 0.10,
        "exact_consensus": 0.35,
        "pc_consensus": 0.30,
        "pc_rank_bonus": 0.10,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "External reconciliation layer between the shared ownership window and local DSP chains. "
            "It first aligns pitch classes, then chooses note identity inside the aligned class."
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
        choices=["balanced", "main-note", "candidate-cloud"],
        default="balanced",
    )
    ap.add_argument("--ownership-topk", type=int, default=8)
    ap.add_argument("--dsp-topk", type=int, default=6)
    args = ap.parse_args()

    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    dsp_rows = _load_csv(Path(args.dsp_group_audit_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    truth_by_group = _group_truth(midi_rows)
    weights = _profile_weights(args.profile)

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    dsp_by_group = {
        str(row.get("onset_group", "")).strip(): row
        for row in dsp_rows
    }

    audit_rows: list[dict[str, Any]] = []
    predicted_by_group: dict[str, list[str]] = {}
    top1_changed = 0
    pc_consensus_counter: Counter[str] = Counter()

    for gid, own_rows in own_by_group.items():
        dsp_row = dsp_by_group.get(gid, {})
        dsp_chain_rows = _json_list(dsp_row.get("dsp_chain_rows_json", "[]"))

        own_top = own_rows[: int(args.ownership_topk)]
        dsp_top = [row for row in dsp_chain_rows if isinstance(row, dict)][: int(args.dsp_topk)]

        own_top_support = max((_safe_float(row.get("support_weight"), 0.0) for row in own_top), default=1.0)
        dsp_top_score = max((_safe_float(row.get("score"), 0.0) for row in dsp_top), default=1.0)
        dsp_top_hcount = max((_safe_int(row.get("harmonic_count"), 0) for row in dsp_top), default=1)

        notes: dict[str, dict[str, Any]] = {}
        own_pc_sums: dict[str, float] = defaultdict(float)
        dsp_pc_sums: dict[str, float] = defaultdict(float)
        own_pc_best_rank: dict[str, int] = {}
        dsp_pc_best_rank: dict[str, int] = {}

        for row in own_top:
            note = _normalize_note(row.get("note_token", ""))
            if not note:
                continue
            own_rank = _safe_int(row.get("candidate_rank"), 0)
            own_norm = _safe_float(row.get("support_weight"), 0.0) / max(own_top_support, 1e-9)
            pc = _pitch_class(note)
            own_pc_sums[pc] += own_norm
            own_pc_best_rank[pc] = min(own_rank, own_pc_best_rank.get(pc, 10**9))
            entry = notes.setdefault(
                note,
                {
                    "note_token": note,
                    "pitch_class": pc,
                    "own_norm": 0.0,
                    "own_rank": 10**9,
                    "dsp_norm": 0.0,
                    "dsp_rank": 10**9,
                    "harmonic_count_norm": 0.0,
                },
            )
            entry["own_norm"] = max(entry["own_norm"], own_norm)
            entry["own_rank"] = min(entry["own_rank"], own_rank)

        for idx, row in enumerate(dsp_top, start=1):
            note = _normalize_note(row.get("note_token", ""))
            if not note:
                continue
            dsp_norm = _safe_float(row.get("score"), 0.0) / max(dsp_top_score, 1e-9)
            hcount_norm = _safe_int(row.get("harmonic_count"), 0) / max(dsp_top_hcount, 1)
            pc = _pitch_class(note)
            dsp_pc_sums[pc] += dsp_norm
            dsp_pc_best_rank[pc] = min(idx, dsp_pc_best_rank.get(pc, 10**9))
            entry = notes.setdefault(
                note,
                {
                    "note_token": note,
                    "pitch_class": pc,
                    "own_norm": 0.0,
                    "own_rank": 10**9,
                    "dsp_norm": 0.0,
                    "dsp_rank": 10**9,
                    "harmonic_count_norm": 0.0,
                },
            )
            entry["dsp_norm"] = max(entry["dsp_norm"], dsp_norm)
            entry["dsp_rank"] = min(entry["dsp_rank"], idx)
            entry["harmonic_count_norm"] = max(entry["harmonic_count_norm"], hcount_norm)

        pitchclasses = set(own_pc_sums) | set(dsp_pc_sums)
        pc_scores: dict[str, float] = {}
        for pc in pitchclasses:
            own_pc = own_pc_sums.get(pc, 0.0)
            dsp_pc = dsp_pc_sums.get(pc, 0.0)
            score = own_pc * weights["ownership_pc"] + dsp_pc * weights["dsp_pc"]
            if own_pc > 0.0 and dsp_pc > 0.0:
                score += weights["pc_consensus"]
                pc_consensus_counter["shared_pc"] += 1
            elif own_pc > 0.0:
                pc_consensus_counter["ownership_only_pc"] += 1
            elif dsp_pc > 0.0:
                pc_consensus_counter["dsp_only_pc"] += 1
            if args.profile == "candidate-cloud":
                rank_bonus = 0.0
                if pc in own_pc_best_rank:
                    rank_bonus += 1.0 / own_pc_best_rank[pc]
                if pc in dsp_pc_best_rank:
                    rank_bonus += 1.0 / dsp_pc_best_rank[pc]
                score += rank_bonus * weights["pc_rank_bonus"]
            pc_scores[pc] = score

        scored_notes: list[dict[str, Any]] = []
        for note, entry in notes.items():
            pc = str(entry["pitch_class"])
            total = (
                entry["own_norm"] * weights["ownership_note"]
                + entry["dsp_norm"] * weights["dsp_note"]
                + pc_scores.get(pc, 0.0)
                + entry["harmonic_count_norm"] * weights["harmonic_bonus"]
            )
            if entry["own_norm"] > 0.0 and entry["dsp_norm"] > 0.0:
                total += weights["exact_consensus"]
            scored_notes.append(
                {
                    "note_token": note,
                    "pitch_class": pc,
                    "total_score": total,
                    "own_norm": entry["own_norm"],
                    "dsp_norm": entry["dsp_norm"],
                    "pc_score": pc_scores.get(pc, 0.0),
                    "harmonic_count_norm": entry["harmonic_count_norm"],
                    "own_rank": entry["own_rank"],
                    "dsp_rank": entry["dsp_rank"],
                }
            )

        scored_notes.sort(
            key=lambda row: (
                -_safe_float(row["pc_score"], 0.0),
                -_safe_float(row["total_score"], 0.0),
                _safe_int(row["own_rank"], 10**9),
                _safe_int(row["dsp_rank"], 10**9),
                str(row["note_token"]),
            )
        )

        predicted = [str(row["note_token"]) for row in scored_notes if str(row["note_token"]).strip()]
        predicted_by_group[gid] = predicted

        old_top = _normalize_note(own_rows[0].get("note_token", "")) if own_rows else ""
        new_top = predicted[0] if predicted else ""
        if old_top != new_top:
            top1_changed += 1

        for new_rank, row in enumerate(scored_notes, start=1):
            audit_rows.append(
                {
                    "onset_group": gid,
                    "new_rank": new_rank,
                    "note_token": row["note_token"],
                    "pitch_class": row["pitch_class"],
                    "total_score": f"{_safe_float(row['total_score']):.9f}",
                    "pc_score": f"{_safe_float(row['pc_score']):.9f}",
                    "own_norm": f"{_safe_float(row['own_norm']):.9f}",
                    "dsp_norm": f"{_safe_float(row['dsp_norm']):.9f}",
                    "harmonic_count_norm": f"{_safe_float(row['harmonic_count_norm']):.9f}",
                    "own_rank": _safe_int(row["own_rank"], 0) if _safe_int(row["own_rank"], 10**9) < 10**9 else "",
                    "dsp_rank": _safe_int(row["dsp_rank"], 0) if _safe_int(row["dsp_rank"], 10**9) < 10**9 else "",
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
        "OWNERSHIP DSP PITCHCLASS RECONCILER",
        "=" * 72,
        f"onset_groups                 : {len(predicted_by_group)}",
        f"profile                      : {args.profile}",
        f"ownership_topk               : {args.ownership_topk}",
        f"dsp_topk                     : {args.dsp_topk}",
        f"groups_with_top1_change      : {top1_changed}",
        "",
        "PITCHCLASS CONSENSUS",
        "-" * 72,
    ]
    for key in sorted(pc_consensus_counter):
        lines.append(f"{key:28s}: {pc_consensus_counter[key]}")
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
        "stage": "ownership_dsp_pitchclass_reconciler",
        "inputs": {
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "dsp_group_audit_csv": args.dsp_group_audit_csv,
            "midi_events_csv": args.midi_events_csv,
        },
        "parameters": {
            "profile": args.profile,
            "ownership_topk": int(args.ownership_topk),
            "dsp_topk": int(args.dsp_topk),
            "weights": weights,
        },
        "result": {
            "top1_changed": top1_changed,
            "pc_consensus_counter": dict(pc_consensus_counter),
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
