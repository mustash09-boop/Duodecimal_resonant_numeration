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


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []
    return []


def _add_support(
    support: dict[str, dict[str, Any]],
    *,
    note: str,
    weight: float,
    role: str,
    source: str,
    ref_id: str,
) -> None:
    note = _normalize_note(note)
    if not note:
        return
    row = support.setdefault(
        note,
        {
            "note_token": note,
            "support_weight": 0.0,
            "role_counter": Counter(),
            "source_counter": Counter(),
            "refs": [],
        },
    )
    row["support_weight"] += weight
    row["role_counter"][role] += 1
    row["source_counter"][source] += 1
    row["refs"].append(ref_id)


def _proto_weight(branch_label: str, route_label: str, confidence: float, age_hint: str) -> float:
    weight = confidence
    if route_label == "notechain":
        weight += 0.45
    elif route_label == "notechain_fallback":
        weight += 0.25
    elif route_label == "event_field_candidate":
        weight += 0.10

    if branch_label == "pitched":
        weight += 0.25
    elif branch_label == "unresolved":
        weight += 0.10
    elif branch_label == "event":
        weight -= 0.05

    if age_hint == "tail":
        weight *= 0.55
    elif age_hint == "birth":
        weight *= 0.90
    return max(weight, 0.05)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve an early shared ownership window for each onset group, so tail, new birth, and field reaction can coexist before a single dominant note is forced."
    )
    ap.add_argument("--fused-events-anchored-csv", required=True)
    ap.add_argument("--fused-onset-groups-csv", required=True)
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--branch-analysis-csv", required=True)
    ap.add_argument("--out-window-candidates-csv", required=True)
    ap.add_argument("--out-window-groups-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--lookback-frames", type=int, default=3)
    ap.add_argument("--lookahead-frames", type=int, default=5)
    args = ap.parse_args()

    fused_rows = _load_csv(Path(args.fused_events_anchored_csv))
    onset_rows = _load_csv(Path(args.fused_onset_groups_csv))
    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    branch_rows = _load_csv(Path(args.branch_analysis_csv))

    branch_by_proto = {
        str(row.get("proto_exciter_id", "")).strip(): row
        for row in branch_rows
    }

    candidate_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    support_count_counter: Counter[str] = Counter()

    for onset in onset_rows:
        onset_group_id = str(onset.get("onset_group_id", "")).strip()
        anchor_frame = _safe_int(onset.get("anchor_frame"), 0)
        start_min = _safe_int(onset.get("start_min_frame"), anchor_frame)
        start_max = _safe_int(onset.get("start_max_frame"), anchor_frame)
        support: dict[str, dict[str, Any]] = {}

        group_fused = [
            row for row in fused_rows
            if str(row.get("onset_group_id", "")).strip() == onset_group_id
        ]
        for row in group_fused:
            note = _normalize_note(row.get("main_note_token", ""))
            event_kind = str(row.get("event_kind", "")).strip()
            support_kind = str(row.get("field_support_kind", "")).strip()
            ref_id = str(row.get("fused_event_id", "")).strip()

            if event_kind == "notechain_backbone":
                _add_support(
                    support,
                    note=note,
                    weight=1.00,
                    role="birth_backbone",
                    source="fused_notechain",
                    ref_id=ref_id,
                )
            elif event_kind == "event_field_only":
                _add_support(
                    support,
                    note=note,
                    weight=0.55,
                    role="field_birth",
                    source="fused_field",
                    ref_id=ref_id,
                )
            elif event_kind == "ambient_field_residue":
                _add_support(
                    support,
                    note=note,
                    weight=0.20,
                    role="ambient_residue",
                    source="fused_residue",
                    ref_id=ref_id,
                )

            if support_kind == "exact_onset_support":
                _add_support(
                    support,
                    note=note,
                    weight=0.45,
                    role="exact_support",
                    source="field_support",
                    ref_id=ref_id,
                )
            elif support_kind == "pitchclass_onset_support":
                _add_support(
                    support,
                    note=note,
                    weight=0.25,
                    role="pitchclass_support",
                    source="field_support",
                    ref_id=ref_id,
                )

        for proto in proto_rows:
            proto_start = _safe_int(proto.get("start_frame"), 0)
            proto_end = _safe_int(proto.get("end_frame"), proto_start)
            if proto_start > start_max + int(args.lookahead_frames):
                continue
            if proto_end < start_min - int(args.lookback_frames):
                continue

            proto_id = str(proto.get("proto_exciter_id", "")).strip()
            branch = branch_by_proto.get(proto_id, {})
            note = _normalize_note(
                proto.get("rescue_group_dominant_note", "")
                or proto.get("coarse_note", "")
            )
            if not note:
                continue

            if proto_end < anchor_frame:
                age_hint = "tail"
                role = "previous_tail"
            elif proto_start > anchor_frame:
                age_hint = "future_birth"
                role = "future_birth"
            else:
                age_hint = "birth"
                role = "local_birth"

            weight = _proto_weight(
                str(branch.get("branch_label", "")).strip(),
                str(branch.get("route_label", "")).strip(),
                _safe_float(proto.get("exciter_confidence"), 0.0),
                age_hint,
            )
            _add_support(
                support,
                note=note,
                weight=weight,
                role=role,
                source="proto_window",
                ref_id=proto_id,
            )

        ranked = sorted(
            support.values(),
            key=lambda row: (-_safe_float(row.get("support_weight"), 0.0), row.get("note_token", "")),
        )
        support_count_counter[str(len(ranked))] += 1

        candidate_notes = []
        for rank, row in enumerate(ranked, start=1):
            candidate_notes.append(str(row.get("note_token", "")))
            candidate_rows.append(
                {
                    "onset_group_id": onset_group_id,
                    "anchor_frame": anchor_frame,
                    "candidate_rank": rank,
                    "note_token": row["note_token"],
                    "support_weight": f"{row['support_weight']:.9f}",
                    "role_counts_json": json.dumps(dict(row["role_counter"]), ensure_ascii=False, sort_keys=True),
                    "source_counts_json": json.dumps(dict(row["source_counter"]), ensure_ascii=False, sort_keys=True),
                    "refs_json": json.dumps(row["refs"], ensure_ascii=False),
                }
            )

        group_rows.append(
            {
                "onset_group_id": onset_group_id,
                "anchor_frame": anchor_frame,
                "start_min_frame": start_min,
                "start_max_frame": start_max,
                "candidate_count": len(ranked),
                "candidate_notes_json": json.dumps(candidate_notes, ensure_ascii=False),
                "top_note_token": candidate_notes[0] if candidate_notes else "",
                "top_support_weight": f"{_safe_float(ranked[0]['support_weight'], 0.0):.9f}" if ranked else "",
            }
        )

    out_candidates = Path(args.out_window_candidates_csv)
    out_groups = Path(args.out_window_groups_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)

    if candidate_rows:
        with out_candidates.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
            w.writeheader()
            w.writerows(candidate_rows)

    if group_rows:
        with out_groups.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            w.writeheader()
            w.writerows(group_rows)

    lines = [
        "EARLY OWNERSHIP WINDOW RESOLVER",
        "=" * 72,
        f"onset_group_count         : {len(group_rows)}",
        f"candidate_row_count       : {len(candidate_rows)}",
        "",
        "CANDIDATE COUNT PER GROUP",
        "-" * 72,
    ]
    for key in sorted(support_count_counter, key=lambda x: int(x)):
        lines.append(f"{key:>3} candidates : {support_count_counter[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "early_ownership_window_resolver",
        "inputs": {
            "fused_events_anchored_csv": args.fused_events_anchored_csv,
            "fused_onset_groups_csv": args.fused_onset_groups_csv,
            "proto_exciters_csv": args.proto_exciters_csv,
            "branch_analysis_csv": args.branch_analysis_csv,
        },
        "parameters": {
            "lookback_frames": int(args.lookback_frames),
            "lookahead_frames": int(args.lookahead_frames),
        },
        "result": {
            "onset_group_count": len(group_rows),
            "candidate_row_count": len(candidate_rows),
            "candidate_count_counter": dict(support_count_counter),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
