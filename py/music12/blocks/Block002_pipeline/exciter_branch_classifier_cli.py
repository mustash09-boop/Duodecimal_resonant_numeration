from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
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


def _coarse(note: str) -> str:
    return _normalize_note(note)


def _build_families_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: _safe_int(r.get("family_rank"), 999999))
    return out


def _window_frames(start_frame: int, end_frame: int, lookahead_frames: int) -> list[int]:
    return list(range(start_frame, end_frame + lookahead_frames + 1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classify proto-exciters into pitched-like, event-like, or unresolved continuation branches."
    )
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-branch-analysis-csv", required=True)
    ap.add_argument("--out-pitched-proto-exciters-csv", required=True)
    ap.add_argument("--out-event-proto-exciters-csv", required=True)
    ap.add_argument("--out-event-field-proto-exciters-csv", required=True)
    ap.add_argument("--out-event-only-proto-exciters-csv", required=True)
    ap.add_argument("--out-unresolved-proto-exciters-csv", required=True)
    ap.add_argument("--out-notechain-proto-exciters-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--lookahead-frames", type=int, default=10)
    ap.add_argument("--family-rank-limit", type=int, default=6)
    ap.add_argument("--min-family-score", type=float, default=0.80)
    ap.add_argument("--min-pitched-support-ratio", type=float, default=0.28)
    ap.add_argument("--min-pitched-match-frames", type=int, default=3)
    ap.add_argument("--min-root-micro-count", type=int, default=10)
    ap.add_argument("--max-foreign-dominance-ratio", type=float, default=1.45)
    ap.add_argument("--fallback-duration-frames", type=int, default=4)
    ap.add_argument("--fallback-exciter-confidence", type=float, default=0.45)
    ap.add_argument("--fallback-total-seed-score", type=float, default=1.0)
    ap.add_argument("--event-field-candidate-max-duration-frames", type=int, default=2)
    ap.add_argument("--event-field-candidate-max-matched-frames", type=int, default=1)
    ap.add_argument("--event-field-candidate-max-support-ratio", type=float, default=0.10)
    ap.add_argument("--event-field-candidate-max-total-seed-score", type=float, default=0.30)
    args = ap.parse_args()

    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    family_rows = _load_csv(Path(args.micro_families_csv))
    families_by_frame = _build_families_by_frame(family_rows)

    analysis_rows: list[dict[str, Any]] = []
    pitched_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_field_rows: list[dict[str, Any]] = []
    event_only_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []

    branch_counter: Counter[str] = Counter()

    for proto in proto_rows:
        proto_id = _safe_int(proto.get("proto_exciter_id"), 0)
        coarse_note = _coarse(proto.get("coarse_note", ""))
        start_frame = _safe_int(proto.get("start_frame"), 0)
        end_frame = _safe_int(proto.get("end_frame"), start_frame)
        duration_frames = _safe_int(proto.get("duration_frames"), max(1, end_frame - start_frame + 1))
        exciter_confidence = _safe_float(proto.get("exciter_confidence"), 0.0)
        total_seed_score = _safe_float(proto.get("total_seed_score"), 0.0)

        frames = _window_frames(start_frame, end_frame, int(args.lookahead_frames))
        frame_count = max(len(frames), 1)
        matched_frames = 0
        foreign_frames = 0
        exact_score_sum = 0.0
        foreign_score_sum = 0.0
        best_exact_score = 0.0
        best_foreign_score = 0.0
        max_root_micro_count = 0
        max_root_micro_diversity = 0
        dominant_counter: Counter[str] = Counter()

        for frame in frames:
            rows = families_by_frame.get(frame, [])[: int(args.family_rank_limit)]
            if not rows:
                continue
            best_exact = None
            best_foreign = None
            for row in rows:
                root_coarse = _coarse(row.get("family_root_note_coarse", ""))
                family_score = _safe_float(row.get("family_score"), 0.0)
                if root_coarse == coarse_note:
                    if best_exact is None or family_score > _safe_float(best_exact.get("family_score"), 0.0):
                        best_exact = row
                else:
                    if best_foreign is None or family_score > _safe_float(best_foreign.get("family_score"), 0.0):
                        best_foreign = row
            if best_exact is not None and _safe_float(best_exact.get("family_score"), 0.0) >= float(args.min_family_score):
                matched_frames += 1
                score = _safe_float(best_exact.get("family_score"), 0.0)
                exact_score_sum += score
                best_exact_score = max(best_exact_score, score)
                max_root_micro_count = max(max_root_micro_count, _safe_int(best_exact.get("root_micro_count"), 0))
                max_root_micro_diversity = max(max_root_micro_diversity, _safe_int(best_exact.get("root_micro_diversity"), 0))
                dominant_counter[_coarse(best_exact.get("family_root_note_micro", ""))] += 1
            if best_foreign is not None:
                score = _safe_float(best_foreign.get("family_score"), 0.0)
                foreign_score_sum += score
                best_foreign_score = max(best_foreign_score, score)
                if score >= float(args.min_family_score):
                    foreign_frames += 1

        support_ratio = matched_frames / frame_count
        foreign_ratio = foreign_frames / frame_count
        mean_exact_score = exact_score_sum / matched_frames if matched_frames else 0.0
        mean_foreign_score = foreign_score_sum / max(frame_count, 1)
        foreign_dominance_ratio = (
            (best_foreign_score / best_exact_score)
            if best_exact_score > 0.0 and best_foreign_score > 0.0
            else (99.0 if best_exact_score <= 0.0 and best_foreign_score > 0.0 else 0.0)
        )

        pitched_signal = (
            matched_frames >= int(args.min_pitched_match_frames)
            and support_ratio >= float(args.min_pitched_support_ratio)
            and max_root_micro_count >= int(args.min_root_micro_count)
            and foreign_dominance_ratio <= float(args.max_foreign_dominance_ratio)
        )

        event_signal = (
            matched_frames == 0
            or (
                support_ratio < float(args.min_pitched_support_ratio) * 0.6
                and foreign_ratio >= support_ratio
                and best_foreign_score > best_exact_score
            )
        )

        if pitched_signal:
            branch = "pitched"
            reason = "stable_root_bearing_chain"
            pitched_rows.append(proto)
        elif event_signal:
            branch = "event"
            reason = "no_stable_note_chain"
            event_rows.append(proto)
        else:
            branch = "unresolved"
            reason = "borderline_chain_emergence"
            unresolved_rows.append(proto)

        route_label = "notechain"
        route_reason = "branch_kept"
        if branch == "event":
            event_field_candidate = (
                duration_frames <= int(args.event_field_candidate_max_duration_frames)
                and matched_frames <= int(args.event_field_candidate_max_matched_frames)
                and support_ratio < float(args.event_field_candidate_max_support_ratio)
                and total_seed_score < float(args.event_field_candidate_max_total_seed_score)
            )
            if event_field_candidate:
                route_label = "event_field_candidate"
                route_reason = "short_weak_event_field_like"
                event_field_rows.append(proto)
            elif (
                duration_frames >= int(args.fallback_duration_frames)
                or exciter_confidence >= float(args.fallback_exciter_confidence)
                or total_seed_score >= float(args.fallback_total_seed_score)
            ):
                route_label = "notechain_fallback"
                route_reason = "event_but_structurally_strong"
            else:
                route_label = "event_only"
                route_reason = "event_without_notechain_strength"
                event_field_rows.append(proto)

        branch_counter[branch] += 1
        analysis_rows.append(
            {
                "proto_exciter_id": proto_id,
                "coarse_note": coarse_note,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "support_ratio": f"{support_ratio:.9f}",
                "matched_frames": matched_frames,
                "foreign_ratio": f"{foreign_ratio:.9f}",
                "best_exact_score": f"{best_exact_score:.9f}",
                "best_foreign_score": f"{best_foreign_score:.9f}",
                "mean_exact_score": f"{mean_exact_score:.9f}",
                "mean_foreign_score": f"{mean_foreign_score:.9f}",
                "foreign_dominance_ratio": f"{foreign_dominance_ratio:.9f}",
                "max_root_micro_count": max_root_micro_count,
                "max_root_micro_diversity": max_root_micro_diversity,
                "dominant_supported_micro": dominant_counter.most_common(1)[0][0] if dominant_counter else "",
                "branch_label": branch,
                "reason": reason,
                "route_label": route_label,
                "route_reason": route_reason,
            }
        )

    analysis_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))

    def _write_csv(path_str: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fieldnames is None:
            if not rows:
                return
            fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    analysis_fields = [
        "proto_exciter_id",
        "coarse_note",
        "start_frame",
        "end_frame",
        "support_ratio",
        "matched_frames",
        "foreign_ratio",
        "best_exact_score",
        "best_foreign_score",
        "mean_exact_score",
        "mean_foreign_score",
        "foreign_dominance_ratio",
        "max_root_micro_count",
        "max_root_micro_diversity",
        "dominant_supported_micro",
        "branch_label",
        "reason",
        "route_label",
        "route_reason",
    ]
    _write_csv(args.out_branch_analysis_csv, analysis_rows, analysis_fields)
    if proto_rows:
        proto_fields = list(proto_rows[0].keys())
        _write_csv(args.out_pitched_proto_exciters_csv, pitched_rows, proto_fields)
        _write_csv(args.out_event_proto_exciters_csv, event_rows, proto_fields)
        _write_csv(args.out_unresolved_proto_exciters_csv, unresolved_rows, proto_fields)
        event_field_ids = {
            _safe_int(row.get("proto_exciter_id"), 0)
            for row in analysis_rows
            if str(row.get("route_label", "")) in {"event_only", "event_field_candidate"}
        }
        event_field_rows = [
            proto
            for proto in proto_rows
            if _safe_int(proto.get("proto_exciter_id"), 0) in event_field_ids
        ]
        event_field_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))
        _write_csv(args.out_event_field_proto_exciters_csv, event_field_rows, proto_fields)
        event_only_ids = {
            _safe_int(row.get("proto_exciter_id"), 0)
            for row in analysis_rows
            if str(row.get("route_label", "")) == "event_only"
        }
        event_only_rows = [
            proto
            for proto in proto_rows
            if _safe_int(proto.get("proto_exciter_id"), 0) in event_only_ids
        ]
        event_only_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))
        _write_csv(args.out_event_only_proto_exciters_csv, event_only_rows, proto_fields)
        notechain_ids = {
            _safe_int(row.get("proto_exciter_id"), 0)
            for row in analysis_rows
            if str(row.get("route_label", "")) in {"notechain", "notechain_fallback"}
        }
        notechain_rows = [
            proto
            for proto in proto_rows
            if _safe_int(proto.get("proto_exciter_id"), 0) in notechain_ids
        ]
        notechain_rows.sort(key=lambda r: (_safe_int(r.get("start_frame"), 0), _safe_int(r.get("proto_exciter_id"), 0)))
        _write_csv(args.out_notechain_proto_exciters_csv, notechain_rows, proto_fields)
    else:
        notechain_rows = []

    summary_lines = [
        "EXCITER BRANCH CLASSIFIER",
        "=" * 72,
        f"proto_exciters : {len(proto_rows)}",
        f"pitched        : {branch_counter['pitched']}",
        f"event          : {branch_counter['event']}",
        f"unresolved     : {branch_counter['unresolved']}",
        f"event_field_rows: {len(event_field_rows)}",
        f"event_only_rows: {len(event_only_rows)}",
        f"notechain_rows : {len(notechain_rows)}",
    ]
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    meta = {
        "stage": "exciter_branch_classifier",
        "inputs": {
            "proto_exciters_csv": args.proto_exciters_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "lookahead_frames": int(args.lookahead_frames),
            "family_rank_limit": int(args.family_rank_limit),
            "min_family_score": float(args.min_family_score),
            "min_pitched_support_ratio": float(args.min_pitched_support_ratio),
            "min_pitched_match_frames": int(args.min_pitched_match_frames),
            "min_root_micro_count": int(args.min_root_micro_count),
            "max_foreign_dominance_ratio": float(args.max_foreign_dominance_ratio),
            "fallback_duration_frames": int(args.fallback_duration_frames),
            "fallback_exciter_confidence": float(args.fallback_exciter_confidence),
            "fallback_total_seed_score": float(args.fallback_total_seed_score),
            "event_field_candidate_max_duration_frames": int(args.event_field_candidate_max_duration_frames),
            "event_field_candidate_max_matched_frames": int(args.event_field_candidate_max_matched_frames),
            "event_field_candidate_max_support_ratio": float(args.event_field_candidate_max_support_ratio),
            "event_field_candidate_max_total_seed_score": float(args.event_field_candidate_max_total_seed_score),
        },
        "result": {
            "proto_exciters": len(proto_rows),
            "branch_counts": dict(branch_counter),
            "event_field_rows": len(event_field_rows),
            "event_only_rows": len(event_only_rows),
            "notechain_rows": len(notechain_rows),
        },
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
