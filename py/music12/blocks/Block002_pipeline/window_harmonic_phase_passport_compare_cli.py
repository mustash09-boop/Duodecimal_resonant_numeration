# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max_run(sorted_frames: list[int]) -> int:
    if not sorted_frames:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(sorted_frames)):
        if sorted_frames[idx] == sorted_frames[idx - 1] + 1:
            current += 1
        else:
            if current > best:
                best = current
            current = 1
    if current > best:
        best = current
    return best


def _load_candidate_harmonic_defs(path: Path) -> dict[int, dict[str, Any]]:
    rows = _load_csv(path)
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        harmonic_index = _safe_int(row.get("harmonic_index"), 0)
        if harmonic_index <= 0 or harmonic_index in by_index:
            continue
        by_index[harmonic_index] = {
            "harmonic_index": harmonic_index,
            "theoretical_token": str(row.get("theoretical_token", "")).strip(),
            "theoretical_hz": _safe_float(row.get("theoretical_hz"), 0.0),
            "lower_hz_tolerance": _safe_float(row.get("lower_hz_tolerance"), 0.0),
            "upper_hz_tolerance": _safe_float(row.get("upper_hz_tolerance"), 0.0),
        }
    return by_index


def _get_best_track_meta(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    best = dict(data.get("best_track", {}) or {})
    hits = best.get("representative_hits", []) or []
    hit_by_index = {int(hit.get("harmonic_index", 0)): hit for hit in hits}
    return {
        "root_note_token": str(best.get("root_note_token", "")).strip(),
        "root_hz_mean": _safe_float(best.get("root_hz_mean"), 0.0),
        "start_time": _safe_float(best.get("start_time"), 0.0),
        "end_time": _safe_float(best.get("end_time"), 0.0),
        "frame_count": _safe_int(best.get("frame_count"), 0),
        "harmonic_presence_profile": dict(best.get("harmonic_presence_profile", {}) or {}),
        "harmonic_amplitude_map": {
            int(idx): _safe_float(hit.get("matched_amplitude"), 0.0)
            for idx, hit in hit_by_index.items()
            if idx > 0
        },
    }


def _pick_frame_best_matches(
    frame_rows_map: dict[int, list[dict[str, Any]]],
    harmonic_defs: dict[int, dict[str, Any]],
    value_key: str,
) -> dict[int, list[dict[str, Any]]]:
    picked: dict[int, list[dict[str, Any]]] = {}
    for frame_index, frame_rows in frame_rows_map.items():
        per_harmonic: list[dict[str, Any]] = []
        for harmonic_index in sorted(harmonic_defs):
            harmonic_def = harmonic_defs[harmonic_index]
            low = float(harmonic_def["lower_hz_tolerance"])
            high = float(harmonic_def["upper_hz_tolerance"])
            matches = [
                row
                for row in frame_rows
                if low <= _safe_float(row.get("frequency_hz" if value_key == "energy" else "freq_hz"), 0.0) <= high
            ]
            if not matches:
                continue
            best = max(matches, key=lambda row: _safe_float(row.get(value_key), 0.0))
            per_harmonic.append(
                {
                    "frame_index": frame_index,
                    "time_sec": _safe_float(best.get("time_sec"), 0.0),
                    "harmonic_index": harmonic_index,
                    "theoretical_token": harmonic_def["theoretical_token"],
                    "theoretical_hz": harmonic_def["theoretical_hz"],
                    "observed_token": str(
                        best.get("observed_coarse_symbol", best.get("observed_token", ""))
                    ).strip(),
                    "observed_hz": _safe_float(best.get("frequency_hz" if value_key == "energy" else "freq_hz"), 0.0),
                    "observed_strength": _safe_float(best.get(value_key), 0.0),
                    "owner_label": str(best.get("owner_label", "PASSPORT_DENSE")).strip(),
                }
            )
        if per_harmonic:
            picked[frame_index] = per_harmonic
    return picked


def _build_timeline_rows(
    picked_by_frame: dict[int, list[dict[str, Any]]],
    frame_indices: list[int],
    frame_times: dict[int, float],
    harmonic_defs: dict[int, dict[str, Any]],
    reference_start_time: float,
    total_frame_count: int,
    source_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    harmonic_hits_map: dict[int, list[dict[str, Any]]] = {idx: [] for idx in harmonic_defs}
    for frame_index, matches in picked_by_frame.items():
        for match in matches:
            harmonic_hits_map[int(match["harmonic_index"])].append(match)

    onset_pairs: list[tuple[int, int]] = []
    for harmonic_index, hits in harmonic_hits_map.items():
        if hits:
            onset_pairs.append((harmonic_index, min(_safe_int(hit.get("frame_index"), 0) for hit in hits)))
    onset_pairs.sort(key=lambda item: (item[1], item[0]))
    onset_rank_map = {harmonic_index: rank + 1 for rank, (harmonic_index, _) in enumerate(onset_pairs)}

    for harmonic_index in sorted(harmonic_defs):
        harmonic_def = harmonic_defs[harmonic_index]
        hits = harmonic_hits_map[harmonic_index]
        hit_frames = sorted({_safe_int(hit.get("frame_index"), 0) for hit in hits})
        strengths = [_safe_float(hit.get("observed_strength"), 0.0) for hit in hits]
        first_time = min((_safe_float(hit.get("time_sec"), 0.0) for hit in hits), default=0.0)
        last_time = max((_safe_float(hit.get("time_sec"), 0.0) for hit in hits), default=0.0)
        owner_counts = Counter(str(hit.get("owner_label", "")).strip() for hit in hits)
        rows.append(
            {
                "source_label": source_label,
                "harmonic_index": harmonic_index,
                "theoretical_token": harmonic_def["theoretical_token"],
                "theoretical_hz": harmonic_def["theoretical_hz"],
                "active_frames": len(hit_frames),
                "coverage_ratio": (float(len(hit_frames)) / float(total_frame_count)) if total_frame_count > 0 else 0.0,
                "first_frame_index": hit_frames[0] if hit_frames else "",
                "first_time_sec": first_time if hits else "",
                "first_relative_sec": (first_time - reference_start_time) if hits else "",
                "last_frame_index": hit_frames[-1] if hit_frames else "",
                "last_time_sec": last_time if hits else "",
                "mean_strength": _mean(strengths),
                "max_strength": max(strengths) if strengths else 0.0,
                "max_consecutive_frames": _max_run(hit_frames),
                "onset_rank": onset_rank_map.get(harmonic_index, ""),
                "dominant_owner_label": owner_counts.most_common(1)[0][0] if owner_counts else "",
                "owner_label_counts_json": json.dumps(dict(owner_counts), ensure_ascii=False),
            }
        )
    return rows


def _normalize_strength_map(timeline_rows: list[dict[str, Any]]) -> dict[int, float]:
    strengths = {int(row["harmonic_index"]): _safe_float(row.get("mean_strength"), 0.0) for row in timeline_rows}
    max_strength = max(strengths.values()) if strengths else 0.0
    if max_strength <= 0.0:
        return {harmonic_index: 0.0 for harmonic_index in strengths}
    return {harmonic_index: (value / max_strength) for harmonic_index, value in strengths.items()}


def _row_by_harmonic(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["harmonic_index"]): row for row in rows}


def _score_candidate_compare(
    window_rows: list[dict[str, Any]],
    passport_rows: list[dict[str, Any]],
    harmonic_presence_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window_by_h = _row_by_harmonic(window_rows)
    passport_by_h = _row_by_harmonic(passport_rows)
    window_norm = _normalize_strength_map(window_rows)
    passport_norm = _normalize_strength_map(passport_rows)
    weights = {
        1: 1.3,
        2: 1.4,
        3: 1.0,
        4: 0.6,
        5: 1.0,
        6: 0.5,
        7: 0.9,
        8: 0.3,
        9: 0.3,
        10: 0.2,
        11: 0.2,
        12: 0.2,
    }
    compare_rows: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for harmonic_index in sorted(weights):
        window_row = window_by_h[harmonic_index]
        passport_row = passport_by_h[harmonic_index]
        w_cov = _safe_float(window_row.get("coverage_ratio"), 0.0)
        p_cov = _safe_float(passport_row.get("coverage_ratio"), 0.0)
        coverage_match = 1.0 - min(1.0, abs(w_cov - p_cov) / 0.75)

        w_rel = _safe_float(window_row.get("first_relative_sec"), -1.0)
        p_rel = _safe_float(passport_row.get("first_relative_sec"), -1.0)
        onset_match = 0.0
        if w_rel >= 0.0 and p_rel >= 0.0:
            onset_match = 1.0 - min(1.0, abs(w_rel - p_rel) / 0.25)

        w_strength = window_norm.get(harmonic_index, 0.0)
        p_strength = passport_norm.get(harmonic_index, 0.0)
        strength_match = 1.0 - min(1.0, abs(w_strength - p_strength))

        p_presence_count = _safe_int(harmonic_presence_profile.get(str(harmonic_index)), 0)
        presence_bias = 1.0 if p_presence_count > 0 else 0.5
        harmonic_score = presence_bias * (0.40 * coverage_match + 0.35 * onset_match + 0.25 * strength_match)

        weight = weights[harmonic_index]
        weighted_sum += harmonic_score * weight
        weight_total += weight
        compare_rows.append(
            {
                "harmonic_index": harmonic_index,
                "window_active_frames": window_row.get("active_frames"),
                "window_coverage_ratio": window_row.get("coverage_ratio"),
                "window_first_relative_sec": window_row.get("first_relative_sec"),
                "window_mean_strength_norm": w_strength,
                "passport_active_frames": passport_row.get("active_frames"),
                "passport_coverage_ratio": passport_row.get("coverage_ratio"),
                "passport_first_relative_sec": passport_row.get("first_relative_sec"),
                "passport_mean_strength_norm": p_strength,
                "coverage_match_score": coverage_match,
                "onset_match_score": onset_match,
                "strength_match_score": strength_match,
                "passport_presence_count": p_presence_count,
                "harmonic_weight": weight,
                "harmonic_score": harmonic_score,
            }
        )

    def _ratio(rows_by_h: dict[int, dict[str, Any]], top_idx: int, bottom_idx: int) -> float:
        top = _safe_float(rows_by_h.get(top_idx, {}).get("mean_strength"), 0.0)
        bottom = _safe_float(rows_by_h.get(bottom_idx, {}).get("mean_strength"), 0.0)
        return top / bottom if bottom > 0.0 else 0.0

    window_h2_h1 = _ratio(window_by_h, 2, 1)
    passport_h2_h1 = _ratio(passport_by_h, 2, 1)
    ratio_match = 1.0 - min(1.0, abs(window_h2_h1 - passport_h2_h1) / 1.5)

    window_h357 = _mean([window_norm.get(idx, 0.0) for idx in (3, 5, 7)])
    passport_h357 = _mean([passport_norm.get(idx, 0.0) for idx in (3, 5, 7)])
    h357_match = 1.0 - min(1.0, abs(window_h357 - passport_h357))

    overall_score = ((weighted_sum / weight_total) if weight_total > 0.0 else 0.0)
    overall_score = 0.70 * overall_score + 0.15 * ratio_match + 0.15 * h357_match

    summary = {
        "overall_score": overall_score,
        "window_h2_over_h1_mean_strength_ratio": window_h2_h1,
        "passport_h2_over_h1_mean_strength_ratio": passport_h2_h1,
        "h2_h1_ratio_match_score": ratio_match,
        "window_h357_mean_strength_norm": window_h357,
        "passport_h357_mean_strength_norm": passport_h357,
        "h357_match_score": h357_match,
    }
    return compare_rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare the bowed-layer window against violin/cello A.5 note passports by harmonic appearance timeline."
    )
    ap.add_argument("--window-observations-csv", required=True)
    ap.add_argument("--data-grounded-owner-csv", required=True)
    ap.add_argument("--violin-dense-vs-theory-csv", required=True)
    ap.add_argument("--violin-dense-csv", required=True)
    ap.add_argument("--violin-chain-json", required=True)
    ap.add_argument("--cello-dense-vs-theory-csv", required=True)
    ap.add_argument("--cello-dense-csv", required=True)
    ap.add_argument("--cello-chain-json", required=True)
    ap.add_argument("--out-window-harmonic-csv", required=True)
    ap.add_argument("--out-candidate-compare-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(args.progress_json, {"status": "running", "phase": "loading_inputs"})

    window_rows = _load_csv(Path(args.window_observations_csv))
    grounded_rows = _load_csv(Path(args.data_grounded_owner_csv))
    selected_frames = sorted({_safe_int(row.get("frame_index"), -1) for row in grounded_rows if _safe_int(row.get("frame_index"), -1) >= 0})
    selected_frame_set = set(selected_frames)
    window_selected_rows = [row for row in window_rows if _safe_int(row.get("frame_index"), -1) in selected_frame_set]
    frame_times = {
        frame_index: min(
            _safe_float(row.get("time_sec"), 0.0)
            for row in window_selected_rows
            if _safe_int(row.get("frame_index"), -1) == frame_index
        )
        for frame_index in selected_frames
    }
    window_frame_map: dict[int, list[dict[str, Any]]] = {}
    for row in window_selected_rows:
        frame_index = _safe_int(row.get("frame_index"), -1)
        window_frame_map.setdefault(frame_index, []).append(row)

    violin_defs = _load_candidate_harmonic_defs(Path(args.violin_dense_vs_theory_csv))
    cello_defs = _load_candidate_harmonic_defs(Path(args.cello_dense_vs_theory_csv))
    violin_track_meta = _get_best_track_meta(Path(args.violin_chain_json))
    cello_track_meta = _get_best_track_meta(Path(args.cello_chain_json))

    violin_dense_rows = _load_csv(Path(args.violin_dense_csv))
    cello_dense_rows = _load_csv(Path(args.cello_dense_csv))

    violin_passport_rows = [
        row
        for row in violin_dense_rows
        if violin_track_meta["start_time"] <= _safe_float(row.get("time_sec"), 0.0) <= violin_track_meta["end_time"]
    ]
    cello_passport_rows = [
        row
        for row in cello_dense_rows
        if cello_track_meta["start_time"] <= _safe_float(row.get("time_sec"), 0.0) <= cello_track_meta["end_time"]
    ]

    violin_passport_frame_map: dict[int, list[dict[str, Any]]] = {}
    for row in violin_passport_rows:
        violin_passport_frame_map.setdefault(_safe_int(row.get("frame_index"), -1), []).append(row)
    cello_passport_frame_map: dict[int, list[dict[str, Any]]] = {}
    for row in cello_passport_rows:
        cello_passport_frame_map.setdefault(_safe_int(row.get("frame_index"), -1), []).append(row)

    _write_progress(args.progress_json, {"status": "running", "phase": "building_harmonic_timelines"})

    violin_window_picked = _pick_frame_best_matches(window_frame_map, violin_defs, "energy")
    cello_window_picked = _pick_frame_best_matches(window_frame_map, cello_defs, "energy")
    violin_passport_picked = _pick_frame_best_matches(violin_passport_frame_map, violin_defs, "amplitude")
    cello_passport_picked = _pick_frame_best_matches(cello_passport_frame_map, cello_defs, "amplitude")

    window_reference_start = min(frame_times.values()) if frame_times else 0.0
    violin_timeline = _build_timeline_rows(
        violin_window_picked,
        selected_frames,
        frame_times,
        violin_defs,
        window_reference_start,
        len(selected_frames),
        "window_vs_violin_a5",
    )
    cello_timeline = _build_timeline_rows(
        cello_window_picked,
        selected_frames,
        frame_times,
        cello_defs,
        window_reference_start,
        len(selected_frames),
        "window_vs_cello_a5",
    )
    violin_passport_timeline = _build_timeline_rows(
        violin_passport_picked,
        sorted(violin_passport_frame_map),
        {},
        violin_defs,
        violin_track_meta["start_time"],
        violin_track_meta["frame_count"],
        "passport_violin_a5",
    )
    cello_passport_timeline = _build_timeline_rows(
        cello_passport_picked,
        sorted(cello_passport_frame_map),
        {},
        cello_defs,
        cello_track_meta["start_time"],
        cello_track_meta["frame_count"],
        "passport_cello_a5",
    )

    window_harmonic_rows = violin_timeline + cello_timeline + violin_passport_timeline + cello_passport_timeline

    violin_compare_rows, violin_summary = _score_candidate_compare(
        violin_timeline,
        violin_passport_timeline,
        violin_track_meta["harmonic_presence_profile"],
    )
    cello_compare_rows, cello_summary = _score_candidate_compare(
        cello_timeline,
        cello_passport_timeline,
        cello_track_meta["harmonic_presence_profile"],
    )

    candidate_compare_rows: list[dict[str, Any]] = []
    for row in violin_compare_rows:
        row_copy = dict(row)
        row_copy["candidate_label"] = "violin_A5_root"
        candidate_compare_rows.append(row_copy)
    for row in cello_compare_rows:
        row_copy = dict(row)
        row_copy["candidate_label"] = "cello_A5_root"
        candidate_compare_rows.append(row_copy)

    violin_score = _safe_float(violin_summary.get("overall_score"), 0.0)
    cello_score = _safe_float(cello_summary.get("overall_score"), 0.0)
    best_label = "violin_A5_root" if violin_score >= cello_score else "cello_A5_root"

    _write_csv(
        Path(args.out_window_harmonic_csv),
        window_harmonic_rows,
        [
            "source_label",
            "harmonic_index",
            "theoretical_token",
            "theoretical_hz",
            "active_frames",
            "coverage_ratio",
            "first_frame_index",
            "first_time_sec",
            "first_relative_sec",
            "last_frame_index",
            "last_time_sec",
            "mean_strength",
            "max_strength",
            "max_consecutive_frames",
            "onset_rank",
            "dominant_owner_label",
            "owner_label_counts_json",
        ],
    )
    _write_csv(
        Path(args.out_candidate_compare_csv),
        candidate_compare_rows,
        [
            "candidate_label",
            "harmonic_index",
            "window_active_frames",
            "window_coverage_ratio",
            "window_first_relative_sec",
            "window_mean_strength_norm",
            "passport_active_frames",
            "passport_coverage_ratio",
            "passport_first_relative_sec",
            "passport_mean_strength_norm",
            "coverage_match_score",
            "onset_match_score",
            "strength_match_score",
            "passport_presence_count",
            "harmonic_weight",
            "harmonic_score",
        ],
    )

    summary_lines = [
        "WINDOW HARMONIC PHASE PASSPORT COMPARE",
        "=" * 72,
        f"selected_window_frames             : {len(selected_frames)}",
        f"window_time_start_sec              : {window_reference_start:.9f}",
        "",
        "candidate_scores:",
        f"  violin_A5_root                   : {violin_score:.6f}",
        f"    h2_h1_ratio_match              : {violin_summary['h2_h1_ratio_match_score']:.6f}",
        f"    h357_match                      : {violin_summary['h357_match_score']:.6f}",
        f"    window_h2_over_h1               : {violin_summary['window_h2_over_h1_mean_strength_ratio']:.6f}",
        f"    passport_h2_over_h1             : {violin_summary['passport_h2_over_h1_mean_strength_ratio']:.6f}",
        f"  cello_A5_root                    : {cello_score:.6f}",
        f"    h2_h1_ratio_match              : {cello_summary['h2_h1_ratio_match_score']:.6f}",
        f"    h357_match                      : {cello_summary['h357_match_score']:.6f}",
        f"    window_h2_over_h1               : {cello_summary['window_h2_over_h1_mean_strength_ratio']:.6f}",
        f"    passport_h2_over_h1             : {cello_summary['passport_h2_over_h1_mean_strength_ratio']:.6f}",
        "",
        f"best_phase_timeline_match          : {best_label}",
        "",
        "window_harmonic_onset_snapshot:",
    ]
    violin_window_by_h = _row_by_harmonic(violin_timeline)
    for harmonic_index in range(1, 8):
        row = violin_window_by_h[harmonic_index]
        summary_lines.append(
            "  "
            f"h{harmonic_index}: frames={row['active_frames']} "
            f"first_rel={_safe_float(row.get('first_relative_sec'), -1.0):.6f} "
            f"owner={row['dominant_owner_label']}"
        )
    summary_lines.extend(
        [
            "",
            "interpretation:",
            "  This compare does not force an instrument label early.",
            "  It checks whether the window's harmonic entrance order and sustain pattern",
            "  behave more like violin A.5 or cello A.5 isolated-note passports.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_harmonic_phase_passport_compare",
                "selected_window_frames": len(selected_frames),
                "window_reference_start_sec": window_reference_start,
                "violin_summary": violin_summary,
                "cello_summary": cello_summary,
                "best_phase_timeline_match": best_label,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    _write_progress(
        args.progress_json,
        {
            "status": "done",
            "phase": "complete",
            "best_phase_timeline_match": best_label,
            "violin_score": violin_score,
            "cello_score": cello_score,
        },
    )


if __name__ == "__main__":
    main()
