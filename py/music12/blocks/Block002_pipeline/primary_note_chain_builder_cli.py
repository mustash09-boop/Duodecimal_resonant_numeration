from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
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


def _pitch_class(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _octave(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[0]
    except Exception:
        return ""


def _build_families_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: _safe_int(r.get("family_rank"), 999999))
    return out


def _build_proto_by_start_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("start_frame"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(
            key=lambda r: (
                -_safe_float(r.get("exciter_confidence"), 0.0),
                -_safe_int(r.get("duration_frames"), 0),
                -_safe_int(r.get("seed_count"), 0),
            )
        )
    return out


def _build_branch_by_proto_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[_safe_int(row.get("proto_exciter_id"), 0)] = row
    return out


def _build_transition_by_proto_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[_safe_int(row.get("proto_exciter_id"), 0)] = row
    return out


def _transition_mode_is_protective(mode: str) -> bool:
    return mode in {
        "SHARED_CURRENT_LED_TRANSITION",
        "INTERLEAVED_CURRENT_LED",
        "INTERLEAVED_BALANCED",
    }


def _transition_mode_is_tail_risky(mode: str) -> bool:
    return mode in {
        "SHARED_TAIL_LED_TRANSITION",
        "SHARED_BALANCED_TRANSITION",
    }


def _transition_mode_allows_early_birth(mode: str) -> bool:
    return mode in {
        "SHARED_CURRENT_LED_TRANSITION",
        "INTERLEAVED_CURRENT_LED",
        "INTERLEAVED_BALANCED",
    }


def _score_family_row(
    *,
    row: dict[str, Any],
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    prev_selected_note: str,
    frame_decay: float,
) -> tuple[float, str]:
    family_note = _normalize_note(row.get("family_root_note_micro", ""))
    family_pc = _pitch_class(family_note)
    family_oct = _octave(family_note)
    family_score = _safe_float(row.get("family_score"), 0.0)
    root_micro_count = _safe_int(row.get("root_micro_count"), 0)
    root_micro_diversity = _safe_int(row.get("root_micro_diversity"), 0)

    reasons: list[str] = []
    score = 0.0

    # Primary note-chain is not allowed to jump to a foreign pitch class.
    # If that happens, we are already in takeover territory rather than
    # preserving the exciter's own note-bearing chain.
    if family_note != proto_coarse and family_pc != proto_pc:
        return -1.0e9, "foreign_pc"

    if family_note == proto_coarse:
        score += 1.20
        reasons.append("exact_coarse")
    elif family_pc == proto_pc and family_oct == proto_oct:
        score += 0.80
        reasons.append("same_pc_same_oct")
    elif family_pc == proto_pc:
        score += 0.45
        reasons.append("same_pc")

    if prev_selected_note:
        prev_pc = _pitch_class(prev_selected_note)
        if family_note == prev_selected_note:
            score += 0.65
            reasons.append("same_as_prev")
        elif family_pc and family_pc == prev_pc:
            score += 0.25
            reasons.append("pc_continuity")

    score += min(family_score / 6.0, 1.25)
    score += min(root_micro_count / 40.0, 0.60)
    score += min(root_micro_diversity / 32.0, 0.35)
    score += frame_decay

    reason = "|".join(reasons) if reasons else "weak"
    return score, reason


def _current_identity_strength(
    *,
    family_note: str,
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    family_score: float,
    root_micro_count: int,
    root_micro_diversity: int,
) -> float:
    family_pc = _pitch_class(family_note)
    family_oct = _octave(family_note)
    score = 0.0
    if family_note == proto_coarse:
        score += 1.10
    elif family_pc == proto_pc and family_oct == proto_oct:
        score += 0.75
    elif family_pc == proto_pc:
        score += 0.35
    score += min(family_score / 10.0, 0.70)
    score += min(root_micro_count / 48.0, 0.35)
    score += min(root_micro_diversity / 40.0, 0.25)
    return score


def _competing_proto_strength(row: dict[str, Any]) -> float:
    return (
        _safe_float(row.get("exciter_confidence"), 0.0)
        + min(_safe_int(row.get("duration_frames"), 0) / 12.0, 0.25)
        + min(_safe_int(row.get("seed_count"), 0) / 12.0, 0.20)
        + min(_safe_float(row.get("mean_continuation"), 0.0) / 4.0, 0.20)
    )


def _frame_identity_is_stable(
    *,
    family_note: str,
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    family_score: float,
    root_micro_count: int,
    root_micro_diversity: int,
    weak_family_score: float,
    weak_root_micro_count: int,
    weak_root_micro_diversity: int,
) -> bool:
    family_pc = _pitch_class(family_note)
    family_oct = _octave(family_note)
    if family_note == proto_coarse:
        return True
    if family_pc == proto_pc and family_oct == proto_oct:
        return (
            family_score >= weak_family_score
            and root_micro_count >= weak_root_micro_count
            and root_micro_diversity >= weak_root_micro_diversity
        )
    return False


def _find_competing_onsets(
    *,
    proto_by_start_frame: dict[int, list[dict[str, Any]]],
    current_proto_id: int,
    current_proto_pc: str,
    frame_index: int,
    lookahead_frames: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for start_frame in range(frame_index, frame_index + max(lookahead_frames, 0) + 1):
        for row in proto_by_start_frame.get(start_frame, []):
            proto_id = _safe_int(row.get("proto_exciter_id"), 0)
            if proto_id == current_proto_id:
                continue
            proto_note = str(row.get("rescue_group_dominant_note", "")).strip() or str(row.get("coarse_note", "")).strip()
            proto_pc = _pitch_class(proto_note)
            if not proto_pc or proto_pc == current_proto_pc:
                continue
            if _safe_float(row.get("exciter_confidence"), 0.0) < min_confidence:
                continue
            found.append(row)
    found.sort(key=_competing_proto_strength, reverse=True)
    return found


def _should_transfer_priority_to_new_note(
    *,
    family_note: str,
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    family_score: float,
    root_micro_count: int,
    root_micro_diversity: int,
    selected_frames: list[dict[str, Any]],
    competing_rows: list[dict[str, Any]],
    weak_family_score: float,
    weak_root_micro_count: int,
    weak_root_micro_diversity: int,
    min_weak_streak_frames: int,
) -> bool:
    if not competing_rows:
        return False

    family_pc = _pitch_class(family_note)
    family_oct = _octave(family_note)
    current_strength = _current_identity_strength(
        family_note=family_note,
        proto_coarse=proto_coarse,
        proto_pc=proto_pc,
        proto_oct=proto_oct,
        family_score=family_score,
        root_micro_count=root_micro_count,
        root_micro_diversity=root_micro_diversity,
    )
    competing_strength = _competing_proto_strength(competing_rows[0])

    recent = selected_frames[-2:] if len(selected_frames) >= 2 else list(selected_frames)
    recent_exact = sum(1 for row in recent if row.get("selected_note_token") == proto_coarse)
    recent_same_oct = sum(
        1
        for row in recent
        if _pitch_class(str(row.get("selected_note_token", ""))) == proto_pc
        and _octave(str(row.get("selected_note_token", ""))) == proto_oct
    )

    exact_anchor_alive = family_note == proto_coarse
    stable_now = _frame_identity_is_stable(
        family_note=family_note,
        proto_coarse=proto_coarse,
        proto_pc=proto_pc,
        proto_oct=proto_oct,
        family_score=family_score,
        root_micro_count=root_micro_count,
        root_micro_diversity=root_micro_diversity,
        weak_family_score=weak_family_score,
        weak_root_micro_count=weak_root_micro_count,
        weak_root_micro_diversity=weak_root_micro_diversity,
    )
    stable_same_octave = stable_now and recent_same_oct >= 1

    if exact_anchor_alive and current_strength >= competing_strength:
        return False
    if stable_same_octave and current_strength + 0.15 >= competing_strength:
        return False

    weak_streak = 0
    probe_rows = list(selected_frames[-max(min_weak_streak_frames - 1, 0):])
    probe_rows.append(
        {
            "selected_note_token": family_note,
            "family_score": family_score,
            "root_micro_count": root_micro_count,
            "root_micro_diversity": root_micro_diversity,
        }
    )
    for row in reversed(probe_rows):
        row_note = str(row.get("selected_note_token", ""))
        row_stable = _frame_identity_is_stable(
            family_note=row_note,
            proto_coarse=proto_coarse,
            proto_pc=proto_pc,
            proto_oct=proto_oct,
            family_score=_safe_float(row.get("family_score"), 0.0),
            root_micro_count=_safe_int(row.get("root_micro_count"), 0),
            root_micro_diversity=_safe_int(row.get("root_micro_diversity"), 0),
            weak_family_score=weak_family_score,
            weak_root_micro_count=weak_root_micro_count,
            weak_root_micro_diversity=weak_root_micro_diversity,
        )
        if row_stable:
            break
        weak_streak += 1

    weakened_identity = (
        not exact_anchor_alive
        and recent_exact == 0
        and weak_streak >= min_weak_streak_frames
    )
    if not weakened_identity:
        return False

    return competing_strength > (current_strength + 0.05)


def _retrospective_trim_weak_tail(
    *,
    selected_frames: list[dict[str, Any]],
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    weak_family_score: float,
    weak_root_micro_count: int,
    weak_root_micro_diversity: int,
    max_lookback_frames: int,
) -> int:
    if not selected_frames or max_lookback_frames <= 0:
        return 0

    trim_count = 0
    tail = selected_frames[-max_lookback_frames:]
    for row in reversed(tail):
        row_note = str(row.get("selected_note_token", ""))
        row_stable = _frame_identity_is_stable(
            family_note=row_note,
            proto_coarse=proto_coarse,
            proto_pc=proto_pc,
            proto_oct=proto_oct,
            family_score=_safe_float(row.get("family_score"), 0.0),
            root_micro_count=_safe_int(row.get("root_micro_count"), 0),
            root_micro_diversity=_safe_int(row.get("root_micro_diversity"), 0),
            weak_family_score=weak_family_score,
            weak_root_micro_count=weak_root_micro_count,
            weak_root_micro_diversity=weak_root_micro_diversity,
        )
        if row_stable:
            break
        trim_count += 1

    if trim_count <= 0:
        return 0
    del selected_frames[-trim_count:]
    return trim_count


def _retrospective_confirm_birth(
    *,
    selected_frames: list[dict[str, Any]],
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    weak_family_score: float,
    weak_root_micro_count: int,
    weak_root_micro_diversity: int,
    lookahead_frames: int,
    min_stable_frames: int,
    min_same_note_frames: int,
) -> int:
    if not selected_frames or lookahead_frames <= 0:
        return 0

    trimmed = 0
    while selected_frames:
        first_note = str(selected_frames[0].get("selected_note_token", ""))
        window = selected_frames[: max(1, lookahead_frames + 1)]
        stable_count = 0
        same_note_count = 0
        same_octave_count = 0
        exact_count = 0

        for row in window:
            row_note = str(row.get("selected_note_token", ""))
            if row_note == first_note:
                same_note_count += 1
            if row_note == proto_coarse:
                exact_count += 1
            if _pitch_class(row_note) == proto_pc and _octave(row_note) == proto_oct:
                same_octave_count += 1
            if _frame_identity_is_stable(
                family_note=row_note,
                proto_coarse=proto_coarse,
                proto_pc=proto_pc,
                proto_oct=proto_oct,
                family_score=_safe_float(row.get("family_score"), 0.0),
                root_micro_count=_safe_int(row.get("root_micro_count"), 0),
                root_micro_diversity=_safe_int(row.get("root_micro_diversity"), 0),
                weak_family_score=weak_family_score,
                weak_root_micro_count=weak_root_micro_count,
                weak_root_micro_diversity=weak_root_micro_diversity,
            ):
                stable_count += 1

        confirmed = (
            stable_count >= min_stable_frames
            and same_octave_count >= min_stable_frames
            and (same_note_count >= min_same_note_frames or exact_count >= 1)
        )
        if confirmed:
            break

        selected_frames.pop(0)
        trimmed += 1

    return trimmed


def _birth_is_suspicious(
    *,
    proto_by_start_frame: dict[int, list[dict[str, Any]]],
    branch_by_proto_id: dict[int, dict[str, Any]],
    current_proto_id: int,
    start_frame: int,
    proto_coarse: str,
    proto_pc: str,
    proto_oct: str,
    selected_frames: list[dict[str, Any]],
    min_competing_confidence: float,
    rhythm_window_frames: int,
    inspect_frames: int,
    transition_mode: str,
) -> bool:
    if not selected_frames:
        return False

    branch_row = branch_by_proto_id.get(current_proto_id, {})
    branch_label = str(branch_row.get("branch_label", "")).strip()
    route_label = str(branch_row.get("route_label", "")).strip()
    rescue_source = str(branch_row.get("rescue_source", "")).strip()

    # Strict birth confirmation is only meaningful for uncertain branches.
    # Confident pitched notechain births should not be pruned here.
    uncertain_branch = (
        route_label == "notechain_fallback"
        or branch_label in {"event", "unresolved"}
        or bool(rescue_source)
    )
    protective_transition = _transition_mode_is_protective(transition_mode)
    tail_risky_transition = _transition_mode_is_tail_risky(transition_mode)
    if not uncertain_branch and not tail_risky_transition:
        return False

    inspected = selected_frames[: max(1, inspect_frames)]
    exact_count = sum(1 for row in inspected if str(row.get("selected_note_token", "")) == proto_coarse)
    same_oct_count = sum(
        1
        for row in inspected
        if _pitch_class(str(row.get("selected_note_token", ""))) == proto_pc
        and _octave(str(row.get("selected_note_token", ""))) == proto_oct
    )
    repeated_first = 0
    first_note = str(inspected[0].get("selected_note_token", ""))
    for row in inspected:
        if str(row.get("selected_note_token", "")) == first_note:
            repeated_first += 1

    weak_birth = (
        exact_count == 0
        or same_oct_count <= 1
        or repeated_first <= 1
    )
    if route_label == "notechain_fallback" or branch_label in {"event", "unresolved"} or rescue_source:
        weak_birth = weak_birth or same_oct_count <= 2
    if tail_risky_transition:
        weak_birth = weak_birth or same_oct_count <= 2 or repeated_first <= 2
    if protective_transition and exact_count >= 1 and same_oct_count >= 2 and repeated_first >= 2:
        return False
    if not weak_birth:
        return False

    # For confident pitched births, only apply the strict retrospective check
    # when the birth is truly weak. This keeps real note births from being
    # over-pruned just because the texture is busy.
    previous_tail_pressure = False
    for frame in range(start_frame - rhythm_window_frames, start_frame):
        for row in proto_by_start_frame.get(frame, []):
            proto_id = _safe_int(row.get("proto_exciter_id"), 0)
            if proto_id == current_proto_id:
                continue
            proto_note = str(row.get("rescue_group_dominant_note", "")).strip() or str(row.get("coarse_note", "")).strip()
            other_pc = _pitch_class(proto_note)
            if not other_pc or other_pc == proto_pc:
                continue
            if _safe_float(row.get("exciter_confidence"), 0.0) < min_competing_confidence:
                continue
            if _safe_int(row.get("end_frame"), 0) < start_frame - rhythm_window_frames:
                continue
            previous_tail_pressure = True
            break
        if previous_tail_pressure:
            break

    if not previous_tail_pressure:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build primary note chains from proto-exciters before bridge/companion takeover."
    )
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-chain-frames-csv", required=True)
    ap.add_argument("--out-chains-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--all-proto-exciters-csv", default="")
    ap.add_argument("--branch-analysis-csv", default="")
    ap.add_argument("--transition-prior-csv", default="")
    ap.add_argument("--lookahead-frames", type=int, default=12)
    ap.add_argument("--min-chain-score", type=float, default=1.55)
    ap.add_argument("--min-chain-frames", type=int, default=2)
    ap.add_argument("--max-gap-frames", type=int, default=1)
    ap.add_argument("--strict-initial-frames", type=int, default=4)
    ap.add_argument("--rescue-initial-frames", type=int, default=3)
    ap.add_argument("--rescue-min-chain-score", type=float, default=0.95)
    ap.add_argument("--rescue-min-exciter-confidence", type=float, default=0.45)
    ap.add_argument("--rescue-min-root-micro-count", type=int, default=8)
    ap.add_argument("--rescue-min-root-micro-diversity", type=int, default=6)
    ap.add_argument("--allow-single-frame-rescue", action="store_true")
    ap.add_argument("--new-note-lookahead-frames", type=int, default=2)
    ap.add_argument("--new-note-min-age-frames", type=int, default=2)
    ap.add_argument("--new-note-min-confidence", type=float, default=0.58)
    ap.add_argument("--weakened-family-score", type=float, default=2.40)
    ap.add_argument("--weakened-root-micro-count", type=int, default=14)
    ap.add_argument("--weakened-root-micro-diversity", type=int, default=10)
    ap.add_argument("--new-note-weak-streak-frames", type=int, default=2)
    ap.add_argument("--retro-lookback-frames", type=int, default=3)
    ap.add_argument("--birth-confirm-lookahead-frames", type=int, default=3)
    ap.add_argument("--birth-confirm-min-stable-frames", type=int, default=2)
    ap.add_argument("--birth-confirm-min-same-note-frames", type=int, default=2)
    ap.add_argument("--birth-confirm-rhythm-window-frames", type=int, default=2)
    ap.add_argument("--birth-confirm-inspect-frames", type=int, default=3)
    args = ap.parse_args()

    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    all_proto_rows = (
        _load_csv(Path(args.all_proto_exciters_csv))
        if str(args.all_proto_exciters_csv).strip()
        else proto_rows
    )
    branch_rows = _load_csv(Path(args.branch_analysis_csv)) if str(args.branch_analysis_csv).strip() else []
    transition_rows = _load_csv(Path(args.transition_prior_csv)) if str(args.transition_prior_csv).strip() else []
    family_rows = _load_csv(Path(args.micro_families_csv))
    families_by_frame = _build_families_by_frame(family_rows)
    proto_by_start_frame = _build_proto_by_start_frame(all_proto_rows)
    branch_by_proto_id = _build_branch_by_proto_id(branch_rows)
    transition_by_proto_id = _build_transition_by_proto_id(transition_rows)

    frame_rows_out: list[dict[str, Any]] = []
    chain_rows_out: list[dict[str, Any]] = []

    for proto in proto_rows:
        proto_id = _safe_int(proto.get("proto_exciter_id"), 0)
        proto_coarse = str(proto.get("rescue_group_dominant_note", "")).strip() or str(proto.get("coarse_note", "")).strip()
        proto_pc = _pitch_class(proto_coarse)
        proto_oct = _octave(proto_coarse)
        start_frame = _safe_int(proto.get("start_frame"), 0)
        end_frame = _safe_int(proto.get("end_frame"), start_frame)
        exciter_confidence = _safe_float(proto.get("exciter_confidence"), 0.0)
        transition_row = transition_by_proto_id.get(proto_id, {})
        transition_mode = str(transition_row.get("transition_mode", "")).strip()

        window_end = end_frame + int(args.lookahead_frames)
        prev_selected_note = ""
        selected_frames: list[dict[str, Any]] = []
        gaps = 0
        same_octave_hits = 0
        used_rescue = False
        handoff_to_new_note = False
        handoff_frame = -1
        handoff_note = ""
        handoff_strength = 0.0
        retro_trimmed_frames = 0
        birth_trimmed_frames = 0
        suspicious_birth = False

        for frame_index in range(start_frame, window_end + 1):
            rows = families_by_frame.get(frame_index, [])
            if not rows:
                gaps += 1
                if gaps > int(args.max_gap_frames) and selected_frames:
                    break
                continue

            frame_distance = max(0, frame_index - start_frame)
            frame_decay = max(0.0, 0.35 - 0.03 * frame_distance) * min(exciter_confidence + 0.2, 1.0)
            transition_birth_relax = _transition_mode_allows_early_birth(transition_mode) and frame_distance <= 2
            transition_tail_risk = _transition_mode_is_tail_risky(transition_mode) and frame_distance <= 2
            local_min_chain_score = float(args.min_chain_score)
            if transition_birth_relax:
                local_min_chain_score -= 0.18
            if transition_tail_risk:
                local_min_chain_score += 0.10

            best_row: dict[str, Any] | None = None
            best_score = -1.0
            best_reason = ""
            rescue_row: dict[str, Any] | None = None
            rescue_score = -1.0
            rescue_reason = ""
            for row in rows[:8]:
                family_note = _normalize_note(row.get("family_root_note_micro", ""))
                family_pc = _pitch_class(family_note)
                family_oct = _octave(family_note)
                frame_distance = max(0, frame_index - start_frame)
                root_micro_count = _safe_int(row.get("root_micro_count"), 0)
                root_micro_diversity = _safe_int(row.get("root_micro_diversity"), 0)

                # In the earliest phase of note-chain formation, octave identity
                # must stay tight. Same pitch class in a different octave is too
                # permissive and quickly becomes bridge takeover.
                if frame_distance <= int(args.strict_initial_frames):
                    if not (
                        family_note == proto_coarse
                        or (family_pc == proto_pc and family_oct == proto_oct)
                    ):
                        continue

                # Until we have at least one same-octave confirmation, do not
                # let the chain migrate to another octave of the same pitch class.
                if same_octave_hits == 0:
                    if family_note != proto_coarse and not (
                        family_pc == proto_pc and family_oct == proto_oct
                    ):
                        continue

                score, reason = _score_family_row(
                    row=row,
                    proto_coarse=proto_coarse,
                    proto_pc=proto_pc,
                    proto_oct=proto_oct,
                    prev_selected_note=prev_selected_note,
                    frame_decay=frame_decay,
                )
                if score > best_score:
                    best_score = score
                    best_row = row
                    best_reason = reason

                # Proto rescue: if an exciter is clearly visible at birth but the
                # standard chain threshold is too strict, allow a small exact or
                # same-octave foothold instead of dropping the note entirely.
                if (
                    exciter_confidence >= float(args.rescue_min_exciter_confidence)
                    and frame_distance <= int(args.rescue_initial_frames)
                    and root_micro_count >= int(args.rescue_min_root_micro_count)
                    and root_micro_diversity >= int(args.rescue_min_root_micro_diversity)
                    and (
                        family_note == proto_coarse
                        or (family_pc == proto_pc and family_oct == proto_oct)
                    )
                ):
                    local_rescue = 0.0
                    local_reasons: list[str] = []
                    if family_note == proto_coarse:
                        local_rescue += 0.95
                        local_reasons.append("rescue_exact")
                    else:
                        local_rescue += 0.70
                        local_reasons.append("rescue_same_oct")
                    local_rescue += min(_safe_float(row.get("family_score"), 0.0) / 10.0, 0.80)
                    local_rescue += min(root_micro_count / 48.0, 0.35)
                    local_rescue += min(root_micro_diversity / 40.0, 0.20)
                    local_rescue += frame_decay
                    if prev_selected_note and family_note == prev_selected_note:
                        local_rescue += 0.25
                        local_reasons.append("rescue_prev")
                    if local_rescue > rescue_score:
                        rescue_score = local_rescue
                        rescue_row = row
                        rescue_reason = "|".join(local_reasons)

            rescued_here = False
            if best_row is None or best_score < local_min_chain_score:
                if rescue_row is not None and rescue_score >= float(args.rescue_min_chain_score):
                    best_row = rescue_row
                    best_score = rescue_score
                    best_reason = f"{rescue_reason}|proto_rescue" if rescue_reason else "proto_rescue"
                    rescued_here = True
                    used_rescue = True
                else:
                    gaps += 1
                    if gaps > int(args.max_gap_frames) and selected_frames:
                        break
                    continue

            gaps = 0
            family_note = _normalize_note(best_row.get("family_root_note_micro", ""))
            family_score = _safe_float(best_row.get("family_score"), 0.0)
            root_micro_count = _safe_int(best_row.get("root_micro_count"), 0)
            root_micro_diversity = _safe_int(best_row.get("root_micro_diversity"), 0)

            competing_rows = []
            frame_age = max(0, frame_index - start_frame)
            if frame_age >= int(args.new_note_min_age_frames):
                competing_rows = _find_competing_onsets(
                    proto_by_start_frame=proto_by_start_frame,
                    current_proto_id=proto_id,
                    current_proto_pc=proto_pc,
                    frame_index=frame_index,
                    lookahead_frames=int(args.new_note_lookahead_frames),
                    min_confidence=float(args.new_note_min_confidence),
                )
            if _should_transfer_priority_to_new_note(
                family_note=family_note,
                proto_coarse=proto_coarse,
                proto_pc=proto_pc,
                proto_oct=proto_oct,
                family_score=family_score,
                root_micro_count=root_micro_count,
                root_micro_diversity=root_micro_diversity,
                    selected_frames=selected_frames,
                    competing_rows=competing_rows,
                    weak_family_score=float(args.weakened_family_score),
                    weak_root_micro_count=int(args.weakened_root_micro_count),
                    weak_root_micro_diversity=int(args.weakened_root_micro_diversity),
                    min_weak_streak_frames=int(args.new_note_weak_streak_frames),
                ):
                strongest = competing_rows[0]
                handoff_to_new_note = True
                handoff_frame = frame_index
                handoff_note = str(strongest.get("rescue_group_dominant_note", "")).strip() or str(strongest.get("coarse_note", "")).strip()
                handoff_strength = _competing_proto_strength(strongest)
                retro_trimmed_frames = _retrospective_trim_weak_tail(
                    selected_frames=selected_frames,
                    proto_coarse=proto_coarse,
                    proto_pc=proto_pc,
                    proto_oct=proto_oct,
                    weak_family_score=float(args.weakened_family_score),
                    weak_root_micro_count=int(args.weakened_root_micro_count),
                    weak_root_micro_diversity=int(args.weakened_root_micro_diversity),
                    max_lookback_frames=int(args.retro_lookback_frames),
                )
                break

            prev_selected_note = family_note
            if _octave(family_note) == proto_oct:
                same_octave_hits += 1

            selected_frames.append(
                {
                    "proto_exciter_id": proto_id,
                    "frame_index": frame_index,
                    "coarse_note": proto_coarse,
                    "selected_note_token": family_note,
                    "selected_pitch_class": _pitch_class(family_note),
                    "selected_octave": _octave(family_note),
                    "chain_score": best_score,
                    "selection_reason": best_reason,
                    "family_score": family_score,
                    "root_micro_count": root_micro_count,
                    "root_micro_diversity": root_micro_diversity,
                    "rescued": int(rescued_here),
                }
            )

        min_required_frames = int(args.min_chain_frames)
        chain_mode = "standard"
        suspicious_birth = _birth_is_suspicious(
            proto_by_start_frame=proto_by_start_frame,
            branch_by_proto_id=branch_by_proto_id,
            current_proto_id=proto_id,
            start_frame=start_frame,
            proto_coarse=proto_coarse,
            proto_pc=proto_pc,
            proto_oct=proto_oct,
            selected_frames=selected_frames,
            min_competing_confidence=float(args.new_note_min_confidence),
            rhythm_window_frames=int(args.birth_confirm_rhythm_window_frames),
            inspect_frames=int(args.birth_confirm_inspect_frames),
            transition_mode=transition_mode,
        )
        if suspicious_birth:
            birth_trimmed_frames = _retrospective_confirm_birth(
                selected_frames=selected_frames,
                proto_coarse=proto_coarse,
                proto_pc=proto_pc,
                proto_oct=proto_oct,
                weak_family_score=float(args.weakened_family_score),
                weak_root_micro_count=int(args.weakened_root_micro_count),
                weak_root_micro_diversity=int(args.weakened_root_micro_diversity),
                lookahead_frames=int(args.birth_confirm_lookahead_frames),
                min_stable_frames=int(args.birth_confirm_min_stable_frames),
                min_same_note_frames=int(args.birth_confirm_min_same_note_frames),
            )
        if used_rescue:
            chain_mode = "proto_rescue"
            if bool(args.allow_single_frame_rescue):
                min_required_frames = 1
        if len(selected_frames) < min_required_frames:
            continue

        frame_rows_out.extend(
            {
                "proto_exciter_id": row["proto_exciter_id"],
                "frame_index": row["frame_index"],
                "coarse_note": row["coarse_note"],
                "selected_note_token": row["selected_note_token"],
                "selected_pitch_class": row["selected_pitch_class"],
                "selected_octave": row["selected_octave"],
                "chain_score": f"{row['chain_score']:.9f}",
                "selection_reason": row["selection_reason"],
                "family_score": f"{row['family_score']:.9f}",
                "root_micro_count": row["root_micro_count"],
                "root_micro_diversity": row["root_micro_diversity"],
                "rescued": row["rescued"],
            }
            for row in selected_frames
        )

        note_counts = Counter(row["selected_note_token"] for row in selected_frames if row["selected_note_token"])
        exact_coarse_frames = sum(1 for row in selected_frames if row["selected_note_token"] == proto_coarse)
        pitchclass_frames = sum(1 for row in selected_frames if row["selected_pitch_class"] == proto_pc)
        same_octave_frames = sum(1 for row in selected_frames if row["selected_octave"] == proto_oct)
        bridge_resistance = exact_coarse_frames / max(len(selected_frames), 1)
        chain_rows_out.append(
            {
                "proto_exciter_id": proto_id,
                "coarse_note": proto_coarse,
                "chain_start_frame": selected_frames[0]["frame_index"],
                "chain_end_frame": selected_frames[-1]["frame_index"],
                "chain_duration_frames": selected_frames[-1]["frame_index"] - selected_frames[0]["frame_index"] + 1,
                "chain_frame_count": len(selected_frames),
                "chain_mode": chain_mode,
                "dominant_note_token": note_counts.most_common(1)[0][0] if note_counts else "",
                "handoff_to_new_note": int(handoff_to_new_note),
                "handoff_frame": handoff_frame if handoff_to_new_note else "",
                "handoff_note": handoff_note,
                "handoff_strength": f"{handoff_strength:.9f}" if handoff_to_new_note else "",
                "retro_trimmed_frames": retro_trimmed_frames,
                "birth_trimmed_frames": birth_trimmed_frames,
                "suspicious_birth": int(suspicious_birth),
                "transition_prior_mode": transition_mode,
                "exact_coarse_frames": exact_coarse_frames,
                "pitchclass_frames": pitchclass_frames,
                "same_octave_frames": same_octave_frames,
                "mean_chain_score": f"{(sum(row['chain_score'] for row in selected_frames) / len(selected_frames)):.9f}",
                "bridge_resistance": f"{bridge_resistance:.9f}",
                "selected_notes_json": json.dumps(dict(note_counts), ensure_ascii=False, sort_keys=True),
            }
        )

    out_chain_frames = Path(args.out_chain_frames_csv)
    out_chains = Path(args.out_chains_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_chain_frames.parent.mkdir(parents=True, exist_ok=True)

    frame_fields = [
        "proto_exciter_id",
        "frame_index",
        "coarse_note",
        "selected_note_token",
        "selected_pitch_class",
        "selected_octave",
        "chain_score",
        "selection_reason",
        "family_score",
        "root_micro_count",
        "root_micro_diversity",
        "rescued",
    ]
    with out_chain_frames.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_fields)
        w.writeheader()
        w.writerows(frame_rows_out)

    chain_fields = [
        "proto_exciter_id",
        "coarse_note",
        "chain_start_frame",
        "chain_end_frame",
        "chain_duration_frames",
        "chain_frame_count",
        "chain_mode",
        "dominant_note_token",
        "handoff_to_new_note",
        "handoff_frame",
        "handoff_note",
        "handoff_strength",
        "retro_trimmed_frames",
        "birth_trimmed_frames",
        "suspicious_birth",
        "transition_prior_mode",
        "exact_coarse_frames",
        "pitchclass_frames",
        "same_octave_frames",
        "mean_chain_score",
        "bridge_resistance",
        "selected_notes_json",
    ]
    with out_chains.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=chain_fields)
        w.writeheader()
        w.writerows(chain_rows_out)

    bridge_res_values = [_safe_float(row.get("bridge_resistance"), 0.0) for row in chain_rows_out]
    summary = {
        "stage": "primary_note_chain_builder",
        "inputs": {
            "proto_exciters_csv": args.proto_exciters_csv,
            "all_proto_exciters_csv": args.all_proto_exciters_csv,
            "branch_analysis_csv": args.branch_analysis_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "lookahead_frames": int(args.lookahead_frames),
            "min_chain_score": float(args.min_chain_score),
            "min_chain_frames": int(args.min_chain_frames),
            "max_gap_frames": int(args.max_gap_frames),
            "strict_initial_frames": int(args.strict_initial_frames),
            "rescue_initial_frames": int(args.rescue_initial_frames),
            "rescue_min_chain_score": float(args.rescue_min_chain_score),
            "rescue_min_exciter_confidence": float(args.rescue_min_exciter_confidence),
            "rescue_min_root_micro_count": int(args.rescue_min_root_micro_count),
            "rescue_min_root_micro_diversity": int(args.rescue_min_root_micro_diversity),
            "allow_single_frame_rescue": bool(args.allow_single_frame_rescue),
            "new_note_lookahead_frames": int(args.new_note_lookahead_frames),
            "new_note_min_age_frames": int(args.new_note_min_age_frames),
            "new_note_min_confidence": float(args.new_note_min_confidence),
            "weakened_family_score": float(args.weakened_family_score),
            "weakened_root_micro_count": int(args.weakened_root_micro_count),
            "weakened_root_micro_diversity": int(args.weakened_root_micro_diversity),
            "new_note_weak_streak_frames": int(args.new_note_weak_streak_frames),
            "retro_lookback_frames": int(args.retro_lookback_frames),
            "birth_confirm_lookahead_frames": int(args.birth_confirm_lookahead_frames),
            "birth_confirm_min_stable_frames": int(args.birth_confirm_min_stable_frames),
            "birth_confirm_min_same_note_frames": int(args.birth_confirm_min_same_note_frames),
            "birth_confirm_rhythm_window_frames": int(args.birth_confirm_rhythm_window_frames),
            "birth_confirm_inspect_frames": int(args.birth_confirm_inspect_frames),
        },
        "result": {
            "chain_count": len(chain_rows_out),
            "chain_frame_rows": len(frame_rows_out),
            "mean_bridge_resistance": sum(bridge_res_values) / max(len(bridge_res_values), 1),
            "rescued_chain_count": sum(1 for row in chain_rows_out if str(row.get("chain_mode", "")) == "proto_rescue"),
            "new_note_handoff_count": sum(1 for row in chain_rows_out if _safe_int(row.get("handoff_to_new_note"), 0) == 1),
            "retro_trimmed_total_frames": sum(_safe_int(row.get("retro_trimmed_frames"), 0) for row in chain_rows_out),
            "birth_trimmed_total_frames": sum(_safe_int(row.get("birth_trimmed_frames"), 0) for row in chain_rows_out),
            "suspicious_birth_count": sum(_safe_int(row.get("suspicious_birth"), 0) for row in chain_rows_out),
        },
    }

    lines = [
        "PRIMARY NOTE CHAIN BUILD",
        "=" * 72,
        f"chain_count            : {len(chain_rows_out)}",
        f"chain_frame_rows       : {len(frame_rows_out)}",
        f"mean_bridge_resistance : {summary['result']['mean_bridge_resistance']:.6f}",
        f"rescued_chain_count    : {summary['result']['rescued_chain_count']}",
        f"new_note_handoff_count : {summary['result']['new_note_handoff_count']}",
        f"retro_trimmed_frames   : {summary['result']['retro_trimmed_total_frames']}",
        f"birth_trimmed_frames   : {summary['result']['birth_trimmed_total_frames']}",
        f"suspicious_birth_count : {summary['result']['suspicious_birth_count']}",
    ]
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
