from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from music12.core.notation12 import token_to_abs_semitone_index


SEMANTIC_NOTE = (
    "This stage does NOT produce true f0. "
    "It stabilizes one or more representative chain candidates in short windows. "
    "No strongest peak is used for decision making. "
    "Any strongest peak data, if present in source rows, is treated as debug only."
)


@dataclass(frozen=True)
class FrameRow:
    raw: dict[str, str]

    frame_index: int
    time_sec: float
    candidate_count: int

    chosen_rc_note: str
    chosen_rc_hz: float
    chosen_rc_energy: float
    chain_score: float

    support_hits: int

    best_theoretical_root_token: str
    best_theoretical_root_score: float
    best_theoretical_chain_string: str
    matched_harmonics_same_frame: str
    matched_harmonics_window: str
    missing_harmonics_window: str
    extra_tokens_window: str
    spiral_match_count: int
    spiral_consistency_score: float
    window_chain_match_score: float
    theoretical_chain_verdict: str

    @property
    def has_rc(self) -> bool:
        return bool(self.chosen_rc_note)

    @property
    def abs_rc_position(self) -> Optional[int]:
        if not self.chosen_rc_note:
            return None
        try:
            return token_to_abs_semitone_index(self.chosen_rc_note)
        except Exception:
            return None


@dataclass(frozen=True)
class PolyCandidate:
    frame_index: int
    time_sec: float

    candidate_note: str
    candidate_hz: float
    candidate_energy: float
    candidate_chain_score: float

    best_theoretical_root_token: str
    best_theoretical_root_score: float
    best_theoretical_chain_string: str

    matched_harmonics_same_frame: str
    matched_harmonics_window: str
    missing_harmonics_window: str
    extra_tokens_window: str

    spiral_match_count: int
    spiral_consistency_score: float
    window_chain_match_score: float
    theoretical_chain_verdict: str

    source_row_chosen_rc_note: str
    source_row_chosen_rc_hz: float
    source_row_chosen_rc_energy: float
    source_row_chain_score: float

    @property
    def has_candidate(self) -> bool:
        return bool(self.candidate_note)

    @property
    def abs_candidate_position(self) -> Optional[int]:
        if not self.candidate_note:
            return None
        try:
            return token_to_abs_semitone_index(self.candidate_note)
        except Exception:
            return None

    @property
    def derived_support_hits(self) -> int:
        return _count_harmonics_field(self.matched_harmonics_window)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _count_harmonics_field(value: str) -> int:
    value = (value or "").strip()
    if not value:
        return 0
    return len([x for x in value.split() if x.strip()])


def load_framewise_with_theory_csv(path: Path) -> List[FrameRow]:
    rows: List[FrameRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            support_hits = 0
            for h in [2, 3, 4, 5, 6, 7, 8]:
                hit = _to_int(r.get(f"support_h{h}_hit", ""), 0)
                support_hits += 1 if hit else 0

            rows.append(
                FrameRow(
                    raw=dict(r),
                    frame_index=_to_int(r.get("frame_index", "")),
                    time_sec=_to_float(r.get("time_sec", "")),
                    candidate_count=_to_int(r.get("candidate_count", "")),

                    chosen_rc_note=_to_str(r.get("chosen_rc_note", "")),
                    chosen_rc_hz=_to_float(r.get("chosen_rc_hz", "")),
                    chosen_rc_energy=_to_float(r.get("chosen_rc_energy", "")),
                    chain_score=_to_float(r.get("chain_score", "")),

                    support_hits=support_hits,

                    best_theoretical_root_token=_to_str(r.get("best_theoretical_root_token", "")),
                    best_theoretical_root_score=_to_float(r.get("best_theoretical_root_score", "")),
                    best_theoretical_chain_string=_to_str(r.get("best_theoretical_chain_string", "")),
                    matched_harmonics_same_frame=_to_str(r.get("matched_harmonics_same_frame", "")),
                    matched_harmonics_window=_to_str(r.get("matched_harmonics_window", "")),
                    missing_harmonics_window=_to_str(r.get("missing_harmonics_window", "")),
                    extra_tokens_window=_to_str(r.get("extra_tokens_window", "")),
                    spiral_match_count=_to_int(r.get("spiral_match_count", "")),
                    spiral_consistency_score=_to_float(r.get("spiral_consistency_score", "")),
                    window_chain_match_score=_to_float(r.get("window_chain_match_score", "")),
                    theoretical_chain_verdict=_to_str(r.get("theoretical_chain_verdict", "")),
                )
            )

    rows.sort(key=lambda x: x.frame_index)
    return rows


def _extract_polyphonic_candidates(row: FrameRow) -> List[PolyCandidate]:
    raw = _to_str(row.raw.get("polyphonic_theory_json", ""))
    if not raw:
        return []

    try:
        payload = json.loads(raw)
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    out: List[PolyCandidate] = []

    for item in payload:
        if not isinstance(item, dict):
            continue

        candidate_note = _to_str(item.get("candidate_note", ""))
        best_root = _to_str(item.get("best_theoretical_root_token", ""))

        if not candidate_note:
            continue
        if not best_root:
            continue

        out.append(
            PolyCandidate(
                frame_index=row.frame_index,
                time_sec=row.time_sec,

                candidate_note=candidate_note,
                candidate_hz=_to_float(item.get("candidate_hz", "")),
                candidate_energy=_to_float(item.get("candidate_energy", "")),
                candidate_chain_score=_to_float(item.get("candidate_chain_score", "")),

                best_theoretical_root_token=best_root,
                best_theoretical_root_score=_to_float(item.get("best_theoretical_root_score", "")),
                best_theoretical_chain_string=_to_str(item.get("best_theoretical_chain_string", "")),

                matched_harmonics_same_frame=_to_str(item.get("matched_harmonics_same_frame", "")),
                matched_harmonics_window=_to_str(item.get("matched_harmonics_window", "")),
                missing_harmonics_window=_to_str(item.get("missing_harmonics_window", "")),
                extra_tokens_window=_to_str(item.get("extra_tokens_window", "")),

                spiral_match_count=_to_int(item.get("spiral_match_count", "")),
                spiral_consistency_score=_to_float(item.get("spiral_consistency_score", "")),
                window_chain_match_score=_to_float(item.get("window_chain_match_score", "")),
                theoretical_chain_verdict=_to_str(item.get("theoretical_chain_verdict", "")),

                source_row_chosen_rc_note=row.chosen_rc_note,
                source_row_chosen_rc_hz=row.chosen_rc_hz,
                source_row_chosen_rc_energy=row.chosen_rc_energy,
                source_row_chain_score=row.chain_score,
            )
        )

    return out


def local_candidate_stability(center: PolyCandidate, neighbors: List[PolyCandidate]) -> float:
    c = center.abs_candidate_position
    if c is None:
        return 0.0

    valid = []
    for n in neighbors:
        if n.abs_candidate_position is None:
            continue
        valid.append(abs(n.abs_candidate_position - c))

    if not valid:
        return 0.0

    mean_delta = sum(valid) / len(valid)
    return max(0.0, 4.0 - mean_delta)


def classify_candidate_role(
    center: PolyCandidate,
    prev_candidate: Optional[PolyCandidate],
    next_candidate: Optional[PolyCandidate],
) -> str:
    if not center.has_candidate:
        return "NO_CHAIN"

    prev_energy = center.candidate_energy if prev_candidate is None else prev_candidate.candidate_energy
    next_energy = center.candidate_energy if next_candidate is None else next_candidate.candidate_energy

    rising = center.candidate_energy > prev_energy
    falling = next_energy < center.candidate_energy

    if center.theoretical_chain_verdict in {"CHAIN_CONFIRMED"}:
        if not rising:
            return "STABLE"

    if rising and center.derived_support_hits <= 2:
        return "ATTACK"

    if center.derived_support_hits >= 3 and not rising:
        return "STABLE"

    if center.derived_support_hits >= 2:
        return "CONVERGING"

    if falling:
        return "DECAY"

    return "UNDEFINED"


def build_candidate_score(candidate: PolyCandidate, role: str, stability: float) -> tuple[float, str]:
    verdict_bonus_map = {
        "CHAIN_CONFIRMED": 8.0,
        "CHAIN_PARTIAL": 4.0,
        "CHAIN_WEAK": 1.0,
        "CHAIN_UNCERTAIN": 0.0,
    }

    role_bonus = {
        "STABLE": 6.0,
        "CONVERGING": 3.0,
        "DECAY": 1.0,
        "ATTACK": -1.0,
        "UNDEFINED": 0.0,
        "NO_CHAIN": -5.0,
    }.get(role, 0.0)

    verdict_bonus = verdict_bonus_map.get(candidate.theoretical_chain_verdict, 0.0)
    matched_same_frame_count = _count_harmonics_field(candidate.matched_harmonics_same_frame)
    matched_window_count = _count_harmonics_field(candidate.matched_harmonics_window)
    missing_window_count = _count_harmonics_field(candidate.missing_harmonics_window)
    extra_window_count = len([x for x in candidate.extra_tokens_window.split("|") if x.strip()])

    score = (
        role_bonus
        + verdict_bonus
        + candidate.derived_support_hits * 1.5
        + candidate.candidate_chain_score * 1.0
        + stability * 1.0
        + candidate.window_chain_match_score * 1.2
        + candidate.spiral_consistency_score * 6.0
        + candidate.spiral_match_count * 0.8
        + matched_same_frame_count * 0.8
        + matched_window_count * 0.6
        - missing_window_count * 0.35
        - extra_window_count * 0.20
        - candidate.candidate_energy * 0.05
    )

    reason = (
        f"{role}; "
        f"verdict={candidate.theoretical_chain_verdict}; "
        f"support_hits={candidate.derived_support_hits}; "
        f"stability={stability:.3f}; "
        f"candidate_chain_score={candidate.candidate_chain_score:.3f}; "
        f"window_chain_match_score={candidate.window_chain_match_score:.3f}; "
        f"spiral_consistency={candidate.spiral_consistency_score:.3f}; "
        f"best_root={candidate.best_theoretical_root_token}"
    )
    return score, reason


def _is_too_close_in_pitch(
    cand_a: PolyCandidate,
    cand_b: PolyCandidate,
    min_semitone_distance: int,
) -> bool:
    if cand_a.abs_candidate_position is None or cand_b.abs_candidate_position is None:
        return False
    return abs(cand_a.abs_candidate_position - cand_b.abs_candidate_position) < min_semitone_distance


def _same_root_token(cand_a: PolyCandidate, cand_b: PolyCandidate) -> bool:
    return bool(cand_a.best_theoretical_root_token) and cand_a.best_theoretical_root_token == cand_b.best_theoretical_root_token


def choose_polyphonic_representatives(
    window_rows: List[FrameRow],
    *,
    max_notes_per_window: int,
    min_score: float,
    min_support_hits: int,
    min_chain_score: float,
    min_window_chain_match_score: float,
    min_semitone_distance: int,
) -> List[dict]:
    expanded_candidates: List[PolyCandidate] = []
    for row in window_rows:
        expanded_candidates.extend(_extract_polyphonic_candidates(row))

    scored_candidates: List[dict] = []

    for i, cand in enumerate(expanded_candidates):
        prev_candidate = expanded_candidates[i - 1] if i > 0 else None
        next_candidate = expanded_candidates[i + 1] if i + 1 < len(expanded_candidates) else None
        role = classify_candidate_role(cand, prev_candidate, next_candidate)

        stability = local_candidate_stability(
            cand,
            expanded_candidates[max(0, i - 1): min(len(expanded_candidates), i + 2)],
        )

        score, reason = build_candidate_score(cand, role, stability)

        scored_candidates.append(
            {
                "candidate": cand,
                "role": role,
                "stability": stability,
                "score": score,
                "reason": reason,
            }
        )

    scored_candidates.sort(
        key=lambda x: (
            x["score"],
            x["candidate"].window_chain_match_score,
            x["candidate"].spiral_consistency_score,
            x["candidate"].derived_support_hits,
            x["candidate"].candidate_chain_score,
        ),
        reverse=True,
    )

    selected: List[dict] = []

    for cand_info in scored_candidates:
        cand = cand_info["candidate"]

        if not cand.has_candidate:
            continue
        if cand.abs_candidate_position is None:
            continue
        if cand_info["score"] < min_score:
            continue
        if cand.derived_support_hits < min_support_hits:
            continue
        if cand.candidate_chain_score < min_chain_score:
            continue
        if cand.window_chain_match_score < min_window_chain_match_score:
            continue

        conflicted = False

        for kept in selected:
            kept_cand = kept["candidate"]

            if _same_root_token(cand, kept_cand):
                conflicted = True
                break

            if _is_too_close_in_pitch(cand, kept_cand, min_semitone_distance):
                conflicted = True
                break

        if conflicted:
            continue

        selected.append(cand_info)

        if len(selected) >= max_notes_per_window:
            break

    return selected


def _format_top_notes(selected: List[dict], max_items: int = 8) -> tuple[str, str]:
    notes = []
    scores = []

    for item in selected[:max_items]:
        cand: PolyCandidate = item["candidate"]
        notes.append(cand.best_theoretical_root_token)
        scores.append(f"{cand.best_theoretical_root_token}:{item['score']:.2f}")

    return " | ".join(notes), " | ".join(scores)


def stabilize_rows(
    rows: List[FrameRow],
    window_frames: int,
    *,
    max_notes_per_window: int,
    min_score: float,
    min_support_hits: int,
    min_chain_score: float,
    min_window_chain_match_score: float,
    min_semitone_distance: int,
) -> List[dict]:
    out: List[dict] = []

    i = 0
    segment_counter = 0

    while i < len(rows):
        start = i
        end = min(len(rows), i + window_frames)
        window = rows[start:end]

        selected = choose_polyphonic_representatives(
            window,
            max_notes_per_window=max_notes_per_window,
            min_score=min_score,
            min_support_hits=min_support_hits,
            min_chain_score=min_chain_score,
            min_window_chain_match_score=min_window_chain_match_score,
            min_semitone_distance=min_semitone_distance,
        )

        top_notes, top_scores = _format_top_notes(selected, max_items=8)
        active_note_count = len(selected)

        if not selected:
            out.append(
                {
                    "segment_index": segment_counter,
                    "window_start_frame": window[0].frame_index,
                    "window_end_frame": window[-1].frame_index,
                    "window_start_sec": window[0].time_sec,
                    "window_end_sec": window[-1].time_sec,

                    "active_note_rank": 0,
                    "active_note_count": 0,
                    "top_8_notes": "",
                    "top_8_scores": "",

                    "chosen_frame_index": "",
                    "chosen_time_sec": "",

                    "representative_rc_note": "",
                    "representative_rc_hz": "",
                    "representative_rc_energy": "",

                    "representative_source_row_note": "",
                    "representative_source_row_hz": "",
                    "representative_source_row_energy": "",

                    "rc_chain_score": "",
                    "support_hits": "",

                    "best_theoretical_root_token": "",
                    "best_theoretical_root_score": "",
                    "best_theoretical_chain_string": "",
                    "matched_harmonics_same_frame": "",
                    "matched_harmonics_window": "",
                    "missing_harmonics_window": "",
                    "extra_tokens_window": "",
                    "spiral_match_count": "",
                    "spiral_consistency_score": "",
                    "window_chain_match_score": "",
                    "theoretical_chain_verdict": "",

                    "stabilization_role": "NO_CHAIN",
                    "stabilization_reason": "no_polyphonic_candidate_passed_thresholds",
                    "stabilization_score": "",
                }
            )
            segment_counter += 1
            i += window_frames
            continue

        for rank, item in enumerate(selected, start=1):
            cand: PolyCandidate = item["candidate"]

            out.append(
                {
                    "segment_index": segment_counter,
                    "window_start_frame": window[0].frame_index,
                    "window_end_frame": window[-1].frame_index,
                    "window_start_sec": window[0].time_sec,
                    "window_end_sec": window[-1].time_sec,

                    "active_note_rank": rank,
                    "active_note_count": active_note_count,
                    "top_8_notes": top_notes,
                    "top_8_scores": top_scores,

                    "chosen_frame_index": cand.frame_index,
                    "chosen_time_sec": cand.time_sec,

                    "representative_rc_note": cand.candidate_note,
                    "representative_rc_hz": cand.candidate_hz,
                    "representative_rc_energy": cand.candidate_energy,

                    "representative_source_row_note": cand.source_row_chosen_rc_note,
                    "representative_source_row_hz": cand.source_row_chosen_rc_hz,
                    "representative_source_row_energy": cand.source_row_chosen_rc_energy,

                    "rc_chain_score": cand.candidate_chain_score,
                    "support_hits": cand.derived_support_hits,

                    "best_theoretical_root_token": cand.best_theoretical_root_token,
                    "best_theoretical_root_score": cand.best_theoretical_root_score,
                    "best_theoretical_chain_string": cand.best_theoretical_chain_string,
                    "matched_harmonics_same_frame": cand.matched_harmonics_same_frame,
                    "matched_harmonics_window": cand.matched_harmonics_window,
                    "missing_harmonics_window": cand.missing_harmonics_window,
                    "extra_tokens_window": cand.extra_tokens_window,
                    "spiral_match_count": cand.spiral_match_count,
                    "spiral_consistency_score": cand.spiral_consistency_score,
                    "window_chain_match_score": cand.window_chain_match_score,
                    "theoretical_chain_verdict": cand.theoretical_chain_verdict,

                    "stabilization_role": item["role"],
                    "stabilization_reason": item["reason"],
                    "stabilization_score": item["score"],
                }
            )

        segment_counter += 1
        i += window_frames

    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_meta(
    path: Path,
    *,
    input_csv: Path,
    out_csv: Path,
    window_frames: int,
    row_count: int,
    max_notes_per_window: int,
    min_score: float,
    min_support_hits: int,
    min_chain_score: float,
    min_window_chain_match_score: float,
    min_semitone_distance: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "framewise_with_theory_csv": str(input_csv),
        },
        "params": {
            "window_frames": window_frames,
            "frame_rate_hz": 60.0,
            "max_notes_per_window": max_notes_per_window,
            "min_score": min_score,
            "min_support_hits": min_support_hits,
            "min_chain_score": min_chain_score,
            "min_window_chain_match_score": min_window_chain_match_score,
            "min_semitone_distance": min_semitone_distance,
            "semantic_note": SEMANTIC_NOTE,
        },
        "derived": {
            "output_row_count": row_count,
        },
        "outputs": {
            "stabilized_csv": str(out_csv),
            "meta_json": str(path),
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Choose one or more representative chain candidates by stabilization of local chain evidence "
            "in short frame windows. This stage does NOT produce true f0 and does NOT use strongest peak."
        )
    )
    ap.add_argument("--framewise_with_theory_csv", required=True)
    ap.add_argument("--out_stabilized_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--window_frames", type=int, default=4)

    ap.add_argument("--max_notes_per_window", type=int, default=8)
    ap.add_argument("--min_score", type=float, default=6.0)
    ap.add_argument("--min_support_hits", type=int, default=1)
    ap.add_argument("--min_chain_score", type=float, default=0.0)
    ap.add_argument("--min_window_chain_match_score", type=float, default=0.0)
    ap.add_argument("--min_semitone_distance", type=int, default=1)

    args = ap.parse_args()

    input_csv = Path(args.framewise_with_theory_csv).resolve()
    out_csv = Path(args.out_stabilized_csv).resolve()
    out_meta = Path(args.out_meta_json).resolve()

    rows = load_framewise_with_theory_csv(input_csv)
    stabilized = stabilize_rows(
        rows,
        args.window_frames,
        max_notes_per_window=args.max_notes_per_window,
        min_score=args.min_score,
        min_support_hits=args.min_support_hits,
        min_chain_score=args.min_chain_score,
        min_window_chain_match_score=args.min_window_chain_match_score,
        min_semitone_distance=args.min_semitone_distance,
    )

    write_csv(out_csv, stabilized)
    write_meta(
        out_meta,
        input_csv=input_csv,
        out_csv=out_csv,
        window_frames=args.window_frames,
        row_count=len(stabilized),
        max_notes_per_window=args.max_notes_per_window,
        min_score=args.min_score,
        min_support_hits=args.min_support_hits,
        min_chain_score=args.min_chain_score,
        min_window_chain_match_score=args.min_window_chain_match_score,
        min_semitone_distance=args.min_semitone_distance,
    )

    print("stabilize chain complete")
    print(json.dumps(
        {
            "output_row_count": len(stabilized),
            "window_frames": args.window_frames,
            "max_notes_per_window": args.max_notes_per_window,
            "min_score": args.min_score,
            "min_support_hits": args.min_support_hits,
            "min_chain_score": args.min_chain_score,
            "min_window_chain_match_score": args.min_window_chain_match_score,
            "min_semitone_distance": args.min_semitone_distance,
            "out_stabilized_csv": str(out_csv),
            "out_meta_json": str(out_meta),
            "semantic_note": SEMANTIC_NOTE,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()