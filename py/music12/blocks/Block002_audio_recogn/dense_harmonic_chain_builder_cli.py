from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


# ============================================================
# 12-radix helpers
# ============================================================

def normalize_letters(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("А", "A").replace("В", "B").replace("С", "C")
    s = s.replace("а", "A").replace("в", "B").replace("с", "C")
    return s


def bij12_to_int(s: str) -> int:
    s = normalize_letters(s).upper()
    if not s or any(ch not in _VAL12 for ch in s):
        raise ValueError(f"Bad bij12 number: {s!r}")
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def int_to_bij12(n: int) -> str:
    n = int(n)
    if n <= 0:
        raise ValueError("int_to_bij12 expects n >= 1")
    out: list[str] = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def int_to_base12_digit(i0: int) -> str:
    i0 = int(i0)
    if not 0 <= i0 < 12:
        raise ValueError("int_to_base12_digit expects 0..11")
    return _CH12[i0 + 1]


def parse_base_note_token(tok: str) -> tuple[str, str]:
    tok = normalize_letters(tok).upper().strip()
    tok = tok.replace("’-", "'-").replace("'", "")
    tok = tok.rstrip("-")

    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")

    oct_s, step = tok.split(".", 1)
    step = step[:1]
    if not oct_s or any(ch not in _VAL12 for ch in oct_s):
        raise ValueError(f"Bad octave in token: {tok!r}")
    if step not in _VAL12:
        raise ValueError(f"Bad step in token: {tok!r}")
    return oct_s, step


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_base_note_token(token)
    oct0 = bij12_to_int(oct_s) - 1
    step0 = _VAL12[step] - 1
    return oct0 * 12 + step0


def abs_step_to_token(abs_step: int, micro: str = "-") -> str:
    abs_step = int(abs_step)
    if abs_step < 0:
        raise ValueError("abs_step must be >= 0")
    oct0, step0 = divmod(abs_step, 12)
    oct_s = int_to_bij12(oct0 + 1)
    step = int_to_base12_digit(step0)
    if micro:
        return f"{oct_s}.{step}'{micro}"
    return f"{oct_s}.{step}"


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str = "9.A-",
    anchor_hz: float = 440.0,
    micro_depth: int = 2,
    exact_mark: bool = True,
) -> str:
    """
    Project-local acoustic projection Hz -> music12 token.

    IMPORTANT:
    This is NOT the notation grammar layer.
    notation12.py is the SSOT for token language, while this helper is a
    temporary projection bridge for dense harmonic chain analysis.

    micro_depth:
      0 -> coarse semitone only
      1 -> 12 micro-units per semitone
      2 -> 144 micro-units per semitone
      3 -> 1728 micro-units per semitone
      ...

    For the current v2 pipeline we use micro_depth=2 to avoid collapsing
    harmonic orbit / drift information into a coarse pitch token.
    """
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")
    if micro_depth < 0:
        raise ValueError("micro_depth must be >= 0")

    abs_anchor = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest_semitone = int(round(semitone_offset))
    residual = semitone_offset - nearest_semitone

    abs_note = abs_anchor + nearest_semitone

    if micro_depth == 0:
        return abs_step_to_token(abs_note, micro="-" if exact_mark else "")

    micro_steps_per_semitone = 12 ** int(micro_depth)
    micro_float = residual * micro_steps_per_semitone
    micro_rounded = int(round(micro_float))

    # If rounding crosses a semitone boundary, carry it into the coarse note.
    while micro_rounded >= micro_steps_per_semitone:
        abs_note += 1
        micro_rounded -= micro_steps_per_semitone

    while micro_rounded <= -micro_steps_per_semitone:
        abs_note -= 1
        micro_rounded += micro_steps_per_semitone

    base_token = abs_step_to_token(abs_note, micro="")

    if micro_rounded == 0:
        return f"{base_token}'-" if exact_mark else base_token

    sign = "i" if micro_rounded > 0 else "a"
    magnitude = abs(micro_rounded)

    # Bijective base-12 fraction digits preserve the no-zero language rule.
    frac = int_to_bij12(magnitude)
    return f"{base_token}'{sign}{frac}"


def token_coarse_part(token: str) -> str:
    """
    Coarse token is intentionally derived as a separate projection.
    It must never replace the full micro token in the pipeline.
    """
    return str(token).split("'", 1)[0]


# ============================================================
# Utilities
# ============================================================

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def cents_error(observed_hz: float, target_hz: float) -> float:
    if observed_hz <= 0 or target_hz <= 0:
        return 1e9
    return 1200.0 * math.log2(observed_hz / target_hz)


# ============================================================
# Dense rows
# ============================================================

def load_dense_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "time_sec": safe_float(row.get("time_sec", 0.0)),
                    "freq_hz": safe_float(row.get("freq_hz", 0.0)),
                    "amplitude": safe_float(row.get("amplitude", 0.0)),
                    "phase_rad": safe_float(row.get("phase_rad", 0.0)),
                    "frame_index": safe_int(row.get("frame_index", 0)),
                    "peak_index": safe_int(row.get("peak_index", 0)),
                }
            )
    return rows


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["frame_index"]].append(row)
    return out


# ============================================================
# Per-frame root candidates
# ============================================================

@dataclass
class FrameCandidate:
    frame_index: int
    time_sec: float
    root_hz: float
    root_note_token_micro: str
    root_note_token_coarse: str
    chain_score: float
    weighted_support_score: float
    root_plausibility_score: float
    subharmonic_penalty: float
    harmonic_count_found: int
    harmonic_indices_found: list[int]
    harmonic_indices_missing: list[int]
    hits: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RootTrack:
    track_id: int
    candidates: list[FrameCandidate] = field(default_factory=list)

    def append(self, c: FrameCandidate) -> None:
        self.candidates.append(c)

    @property
    def start_time(self) -> float:
        return self.candidates[0].time_sec if self.candidates else 0.0

    @property
    def end_time(self) -> float:
        return self.candidates[-1].time_sec if self.candidates else 0.0

    @property
    def duration(self) -> float:
        if not self.candidates:
            return 0.0
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return len(self.candidates)

    @property
    def last(self) -> FrameCandidate:
        return self.candidates[-1]


def find_best_match_for_harmonic(
    frame_rows: list[dict[str, Any]],
    target_hz: float,
    tolerance_cents: float,
) -> dict[str, Any] | None:
    best = None
    best_abs_cents = 1e18

    for row in frame_rows:
        hz = row["freq_hz"]
        ce = cents_error(hz, target_hz)
        ace = abs(ce)
        if ace <= tolerance_cents and ace < best_abs_cents:
            best_abs_cents = ace
            best = {
                "matched_hz": hz,
                "matched_amplitude": row["amplitude"],
                "matched_phase_rad": row["phase_rad"],
                "matched_peak_index": row["peak_index"],
                "matched_frame_index": row["frame_index"],
                "delta_cents": ce,
            }

    return best


def candidate_roots_from_frame(
    frame_rows: list[dict[str, Any]],
    *,
    max_harmonic: int,
    root_min_hz: float,
    root_max_hz: float,
) -> list[float]:
    candidates: list[float] = []

    for row in frame_rows:
        f = row["freq_hz"]
        if f <= 0:
            continue
        for h in range(1, max_harmonic + 1):
            root = f / h
            if root_min_hz <= root <= root_max_hz:
                candidates.append(root)

    return candidates


def cluster_root_candidates(
    candidates: list[float],
    cluster_cents: float = 25.0,
) -> list[float]:
    if not candidates:
        return []

    candidates = sorted(candidates)
    clusters: list[list[float]] = []

    for f in candidates:
        if not clusters:
            clusters.append([f])
            continue

        ref = mean(clusters[-1])
        if abs(cents_error(f, ref)) <= cluster_cents:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    return [mean(c) for c in clusters]


def score_root_candidate(
    root_hz: float,
    frame_rows: list[dict[str, Any]],
    *,
    max_harmonic: int,
    tolerance_cents: float,
    anchor_token: str,
    anchor_hz: float,
) -> FrameCandidate:
    hits: list[dict[str, Any]] = []
    harmonic_indices_found: list[int] = []
    harmonic_indices_missing: list[int] = []

    chain_energy_sum = 0.0
    weighted_support_score = 0.0

    for h in range(1, max_harmonic + 1):
        target_hz = root_hz * h
        match = find_best_match_for_harmonic(frame_rows, target_hz, tolerance_cents)

        if match is None:
            harmonic_indices_missing.append(h)
            continue

        harmonic_indices_found.append(h)

        amp = float(match["matched_amplitude"])
        harmonic_weight = 1.0 / math.sqrt(h)
        weighted_support_score += amp * harmonic_weight
        chain_energy_sum += amp

        hits.append(
            {
                "harmonic_index": h,
                "theoretical_hz": target_hz,
                "theoretical_token_micro": hz_to_token_with_micro(
                    target_hz,
                    anchor_token=anchor_token,
                    anchor_hz=anchor_hz,
                ),
                "matched_hz": match["matched_hz"],
                "matched_token_micro": hz_to_token_with_micro(
                    match["matched_hz"],
                    anchor_token=anchor_token,
                    anchor_hz=anchor_hz,
                ),
                "matched_amplitude": amp,
                "matched_phase_rad": match["matched_phase_rad"],
                "matched_peak_index": match["matched_peak_index"],
                "delta_cents": match["delta_cents"],
            }
        )

    harmonic_count_found = len(harmonic_indices_found)
    found_set = set(harmonic_indices_found)

    root_plausibility_score = 0.0
    if 1 in found_set:
        root_plausibility_score += 1.40
    if 2 in found_set:
        root_plausibility_score += 0.90
    if 3 in found_set:
        root_plausibility_score += 0.80
    if 4 in found_set:
        root_plausibility_score += 0.45
    if 5 in found_set:
        root_plausibility_score += 0.35
    if 6 in found_set:
        root_plausibility_score += 0.25
    if 7 in found_set:
        root_plausibility_score += 0.20

    shape_score = 0.0

    early_run = 0
    for h in range(1, min(max_harmonic, 8) + 1):
        if h in found_set:
            early_run += 1
        else:
            break
    shape_score += 0.55 * early_run

    if 1 not in found_set and 3 in found_set and 4 in found_set:
        shape_score += 0.35

    if 3 in found_set and 4 in found_set and 5 in found_set:
        shape_score += 0.50

    if 4 in found_set and 5 in found_set and 6 in found_set:
        shape_score += 0.35

    subharmonic_penalty = 0.0

    if harmonic_count_found > 0:
        first_found = min(found_set)
    else:
        first_found = 99

    if first_found >= 4:
        subharmonic_penalty += 1.40
    elif first_found == 3:
        subharmonic_penalty += 0.70
    elif first_found == 2:
        subharmonic_penalty += 0.20

    if 1 not in found_set and 2 not in found_set:
        subharmonic_penalty += 0.90

    if harmonic_count_found > 0:
        early_count = sum(1 for h in found_set if h <= 6)
        late_count = sum(1 for h in found_set if h >= 7)
        if early_count == 0 and late_count >= 2:
            subharmonic_penalty += 1.20

    continuity_bias = 1.0 / max(root_hz, 1e-9)

    chain_score = (
        weighted_support_score
        + 1.30 * root_plausibility_score
        + 1.10 * shape_score
        + 10.0 * continuity_bias
        - subharmonic_penalty
    )

    frame_index = safe_int(frame_rows[0]["frame_index"], 0) if frame_rows else 0
    time_sec = safe_float(frame_rows[0]["time_sec"], 0.0) if frame_rows else 0.0

    return FrameCandidate(
        frame_index=frame_index,
        time_sec=time_sec,
        root_hz=root_hz,
        root_note_token_micro=hz_to_token_with_micro(
            root_hz,
            anchor_token=anchor_token,
            anchor_hz=anchor_hz,
        ),
        root_note_token_coarse=token_coarse_part(
            hz_to_token_with_micro(
                root_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
            )
        ),
        chain_score=chain_score,
        weighted_support_score=weighted_support_score,
        root_plausibility_score=root_plausibility_score + shape_score,
        subharmonic_penalty=subharmonic_penalty,
        harmonic_count_found=harmonic_count_found,
        harmonic_indices_found=harmonic_indices_found,
        harmonic_indices_missing=harmonic_indices_missing,
        hits=hits,
    )


def build_frame_candidates(
    dense_rows: list[dict[str, Any]],
    *,
    max_harmonic: int,
    tolerance_cents: float,
    root_min_hz: float,
    root_max_hz: float,
    cluster_cents: float,
    anchor_token: str,
    anchor_hz: float,
    top_n_per_frame: int,
) -> list[FrameCandidate]:
    by_frame = group_by_frame(dense_rows)
    out: list[FrameCandidate] = []

    for frame_index, frame_rows in sorted(by_frame.items()):
        if not frame_rows:
            continue

        root_cands = candidate_roots_from_frame(
            frame_rows,
            max_harmonic=max_harmonic,
            root_min_hz=root_min_hz,
            root_max_hz=root_max_hz,
        )
        clustered = cluster_root_candidates(root_cands, cluster_cents=cluster_cents)

        scored: list[FrameCandidate] = []
        for root_hz in clustered:
            c = score_root_candidate(
                root_hz,
                frame_rows,
                max_harmonic=max_harmonic,
                tolerance_cents=tolerance_cents,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
            )
            scored.append(c)

        scored.sort(
            key=lambda c: (
                -c.chain_score,
                -c.harmonic_count_found,
                c.root_hz,
            )
        )

        out.extend(scored[:max(1, top_n_per_frame)])

    return out


# ============================================================
# Track building across all frames
# ============================================================

def can_link(a: FrameCandidate, b: FrameCandidate, *, max_gap_frames: int, max_root_jump_cents: float) -> bool:
    frame_gap = b.frame_index - a.frame_index
    if frame_gap <= 0:
        return False
    if frame_gap > max_gap_frames:
        return False

    if abs(cents_error(b.root_hz, a.root_hz)) > max_root_jump_cents:
        return False

    return True


def build_tracks(
    frame_candidates: list[FrameCandidate],
    *,
    max_gap_frames: int,
    max_root_jump_cents: float,
    min_link_score: float,
) -> list[RootTrack]:
    by_frame: dict[int, list[FrameCandidate]] = defaultdict(list)
    for c in frame_candidates:
        by_frame[c.frame_index].append(c)

    tracks: list[RootTrack] = []

    for frame_index in sorted(by_frame):
        for cand in by_frame[frame_index]:
            best_track = None
            best_score = None

            for track in tracks:
                prev = track.last
                if not can_link(prev, cand, max_gap_frames=max_gap_frames, max_root_jump_cents=max_root_jump_cents):
                    continue

                root_jump = abs(cents_error(cand.root_hz, prev.root_hz))
                continuity = 1.0 - min(1.0, root_jump / max_root_jump_cents)
                score = (
                    2.0 * continuity
                    + 0.7 * cand.harmonic_count_found
                    + 0.05 * cand.chain_score
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None and best_score is not None and best_score >= min_link_score:
                best_track.append(cand)
            else:
                track = RootTrack(track_id=len(tracks))
                track.append(cand)
                tracks.append(track)

    return tracks


# ============================================================
# Track summarization
# ============================================================

def summarize_track(track: RootTrack) -> dict[str, Any]:
    roots = [c.root_hz for c in track.candidates]
    chain_scores = [c.chain_score for c in track.candidates]

    harmonic_presence: dict[int, int] = defaultdict(int)
    all_hits_by_h: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for c in track.candidates:
        for h in c.harmonic_indices_found:
            harmonic_presence[h] += 1
        for hit in c.hits:
            all_hits_by_h[safe_int(hit["harmonic_index"], 0)].append(hit)

    representative_hits: list[dict[str, Any]] = []
    for h in sorted(all_hits_by_h):
        candidates = all_hits_by_h[h]
        best = max(candidates, key=lambda x: safe_float(x.get("matched_amplitude", 0.0), 0.0))
        representative_hits.append(best)

    root_hz_mean = mean(roots)
    root_hz_median = median(roots)
    root_note_token_micro = hz_to_token_with_micro(root_hz_median)
    root_note_token_coarse = token_coarse_part(root_note_token_micro)

    found_set = set(harmonic_presence.keys())

    coverage_score = float(track.frame_count)
    persistence_bonus = min(2.5, track.duration * 4.0)

    harmonic_shape_score = 0.0

    if 1 in found_set:
        harmonic_shape_score += 2.0
    if 2 in found_set:
        harmonic_shape_score += 1.4
    if 3 in found_set:
        harmonic_shape_score += 1.2
    if 4 in found_set:
        harmonic_shape_score += 0.7
    if 5 in found_set:
        harmonic_shape_score += 0.6
    if 6 in found_set:
        harmonic_shape_score += 0.4
    if 7 in found_set:
        harmonic_shape_score += 0.3

    early_run = 0
    for h in range(1, 8):
        if h in found_set:
            early_run += 1
        else:
            break
    harmonic_shape_score += 0.9 * early_run

    if 1 not in found_set and 3 in found_set and 4 in found_set and 5 in found_set:
        harmonic_shape_score += 1.0

    false_root_penalty = 0.0

    if found_set:
        first_found = min(found_set)
    else:
        first_found = 99

    if first_found >= 4:
        false_root_penalty += 4.0
    elif first_found == 3:
        false_root_penalty += 1.8
    elif first_found == 2:
        false_root_penalty += 0.5

    if 1 not in found_set and 2 not in found_set:
        false_root_penalty += 2.0

    early_count = sum(1 for h in found_set if h <= 6)
    late_count = sum(1 for h in found_set if h >= 7)
    if early_count <= 1 and late_count >= 3:
        false_root_penalty += 2.5

    harmonic_density_score = mean([len(c.harmonic_indices_found) for c in track.candidates]) if track.candidates else 0.0

    track_score = (
        0.70 * mean(chain_scores)
        + 0.20 * coverage_score
        + 0.60 * persistence_bonus
        + 2.40 * harmonic_shape_score
        + 0.50 * harmonic_density_score
        - false_root_penalty
    )

    return {
        "track_id": track.track_id,
        "start_time": track.start_time,
        "end_time": track.end_time,
        "duration": track.duration,
        "frame_count": track.frame_count,
        "root_hz_mean": root_hz_mean,
        "root_hz_median": root_hz_median,
        "root_note_token_micro": root_note_token_micro,
        "root_note_token_coarse": root_note_token_coarse,
        "chain_score_mean": mean(chain_scores),
        "track_score": track_score,
        "harmonic_presence_profile": dict(sorted(harmonic_presence.items())),
        "representative_hits": representative_hits,
    }


def summarize_tracks(tracks: list[RootTrack]) -> dict[str, Any]:
    if not tracks:
        return {
            "total_tracks": 0,
            "best_track": None,
            "top_roots": [],
        }

    summaries = [summarize_track(t) for t in tracks]
    summaries.sort(key=lambda s: (-s["track_score"], -s["frame_count"], s["root_hz_median"]))

    root_counter = Counter(s["root_note_token_micro"] for s in summaries)

    return {
        "total_tracks": len(summaries),
        "top_roots": [
            {"root_note_token_micro": k, "root_note_token_coarse": token_coarse_part(k), "count": v}
            for k, v in root_counter.most_common(20)
        ],
        "best_track": summaries[0],
        "all_tracks": summaries,
    }


# ============================================================
# Writers
# ============================================================

def write_candidates_csv(path: Path, frame_candidates: list[FrameCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_index",
        "time_sec",
        "root_note_token_micro",
        "root_note_token_coarse",
        "root_hz",
        "harmonic_count_found",
        "harmonic_indices_found",
        "harmonic_indices_missing",
        "weighted_support_score",
        "root_plausibility_score",
        "subharmonic_penalty",
        "chain_score",
        "hits_json",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in frame_candidates:
            writer.writerow(
                {
                    "frame_index": c.frame_index,
                    "time_sec": c.time_sec,
                    "root_note_token_micro": c.root_note_token_micro,
                    "root_note_token_coarse": c.root_note_token_coarse,
                    "root_hz": c.root_hz,
                    "harmonic_count_found": c.harmonic_count_found,
                    "harmonic_indices_found": json.dumps(c.harmonic_indices_found, ensure_ascii=False),
                    "harmonic_indices_missing": json.dumps(c.harmonic_indices_missing, ensure_ascii=False),
                    "weighted_support_score": c.weighted_support_score,
                    "root_plausibility_score": c.root_plausibility_score,
                    "subharmonic_penalty": c.subharmonic_penalty,
                    "chain_score": c.chain_score,
                    "hits_json": json.dumps(c.hits, ensure_ascii=False),
                }
            )


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_txt(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("DENSE HARMONIC CHAIN SUMMARY (TIME-TRACKED)")
    lines.append("=" * 100)
    lines.append(f"Total tracks: {summary['total_tracks']}")
    lines.append("Top roots:")
    for item in summary["top_roots"]:
        lines.append(f"  {item.get('root_note_token_micro', item.get('root_note_token', ''))}: {item['count']}")

    lines.append("")
    lines.append("Best track:")

    best = summary["best_track"]
    if not best:
        lines.append("  None")
    else:
        lines.append(f"  track_id               : {best['track_id']}")
        lines.append(f"  start_time             : {best['start_time']}")
        lines.append(f"  end_time               : {best['end_time']}")
        lines.append(f"  duration               : {best['duration']}")
        lines.append(f"  frame_count            : {best['frame_count']}")
        lines.append(f"  root_note_token_micro  : {best['root_note_token_micro']}")
        lines.append(f"  root_hz_mean           : {best['root_hz_mean']}")
        lines.append(f"  root_hz_median         : {best['root_hz_median']}")
        lines.append(f"  chain_score_mean       : {best['chain_score_mean']}")
        lines.append(f"  track_score            : {best['track_score']}")
        lines.append(f"  harmonic_presence      : {best['harmonic_presence_profile']}")
        lines.append("")
        lines.append("  representative_hits:")
        for hit in best["representative_hits"]:
            lines.append(
                f"    h{hit['harmonic_index']}: "
                f"theory={hit.get('theoretical_token_micro', hit.get('theoretical_token', ''))} ({hit['theoretical_hz']:.6f})  "
                f"obs={hit.get('matched_token_micro', hit.get('matched_token', ''))} ({hit['matched_hz']:.6f})  "
                f"amp={hit['matched_amplitude']:.6f}  "
                f"phase={hit['matched_phase_rad']:.6f}  "
                f"delta_cents={hit['delta_cents']:.6f}"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build harmonic chains directly from dense spectral observer CSV across all frames."
    )
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--out_chain_candidates_csv", required=True)
    ap.add_argument("--out_chain_summary_json", required=True)
    ap.add_argument("--out_chain_summary_txt", required=True)

    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--tolerance_cents", type=float, default=35.0)
    ap.add_argument("--cluster_cents", type=float, default=25.0)
    ap.add_argument("--root_min_hz", type=float, default=20.0)
    ap.add_argument("--root_max_hz", type=float, default=6000.0)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)

    ap.add_argument("--top_n_per_frame", type=int, default=6)
    ap.add_argument("--max_gap_frames", type=int, default=3)
    ap.add_argument("--max_root_jump_cents", type=float, default=40.0)
    ap.add_argument("--min_link_score", type=float, default=2.0)

    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    out_chain_candidates_csv = Path(args.out_chain_candidates_csv).resolve()
    out_chain_summary_json = Path(args.out_chain_summary_json).resolve()
    out_chain_summary_txt = Path(args.out_chain_summary_txt).resolve()

    dense_rows = load_dense_rows(dense_csv)

    frame_candidates = build_frame_candidates(
        dense_rows,
        max_harmonic=int(args.max_harmonic),
        tolerance_cents=float(args.tolerance_cents),
        root_min_hz=float(args.root_min_hz),
        root_max_hz=float(args.root_max_hz),
        cluster_cents=float(args.cluster_cents),
        anchor_token=str(args.anchor_token),
        anchor_hz=float(args.anchor_hz),
        top_n_per_frame=int(args.top_n_per_frame),
    )

    tracks = build_tracks(
        frame_candidates,
        max_gap_frames=int(args.max_gap_frames),
        max_root_jump_cents=float(args.max_root_jump_cents),
        min_link_score=float(args.min_link_score),
    )

    summary = summarize_tracks(tracks)

    write_candidates_csv(out_chain_candidates_csv, frame_candidates)
    write_summary_json(out_chain_summary_json, summary)
    write_summary_txt(out_chain_summary_txt, summary)

    print(f"Wrote candidates CSV: {out_chain_candidates_csv}")
    print(f"Wrote summary JSON  : {out_chain_summary_json}")
    print(f"Wrote summary TXT   : {out_chain_summary_txt}")
    print(f"Total frame candidates: {len(frame_candidates)}")
    print(f"Total tracks          : {summary['total_tracks']}")
    if summary["best_track"]:
        print(f"Best root             : {summary['best_track']['root_note_token_micro']}")
        print(f"Best root Hz median   : {summary['best_track']['root_hz_median']}")


if __name__ == "__main__":
    main()