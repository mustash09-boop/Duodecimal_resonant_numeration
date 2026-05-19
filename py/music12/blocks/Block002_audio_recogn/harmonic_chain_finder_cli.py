from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ProbeCoord:
    probe_index: int
    frequency_hz: float
    note_token: str


@dataclass
class FramePeak:
    frame_index: int
    time_sec: float
    probe_index: int
    frequency_hz: float
    note_token: str
    response_value: float


@dataclass
class HarmonicHit:
    harmonic_index: int
    expected_hz: float
    matched_hz: float
    matched_note_token: str
    matched_probe_index: int
    matched_response_value: float
    rel_error: float


@dataclass
class ChainCandidate:
    frame_index: int
    time_sec: float
    root_probe_index: int
    root_hz: float
    root_note_token: str
    root_response_value: float
    harmonic_count_found: int
    harmonic_indices_found: List[int]
    harmonic_indices_missing: List[int]
    chain_energy_sum: float
    chain_score: float
    weighted_support_score: float
    subharmonic_penalty: float
    root_plausibility_score: float
    lowest_present_harmonic: int
    highest_present_harmonic: int
    hits: List[HarmonicHit]


# ============================================================
# CSV LOADERS
# ============================================================

def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def load_times_csv(path: Path) -> List[float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: List[float] = []
        for row in reader:
            if "time_seconds" in row:
                out.append(safe_float(row["time_seconds"]))
            elif "time_sec" in row:
                out.append(safe_float(row["time_sec"]))
            else:
                vals = list(row.values())
                out.append(safe_float(vals[1] if len(vals) > 1 else "0"))
        return out


def load_coords_csv(path: Path) -> Dict[int, ProbeCoord]:
    out: Dict[int, ProbeCoord] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            probe_index = safe_int(row.get("probe_index", "0"))
            frequency_hz = safe_float(row.get("frequency_hz", "0"))
            note_token = str(row.get("note_token", "")).strip()

            out[probe_index] = ProbeCoord(
                probe_index=probe_index,
                frequency_hz=frequency_hz,
                note_token=note_token,
            )

    return out


def load_matrix_csv(path: Path) -> Tuple[List[int], List[List[float]]]:
    probe_indices_by_row: List[int] = []
    matrix: List[List[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        _header = next(reader)

        for row in reader:
            if not row:
                continue
            probe_idx = safe_int(row[0], 0)
            values = [safe_float(x, 0.0) for x in row[1:]]
            probe_indices_by_row.append(probe_idx)
            matrix.append(values)

    return probe_indices_by_row, matrix


# ============================================================
# PEAK EXTRACTION
# ============================================================

def extract_frame_peaks(
    probe_indices_by_row: List[int],
    matrix: List[List[float]],
    coords: Dict[int, ProbeCoord],
    times: List[float],
    top_k_roots: int,
    min_response: float,
    root_min_hz: float,
    root_max_hz: float,
) -> List[List[FramePeak]]:
    """
    For each frame:
      returns the strongest top_k_roots peaks above min_response
      and only within allowed root frequency range.
    """
    if not matrix:
        return []

    frame_count = len(matrix[0])
    out: List[List[FramePeak]] = []

    for frame_idx in range(frame_count):
        peaks: List[FramePeak] = []

        for row_idx, probe_idx in enumerate(probe_indices_by_row):
            value = matrix[row_idx][frame_idx]
            if value < min_response:
                continue

            coord = coords.get(probe_idx)
            if coord is None:
                continue

            if coord.frequency_hz < root_min_hz or coord.frequency_hz > root_max_hz:
                continue

            peaks.append(
                FramePeak(
                    frame_index=frame_idx,
                    time_sec=times[frame_idx] if frame_idx < len(times) else 0.0,
                    probe_index=probe_idx,
                    frequency_hz=coord.frequency_hz,
                    note_token=coord.note_token,
                    response_value=value,
                )
            )

        peaks.sort(key=lambda p: p.response_value, reverse=True)
        out.append(peaks[:top_k_roots])

    return out


# ============================================================
# HARMONIC CHAIN SEARCH
# ============================================================

def relative_error(expected_hz: float, observed_hz: float) -> float:
    if expected_hz <= 0:
        return 999.0
    return abs(observed_hz - expected_hz) / expected_hz


def find_best_harmonic_match(
    frame_idx: int,
    expected_hz: float,
    probe_indices_by_row: List[int],
    matrix: List[List[float]],
    coords: Dict[int, ProbeCoord],
    hz_tolerance_ratio: float,
    min_response: float,
) -> Optional[HarmonicHit]:
    best: Optional[HarmonicHit] = None

    for row_idx, probe_idx in enumerate(probe_indices_by_row):
        value = matrix[row_idx][frame_idx]
        if value < min_response:
            continue

        coord = coords.get(probe_idx)
        if coord is None:
            continue

        err = relative_error(expected_hz, coord.frequency_hz)
        if err > hz_tolerance_ratio:
            continue

        hit = HarmonicHit(
            harmonic_index=0,  # filled later
            expected_hz=expected_hz,
            matched_hz=coord.frequency_hz,
            matched_note_token=coord.note_token,
            matched_probe_index=probe_idx,
            matched_response_value=value,
            rel_error=err,
        )

        if best is None:
            best = hit
        else:
            if hit.rel_error < best.rel_error:
                best = hit
            elif math.isclose(hit.rel_error, best.rel_error, abs_tol=1e-12):
                if hit.matched_response_value > best.matched_response_value:
                    best = hit

    return best


def compute_subharmonic_penalty(found_indices: List[int]) -> float:
    """
    Penalize roots that are supported mainly by high harmonics,
    which is a typical false explanation by a too-low root.
    """
    if not found_indices:
        return 999.0

    low_band = sum(1 for h in found_indices if h <= 3)
    high_band = sum(1 for h in found_indices if h >= 5)

    penalty = 0.0

    if low_band == 0:
        penalty += 2.0
    elif low_band == 1:
        penalty += 0.8

    if high_band > low_band:
        penalty += 0.5 * (high_band - low_band)

    if min(found_indices) > 1:
        penalty += 1.0

    return penalty


def compute_root_plausibility_score(
    root_hz: float,
    root_response_value: float,
    found_indices: List[int],
    expected_note_hz: Optional[float],
) -> float:
    """
    Prefer roots that:
    - are close to the expected note center if provided
    - have strong own response
    - have low-order harmonics present
    """
    score = 0.0

    score += root_response_value

    low_hits = sum(1 for h in found_indices if h <= 4)
    score += 0.25 * low_hits

    if expected_note_hz and expected_note_hz > 0:
        err = relative_error(expected_note_hz, root_hz)
        score += max(0.0, 1.0 - 8.0 * err)

    return score


def build_chain_for_root(
    root_peak: FramePeak,
    frame_idx: int,
    probe_indices_by_row: List[int],
    matrix: List[List[float]],
    coords: Dict[int, ProbeCoord],
    max_harmonic: int,
    hz_tolerance_ratio: float,
    min_response: float,
    harmonic_weight_power: float,
    expected_note_hz: Optional[float],
) -> ChainCandidate:
    hits: List[HarmonicHit] = []
    missing: List[int] = []
    found_indices: List[int] = []
    chain_energy_sum = 0.0
    weighted_support_score = 0.0

    # harmonic 1 = root itself
    root_hit = HarmonicHit(
        harmonic_index=1,
        expected_hz=root_peak.frequency_hz,
        matched_hz=root_peak.frequency_hz,
        matched_note_token=root_peak.note_token,
        matched_probe_index=root_peak.probe_index,
        matched_response_value=root_peak.response_value,
        rel_error=0.0,
    )
    hits.append(root_hit)
    found_indices.append(1)
    chain_energy_sum += root_peak.response_value
    weighted_support_score += root_peak.response_value

    for h in range(2, max_harmonic + 1):
        expected_hz = root_peak.frequency_hz * h
        best_hit = find_best_harmonic_match(
            frame_idx=frame_idx,
            expected_hz=expected_hz,
            probe_indices_by_row=probe_indices_by_row,
            matrix=matrix,
            coords=coords,
            hz_tolerance_ratio=hz_tolerance_ratio,
            min_response=min_response,
        )

        if best_hit is None:
            missing.append(h)
            continue

        best_hit.harmonic_index = h
        hits.append(best_hit)
        found_indices.append(h)

        weight = 1.0 / (h ** harmonic_weight_power)
        chain_energy_sum += best_hit.matched_response_value
        weighted_support_score += best_hit.matched_response_value * weight

    subharmonic_penalty = compute_subharmonic_penalty(found_indices)
    root_plausibility_score = compute_root_plausibility_score(
        root_hz=root_peak.frequency_hz,
        root_response_value=root_peak.response_value,
        found_indices=found_indices,
        expected_note_hz=expected_note_hz,
    )

    chain_score = weighted_support_score + root_plausibility_score - subharmonic_penalty

    return ChainCandidate(
        frame_index=root_peak.frame_index,
        time_sec=root_peak.time_sec,
        root_probe_index=root_peak.probe_index,
        root_hz=root_peak.frequency_hz,
        root_note_token=root_peak.note_token,
        root_response_value=root_peak.response_value,
        harmonic_count_found=len(found_indices),
        harmonic_indices_found=found_indices,
        harmonic_indices_missing=missing,
        chain_energy_sum=chain_energy_sum,
        chain_score=chain_score,
        weighted_support_score=weighted_support_score,
        subharmonic_penalty=subharmonic_penalty,
        root_plausibility_score=root_plausibility_score,
        lowest_present_harmonic=min(found_indices) if found_indices else 0,
        highest_present_harmonic=max(found_indices) if found_indices else 0,
        hits=hits,
    )


def build_all_chains(
    frame_peaks: List[List[FramePeak]],
    probe_indices_by_row: List[int],
    matrix: List[List[float]],
    coords: Dict[int, ProbeCoord],
    max_harmonic: int,
    hz_tolerance_ratio: float,
    min_response: float,
    harmonic_weight_power: float,
    expected_note_hz: Optional[float],
) -> List[ChainCandidate]:
    out: List[ChainCandidate] = []

    for frame_idx, peaks in enumerate(frame_peaks):
        for peak in peaks:
            chain = build_chain_for_root(
                root_peak=peak,
                frame_idx=frame_idx,
                probe_indices_by_row=probe_indices_by_row,
                matrix=matrix,
                coords=coords,
                max_harmonic=max_harmonic,
                hz_tolerance_ratio=hz_tolerance_ratio,
                min_response=min_response,
                harmonic_weight_power=harmonic_weight_power,
                expected_note_hz=expected_note_hz,
            )
            out.append(chain)

    return out


# ============================================================
# OUTPUTS
# ============================================================

def chain_to_row(c: ChainCandidate) -> dict:
    return {
        "frame_index": c.frame_index,
        "time_sec": c.time_sec,
        "root_probe_index": c.root_probe_index,
        "root_hz": c.root_hz,
        "root_note_token": c.root_note_token,
        "root_response_value": c.root_response_value,
        "harmonic_count_found": c.harmonic_count_found,
        "harmonic_indices_found": json.dumps(c.harmonic_indices_found, ensure_ascii=False),
        "harmonic_indices_missing": json.dumps(c.harmonic_indices_missing, ensure_ascii=False),
        "chain_energy_sum": c.chain_energy_sum,
        "weighted_support_score": c.weighted_support_score,
        "subharmonic_penalty": c.subharmonic_penalty,
        "root_plausibility_score": c.root_plausibility_score,
        "chain_score": c.chain_score,
        "lowest_present_harmonic": c.lowest_present_harmonic,
        "highest_present_harmonic": c.highest_present_harmonic,
        "hits_json": json.dumps(
            [
                {
                    "harmonic_index": h.harmonic_index,
                    "expected_hz": h.expected_hz,
                    "matched_hz": h.matched_hz,
                    "matched_note_token": h.matched_note_token,
                    "matched_probe_index": h.matched_probe_index,
                    "matched_response_value": h.matched_response_value,
                    "rel_error": h.rel_error,
                }
                for h in c.hits
            ],
            ensure_ascii=False,
        ),
    }


def write_chain_candidates_csv(path: Path, chains: List[ChainCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [chain_to_row(c) for c in chains]
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_chain_summary_json(path: Path, chains: List[ChainCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not chains:
        data = {
            "chain_count": 0,
            "top_roots": [],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    root_counter: Dict[str, int] = {}
    for c in chains:
        root_counter[c.root_note_token] = root_counter.get(c.root_note_token, 0) + 1

    top_roots = sorted(root_counter.items(), key=lambda kv: kv[1], reverse=True)[:20]
    best = max(chains, key=lambda c: c.chain_score)

    data = {
        "chain_count": len(chains),
        "top_roots": [
            {"root_note_token": k, "count": v}
            for k, v in top_roots
        ],
        "best_chain": chain_to_row(best),
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_chain_summary_txt(path: Path, chains: List[ChainCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("HARMONIC CHAIN SUMMARY")
    lines.append("=" * 80)

    if not chains:
        lines.append("No chains found.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    root_counter: Dict[str, int] = {}
    for c in chains:
        root_counter[c.root_note_token] = root_counter.get(c.root_note_token, 0) + 1

    lines.append(f"Total chains: {len(chains)}")
    lines.append("Top roots:")
    for root, count in sorted(root_counter.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"  {root}: {count}")

    best = max(chains, key=lambda c: c.chain_score)
    lines.append("")
    lines.append("Best chain:")
    lines.append(f"  frame_index             : {best.frame_index}")
    lines.append(f"  time_sec                : {best.time_sec}")
    lines.append(f"  root_note_token         : {best.root_note_token}")
    lines.append(f"  root_hz                 : {best.root_hz}")
    lines.append(f"  harmonic_count_found    : {best.harmonic_count_found}")
    lines.append(f"  harmonic_indices_found  : {best.harmonic_indices_found}")
    lines.append(f"  harmonic_indices_missing: {best.harmonic_indices_missing}")
    lines.append(f"  chain_energy_sum        : {best.chain_energy_sum}")
    lines.append(f"  weighted_support_score  : {best.weighted_support_score}")
    lines.append(f"  subharmonic_penalty     : {best.subharmonic_penalty}")
    lines.append(f"  root_plausibility_score : {best.root_plausibility_score}")
    lines.append(f"  chain_score             : {best.chain_score}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find harmonic chains directly from probe outputs using simple multiplicity."
    )
    ap.add_argument("--matrix_csv", required=True)
    ap.add_argument("--times_csv", required=True)
    ap.add_argument("--coords_csv", required=True)
    ap.add_argument("--out_chain_candidates_csv", required=True)
    ap.add_argument("--out_chain_summary_json", required=True)
    ap.add_argument("--out_chain_summary_txt", required=True)

    ap.add_argument("--top_k_roots", type=int, default=8)
    ap.add_argument("--min_response", type=float, default=0.01)
    ap.add_argument("--max_harmonic", type=int, default=8)
    ap.add_argument("--hz_tolerance_ratio", type=float, default=0.03)
    ap.add_argument("--harmonic_weight_power", type=float, default=1.0)

    ap.add_argument("--root_min_hz", type=float, default=80.0)
    ap.add_argument("--root_max_hz", type=float, default=2000.0)
    ap.add_argument("--expected_note_hz", type=float, default=0.0)

    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv).resolve()
    times_csv = Path(args.times_csv).resolve()
    coords_csv = Path(args.coords_csv).resolve()

    out_chain_candidates_csv = Path(args.out_chain_candidates_csv).resolve()
    out_chain_summary_json = Path(args.out_chain_summary_json).resolve()
    out_chain_summary_txt = Path(args.out_chain_summary_txt).resolve()

    probe_indices_by_row, matrix = load_matrix_csv(matrix_csv)
    times = load_times_csv(times_csv)
    coords = load_coords_csv(coords_csv)

    expected_note_hz = args.expected_note_hz if args.expected_note_hz > 0 else None

    frame_peaks = extract_frame_peaks(
        probe_indices_by_row=probe_indices_by_row,
        matrix=matrix,
        coords=coords,
        times=times,
        top_k_roots=args.top_k_roots,
        min_response=args.min_response,
        root_min_hz=args.root_min_hz,
        root_max_hz=args.root_max_hz,
    )

    chains = build_all_chains(
        frame_peaks=frame_peaks,
        probe_indices_by_row=probe_indices_by_row,
        matrix=matrix,
        coords=coords,
        max_harmonic=args.max_harmonic,
        hz_tolerance_ratio=args.hz_tolerance_ratio,
        min_response=args.min_response,
        harmonic_weight_power=args.harmonic_weight_power,
        expected_note_hz=expected_note_hz,
    )

    write_chain_candidates_csv(out_chain_candidates_csv, chains)
    write_chain_summary_json(out_chain_summary_json, chains)
    write_chain_summary_txt(out_chain_summary_txt, chains)

    print("DONE (harmonic chain finder v2)")


if __name__ == "__main__":
    main()