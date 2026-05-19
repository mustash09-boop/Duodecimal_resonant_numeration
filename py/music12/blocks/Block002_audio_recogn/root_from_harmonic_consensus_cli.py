from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
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
    micro_steps_per_semitone: int = 12,
    exact_mark: bool = True,
) -> str:
    if freq_hz <= 0:
        return ""

    abs_anchor = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest_semitone = int(round(semitone_offset))
    residual = semitone_offset - nearest_semitone

    abs_note = abs_anchor + nearest_semitone
    base_token = abs_step_to_token(abs_note, micro="")

    micro_float = residual * micro_steps_per_semitone
    micro_rounded = int(round(micro_float))

    if micro_rounded == 0:
        return f"{base_token}'-" if exact_mark else base_token

    sign = "i" if micro_rounded > 0 else "a"
    magnitude = abs(micro_rounded)

    while magnitude >= micro_steps_per_semitone:
        if sign == "i":
            abs_note += 1
        else:
            abs_note -= 1
        magnitude -= micro_steps_per_semitone

    if magnitude == 0:
        base_token = abs_step_to_token(abs_note, micro="")
        return f"{base_token}'-" if exact_mark else base_token

    digit = int_to_base12_digit(magnitude)
    base_token = abs_step_to_token(abs_note, micro="")
    return f"{base_token}'{sign}{digit}"


def root_hz_from_note_token(token: str, anchor_token: str = "9.A-", anchor_hz: float = 440.0) -> float:
    semitone_delta = token_to_abs_step(token) - token_to_abs_step(anchor_token)
    return float(anchor_hz * (2.0 ** (semitone_delta / 12.0)))


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


def std(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0


def cents_error(observed_hz: float, target_hz: float) -> float:
    if observed_hz <= 0 or target_hz <= 0:
        return 1e9
    return 1200.0 * math.log2(observed_hz / target_hz)


# ============================================================
# Dense loading
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


# ============================================================
# Consensus logic
# ============================================================

def cluster_values_by_cents(values: list[float], cluster_cents: float) -> list[list[float]]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = []

    for v in values:
        if not clusters:
            clusters.append([v])
            continue

        ref = mean(clusters[-1])
        if abs(cents_error(v, ref)) <= cluster_cents:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    return clusters


def build_root_candidates_from_dense(
    dense_rows: list[dict[str, Any]],
    *,
    harmonic_min: int,
    harmonic_max: int,
    root_min_hz: float,
    root_max_hz: float,
    min_amplitude: float,
    expected_root_hz: float,
    expected_root_tolerance_cents: float,
) -> list[dict[str, Any]]:
    """
    For every observed peak f, generate root candidates f / n,
    but keep only those candidates that are close to the expected root.
    """
    out: list[dict[str, Any]] = []

    for row in dense_rows:
        freq_hz = safe_float(row["freq_hz"], 0.0)
        amplitude = safe_float(row["amplitude"], 0.0)
        if freq_hz <= 0 or amplitude < min_amplitude:
            continue

        for h in range(harmonic_min, harmonic_max + 1):
            root_hz = freq_hz / h
            if not (root_min_hz <= root_hz <= root_max_hz):
                continue

            if abs(cents_error(root_hz, expected_root_hz)) > expected_root_tolerance_cents:
                continue

            out.append(
                {
                    "frame_index": safe_int(row["frame_index"], 0),
                    "time_sec": safe_float(row["time_sec"], 0.0),
                    "observed_freq_hz": freq_hz,
                    "observed_amplitude": amplitude,
                    "observed_phase_rad": safe_float(row["phase_rad"], 0.0),
                    "assumed_harmonic_index": h,
                    "root_hz_candidate": root_hz,
                    "root_delta_cents_vs_expected": cents_error(root_hz, expected_root_hz),
                }
            )

    return out


def cluster_root_candidates(
    root_candidates: list[dict[str, Any]],
    *,
    cluster_cents: float,
) -> list[dict[str, Any]]:
    if not root_candidates:
        return []

    values = [safe_float(x["root_hz_candidate"], 0.0) for x in root_candidates]
    clusters_raw = cluster_values_by_cents(values, cluster_cents)

    clustered: list[dict[str, Any]] = []
    for cluster_vals in clusters_raw:
        center_hz = mean(cluster_vals)

        members = []
        for cand in root_candidates:
            v = safe_float(cand["root_hz_candidate"], 0.0)
            if abs(cents_error(v, center_hz)) <= cluster_cents:
                members.append(cand)

        if not members:
            continue

        frames = {safe_int(m["frame_index"], 0) for m in members}
        harmonics = [safe_int(m["assumed_harmonic_index"], 0) for m in members]
        amps = [safe_float(m["observed_amplitude"], 0.0) for m in members]
        deltas = [safe_float(m["root_delta_cents_vs_expected"], 0.0) for m in members]

        harmonic_counts = Counter(harmonics)

        clustered.append(
            {
                "cluster_center_hz": center_hz,
                "member_count": len(members),
                "unique_frame_count": len(frames),
                "harmonic_index_counts": dict(sorted(harmonic_counts.items())),
                "mean_observed_amplitude": mean(amps),
                "median_observed_amplitude": median(amps),
                "mean_root_delta_cents_vs_expected": mean(deltas),
                "median_root_delta_cents_vs_expected": median(deltas),
                "members": members,
            }
        )

    return clustered


def summarize_cluster_against_theory(
    cluster: dict[str, Any],
    *,
    expected_note: str,
    anchor_token: str,
    anchor_hz: float,
) -> dict[str, Any]:
    root_hz_theory = root_hz_from_note_token(expected_note, anchor_token=anchor_token, anchor_hz=anchor_hz)
    root_hz_consensus = safe_float(cluster["cluster_center_hz"], 0.0)

    delta_cents = cents_error(root_hz_consensus, root_hz_theory)
    note_token = hz_to_token_with_micro(root_hz_consensus, anchor_token=anchor_token, anchor_hz=anchor_hz)

    harmonic_counts = cluster["harmonic_index_counts"]
    present_harmonics = sorted(int(k) for k, v in harmonic_counts.items() if int(v) > 0)

    tuner_confidence = 0.0
    for h in present_harmonics:
        if h == 1:
            tuner_confidence += 0.6
        elif h == 2:
            tuner_confidence += 0.7
        elif h == 3:
            tuner_confidence += 1.6
        elif h == 4:
            tuner_confidence += 1.4
        elif h == 5:
            tuner_confidence += 1.2
        elif h == 6:
            tuner_confidence += 1.1
        elif h == 7:
            tuner_confidence += 0.9
        elif h == 8:
            tuner_confidence += 0.5
        else:
            tuner_confidence += 0.25

    # bonus for low-register useful harmonics
    if 3 in present_harmonics and 6 in present_harmonics:
        tuner_confidence += 1.2
    if 3 in present_harmonics and 11 in present_harmonics:
        tuner_confidence += 0.9
    if 4 in present_harmonics and 5 in present_harmonics:
        tuner_confidence += 0.6

    # penalty for drifting too far from theoretical root
    tuner_confidence -= min(4.0, abs(delta_cents) / 25.0)

    return {
        "expected_note": expected_note,
        "theoretical_root_hz": root_hz_theory,
        "consensus_root_hz": root_hz_consensus,
        "consensus_root_token": note_token,
        "root_delta_cents_vs_theory": delta_cents,
        "member_count": cluster["member_count"],
        "unique_frame_count": cluster["unique_frame_count"],
        "mean_observed_amplitude": cluster["mean_observed_amplitude"],
        "median_observed_amplitude": cluster["median_observed_amplitude"],
        "mean_root_delta_cents_vs_expected": cluster["mean_root_delta_cents_vs_expected"],
        "median_root_delta_cents_vs_expected": cluster["median_root_delta_cents_vs_expected"],
        "harmonic_index_counts": cluster["harmonic_index_counts"],
        "present_harmonics": present_harmonics,
        "tuner_confidence": tuner_confidence,
    }


# ============================================================
# Writers
# ============================================================

def write_root_candidates_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "time_sec",
        "observed_freq_hz",
        "observed_amplitude",
        "observed_phase_rad",
        "assumed_harmonic_index",
        "root_hz_candidate",
        "root_delta_cents_vs_expected",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cluster_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "expected_note",
        "theoretical_root_hz",
        "consensus_root_hz",
        "consensus_root_token",
        "root_delta_cents_vs_theory",
        "member_count",
        "unique_frame_count",
        "mean_observed_amplitude",
        "median_observed_amplitude",
        "mean_root_delta_cents_vs_expected",
        "median_root_delta_cents_vs_expected",
        "harmonic_index_counts",
        "present_harmonics",
        "tuner_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["harmonic_index_counts"] = json.dumps(out["harmonic_index_counts"], ensure_ascii=False)
            out["present_harmonics"] = json.dumps(out["present_harmonics"], ensure_ascii=False)
            writer.writerow(out)


def write_summary_txt(
    path: Path,
    *,
    expected_note: str,
    cluster_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("ROOT FROM HARMONIC CONSENSUS")
    lines.append("=" * 100)
    lines.append(f"expected_note : {expected_note}")
    lines.append("")

    if not cluster_rows:
        lines.append("No consensus clusters found.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    best = cluster_rows[0]

    lines.append("BEST CONSENSUS CLUSTER")
    lines.append("-" * 100)
    lines.append(f"theoretical_root_hz             : {best['theoretical_root_hz']}")
    lines.append(f"consensus_root_hz               : {best['consensus_root_hz']}")
    lines.append(f"consensus_root_token            : {best['consensus_root_token']}")
    lines.append(f"root_delta_cents_vs_theory      : {best['root_delta_cents_vs_theory']}")
    lines.append(f"member_count                    : {best['member_count']}")
    lines.append(f"unique_frame_count              : {best['unique_frame_count']}")
    lines.append(f"mean_observed_amplitude         : {best['mean_observed_amplitude']}")
    lines.append(f"median_observed_amplitude       : {best['median_observed_amplitude']}")
    lines.append(f"mean_root_delta_cents_expected  : {best['mean_root_delta_cents_vs_expected']}")
    lines.append(f"median_root_delta_cents_expected: {best['median_root_delta_cents_vs_expected']}")
    lines.append(f"present_harmonics               : {best['present_harmonics']}")
    lines.append(f"harmonic_index_counts           : {best['harmonic_index_counts']}")
    lines.append(f"tuner_confidence                : {best['tuner_confidence']}")
    lines.append("")

    lines.append("TOP CLUSTERS")
    lines.append("-" * 100)
    for i, row in enumerate(cluster_rows[:15], start=1):
        lines.append(
            f"{i:02d}. token={row['consensus_root_token']:12s}  "
            f"root_hz={row['consensus_root_hz']:.6f}  "
            f"delta_theory={row['root_delta_cents_vs_theory']:.6f}  "
            f"frames={row['unique_frame_count']:4d}  "
            f"members={row['member_count']:4d}  "
            f"harmonics={row['present_harmonics']}  "
            f"confidence={row['tuner_confidence']:.3f}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_meta_json(
    path: Path,
    *,
    dense_csv: Path,
    expected_note: str,
    out_root_candidates_csv: Path,
    out_cluster_summary_csv: Path,
    out_summary_txt: Path,
    harmonic_min: int,
    harmonic_max: int,
    cluster_cents: float,
    expected_root_tolerance_cents: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "dense_csv": str(dense_csv),
            "expected_note": expected_note,
        },
        "outputs": {
            "root_candidates_csv": str(out_root_candidates_csv),
            "cluster_summary_csv": str(out_cluster_summary_csv),
            "summary_txt": str(out_summary_txt),
            "meta_json": str(path),
        },
        "settings": {
            "harmonic_min": int(harmonic_min),
            "harmonic_max": int(harmonic_max),
            "cluster_cents": float(cluster_cents),
            "expected_root_tolerance_cents": float(expected_root_tolerance_cents),
        },
        "semantic_note": (
            "Tuner-style directed root recovery. "
            "Root is reconstructed from consensus of multiple harmonics hn/n, "
            "but only around the theoretical root region of the expected note."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recover root from harmonic consensus around expected note (tuner-style)."
    )
    ap.add_argument("--dense_csv", required=True)
    ap.add_argument("--expected_note", required=True)
    ap.add_argument("--out_root_candidates_csv", required=True)
    ap.add_argument("--out_cluster_summary_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--harmonic_min", type=int, default=3)
    ap.add_argument("--harmonic_max", type=int, default=12)
    ap.add_argument("--root_min_hz", type=float, default=20.0)
    ap.add_argument("--root_max_hz", type=float, default=1000.0)
    ap.add_argument("--min_amplitude", type=float, default=0.0)
    ap.add_argument("--cluster_cents", type=float, default=18.0)
    ap.add_argument("--expected_root_tolerance_cents", type=float, default=80.0)
    args = ap.parse_args()

    dense_csv = Path(args.dense_csv).resolve()
    out_root_candidates_csv = Path(args.out_root_candidates_csv).resolve()
    out_cluster_summary_csv = Path(args.out_cluster_summary_csv).resolve()
    out_summary_txt = Path(args.out_summary_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    expected_root_hz = root_hz_from_note_token(
        str(args.expected_note),
        anchor_token=str(args.anchor_token),
        anchor_hz=float(args.anchor_hz),
    )

    dense_rows = load_dense_rows(dense_csv)

    root_candidates = build_root_candidates_from_dense(
        dense_rows,
        harmonic_min=int(args.harmonic_min),
        harmonic_max=int(args.harmonic_max),
        root_min_hz=float(args.root_min_hz),
        root_max_hz=float(args.root_max_hz),
        min_amplitude=float(args.min_amplitude),
        expected_root_hz=expected_root_hz,
        expected_root_tolerance_cents=float(args.expected_root_tolerance_cents),
    )

    clusters = cluster_root_candidates(
        root_candidates,
        cluster_cents=float(args.cluster_cents),
    )

    cluster_rows = [
        summarize_cluster_against_theory(
            c,
            expected_note=str(args.expected_note),
            anchor_token=str(args.anchor_token),
            anchor_hz=float(args.anchor_hz),
        )
        for c in clusters
    ]

    cluster_rows.sort(
        key=lambda r: (
            abs(safe_float(r["root_delta_cents_vs_theory"], 1e9)),
            -safe_int(r["unique_frame_count"], 0),
            -safe_float(r["tuner_confidence"], 0.0),
        )
    )

    write_root_candidates_csv(out_root_candidates_csv, root_candidates)
    write_cluster_summary_csv(out_cluster_summary_csv, cluster_rows)
    write_summary_txt(
        out_summary_txt,
        expected_note=str(args.expected_note),
        cluster_rows=cluster_rows,
    )
    write_meta_json(
        out_meta_json,
        dense_csv=dense_csv,
        expected_note=str(args.expected_note),
        out_root_candidates_csv=out_root_candidates_csv,
        out_cluster_summary_csv=out_cluster_summary_csv,
        out_summary_txt=out_summary_txt,
        harmonic_min=int(args.harmonic_min),
        harmonic_max=int(args.harmonic_max),
        cluster_cents=float(args.cluster_cents),
        expected_root_tolerance_cents=float(args.expected_root_tolerance_cents),
    )

    print(f"Wrote root candidates CSV : {out_root_candidates_csv}")
    print(f"Wrote cluster summary CSV : {out_cluster_summary_csv}")
    print(f"Wrote summary TXT         : {out_summary_txt}")
    print(f"Wrote meta JSON           : {out_meta_json}")


if __name__ == "__main__":
    main()