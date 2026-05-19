from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================
# HELPERS
# ============================================================

def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def parse_harmonic_list_field(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [x.strip() for x in value.split() if x.strip()]


def harmonic_sort_key(h: str) -> tuple[int, str]:
    h = h.strip().lower()
    if h.startswith("h"):
        try:
            return (int(h[1:]), h)
        except Exception:
            pass
    return (999, h)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ConvergenceRow:
    segment_index: int
    chosen_time_sec: float
    target_state: str
    target_is_best_root: bool
    target_convergence_score: float
    best_theoretical_root_token: str
    matched_harmonics_window: list[str]


@dataclass(frozen=True)
class HarmonicClusterRow:
    harmonic: str
    start_segment: int
    end_segment: int
    length: int


@dataclass(frozen=True)
class HarmonicSequenceRow:
    cluster_id: int
    start_segment: int
    end_segment: int
    length: int
    harmonics_present: list[str]
    harmonic_arrival_sequence: list[str]


# ============================================================
# LOADERS
# ============================================================

def load_convergence_csv(path: Path) -> list[ConvergenceRow]:
    rows: list[ConvergenceRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ConvergenceRow(
                    segment_index=safe_int(r.get("segment_index", ""), 0),
                    chosen_time_sec=safe_float(r.get("chosen_time_sec", ""), 0.0),
                    target_state=(r.get("target_state", "") or "").strip(),
                    target_is_best_root=str(r.get("target_is_best_root", "")).strip().lower() in {"true", "1"},
                    target_convergence_score=safe_float(r.get("target_convergence_score", ""), 0.0),
                    best_theoretical_root_token=(r.get("best_theoretical_root_token", "") or "").strip(),
                    matched_harmonics_window=parse_harmonic_list_field(r.get("matched_harmonics_window", "")),
                )
            )

    rows.sort(key=lambda x: x.segment_index)
    return rows


def load_harmonic_clusters_csv(path: Path) -> list[HarmonicClusterRow]:
    rows: list[HarmonicClusterRow] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                HarmonicClusterRow(
                    harmonic=(r.get("harmonic", "") or "").strip(),
                    start_segment=safe_int(r.get("start_segment", ""), 0),
                    end_segment=safe_int(r.get("end_segment", ""), 0),
                    length=safe_int(r.get("length", ""), 0),
                )
            )
    return rows


def load_harmonic_sequences_csv(path: Path) -> list[HarmonicSequenceRow]:
    rows: list[HarmonicSequenceRow] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                HarmonicSequenceRow(
                    cluster_id=safe_int(r.get("cluster_id", ""), 0),
                    start_segment=safe_int(r.get("start_segment", ""), 0),
                    end_segment=safe_int(r.get("end_segment", ""), 0),
                    length=safe_int(r.get("length", ""), 0),
                    harmonics_present=parse_harmonic_list_field(r.get("harmonics_present", "")),
                    harmonic_arrival_sequence=[
                        x.strip() for x in (r.get("harmonic_arrival_sequence", "") or "").split("->") if x.strip()
                    ],
                )
            )
    return rows


# ============================================================
# SIGNATURE BUILDING
# ============================================================

TARGET_ZONE_STATES = {
    "PHASE_LOCK_TO_TARGET",
    "APPROACHING_TARGET",
    "TARGET_PHASE_NEAR",
    "TARGET_RADIAL_NEAR",
}

CORE_STATES = {
    "PHASE_LOCK_TO_TARGET",
}


def build_signature(
    *,
    target_root_token: str,
    convergence_rows: list[ConvergenceRow],
    cluster_rows: list[HarmonicClusterRow],
    sequence_rows: list[HarmonicSequenceRow],
) -> dict:
    total_segments = len(convergence_rows)
    state_counts = Counter(r.target_state for r in convergence_rows)

    target_zone_rows = [r for r in convergence_rows if r.target_state in TARGET_ZONE_STATES]
    core_rows = [r for r in convergence_rows if r.target_state in CORE_STATES]

    target_zone_ratio = len(target_zone_rows) / total_segments if total_segments else 0.0
    core_ratio = len(core_rows) / total_segments if total_segments else 0.0

    target_best_root_segments = [r for r in convergence_rows if r.target_is_best_root]
    target_best_root_ratio = len(target_best_root_segments) / total_segments if total_segments else 0.0

    mean_target_convergence_score = (
        sum(r.target_convergence_score for r in convergence_rows) / total_segments
        if total_segments else 0.0
    )

    harmonic_freq_total = Counter()
    harmonic_freq_target_zone = Counter()
    harmonic_freq_core = Counter()

    for r in convergence_rows:
        for h in r.matched_harmonics_window:
            harmonic_freq_total[h] += 1
            if r.target_state in TARGET_ZONE_STATES:
                harmonic_freq_target_zone[h] += 1
            if r.target_state in CORE_STATES:
                harmonic_freq_core[h] += 1

    dominant_total = [h for h, _ in harmonic_freq_total.most_common(8)]
    dominant_target_zone = [h for h, _ in harmonic_freq_target_zone.most_common(8)]
    dominant_core = [h for h, _ in harmonic_freq_core.most_common(8)]

    cluster_lengths_by_harmonic: dict[str, list[int]] = defaultdict(list)
    for c in cluster_rows:
        cluster_lengths_by_harmonic[c.harmonic].append(c.length)

    harmonic_cluster_summary = {}
    for h, lens in cluster_lengths_by_harmonic.items():
        harmonic_cluster_summary[h] = {
            "cluster_count": len(lens),
            "max_cluster_length": max(lens),
            "mean_cluster_length": sum(lens) / len(lens),
        }

    arrival_sequence_counter = Counter()
    arrival_prefix_counter = Counter()

    for s in sequence_rows:
        seq = [x.strip() for x in s.harmonic_arrival_sequence if x.strip()]
        if not seq:
            continue
        arrival_sequence_counter[" -> ".join(seq)] += 1
        if len(seq) >= 3:
            arrival_prefix_counter[" -> ".join(seq[:3])] += 1
        elif len(seq) == 2:
            arrival_prefix_counter[" -> ".join(seq[:2])] += 1
        else:
            arrival_prefix_counter[seq[0]] += 1

    strongest_sequences = arrival_sequence_counter.most_common(20)
    strongest_prefixes = arrival_prefix_counter.most_common(20)

    # Простая формулировка сигнатуры
    signature_rules = []

    if dominant_total:
        signature_rules.append(
            f"Общее ядро проявления: {' + '.join(dominant_total[:3])}"
        )

    if dominant_target_zone:
        signature_rules.append(
            f"В зоне притяжения цели чаще всего проявляются: {' + '.join(dominant_target_zone[:3])}"
        )

    if dominant_core:
        signature_rules.append(
            f"В ядре цели чаще всего удерживаются: {' + '.join(dominant_core[:3])}"
        )

    if strongest_prefixes:
        signature_rules.append(
            f"Типичный порядок прихода гармоник: {strongest_prefixes[0][0]}"
        )

    long_harmonics = [
        h for h, info in harmonic_cluster_summary.items()
        if info["max_cluster_length"] >= 5
    ]
    long_harmonics = sorted(long_harmonics, key=harmonic_sort_key)

    if long_harmonics:
        signature_rules.append(
            f"Гармоники с длинными временными кластерами: {' + '.join(long_harmonics)}"
        )

    signature = {
        "target_root_token": target_root_token,
        "segment_statistics": {
            "total_segments": total_segments,
            "target_zone_count": len(target_zone_rows),
            "target_zone_ratio": target_zone_ratio,
            "core_count": len(core_rows),
            "core_ratio": core_ratio,
            "target_best_root_count": len(target_best_root_segments),
            "target_best_root_ratio": target_best_root_ratio,
            "mean_target_convergence_score": mean_target_convergence_score,
            "state_counts": dict(state_counts),
        },
        "harmonic_frequency": {
            "total": dict(harmonic_freq_total),
            "target_zone": dict(harmonic_freq_target_zone),
            "core": dict(harmonic_freq_core),
            "dominant_total": dominant_total,
            "dominant_target_zone": dominant_target_zone,
            "dominant_core": dominant_core,
        },
        "harmonic_clusters": harmonic_cluster_summary,
        "harmonic_arrival": {
            "strongest_sequences": strongest_sequences,
            "strongest_prefixes": strongest_prefixes,
        },
        "signature_rules": signature_rules,
    }

    return signature


# ============================================================
# OUTPUT
# ============================================================

def write_signature_json(path: Path, signature: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")


def write_signature_txt(path: Path, signature: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    seg = signature["segment_statistics"]
    harm = signature["harmonic_frequency"]
    arrival = signature["harmonic_arrival"]

    with path.open("w", encoding="utf-8") as f:
        f.write("NOTE EMERGENCE SIGNATURE\n")
        f.write("=" * 80 + "\n")
        f.write(f"target_root_token: {signature['target_root_token']}\n\n")

        f.write("SEGMENT STATISTICS\n")
        for k, v in seg.items():
            if k == "state_counts":
                continue
            f.write(f"  {k}: {v}\n")

        f.write("\nSTATE COUNTS\n")
        for k, v in seg["state_counts"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\nDOMINANT HARMONICS\n")
        f.write(f"  total: {harm['dominant_total']}\n")
        f.write(f"  target_zone: {harm['dominant_target_zone']}\n")
        f.write(f"  core: {harm['dominant_core']}\n")

        f.write("\nTOP ARRIVAL SEQUENCES\n")
        for seq, cnt in arrival["strongest_sequences"][:15]:
            f.write(f"  {seq}: {cnt}\n")

        f.write("\nTOP ARRIVAL PREFIXES\n")
        for seq, cnt in arrival["strongest_prefixes"][:15]:
            f.write(f"  {seq}: {cnt}\n")

        f.write("\nFORMULATED SIGNATURE RULES\n")
        for rule in signature["signature_rules"]:
            f.write(f"  - {rule}\n")


def write_meta_json(path: Path, *, inputs: dict, outputs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": inputs,
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a note-emergence signature from target-root convergence analysis "
            "and harmonic temporal statistics. This is a laboratory rule-building layer."
        )
    )
    ap.add_argument("--target_root_token", required=True)
    ap.add_argument("--convergence_csv", required=True)
    ap.add_argument("--harmonic_clusters_csv", required=True)
    ap.add_argument("--harmonic_sequences_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    convergence_csv = Path(args.convergence_csv).resolve()
    harmonic_clusters_csv = Path(args.harmonic_clusters_csv).resolve()
    harmonic_sequences_csv = Path(args.harmonic_sequences_csv).resolve()
    out_json = Path(args.out_json).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    convergence_rows = load_convergence_csv(convergence_csv)
    cluster_rows = load_harmonic_clusters_csv(harmonic_clusters_csv)
    sequence_rows = load_harmonic_sequences_csv(harmonic_sequences_csv)

    signature = build_signature(
        target_root_token=args.target_root_token,
        convergence_rows=convergence_rows,
        cluster_rows=cluster_rows,
        sequence_rows=sequence_rows,
    )

    write_signature_json(out_json, signature)
    write_signature_txt(out_txt, signature)
    write_meta_json(
        out_meta_json,
        inputs={
            "target_root_token": args.target_root_token,
            "convergence_csv": str(convergence_csv),
            "harmonic_clusters_csv": str(harmonic_clusters_csv),
            "harmonic_sequences_csv": str(harmonic_sequences_csv),
        },
        outputs={
            "signature_json": str(out_json),
            "signature_txt": str(out_txt),
        },
    )

    print("note emergence signature build complete")
    print(json.dumps(
        {
            "target_root_token": args.target_root_token,
            "out_json": str(out_json),
            "out_txt": str(out_txt),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()