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


def parse_harmonics_field(value: str) -> list[str]:
    """
    Input examples:
      "h1 h2 h3"
      ""
    """
    value = (value or "").strip()
    if not value:
        return []
    return [x.strip() for x in value.split() if x.strip()]


def normalize_harmonic_label(h: str) -> str:
    h = h.strip().lower()
    if not h.startswith("h"):
        return h
    return h


def harmonic_sort_key(h: str) -> tuple[int, str]:
    try:
        if h.lower().startswith("h"):
            return (int(h[1:]), h)
    except Exception:
        pass
    return (999, h)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class HarmonicRow:
    segment_index: int
    chosen_time_sec: float
    target_state: str
    target_is_best_root: bool
    target_convergence_score: float

    matched_harmonics_same_frame: list[str]
    matched_harmonics_window: list[str]
    missing_harmonics_window: list[str]

    representative_rc_note: str
    best_theoretical_root_token: str


# ============================================================
# LOAD
# ============================================================

def load_rows(path: Path) -> list[HarmonicRow]:
    rows: list[HarmonicRow] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                HarmonicRow(
                    segment_index=safe_int(r.get("segment_index", ""), 0),
                    chosen_time_sec=safe_float(r.get("chosen_time_sec", ""), 0.0),
                    target_state=(r.get("target_state", "") or "").strip(),
                    target_is_best_root=str(r.get("target_is_best_root", "")).strip().lower() in {"true", "1"},
                    target_convergence_score=safe_float(r.get("target_convergence_score", ""), 0.0),

                    matched_harmonics_same_frame=[
                        normalize_harmonic_label(x)
                        for x in parse_harmonics_field(r.get("matched_harmonics_same_frame", ""))
                    ],
                    matched_harmonics_window=[
                        normalize_harmonic_label(x)
                        for x in parse_harmonics_field(r.get("matched_harmonics_window", ""))
                    ],
                    missing_harmonics_window=[
                        normalize_harmonic_label(x)
                        for x in parse_harmonics_field(r.get("missing_harmonics_window", ""))
                    ],

                    representative_rc_note=(r.get("representative_rc_note", "") or "").strip(),
                    best_theoretical_root_token=(r.get("best_theoretical_root_token", "") or "").strip(),
                )
            )

    rows.sort(key=lambda x: x.segment_index)
    return rows


# ============================================================
# ANALYSIS
# ============================================================

def build_global_frequency(rows: list[HarmonicRow]) -> Counter:
    c = Counter()
    for r in rows:
        for h in r.matched_harmonics_window:
            c[h] += 1
    return c


def build_state_frequency(rows: list[HarmonicRow]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for h in r.matched_harmonics_window:
            out[r.target_state][h] += 1
    return out


def build_same_frame_frequency(rows: list[HarmonicRow]) -> Counter:
    c = Counter()
    for r in rows:
        for h in r.matched_harmonics_same_frame:
            c[h] += 1
    return c


def build_cooccurrence(rows: list[HarmonicRow]) -> Counter:
    c = Counter()
    for r in rows:
        hs = sorted(set(r.matched_harmonics_window), key=harmonic_sort_key)
        if len(hs) < 2:
            continue
        key = " + ".join(hs)
        c[key] += 1
    return c


def build_clusters(
    rows: list[HarmonicRow],
    *,
    max_gap_segments: int = 2,
) -> dict[str, list[dict]]:
    """
    Build temporal clusters per harmonic from matched_harmonics_window.
    """
    harmonic_to_segments: dict[str, list[int]] = defaultdict(list)

    for r in rows:
        for h in r.matched_harmonics_window:
            harmonic_to_segments[h].append(r.segment_index)

    out: dict[str, list[dict]] = {}

    for h, segs in harmonic_to_segments.items():
        segs = sorted(set(segs))
        if not segs:
            out[h] = []
            continue

        clusters: list[dict] = []
        current = [segs[0]]

        for s in segs[1:]:
            if s - current[-1] <= max_gap_segments:
                current.append(s)
            else:
                clusters.append(
                    {
                        "harmonic": h,
                        "start_segment": current[0],
                        "end_segment": current[-1],
                        "length": len(current),
                        "segments": current[:],
                    }
                )
                current = [s]

        clusters.append(
            {
                "harmonic": h,
                "start_segment": current[0],
                "end_segment": current[-1],
                "length": len(current),
                "segments": current[:],
            }
        )
        out[h] = clusters

    return out


def build_cluster_sequences(
    rows: list[HarmonicRow],
    *,
    states_of_interest: Optional[set[str]] = None,
    max_gap_segments: int = 2,
) -> list[dict]:
    """
    Build local harmonic arrival sequences inside target-related clusters.
    """
    if states_of_interest is None:
        states_of_interest = {
            "PHASE_LOCK_TO_TARGET",
            "APPROACHING_TARGET",
            "TARGET_PHASE_NEAR",
            "TARGET_RADIAL_NEAR",
        }

    filtered = [r for r in rows if r.target_state in states_of_interest]
    if not filtered:
        return []

    filtered.sort(key=lambda x: x.segment_index)

    clusters: list[list[HarmonicRow]] = []
    current = [filtered[0]]

    for r in filtered[1:]:
        if r.segment_index - current[-1].segment_index <= max_gap_segments:
            current.append(r)
        else:
            clusters.append(current[:])
            current = [r]
    clusters.append(current[:])

    out: list[dict] = []

    for idx, cl in enumerate(clusters):
        first_seen: dict[str, int] = {}
        for row in cl:
            for h in row.matched_harmonics_window:
                if h not in first_seen:
                    first_seen[h] = row.segment_index

        ordered = sorted(first_seen.items(), key=lambda x: (x[1], harmonic_sort_key(x[0])))
        sequence = [h for h, _ in ordered]

        out.append(
            {
                "cluster_id": idx,
                "start_segment": cl[0].segment_index,
                "end_segment": cl[-1].segment_index,
                "length": len(cl),
                "states": " | ".join(r.target_state for r in cl),
                "harmonic_arrival_sequence": " -> ".join(sequence),
                "harmonics_present": " ".join(sorted(first_seen.keys(), key=harmonic_sort_key)),
            }
        )

    return out


# ============================================================
# WRITE
# ============================================================

def write_frequency_csv(path: Path, freq: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["harmonic", "count"])
        for h, v in sorted(freq.items(), key=lambda x: (harmonic_sort_key(x[0]), -x[1])):
            writer.writerow([h, v])


def write_state_frequency_csv(path: Path, state_freq: dict[str, Counter]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    harmonics = sorted(
        {h for c in state_freq.values() for h in c.keys()},
        key=harmonic_sort_key,
    )
    states = sorted(state_freq.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state"] + harmonics)
        for st in states:
            writer.writerow([st] + [state_freq[st].get(h, 0) for h in harmonics])


def write_cooccurrence_csv(path: Path, cooc: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["harmonic_group", "count"])
        for key, value in cooc.most_common():
            writer.writerow([key, value])


def write_clusters_csv(path: Path, clusters: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for h, cls in clusters.items():
        for c in cls:
            rows.append(c)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sequences_csv(path: Path, seqs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not seqs:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seqs[0].keys()))
        writer.writeheader()
        writer.writerows(seqs)


def write_txt_report(
    path: Path,
    *,
    input_csv: Path,
    global_freq: Counter,
    same_frame_freq: Counter,
    state_freq: dict[str, Counter],
    cooc: Counter,
    clusters: dict[str, list[dict]],
    seqs: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("TARGET HARMONIC TEMPORAL ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"input_csv: {input_csv}\n\n")

        f.write("GLOBAL MATCHED_HARMONICS_WINDOW FREQUENCY\n")
        for h, v in sorted(global_freq.items(), key=lambda x: (harmonic_sort_key(x[0]), -x[1])):
            f.write(f"  {h}: {v}\n")

        f.write("\nSAME-FRAME HARMONIC FREQUENCY\n")
        for h, v in sorted(same_frame_freq.items(), key=lambda x: (harmonic_sort_key(x[0]), -x[1])):
            f.write(f"  {h}: {v}\n")

        f.write("\nFREQUENCY BY TARGET_STATE\n")
        for st in sorted(state_freq.keys()):
            f.write(f"\n  {st}\n")
            for h, v in sorted(state_freq[st].items(), key=lambda x: (harmonic_sort_key(x[0]), -x[1])):
                f.write(f"    {h}: {v}\n")

        f.write("\nTOP CO-OCCURRENCE GROUPS\n")
        for key, value in cooc.most_common(20):
            f.write(f"  {key}: {value}\n")

        f.write("\nTEMPORAL CLUSTERS BY HARMONIC\n")
        for h in sorted(clusters.keys(), key=harmonic_sort_key):
            f.write(f"\n  {h}\n")
            for c in clusters[h][:20]:
                f.write(
                    f"    start={c['start_segment']} end={c['end_segment']} "
                    f"length={c['length']} segments={c['segments']}\n"
                )

        f.write("\nCLUSTER ARRIVAL SEQUENCES\n")
        for s in seqs[:50]:
            f.write(
                f"  cluster_id={s['cluster_id']} "
                f"start={s['start_segment']} end={s['end_segment']} "
                f"length={s['length']} "
                f"harmonics_present={s['harmonics_present']} "
                f"arrival={s['harmonic_arrival_sequence']}\n"
            )


def write_meta_json(
    path: Path,
    *,
    input_csv: Path,
    outputs: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "input_csv": str(input_csv),
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze temporal statistics of harmonics for a previously computed target-root convergence table. "
            "This is a laboratory analysis layer, not a recognition algorithm."
        )
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_frequency_csv", required=True)
    ap.add_argument("--out_state_frequency_csv", required=True)
    ap.add_argument("--out_cooccurrence_csv", required=True)
    ap.add_argument("--out_clusters_csv", required=True)
    ap.add_argument("--out_sequences_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_gap_segments", type=int, default=2)
    args = ap.parse_args()

    input_csv = Path(args.input_csv).resolve()

    rows = load_rows(input_csv)

    global_freq = build_global_frequency(rows)
    same_frame_freq = build_same_frame_frequency(rows)
    state_freq = build_state_frequency(rows)
    cooc = build_cooccurrence(rows)
    clusters = build_clusters(rows, max_gap_segments=args.max_gap_segments)
    seqs = build_cluster_sequences(rows, max_gap_segments=args.max_gap_segments)

    out_frequency_csv = Path(args.out_frequency_csv).resolve()
    out_state_frequency_csv = Path(args.out_state_frequency_csv).resolve()
    out_cooccurrence_csv = Path(args.out_cooccurrence_csv).resolve()
    out_clusters_csv = Path(args.out_clusters_csv).resolve()
    out_sequences_csv = Path(args.out_sequences_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    write_frequency_csv(out_frequency_csv, global_freq)
    write_state_frequency_csv(out_state_frequency_csv, state_freq)
    write_cooccurrence_csv(out_cooccurrence_csv, cooc)
    write_clusters_csv(out_clusters_csv, clusters)
    write_sequences_csv(out_sequences_csv, seqs)
    write_txt_report(
        out_txt,
        input_csv=input_csv,
        global_freq=global_freq,
        same_frame_freq=same_frame_freq,
        state_freq=state_freq,
        cooc=cooc,
        clusters=clusters,
        seqs=seqs,
    )
    write_meta_json(
        out_meta_json,
        input_csv=input_csv,
        outputs={
            "frequency_csv": str(out_frequency_csv),
            "state_frequency_csv": str(out_state_frequency_csv),
            "cooccurrence_csv": str(out_cooccurrence_csv),
            "clusters_csv": str(out_clusters_csv),
            "sequences_csv": str(out_sequences_csv),
            "txt": str(out_txt),
        },
    )

    print("target harmonic temporal analysis complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_frequency_csv": str(out_frequency_csv),
            "out_state_frequency_csv": str(out_state_frequency_csv),
            "out_cooccurrence_csv": str(out_cooccurrence_csv),
            "out_clusters_csv": str(out_clusters_csv),
            "out_sequences_csv": str(out_sequences_csv),
            "out_txt": str(out_txt),
            "out_meta_json": str(out_meta_json),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()