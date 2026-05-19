from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _to_float(v: str, default: Optional[float] = None) -> Optional[float]:
    s = (v or "").strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _to_int(v: str, default: Optional[int] = None) -> Optional[int]:
    s = (v or "").strip()
    if s == "":
        return default
    try:
        return int(s)
    except Exception:
        return default


def load_compare_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_summary_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def save_bar_counts(rows: List[dict], out_png: Path) -> None:
    cnt = Counter(r["classification"] for r in rows)

    labels = list(cnt.keys())
    values = [cnt[k] for k in labels]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.title("Classification counts")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_abs_delta_vs_time(rows: List[dict], out_png: Path) -> None:
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": []})

    for r in rows:
        cls = r["classification"]
        x = _to_float(r.get("det_time_start_sec", ""))
        y = _to_float(r.get("abs_delta_steps", ""))
        if x is None or y is None:
            continue
        groups[cls]["x"].append(x)
        groups[cls]["y"].append(y)

    plt.figure(figsize=(14, 6))
    for cls, data in groups.items():
        if not data["x"]:
            continue
        plt.scatter(data["x"], data["y"], s=8, label=cls)
    plt.xlabel("detected time start (sec)")
    plt.ylabel("abs delta steps")
    plt.title("Absolute pitch delta vs time")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_octave_delta_vs_etalon_octave(rows: List[dict], out_png: Path) -> None:
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": []})

    for r in rows:
        cls = r["classification"]
        x = _to_float(r.get("et_octave_index1", ""))
        y = _to_float(r.get("octave_delta", ""))
        if x is None or y is None:
            continue
        groups[cls]["x"].append(x)
        groups[cls]["y"].append(y)

    plt.figure(figsize=(12, 6))
    for cls, data in groups.items():
        if not data["x"]:
            continue
        plt.scatter(data["x"], data["y"], s=8, label=cls)
    plt.xlabel("etalon octave index")
    plt.ylabel("octave delta")
    plt.title("Octave delta vs etalon octave")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_abs_delta_hist(rows: List[dict], out_png: Path, max_abs: int = 120) -> None:
    vals: List[float] = []

    for r in rows:
        v = _to_float(r.get("abs_delta_steps", ""))
        if v is None:
            continue
        if abs(v) <= max_abs:
            vals.append(v)

    plt.figure(figsize=(12, 5))
    if vals:
        bins = min(80, max(20, len(set(int(v) for v in vals))))
        plt.hist(vals, bins=bins)
    plt.xlabel("abs delta steps")
    plt.ylabel("count")
    plt.title(f"Histogram of abs delta steps (|delta| <= {max_abs})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_nearest_harmonic_hist(rows: List[dict], out_png: Path) -> None:
    vals: List[int] = []

    for r in rows:
        idx = _to_int(r.get("nearest_harmonic_index", ""))
        if idx is None:
            continue
        vals.append(idx)

    cnt = Counter(vals)
    labels = sorted(cnt.keys())
    values = [cnt[k] for k in labels]

    plt.figure(figsize=(10, 5))
    if labels:
        plt.bar([str(x) for x in labels], values)
    plt.xlabel("nearest harmonic index")
    plt.ylabel("count")
    plt.title("Nearest harmonic index histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_no_time_match_timeline(rows: List[dict], out_png: Path, bin_sec: float = 1.0) -> None:
    bins: Dict[int, int] = defaultdict(int)

    for r in rows:
        if r["classification"] != "no_time_match":
            continue
        t = _to_float(r.get("det_time_start_sec", ""))
        if t is None:
            continue
        b = int(t // bin_sec)
        bins[b] += 1

    xs = sorted(bins.keys())
    ys = [bins[x] for x in xs]

    plt.figure(figsize=(14, 4))
    if xs:
        plt.bar(xs, ys)
    plt.xlabel(f"time bin ({bin_sec:.1f} sec)")
    plt.ylabel("no_time_match count")
    plt.title("No-time-match timeline")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_summary_table_txt(summary_rows: List[dict], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = ["COMPARE F0 CHAINS PLOT SUMMARY", ""]
    for r in summary_rows:
        lines.append(f'{r.get("classification","")} : {r.get("count","")}')
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot diagnostic charts from compare_f0_chains outputs."
    )
    ap.add_argument("--compare_csv", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    compare_csv = Path(args.compare_csv).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_compare_rows(compare_csv)
    summary_rows = load_summary_rows(summary_csv)

    save_bar_counts(rows, out_dir / "01_classification_counts.png")
    save_abs_delta_vs_time(rows, out_dir / "02_abs_delta_vs_time.png")
    save_octave_delta_vs_etalon_octave(rows, out_dir / "03_octave_delta_vs_etalon_octave.png")
    save_abs_delta_hist(rows, out_dir / "04_abs_delta_hist.png")
    save_nearest_harmonic_hist(rows, out_dir / "05_nearest_harmonic_index_hist.png")
    save_no_time_match_timeline(rows, out_dir / "06_no_time_match_timeline.png")
    save_summary_table_txt(summary_rows, out_dir / "00_summary.txt")

    print("compare f0 chains plots complete")
    print(str(out_dir))


if __name__ == "__main__":
    main()