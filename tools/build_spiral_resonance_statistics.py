from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from music12.core.harmonic_alphabet12 import harmonic_token_from_root
from music12.core.notation12 import normalize_token


# ============================================================
# HELPERS
# ============================================================

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def extract_root_from_folder(folder_name: str) -> str:
    """
    Example:
      001__RealPiano_1__5.A-  ->  5.A-
    """
    parts = folder_name.split("__")
    if len(parts) >= 3:
        return parts[2].strip()
    return folder_name.strip()


def root_band(root: str) -> str:
    """
    Coarse banding by octave container.
    This is only for summary statistics, not final theory.
    """
    if "." not in root:
        return "unknown"

    octv = root.split(".", 1)[0].strip().upper()

    low = {"5", "6"}
    mid = {"7", "8"}
    high = {"9", "A", "B", "C", "11"}

    if octv in low:
        return "low"
    if octv in mid:
        return "mid"
    if octv in high:
        return "high"
    return "unknown"


def build_ideal_chain(root: str, max_h: int = 8) -> set[str]:
    """
    Ideal theoretical chain for a root note using current project law.
    """
    chain = set()

    try:
        root_norm = normalize_token(root)
    except Exception:
        root_norm = root.strip()

    # Root itself must belong to its own ideal chain
    chain.add(root_norm)

    for h in range(1, max_h + 1):
        try:
            tok = harmonic_token_from_root(root_norm, h)
            chain.add(normalize_token(tok))
        except Exception:
            continue

    return chain


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_note_token_from_row(row: dict[str, str]) -> str:
    """
    Try all realistic token columns that may exist in stabilized report.
    """
    candidates = [
        "note_token",
        "best_theoretical_root_token",
        "representative_rc_note",
        "target_root_token",
    ]
    for c in candidates:
        v = (row.get(c, "") or "").strip()
        if v:
            try:
                return normalize_token(v)
            except Exception:
                return v
    return ""


def get_phase_from_row(row: dict[str, str]) -> float:
    for c in ["mean_phase_deg", "phase_deg", "target_phase_distance"]:
        if c in row and str(row.get(c, "")).strip():
            return safe_float(row.get(c, 0.0))
    return 0.0


def get_radial_from_row(row: dict[str, str]) -> float:
    for c in ["mean_radial_level", "radial_level", "target_radial_distance"]:
        if c in row and str(row.get(c, "")).strip():
            return safe_float(row.get(c, 0.0))
    return 0.0


def get_amplitude_from_row(row: dict[str, str]) -> float:
    """
    Prefer real confidence/energy-like columns.
    """
    for c in [
        "note_confidence",
        "stabilization_score",
        "target_convergence_score",
        "chosen_rc_energy",
        "energy",
    ]:
        if c in row and str(row.get(c, "")).strip():
            return safe_float(row.get(c, 0.0))
    return 0.0


# ============================================================
# CORE
# ============================================================

def build_statistics(input_dir: Path, *, max_h: int = 8):
    """
    Collect non-chain spiral responses from all note folders.
    """

    # per root -> response note -> aggregated stats
    per_root = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "amp_sum": 0.0,
        "phase_sum": 0.0,
        "radial_sum": 0.0,
    }))

    # global response note stats
    global_map = defaultdict(lambda: {
        "count": 0,
        "amp_sum": 0.0,
        "phase_sum": 0.0,
        "radial_sum": 0.0,
    })

    # band -> response note -> aggregated stats
    per_band = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "amp_sum": 0.0,
        "phase_sum": 0.0,
        "radial_sum": 0.0,
    }))

    folders = [p for p in input_dir.iterdir() if p.is_dir()]
    processed_roots = []
    skipped_roots = []

    for folder in sorted(folders):
        root = extract_root_from_folder(folder.name)
        band = root_band(root)
        ideal_chain = build_ideal_chain(root, max_h=max_h)

        csv_files = list(folder.glob("*__stabilized__with_phase.csv"))
        if not csv_files:
            skipped_roots.append({"root": root, "reason": "no_stabilized_with_phase_csv"})
            continue

        processed_roots.append(root)

        for f in csv_files:
            with f.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)

                for row in reader:
                    note = get_note_token_from_row(row)
                    if not note:
                        continue

                    # only non-chain responses
                    if note in ideal_chain:
                        continue

                    amp = get_amplitude_from_row(row)
                    phase = get_phase_from_row(row)
                    radial = get_radial_from_row(row)

                    d1 = per_root[root][note]
                    d1["count"] += 1
                    d1["amp_sum"] += amp
                    d1["phase_sum"] += phase
                    d1["radial_sum"] += radial

                    d2 = global_map[note]
                    d2["count"] += 1
                    d2["amp_sum"] += amp
                    d2["phase_sum"] += phase
                    d2["radial_sum"] += radial

                    d3 = per_band[band][note]
                    d3["count"] += 1
                    d3["amp_sum"] += amp
                    d3["phase_sum"] += phase
                    d3["radial_sum"] += radial

    return {
        "per_root": per_root,
        "global_map": global_map,
        "per_band": per_band,
        "processed_roots": processed_roots,
        "skipped_roots": skipped_roots,
    }


# ============================================================
# FINALIZATION
# ============================================================

def finalize_map(d):
    out = []
    for note, vals in d.items():
        n = vals["count"]
        out.append({
            "note": note,
            "count": n,
            "mean_amplitude": vals["amp_sum"] / n if n else 0.0,
            "mean_phase_deg": vals["phase_sum"] / n if n else 0.0,
            "mean_radial_level": vals["radial_sum"] / n if n else 0.0,
        })
    out.sort(key=lambda x: (-x["count"], -x["mean_amplitude"], x["note"]))
    return out


# ============================================================
# WRITE CSV
# ============================================================

def write_per_root(path: Path, per_root):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "root",
            "response_note",
            "count",
            "mean_amplitude",
            "mean_phase_deg",
            "mean_radial_level",
        ])

        for root in sorted(per_root.keys()):
            items = finalize_map(per_root[root])
            for row in items:
                writer.writerow([
                    root,
                    row["note"],
                    row["count"],
                    row["mean_amplitude"],
                    row["mean_phase_deg"],
                    row["mean_radial_level"],
                ])


def write_global(path: Path, global_map):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "response_note",
            "count",
            "mean_amplitude",
            "mean_phase_deg",
            "mean_radial_level",
        ])

        for row in finalize_map(global_map):
            writer.writerow([
                row["note"],
                row["count"],
                row["mean_amplitude"],
                row["mean_phase_deg"],
                row["mean_radial_level"],
            ])


def write_per_band(path: Path, per_band):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "band",
            "response_note",
            "count",
            "mean_amplitude",
            "mean_phase_deg",
            "mean_radial_level",
        ])

        for band in sorted(per_band.keys()):
            items = finalize_map(per_band[band])
            for row in items:
                writer.writerow([
                    band,
                    row["note"],
                    row["count"],
                    row["mean_amplitude"],
                    row["mean_phase_deg"],
                    row["mean_radial_level"],
                ])


# ============================================================
# WRITE TXT / META
# ============================================================

def write_txt(path: Path, result):
    ensure_parent(path)

    per_root = result["per_root"]
    global_map = result["global_map"]
    per_band = result["per_band"]
    processed_roots = result["processed_roots"]
    skipped_roots = result["skipped_roots"]

    with path.open("w", encoding="utf-8") as f:
        f.write("SPIRAL RESONANCE STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"processed_roots: {len(processed_roots)}\n")
        f.write(f"skipped_roots: {len(skipped_roots)}\n\n")

        if skipped_roots:
            f.write("SKIPPED ROOTS\n")
            for row in skipped_roots[:20]:
                f.write(f"  {row['root']} -> {row['reason']}\n")
            f.write("\n")

        f.write("GLOBAL TOP RESPONSES\n")
        for row in finalize_map(global_map)[:20]:
            f.write(
                f"{row['note']} | "
                f"count={row['count']} | "
                f"amp={row['mean_amplitude']:.2f} | "
                f"phase={row['mean_phase_deg']:.1f} | "
                f"radial={row['mean_radial_level']:.2f}\n"
            )

        f.write("\nTOP RESPONSES BY BAND\n")
        for band in sorted(per_band.keys()):
            f.write(f"\n[{band}]\n")
            for row in finalize_map(per_band[band])[:12]:
                f.write(
                    f"{row['note']} | "
                    f"count={row['count']} | "
                    f"amp={row['mean_amplitude']:.2f} | "
                    f"phase={row['mean_phase_deg']:.1f} | "
                    f"radial={row['mean_radial_level']:.2f}\n"
                )

        f.write("\nTOP RESPONSES PER ROOT\n")
        for root in sorted(per_root.keys()):
            f.write(f"\n[{root}]\n")
            for row in finalize_map(per_root[root])[:10]:
                f.write(
                    f"{row['note']} | "
                    f"count={row['count']} | "
                    f"amp={row['mean_amplitude']:.2f} | "
                    f"phase={row['mean_phase_deg']:.1f} | "
                    f"radial={row['mean_radial_level']:.2f}\n"
                )


def write_meta_json(path: Path, *, input_dir: Path, outputs: dict, result: dict, max_h: int):
    ensure_parent(path)
    meta = {
        "inputs": {
            "input_dir": str(input_dir),
            "max_h": max_h,
        },
        "outputs": outputs,
        "summary": {
            "processed_roots_count": len(result["processed_roots"]),
            "skipped_roots_count": len(result["skipped_roots"]),
        },
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_per_root", required=True)
    ap.add_argument("--out_global", required=True)
    ap.add_argument("--out_per_band", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_h", type=int, default=8)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()

    result = build_statistics(input_dir, max_h=args.max_h)

    write_per_root(Path(args.out_per_root), result["per_root"])
    write_global(Path(args.out_global), result["global_map"])
    write_per_band(Path(args.out_per_band), result["per_band"])
    write_txt(Path(args.out_txt), result)
    write_meta_json(
        Path(args.out_meta_json),
        input_dir=input_dir,
        outputs={
            "per_root_csv": str(Path(args.out_per_root).resolve()),
            "global_csv": str(Path(args.out_global).resolve()),
            "per_band_csv": str(Path(args.out_per_band).resolve()),
            "txt": str(Path(args.out_txt).resolve()),
        },
        result=result,
        max_h=args.max_h,
    )

    print("spiral resonance statistics built")
    print(f"processed_roots={len(result['processed_roots'])}")
    print(f"skipped_roots={len(result['skipped_roots'])}")


if __name__ == "__main__":
    main()