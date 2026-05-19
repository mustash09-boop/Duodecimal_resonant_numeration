from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def sf(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def si(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            return sum(1 for _ in r)
    except Exception:
        return 0


def load_summary_txt(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    text = read_text(path)
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                return dict(row)
    except Exception:
        return {}
    return {}


def extract_root_from_summary(root_summary_path: Path) -> dict[str, Any]:
    text = read_text(root_summary_path)
    out = {
        "consensus_root_token": "",
        "consensus_root_hz": 0.0,
        "root_delta_cents_vs_theory": 0.0,
        "member_count": 0,
        "unique_frame_count": 0,
        "present_harmonics": "",
        "tuner_confidence": 0.0,
    }

    patterns = {
        "consensus_root_token": r"consensus_root_token\s*:\s*([^\s]+)",
        "consensus_root_hz": r"consensus_root_hz\s*:\s*([-0-9.]+)",
        "root_delta_cents_vs_theory": r"root_delta_cents_vs_theory\s*:\s*([-0-9.]+)",
        "member_count": r"member_count\s*:\s*([0-9]+)",
        "unique_frame_count": r"unique_frame_count\s*:\s*([0-9]+)",
        "present_harmonics": r"present_harmonics\s*:\s*(\[.*?\])",
        "tuner_confidence": r"tuner_confidence\s*:\s*([-0-9.]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            continue
        val = m.group(1).strip()
        if key in ("consensus_root_hz", "root_delta_cents_vs_theory", "tuner_confidence"):
            out[key] = sf(val)
        elif key in ("member_count", "unique_frame_count"):
            out[key] = si(val)
        else:
            out[key] = val

    return out


def extract_expected_note(folder: Path, dense_vs_theory_csv: Path) -> str:
    text = read_text(folder / f"{folder.name}__root_consensus_summary.txt")
    m = re.search(r"expected_note\s*:\s*([^\s]+)", text)
    if m:
        return m.group(1).strip()

    row = read_first_csv_row(dense_vs_theory_csv)
    return row.get("expected_note", "").strip()


def load_dense_vs_theory_stats(path: Path) -> dict[str, Any]:
    stats = {
        "theory_rows": 0,
        "theory_match": 0,
        "theory_missing": 0,
        "theory_shifted_down": 0,
        "theory_shifted_up": 0,
        "theory_match_ratio": 0.0,
        "mean_abs_delta_cents": 0.0,
    }

    if not path.exists():
        return stats

    deltas: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stats["theory_rows"] += 1
            status = str(row.get("status", "")).strip()
            if status == "MATCH":
                stats["theory_match"] += 1
            elif status == "MISSING":
                stats["theory_missing"] += 1
            elif status == "SHIFTED_DOWN":
                stats["theory_shifted_down"] += 1
            elif status == "SHIFTED_UP":
                stats["theory_shifted_up"] += 1

            if row.get("delta_cents") not in (None, ""):
                deltas.append(abs(sf(row.get("delta_cents"))))

    if stats["theory_rows"]:
        stats["theory_match_ratio"] = stats["theory_match"] / stats["theory_rows"]
    if deltas:
        stats["mean_abs_delta_cents"] = sum(deltas) / len(deltas)

    return stats


def load_clean_summary(path: Path) -> dict[str, Any]:
    raw = load_summary_txt(path)
    return {
        "note_range": raw.get("note_range", raw.get("range", "")),
        "root_hz": sf(raw.get("root_hz")),
        "protected_harmonics": raw.get("protected_harmonics", ""),
        "total_rows": si(raw.get("total_rows")),
        "clean_rows": si(raw.get("clean_rows")),
        "removed_rows": si(raw.get("removed_rows")),
        "removed_percent": sf(raw.get("removed_percent")),
    }


def load_spiral_stats(path: Path) -> dict[str, Any]:
    stats = {
        "spiral_points": 0,
        "phase_min": 0.0,
        "phase_max": 0.0,
        "radius_min": 0.0,
        "radius_max": 0.0,
    }

    if not path.exists():
        return stats

    phases: list[float] = []
    radii: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stats["spiral_points"] += 1

            for k in ("phase12", "phase", "angle_deg"):
                if row.get(k) not in ("", None):
                    phases.append(sf(row.get(k)))
                    break

            for k in ("radial_level", "radius", "r"):
                if row.get(k) not in ("", None):
                    radii.append(sf(row.get(k)))
                    break

    if phases:
        stats["phase_min"] = min(phases)
        stats["phase_max"] = max(phases)
    if radii:
        stats["radius_min"] = min(radii)
        stats["radius_max"] = max(radii)

    return stats


def load_box_profile(path: Path, top_n: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cluster_hz = sf(row.get("cluster_center_hz"))
            cluster_token = str(row.get("dominant_token", "")).strip()
            note_count = si(row.get("count_notes"))
            percent_notes = sf(row.get("percent_notes"))
            mean_amp = sf(row.get("mean_sum_amplitude"))
            median_amp = sf(row.get("median_sum_amplitude"))
            std_amp = sf(row.get("std_sum_amplitude"))
            relative_amp = sf(row.get("mean_relative_amplitude"))
            examples = str(row.get("note_examples", "")).strip()

            if cluster_hz <= 0 and not cluster_token:
                continue

            rows.append({
                "cluster_hz": cluster_hz,
                "cluster_token": cluster_token,
                "note_count": note_count,
                "percent_notes": percent_notes,
                "mean_amp": mean_amp,
                "median_amp": median_amp,
                "std_amp": std_amp,
                "relative_amp": relative_amp,
                "examples": examples,
            })

    rows.sort(key=lambda x: (-x["percent_notes"], -x["relative_amp"], -x["mean_amp"], x["cluster_hz"]))
    return rows[:top_n]


def load_box_relation(path: Path, top_n: int = 120) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "cluster_hz": sf(row.get("cluster_hz")),
                "token": str(row.get("token", "")).strip(),
                "percent_notes": sf(row.get("percent_notes")),
                "mean_amp": sf(row.get("mean_amp")),
                "class": str(row.get("class", "")).strip(),
                "harmonic_index": si(row.get("harmonic_index")),
                "delta_cents": sf(row.get("delta_cents")),
            })

    class_rank = {"HARMONIC": 0, "NEAR_HARMONIC": 1, "NON_HARMONIC": 2}
    rows.sort(key=lambda x: (class_rank.get(x["class"], 9), -x["percent_notes"], x["delta_cents"]))
    return rows[:top_n]


def collect_note_passport(folder: Path) -> dict[str, Any]:
    stem = folder.name

    root_summary_path = folder / f"{stem}__root_consensus_summary.txt"
    dense_vs_theory_csv = folder / f"{stem}__dense_vs_theory.csv"
    clean_summary_path = folder / f"{stem}__dense_unified_clean_summary.txt"
    clean_csv = folder / f"{stem}__dense_unified_clean.csv"
    removed_csv = folder / f"{stem}__dense_unified_removed_box.csv"
    spiral_csv = folder / f"{stem}__spiral12_clean_points.csv"
    spiral_png = folder / f"{stem}__spiral12_clean.png"
    dense_csv = folder / f"{stem}__dense.csv"
    chain_json = folder / f"{stem}__dense_chain_summary.json"

    root = extract_root_from_summary(root_summary_path)
    clean = load_clean_summary(clean_summary_path)
    theory = load_dense_vs_theory_stats(dense_vs_theory_csv)
    spiral = load_spiral_stats(spiral_csv)
    expected_note = extract_expected_note(folder, dense_vs_theory_csv)

    return {
        "folder_name": stem,
        "expected_note": expected_note,
        "root_consensus_token": root["consensus_root_token"],
        "root_hz": root["consensus_root_hz"],
        "root_delta_cents_vs_theory": root["root_delta_cents_vs_theory"],
        "root_member_count": root["member_count"],
        "root_frame_count": root["unique_frame_count"],
        "present_harmonics": root["present_harmonics"],
        "tuner_confidence": root["tuner_confidence"],

        "theory_rows": theory["theory_rows"],
        "theory_match": theory["theory_match"],
        "theory_match_ratio": theory["theory_match_ratio"],
        "theory_missing": theory["theory_missing"],
        "theory_shifted_down": theory["theory_shifted_down"],
        "theory_shifted_up": theory["theory_shifted_up"],
        "mean_abs_delta_cents": theory["mean_abs_delta_cents"],

        "dense_rows": count_csv_rows(dense_csv),
        "clean_rows": count_csv_rows(clean_csv),
        "removed_box_rows": count_csv_rows(removed_csv),
        "dense_removed_percent": clean["removed_percent"],
        "protected_harmonics": clean["protected_harmonics"],

        "spiral_points": spiral["spiral_points"],
        "phase_min": spiral["phase_min"],
        "phase_max": spiral["phase_max"],
        "radius_min": spiral["radius_min"],
        "radius_max": spiral["radius_max"],

        "dense_csv": str(dense_csv) if dense_csv.exists() else "",
        "chain_summary_json": str(chain_json) if chain_json.exists() else "",
        "dense_vs_theory_csv": str(dense_vs_theory_csv) if dense_vs_theory_csv.exists() else "",
        "clean_dense_csv": str(clean_csv) if clean_csv.exists() else "",
        "removed_box_csv": str(removed_csv) if removed_csv.exists() else "",
        "spiral12_points_csv": str(spiral_csv) if spiral_csv.exists() else "",
        "spiral12_png": str(spiral_png) if spiral_png.exists() else "",
    }


def collect_notes(reports_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder in sorted([p for p in reports_root.iterdir() if p.is_dir()]):
        rows.append(collect_note_passport(folder))
    return rows


def write_notes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_passport_json(
    path: Path,
    *,
    instrument_name: str,
    reports_root: Path,
    box_all: list[dict[str, Any]],
    box_breath: list[dict[str, Any]],
    box_resonance: list[dict[str, Any]],
    box_relation: list[dict[str, Any]],
    note_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "instrument_name": instrument_name,
        "reports_root": str(reports_root),
        "version": "instrument_passport_v3_box_layers",
        "meaning": {
            "note": "Per-note acoustic behavior after root consensus, theory comparison, box-aware cleaning, and spiral projection.",
            "box_all": "Full repeated dense components across notes.",
            "box_breath": "Low-frequency breath/mechanical layer separated from resonant body.",
            "box_resonance": "Main resonant body layer used for instrument identity.",
            "box_harmonic_relation": "Relation of resonance components to harmonic nodes: HARMONIC, NEAR_HARMONIC, NON_HARMONIC.",
            "spiral12": "12-radix spiral coordinates of cleaned dense stream.",
        },
        "summary": {
            "total_notes": len(note_rows),
            "box_all_components": len(box_all),
            "box_breath_components": len(box_breath),
            "box_resonance_components": len(box_resonance),
            "box_relation_components": len(box_relation),
        },
        "box_all_top": box_all,
        "box_breath_top": box_breath,
        "box_resonance_top": box_resonance,
        "box_harmonic_relation_top": box_relation,
        "notes": note_rows,
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def md_box_table(lines: list[str], title: str, rows: list[dict[str, Any]], limit: int = 30) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| token | Hz | notes % | rel amp | mean amp | examples |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in rows[:limit]:
        lines.append(
            f"| {r.get('cluster_token','')} | {sf(r.get('cluster_hz')):.3f} | "
            f"{sf(r.get('percent_notes')):.2f} | {sf(r.get('relative_amp')):.6f} | "
            f"{sf(r.get('mean_amp')):.6f} | {r.get('examples','')} |"
        )
    lines.append("")


def md_relation_table(lines: list[str], rows: list[dict[str, Any]], limit: int = 40) -> None:
    lines.append("## Resonance ↔ harmonic relation")
    lines.append("")
    lines.append("| class | token | Hz | h | Δ cents | notes % | mean amp |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows[:limit]:
        lines.append(
            f"| {r.get('class','')} | {r.get('token','')} | {sf(r.get('cluster_hz')):.3f} | "
            f"{si(r.get('harmonic_index'))} | {sf(r.get('delta_cents')):.2f} | "
            f"{sf(r.get('percent_notes')):.2f} | {sf(r.get('mean_amp')):.6f} |"
        )
    lines.append("")


def write_passport_md(
    path: Path,
    *,
    instrument_name: str,
    box_all: list[dict[str, Any]],
    box_breath: list[dict[str, Any]],
    box_resonance: list[dict[str, Any]],
    box_relation: list[dict[str, Any]],
    note_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_notes = len(note_rows)
    avg_match = sum(sf(r.get("theory_match_ratio")) for r in note_rows) / total_notes if total_notes else 0.0
    avg_conf = sum(sf(r.get("tuner_confidence")) for r in note_rows) / total_notes if total_notes else 0.0

    relation_counts: dict[str, int] = {}
    for r in box_relation:
        cls = r.get("class", "UNKNOWN") or "UNKNOWN"
        relation_counts[cls] = relation_counts.get(cls, 0) + 1

    lines: list[str] = []
    lines.append(f"# Instrument Passport: {instrument_name}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total notes: {total_notes}")
    lines.append(f"- Average theory match ratio: {avg_match:.3f}")
    lines.append(f"- Average tuner confidence: {avg_conf:.3f}")
    lines.append(f"- Box all components: {len(box_all)}")
    lines.append(f"- Breath layer components: {len(box_breath)}")
    lines.append(f"- Resonance layer components: {len(box_resonance)}")
    lines.append(f"- Harmonic relation components: {len(box_relation)}")
    for k, v in sorted(relation_counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    md_box_table(lines, "Breath / mechanical layer", box_breath, limit=25)
    md_box_table(lines, "Resonance body layer", box_resonance, limit=30)
    md_relation_table(lines, box_relation, limit=40)

    lines.append("## Notes")
    lines.append("")
    lines.append("| note file | expected | detected | root Hz | Δ cents | theory match | clean rows | removed box | spiral points |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in note_rows:
        lines.append(
            f"| {r['folder_name']} | {r['expected_note']} | {r['root_consensus_token']} | "
            f"{sf(r['root_hz']):.3f} | {sf(r['root_delta_cents_vs_theory']):.3f} | "
            f"{sf(r['theory_match_ratio']):.3f} | {si(r['clean_rows'])} | "
            f"{si(r['removed_box_rows'])} | {si(r['spiral_points'])} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This passport separates five layers:")
    lines.append("")
    lines.append("1. The note core: consensus root and harmonic support.")
    lines.append("2. The theoretical match: observed dense-chain harmonics against the 12-radix reference table.")
    lines.append("3. The breath/mechanical layer: low-frequency repeated components.")
    lines.append("4. The resonant body layer: repeated resonant components used for instrument identity.")
    lines.append("5. The formation field: cleaned dense stream projected into 12-radix spiral coordinates.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build instrument passport from current pipeline outputs.")
    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--reports_root", required=True)

    ap.add_argument("--box_csv", required=True)
    ap.add_argument("--box_breath_csv", default="")
    ap.add_argument("--box_resonance_csv", default="")
    ap.add_argument("--box_relation_csv", default="")

    ap.add_argument("--out_notes_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--box_top_n", type=int, default=80)
    args = ap.parse_args()

    reports_root = Path(args.reports_root).resolve()

    box_all = load_box_profile(Path(args.box_csv).resolve(), top_n=args.box_top_n)
    box_breath = load_box_profile(Path(args.box_breath_csv).resolve(), top_n=args.box_top_n) if args.box_breath_csv else []
    box_resonance = load_box_profile(Path(args.box_resonance_csv).resolve(), top_n=args.box_top_n) if args.box_resonance_csv else box_all
    box_relation = load_box_relation(Path(args.box_relation_csv).resolve(), top_n=args.box_top_n) if args.box_relation_csv else []

    note_rows = collect_notes(reports_root)

    write_notes_csv(Path(args.out_notes_csv).resolve(), note_rows)
    write_passport_json(
        Path(args.out_json).resolve(),
        instrument_name=args.instrument_name,
        reports_root=reports_root,
        box_all=box_all,
        box_breath=box_breath,
        box_resonance=box_resonance,
        box_relation=box_relation,
        note_rows=note_rows,
    )
    write_passport_md(
        Path(args.out_md).resolve(),
        instrument_name=args.instrument_name,
        box_all=box_all,
        box_breath=box_breath,
        box_resonance=box_resonance,
        box_relation=box_relation,
        note_rows=note_rows,
    )

    print(json.dumps({
        "instrument_name": args.instrument_name,
        "notes": len(note_rows),
        "box_all_components": len(box_all),
        "box_breath_components": len(box_breath),
        "box_resonance_components": len(box_resonance),
        "box_relation_components": len(box_relation),
        "out_notes_csv": args.out_notes_csv,
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()