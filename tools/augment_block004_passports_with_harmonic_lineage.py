from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
BLOCK004_ROOT = PROJECT_ROOT / "Block004_data"


def tonal_dataset_dirs() -> list[Path]:
    out: list[Path] = []
    for d in sorted(BLOCK004_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_") or d.name == "percussion":
            continue
        if (d / "20_range_research").exists():
            out.append(d)
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_lineage_block(summary_df: pd.DataFrame) -> dict:
    def col_sum(name: str) -> int:
        if name not in summary_df.columns:
            return 0
        return int(summary_df[name].fillna(0).sum())

    if len(summary_df) == 0:
        return {
            "notes_built": 0,
            "total_points": 0,
            "total_visual_points": 0,
            "total_core_points": 0,
            "total_note_box_points": 0,
            "total_residual_points": 0,
            "total_unassigned_points": 0,
            "unassigned_ratio": 0.0,
            "top_unassigned_notes": [],
        }

    total_points = col_sum("points")
    total_visual_points = col_sum("visual_points")
    total_core = col_sum("harmonic_core_points")
    total_box = col_sum("spawned_note_box_points")
    total_residual = col_sum("spawned_residual_points")
    total_unassigned = col_sum("unassigned_points")
    unassigned_ratio = (float(total_unassigned) / float(total_points)) if total_points else 0.0

    top_df = summary_df.copy()
    top_df["unassigned_ratio_note"] = top_df["unassigned_points"].fillna(0) / top_df["points"].replace(0, 1)
    top_df = top_df.sort_values(["unassigned_ratio_note", "points"], ascending=[False, False]).head(12)

    return {
        "notes_built": int(len(summary_df)),
        "total_points": total_points,
        "total_visual_points": total_visual_points,
        "total_core_points": total_core,
        "total_note_box_points": total_box,
        "total_residual_points": total_residual,
        "total_unassigned_points": total_unassigned,
        "unassigned_ratio": unassigned_ratio,
        "top_unassigned_notes": [
            {
                "note": str(r["note"]),
                "points": int(r["points"]),
                "unassigned_points": int(r["unassigned_points"]),
                "unassigned_ratio": float(r["unassigned_ratio_note"]),
                "assigned_harmonics": str(r.get("assigned_harmonics", "")),
            }
            for _, r in top_df.iterrows()
        ],
    }


def update_markdown(md_path: Path, lineage: dict) -> None:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    marker_start = "\n## Harmonic lineage spiral 3D\n"
    if marker_start in text:
        text = text.split(marker_start, 1)[0].rstrip() + "\n"

    lines = []
    lines.append("## Harmonic lineage spiral 3D")
    lines.append("")
    lines.append("This section summarizes the additional Block004 layer that traces which harmonic core")
    lines.append("appears to generate which resonance chain and secondary response around each isolated note.")
    lines.append("")
    lines.append(f"- Notes built: {lineage['notes_built']}")
    lines.append(f"- Total lineage points: {lineage['total_points']}")
    lines.append(f"- Visual points kept for PNG/HTML: {lineage['total_visual_points']}")
    lines.append(f"- Harmonic core points: {lineage['total_core_points']}")
    lines.append(f"- Spawned note-box points: {lineage['total_note_box_points']}")
    lines.append(f"- Spawned residual points: {lineage['total_residual_points']}")
    lines.append(f"- Unassigned points: {lineage['total_unassigned_points']}")
    lines.append(f"- Unassigned ratio: {lineage['unassigned_ratio']:.3f}")
    lines.append("")
    lines.append("### Notes with the highest unassigned share")
    lines.append("")
    lines.append("| note | points | unassigned | ratio | assigned harmonics |")
    lines.append("|---|---:|---:|---:|---|")
    for row in lineage["top_unassigned_notes"]:
        lines.append(
            f"| {row['note']} | {row['points']} | {row['unassigned_points']} | "
            f"{row['unassigned_ratio']:.3f} | {row['assigned_harmonics']} |"
        )
    lines.append("")

    md_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    for dataset_dir in tonal_dataset_dirs():
        range_dir = dataset_dir / "20_range_research"
        json_candidates = sorted(range_dir.glob("*__instrument_passport.json"))
        md_candidates = sorted(range_dir.glob("*__instrument_passport.md"))
        summary_candidates = sorted((dataset_dir / "55_harmonic_chain_spiral3d").glob("*__harmonic_chain_spiral3d_summary.csv"))

        if not json_candidates or not md_candidates or not summary_candidates:
            continue

        passport_json = json_candidates[0]
        passport_md = md_candidates[0]
        summary_csv = summary_candidates[0]

        summary_df = pd.read_csv(summary_csv)
        lineage = build_lineage_block(summary_df)

        data = load_json(passport_json)
        meaning = dict(data.get("meaning", {}))
        meaning["harmonic_chain_spiral3d"] = (
            "Per-note lineage layer tracing which harmonic core attracts note-box and residual resonance points."
        )
        data["meaning"] = meaning

        summary = dict(data.get("summary", {}))
        summary["harmonic_chain_notes_built"] = lineage["notes_built"]
        summary["harmonic_chain_total_points"] = lineage["total_points"]
        summary["harmonic_chain_total_visual_points"] = lineage["total_visual_points"]
        summary["harmonic_chain_total_core_points"] = lineage["total_core_points"]
        summary["harmonic_chain_total_note_box_points"] = lineage["total_note_box_points"]
        summary["harmonic_chain_total_residual_points"] = lineage["total_residual_points"]
        summary["harmonic_chain_total_unassigned_points"] = lineage["total_unassigned_points"]
        summary["harmonic_chain_unassigned_ratio"] = lineage["unassigned_ratio"]
        data["summary"] = summary
        data["harmonic_chain_spiral3d"] = lineage

        save_json(passport_json, data)
        update_markdown(passport_md, lineage)
        print(f"UPDATED {dataset_dir.name}")


if __name__ == "__main__":
    main()
