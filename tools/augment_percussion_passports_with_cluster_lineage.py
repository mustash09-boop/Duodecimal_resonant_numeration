from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PERC_ROOT = PROJECT_ROOT / "Block004_data" / "percussion"
PASSPORT_DIR = PERC_ROOT / "40_passports"
LINEAGE_DIR = PERC_ROOT / "55_cluster_lineage_spiral3d"
SUMMARY_CSV = LINEAGE_DIR / "percussion__cluster_lineage_summary.csv"


def update_instrument_passport(passport_path: Path, lineage_df: pd.DataFrame) -> None:
    data = json.loads(passport_path.read_text(encoding="utf-8"))
    instrument_name = data.get("instrument_name", "")
    events = lineage_df[lineage_df["instrument_name"] == instrument_name].copy()

    if len(events) == 0:
        block = {
            "event_count": 0,
            "cluster_core_points": 0,
            "spawned_dense_points": 0,
            "unassigned_dense_points": 0,
            "spawned_ratio_mean": 0.0,
            "unassigned_ratio_mean": 0.0,
            "top_events_by_unassigned_ratio": [],
        }
    else:
        top_events = (
            events.sort_values(["unassigned_ratio", "points"], ascending=[False, False])
            .head(12)
        )
        block = {
            "event_count": int(len(events)),
            "cluster_core_points": int(events["cluster_core_points"].fillna(0).sum()),
            "spawned_dense_points": int(events["spawned_dense_points"].fillna(0).sum()),
            "unassigned_dense_points": int(events["unassigned_dense_points"].fillna(0).sum()),
            "spawned_ratio_mean": float(events["spawned_ratio"].fillna(0.0).mean()),
            "unassigned_ratio_mean": float(events["unassigned_ratio"].fillna(0.0).mean()),
            "top_events_by_unassigned_ratio": [
                {
                    "event": str(r["event"]),
                    "points": int(r["points"]),
                    "spawned_ratio": float(r["spawned_ratio"]),
                    "unassigned_ratio": float(r["unassigned_ratio"]),
                }
                for _, r in top_events.iterrows()
            ],
        }

    summary = dict(data.get("summary", {}))
    summary["cluster_lineage_event_count"] = block["event_count"]
    summary["cluster_lineage_cluster_core_points"] = block["cluster_core_points"]
    summary["cluster_lineage_spawned_dense_points"] = block["spawned_dense_points"]
    summary["cluster_lineage_unassigned_dense_points"] = block["unassigned_dense_points"]
    summary["cluster_lineage_spawned_ratio_mean"] = block["spawned_ratio_mean"]
    summary["cluster_lineage_unassigned_ratio_mean"] = block["unassigned_ratio_mean"]
    data["summary"] = summary

    data["cluster_lineage_spiral3d"] = block
    meanings = dict(data.get("meaning", {}))
    meanings["cluster_lineage_spiral3d"] = (
        "Event-level lineage of percussion resonance clusters and attached dense peaks."
    )
    data["meaning"] = meanings
    passport_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = passport_path.with_suffix(".md")
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8", errors="replace")
        marker = "\n## Cluster lineage spiral 3D\n"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip() + "\n"
        lines = []
        lines.append("## Cluster lineage spiral 3D")
        lines.append("")
        lines.append(f"- Event count: {block['event_count']}")
        lines.append(f"- Cluster core points: {block['cluster_core_points']}")
        lines.append(f"- Spawned dense points: {block['spawned_dense_points']}")
        lines.append(f"- Unassigned dense points: {block['unassigned_dense_points']}")
        lines.append(f"- Mean spawned ratio: {block['spawned_ratio_mean']:.4f}")
        lines.append(f"- Mean unassigned ratio: {block['unassigned_ratio_mean']:.4f}")
        lines.append("")
        lines.append("| event | points | spawned ratio | unassigned ratio |")
        lines.append("|---|---:|---:|---:|")
        for row in block["top_events_by_unassigned_ratio"]:
            lines.append(
                f"| {row['event']} | {row['points']} | {row['spawned_ratio']:.4f} | {row['unassigned_ratio']:.4f} |"
            )
        md_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    lineage_df = pd.read_csv(SUMMARY_CSV)
    lineage_df["instrument_name"] = (
        lineage_df["event"].astype(str).str.split("__", n=1).str[0]
    )
    for passport_path in sorted(PASSPORT_DIR.glob("*__percussion_passport.json")):
        update_instrument_passport(passport_path, lineage_df)
        print(f"UPDATED {passport_path.name}")


if __name__ == "__main__":
    main()
