# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _support_class(source_owner: str, ratio_error: float, frame_offset: int) -> str:
    if source_owner == "BODY_CONTINUATION_OWNER":
        return "BODY_COUPLED_SUPPORT_CANDIDATE"
    if ratio_error <= 0.025 and frame_offset == 0:
        return "HIDDEN_ROOT_SUPPORT_CANDIDATE"
    return "MASKED_SHARED_SUPPORT_CANDIDATE"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Use the upper second-sustain chain as evidence of a bowed layer and search for "
            "linked hidden root/support observations below it from the same window data."
        )
    )
    ap.add_argument("--ownership-observations-csv", required=True)
    ap.add_argument("--window-start-sec", type=float, required=True)
    ap.add_argument("--window-end-sec", type=float, required=True)
    ap.add_argument("--out-links-csv", required=True)
    ap.add_argument("--out-support-groups-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "loading_inputs",
            "window_start_sec": args.window_start_sec,
            "window_end_sec": args.window_end_sec,
        },
    )

    rows = _load_csv(Path(args.ownership_observations_csv))
    selected = [
        row for row in rows
        if args.window_start_sec <= _safe_float(row.get("time_sec"), -1.0) <= args.window_end_sec
    ]

    upper_rows = [
        row for row in selected
        if str(row.get("owner_label", "")).strip() == "SECOND_SUSTAIN_OWNER"
    ]
    frame_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        frame_map[_safe_int(row.get("frame_index"), 0)].append(row)

    _write_progress(
        args.progress_json,
        {
            "status": "running",
            "phase": "link_search",
            "upper_rows": len(upper_rows),
            "window_rows": len(selected),
        },
    )

    allowed_support_owners = {
        "LOCAL_TRANSIENT_OWNER",
        "LOCAL_EVENT_OWNER",
        "BODY_CONTINUATION_OWNER",
        "UNRESOLVED_BACKBONE_OWNER",
    }

    link_rows: list[dict[str, Any]] = []
    group_acc: dict[tuple[str, int], dict[str, Any]] = {}

    for upper in upper_rows:
        upper_frame = _safe_int(upper.get("frame_index"), 0)
        upper_freq = _safe_float(upper.get("frequency_hz"), 0.0)
        upper_energy = _safe_float(upper.get("energy"), 0.0)
        if upper_freq <= 0.0:
            continue
        for frame_offset in (-1, 0, 1):
            frame_index = upper_frame + frame_offset
            for cand in frame_map.get(frame_index, []):
                source_owner = str(cand.get("owner_label", "")).strip()
                if source_owner not in allowed_support_owners:
                    continue
                cand_freq = _safe_float(cand.get("frequency_hz"), 0.0)
                cand_energy = _safe_float(cand.get("energy"), 0.0)
                if cand_freq <= 0.0:
                    continue
                ratio = upper_freq / cand_freq
                if not (1.88 <= ratio <= 2.12):
                    continue
                ratio_error = abs(ratio - 2.0)
                energy_ratio = cand_energy / max(upper_energy, 1e-9)
                link_score = (
                    max(0.0, 1.0 - (ratio_error / 0.12)) * 0.50
                    + max(0.0, 1.0 - (abs(frame_offset) / 2.0)) * 0.20
                    + min(1.0, cand_energy / 0.60) * 0.30
                )
                support_class = _support_class(source_owner, ratio_error, abs(frame_offset))
                row = {
                    "upper_frame_index": upper_frame,
                    "upper_time_sec": upper.get("time_sec", ""),
                    "upper_probe_index": upper.get("probe_index", ""),
                    "upper_micro_symbol": upper.get("observed_micro_symbol", ""),
                    "upper_coarse_symbol": upper.get("observed_coarse_symbol", ""),
                    "upper_frequency_hz": upper.get("frequency_hz", ""),
                    "upper_energy": upper.get("energy", ""),
                    "support_frame_index": frame_index,
                    "support_time_sec": cand.get("time_sec", ""),
                    "support_probe_index": cand.get("probe_index", ""),
                    "support_micro_symbol": cand.get("observed_micro_symbol", ""),
                    "support_coarse_symbol": cand.get("observed_coarse_symbol", ""),
                    "support_frequency_hz": cand.get("frequency_hz", ""),
                    "support_energy": cand.get("energy", ""),
                    "support_source_owner": source_owner,
                    "support_resolved_role": cand.get("resolved_role", ""),
                    "support_backbone_lineage_class": cand.get("backbone_lineage_class", ""),
                    "frame_offset": frame_offset,
                    "octave_ratio": f"{ratio:.9f}",
                    "ratio_error": f"{ratio_error:.9f}",
                    "energy_ratio": f"{energy_ratio:.9f}",
                    "link_score": f"{link_score:.9f}",
                    "support_class": support_class,
                }
                link_rows.append(row)

                key = (
                    str(cand.get("observed_coarse_symbol", "")).strip(),
                    _safe_int(cand.get("probe_index"), 0),
                )
                group = group_acc.setdefault(
                    key,
                    {
                        "support_coarse_symbol": str(cand.get("observed_coarse_symbol", "")).strip(),
                        "support_probe_index": _safe_int(cand.get("probe_index"), 0),
                        "support_source_owner_counts": Counter(),
                        "support_class_counts": Counter(),
                        "frame_hits": set(),
                        "upper_frame_hits": set(),
                        "link_scores": [],
                        "support_freqs": [],
                        "support_energies": [],
                        "upper_coarse_counts": Counter(),
                    },
                )
                group["support_source_owner_counts"][source_owner] += 1
                group["support_class_counts"][support_class] += 1
                group["frame_hits"].add(frame_index)
                group["upper_frame_hits"].add(upper_frame)
                group["link_scores"].append(link_score)
                group["support_freqs"].append(cand_freq)
                group["support_energies"].append(cand_energy)
                group["upper_coarse_counts"][str(upper.get("observed_coarse_symbol", "")).strip()] += 1

    out_links = Path(args.out_links_csv)
    out_links.parent.mkdir(parents=True, exist_ok=True)
    link_fields = list(link_rows[0].keys()) if link_rows else [
        "upper_frame_index",
        "support_frame_index",
        "support_coarse_symbol",
        "link_score",
    ]
    with out_links.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=link_fields)
        writer.writeheader()
        for row in link_rows:
            writer.writerow(row)

    group_rows: list[dict[str, Any]] = []
    for _key, group in sorted(group_acc.items(), key=lambda item: (-len(item[1]["upper_frame_hits"]), -_mean(item[1]["link_scores"]))):
        group_rows.append(
            {
                "support_coarse_symbol": group["support_coarse_symbol"],
                "support_probe_index": group["support_probe_index"],
                "support_frame_hit_count": len(group["frame_hits"]),
                "upper_frame_hit_count": len(group["upper_frame_hits"]),
                "mean_link_score": f"{_mean(group['link_scores']):.9f}",
                "mean_support_frequency_hz": f"{_mean(group['support_freqs']):.9f}",
                "mean_support_energy": f"{_mean(group['support_energies']):.9f}",
                "support_source_owner_counts_json": json.dumps(dict(group["support_source_owner_counts"]), ensure_ascii=False),
                "support_class_counts_json": json.dumps(dict(group["support_class_counts"]), ensure_ascii=False),
                "upper_coarse_counts_json": json.dumps(dict(group["upper_coarse_counts"]), ensure_ascii=False),
            }
        )

    out_groups = Path(args.out_support_groups_csv)
    group_fields = list(group_rows[0].keys()) if group_rows else [
        "support_coarse_symbol",
        "support_probe_index",
        "mean_link_score",
    ]
    with out_groups.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=group_fields)
        writer.writeheader()
        for row in group_rows:
            writer.writerow(row)

    support_class_counter = Counter(str(row["support_class"]) for row in link_rows)
    support_owner_counter = Counter(str(row["support_source_owner"]) for row in link_rows)
    summary_lines = [
        "WINDOW HIDDEN ROOT SUPPORT FINDER",
        "=" * 72,
        f"window_start_sec              : {args.window_start_sec:.6f}",
        f"window_end_sec                : {args.window_end_sec:.6f}",
        f"upper_second_rows             : {len(upper_rows)}",
        f"linked_support_links          : {len(link_rows)}",
        f"support_groups                : {len(group_rows)}",
        "",
        "support_class_counts:",
    ]
    for key, value in support_class_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "support_source_owner_counts:"])
    for key, value in support_owner_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "top_support_groups:"])
    for row in group_rows[:12]:
        summary_lines.append(
            "  "
            f"{row['support_coarse_symbol']} probe={row['support_probe_index']} "
            f"frames={row['upper_frame_hit_count']} mean_score={row['mean_link_score']}"
        )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_hidden_root_support_finder",
                "inputs": {
                    "ownership_observations_csv": args.ownership_observations_csv,
                },
                "window": {
                    "start_sec": args.window_start_sec,
                    "end_sec": args.window_end_sec,
                },
                "result": {
                    "upper_second_rows": len(upper_rows),
                    "linked_support_links": len(link_rows),
                    "support_groups": len(group_rows),
                    "support_class_counts": dict(support_class_counter),
                    "support_source_owner_counts": dict(support_owner_counter),
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    _write_progress(
        args.progress_json,
        {
            "status": "done",
            "phase": "complete",
            "upper_second_rows": len(upper_rows),
            "linked_support_links": len(link_rows),
            "support_groups": len(group_rows),
        },
    )


if __name__ == "__main__":
    main()
