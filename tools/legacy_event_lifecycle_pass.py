# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _phase_to_temporal_state(phase: str) -> str:
    p = str(phase or "").strip().upper()
    if p == "STABILIZATION":
        return "NEW_CAUSAL_EXCITATION"
    if p in {"PRIMARY_CHAIN", "CONTROLLED_SUSTAIN"}:
        return "ACTIVE_SUSTAIN"
    if p == "BOX_TRANSFER":
        return "BOX_OR_SHARED_RESONANCE"
    return ""


def _build_frame_notes_adapter(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        note = str(r.get("selected_note_token", "")).strip()
        if not note:
            continue
        out.append(
            {
                "frame_index": _safe_int(r.get("frame_index"), 0),
                "note_token": note,
                "score": f"{_safe_float(r.get('phase_score'), 0.0):.9f}",
                "temporal_state": _phase_to_temporal_state(r.get("phase")),
                "proto_exciter_id": str(r.get("proto_exciter_id", "")),
                "phase": str(r.get("phase", "")),
                "selection_reason": str(r.get("selection_reason", "")),
            }
        )
    out.sort(key=lambda x: (_safe_int(x["frame_index"]), -_safe_float(x["score"])))
    return out


def _run_py(module_path: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(module_path), *args]
    subprocess.run(cmd, check=True)


def _soft_group_count(rows: List[Dict[str, Any]], frame_key: str, window: int) -> int:
    frames = sorted(_safe_int(r.get(frame_key), 0) for r in rows)
    if not frames:
        return 0
    groups = 1
    anchor = frames[0]
    for frame in frames[1:]:
        if frame - anchor > window:
            groups += 1
            anchor = frame
    return groups


def _count_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, "")).strip() or "<EMPTY>"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="External legacy lifecycle pass over current Block002 outputs."
    )
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--midi-meta-json", required=True)
    ap.add_argument("--onset-window", type=int, default=3)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    report_dir = Path(args.report_dir)
    prefix = args.prefix

    legacy_dir = project_root / "py" / "music12" / "blocks" / "Block002_audio_recogn"
    tracker_py = legacy_dir / "resonance_event_lifecycle_tracker_v2_cli.py"
    merger_py = legacy_dir / "resonance_event_merger_cli.py"
    coherence_py = legacy_dir / "event_internal_coherence_refiner_cli.py"

    sustain_frames_csv = report_dir / f"{prefix}_controlled_sustain_frames_v1.csv"
    if not sustain_frames_csv.exists():
        raise FileNotFoundError(str(sustain_frames_csv))

    adapter_csv = report_dir / f"{prefix}_legacy_lifecycle_input_v1.csv"
    lifecycle_events_csv = report_dir / f"{prefix}_legacy_lifecycle_events_v1.csv"
    lifecycle_event_frames_csv = report_dir / f"{prefix}_legacy_lifecycle_event_frames_v1.csv"
    lifecycle_overlap_csv = report_dir / f"{prefix}_legacy_lifecycle_overlap_v1.csv"
    lifecycle_readable_csv = report_dir / f"{prefix}_legacy_lifecycle_readable_v1.csv"
    lifecycle_meta_json = report_dir / f"{prefix}_legacy_lifecycle_meta_v1.json"
    lifecycle_summary_txt = report_dir / f"{prefix}_legacy_lifecycle_summary_v1.txt"

    merged_events_csv = report_dir / f"{prefix}_legacy_lifecycle_merged_events_v1.csv"
    merged_mapping_csv = report_dir / f"{prefix}_legacy_lifecycle_merged_mapping_v1.csv"
    merged_meta_json = report_dir / f"{prefix}_legacy_lifecycle_merged_meta_v1.json"
    merged_summary_txt = report_dir / f"{prefix}_legacy_lifecycle_merged_summary_v1.txt"

    coherent_events_csv = report_dir / f"{prefix}_legacy_lifecycle_coherent_events_v1.csv"
    coherent_readable_csv = report_dir / f"{prefix}_legacy_lifecycle_coherent_readable_v1.csv"
    coherent_meta_json = report_dir / f"{prefix}_legacy_lifecycle_coherent_meta_v1.json"
    coherent_summary_txt = report_dir / f"{prefix}_legacy_lifecycle_coherent_summary_v1.txt"

    summary_txt = report_dir / f"{prefix}_legacy_lifecycle_pass_summary_v1.txt"
    summary_json = report_dir / f"{prefix}_legacy_lifecycle_pass_meta_v1.json"

    sustain_rows = _load_csv(sustain_frames_csv)
    adapter_rows = _build_frame_notes_adapter(sustain_rows)
    _write_csv(
        adapter_csv,
        adapter_rows,
        ["frame_index", "note_token", "score", "temporal_state", "proto_exciter_id", "phase", "selection_reason"],
    )

    _run_py(
        tracker_py,
        [
            "--frame_notes_csv",
            str(adapter_csv),
            "--out_events_csv",
            str(lifecycle_events_csv),
            "--out_event_frames_csv",
            str(lifecycle_event_frames_csv),
            "--out_overlap_csv",
            str(lifecycle_overlap_csv),
            "--out_readable_csv",
            str(lifecycle_readable_csv),
            "--out_meta_json",
            str(lifecycle_meta_json),
            "--out_summary_txt",
            str(lifecycle_summary_txt),
            "--max_gap_frames",
            "4",
            "--max_same_event_pitch_drift",
            "0",
            "--min_event_frames",
            "3",
            "--min_birth_score",
            "0.95",
            "--attack_delta",
            "0.18",
            "--reexcite_drop",
            "0.35",
        ],
    )

    _run_py(
        merger_py,
        [
            "--events_csv",
            str(lifecycle_events_csv),
            "--out_merged_events_csv",
            str(merged_events_csv),
            "--out_mapping_csv",
            str(merged_mapping_csv),
            "--out_meta_json",
            str(merged_meta_json),
            "--out_summary_txt",
            str(merged_summary_txt),
            "--max_merge_gap_frames",
            "10",
            "--max_birth_jump_ratio",
            "0.28",
            "--min_merged_frames",
            "3",
        ],
    )

    _run_py(
        coherence_py,
        [
            "--event_matches_csv",
            str(merged_events_csv),
            "--out_refined_events_csv",
            str(coherent_events_csv),
            "--out_readable_csv",
            str(coherent_readable_csv),
            "--out_meta_json",
            str(coherent_meta_json),
            "--out_summary_txt",
            str(coherent_summary_txt),
            "--min_coherent_score",
            "0.72",
        ],
    )

    lifecycle_rows = _load_csv(lifecycle_events_csv)
    merged_rows = _load_csv(merged_events_csv)
    coherent_rows = _load_csv(coherent_events_csv)
    midi_meta = json.loads(Path(args.midi_meta_json).read_text(encoding="utf-8"))

    coherent_high = [
        r for r in coherent_rows if _safe_float(r.get("internal_coherence_score"), 0.0) >= 0.72
    ]

    lifecycle_soft = _soft_group_count(lifecycle_rows, "birth_frame", args.onset_window)
    merged_soft = _soft_group_count(merged_rows, "birth_frame", args.onset_window)
    coherent_soft = _soft_group_count(coherent_high, "birth_frame", args.onset_window)

    refined_counts = _count_by_key(coherent_rows, "refined_lifecycle_kind")
    merged_lifecycle_counts = _count_by_key(merged_rows, "lifecycle_kind")

    summary_lines = [
        "LEGACY EVENT LIFECYCLE PASS",
        "=" * 72,
        f"source_frames_csv            : {sustain_frames_csv}",
        f"adapter_rows                 : {len(adapter_rows)}",
        "",
        f"target_event_count           : {midi_meta.get('event_count', 0)}",
        f"target_onset_groups          : {midi_meta.get('onset_group_count', 0)}",
        "",
        f"lifecycle_events             : {len(lifecycle_rows)}",
        f"lifecycle_soft_onset_groups  : {lifecycle_soft}",
        "",
        f"merged_events                : {len(merged_rows)}",
        f"merged_soft_onset_groups     : {merged_soft}",
        "",
        f"coherent_events_total        : {len(coherent_rows)}",
        f"coherent_events_ge_0_72      : {len(coherent_high)}",
        f"coherent_soft_onset_groups   : {coherent_soft}",
        "",
        "merged_lifecycle_counts:",
    ]
    for k, v in merged_lifecycle_counts.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("refined_lifecycle_counts:")
    for k, v in refined_counts.items():
        summary_lines.append(f"  {k}: {v}")

    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "inputs": {
                    "controlled_sustain_frames_csv": str(sustain_frames_csv),
                    "midi_meta_json": args.midi_meta_json,
                },
                "outputs": {
                    "adapter_csv": str(adapter_csv),
                    "lifecycle_events_csv": str(lifecycle_events_csv),
                    "merged_events_csv": str(merged_events_csv),
                    "coherent_events_csv": str(coherent_events_csv),
                    "summary_txt": str(summary_txt),
                },
                "result": {
                    "target_event_count": midi_meta.get("event_count", 0),
                    "target_onset_groups": midi_meta.get("onset_group_count", 0),
                    "lifecycle_events": len(lifecycle_rows),
                    "lifecycle_soft_onset_groups": lifecycle_soft,
                    "merged_events": len(merged_rows),
                    "merged_soft_onset_groups": merged_soft,
                    "coherent_events_ge_0_72": len(coherent_high),
                    "coherent_soft_onset_groups": coherent_soft,
                    "merged_lifecycle_counts": merged_lifecycle_counts,
                    "refined_lifecycle_counts": refined_counts,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
