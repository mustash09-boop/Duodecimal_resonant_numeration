from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_dense_chain_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_voice_event_rows(
    summary: dict[str, Any],
    *,
    duration_sec: float,
    voice_id: int = 0,
) -> list[dict[str, Any]]:
    best = summary.get("best_chain")
    if not best:
        return []

    time_start = _safe_float(best.get("time_sec", 0.0), 0.0)
    time_end = time_start + max(0.001, float(duration_sec))

    root_note_token = _safe_str(best.get("root_note_token", ""))
    root_hz = _safe_float(best.get("root_hz", 0.0), 0.0)
    chain_score = _safe_float(best.get("chain_score", 0.0), 0.0)
    weighted_support_score = _safe_float(best.get("weighted_support_score", 0.0), 0.0)

    rows: list[dict[str, Any]] = []

    hits = best.get("hits", [])
    for i, hit in enumerate(hits):
        matched_token = _safe_str(hit.get("matched_token", "")) or root_note_token
        matched_hz = _safe_float(hit.get("matched_hz", 0.0), 0.0)
        matched_amp = _safe_float(hit.get("matched_amplitude", 0.0), 0.0)
        harmonic_index = int(hit.get("harmonic_index", i + 1) or (i + 1))

        row = {
            "voice_id": int(voice_id),
            "note_index": int(i),
            "note_token": matched_token,
            "time_start": float(time_start),
            "time_end": float(time_end),
            "duration": float(time_end - time_start),
            "segment_start": 0,
            "segment_end": 0,
            "event_count": 1,
            "representative_rc_hz_mean": float(matched_hz),
            "representative_rc_energy_mean": float(matched_amp),
            "best_theoretical_root_score_mean": float(chain_score),
            "support_hits_mean": 1.0,
            "spiral_match_count_mean": 1.0,
            "spiral_consistency_score_mean": 1.0,
            "window_chain_match_score_mean": float(weighted_support_score),
            "stabilization_score_mean": 1.0,
            "theoretical_chain_verdict_mode": f"dense_h{harmonic_index}",
            "stabilization_role_mode": "dense_harmonic",
            "best_theoretical_chain_string_mode": f"root={root_note_token};h={harmonic_index}",
        }
        rows.append(row)

    return rows


def write_voice_events_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "voice_id",
        "note_index",
        "note_token",
        "time_start",
        "time_end",
        "duration",
        "segment_start",
        "segment_end",
        "event_count",
        "representative_rc_hz_mean",
        "representative_rc_energy_mean",
        "best_theoretical_root_score_mean",
        "support_hits_mean",
        "spiral_match_count_mean",
        "spiral_consistency_score_mean",
        "window_chain_match_score_mean",
        "stabilization_score_mean",
        "theoretical_chain_verdict_mode",
        "stabilization_role_mode",
        "best_theoretical_chain_string_mode",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_meta_json(path: Path, *, input_json: Path, output_csv: Path, row_count: int, duration_sec: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "dense_chain_summary_json": str(input_json),
        },
        "outputs": {
            "voice_events_csv": str(output_csv),
            "meta_json": str(path),
        },
        "row_count": int(row_count),
        "duration_sec": float(duration_sec),
        "semantic_note": (
            "Adapter from dense harmonic chain summary to harmonic-level voice_events CSV "
            "for spiral12_voice_plot_cli compatibility."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert dense_chain_summary.json into harmonic-level voice_events.csv for spiral plotting."
    )
    ap.add_argument("--dense_chain_summary_json", required=True)
    ap.add_argument("--out_voice_events_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--duration_sec", type=float, default=0.08)
    args = ap.parse_args()

    dense_chain_summary_json = Path(args.dense_chain_summary_json).resolve()
    out_voice_events_csv = Path(args.out_voice_events_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    summary = load_dense_chain_summary(dense_chain_summary_json)
    rows = build_voice_event_rows(
        summary,
        duration_sec=float(args.duration_sec),
        voice_id=0,
    )

    write_voice_events_csv(out_voice_events_csv, rows)
    write_meta_json(
        out_meta_json,
        input_json=dense_chain_summary_json,
        output_csv=out_voice_events_csv,
        row_count=len(rows),
        duration_sec=float(args.duration_sec),
    )

    print("dense chain -> harmonic voice events complete")
    print(json.dumps(
        {
            "row_count": len(rows),
            "out_voice_events_csv": str(out_voice_events_csv),
            "out_meta_json": str(out_meta_json),
            "duration_sec": float(args.duration_sec),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()