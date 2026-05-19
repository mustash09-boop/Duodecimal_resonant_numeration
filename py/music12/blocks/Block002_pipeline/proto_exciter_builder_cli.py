from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class ProtoExciter:
    proto_exciter_id: int
    coarse_note: str
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    peak_frame: int
    peak_time_sec: float
    peak_note_token: str
    peak_probe_index: int
    peak_frequency_hz: float
    peak_seed_score: float
    total_seed_score: float = 0.0
    seed_count: int = 0
    max_energy: float = 0.0
    rise_sum: float = 0.0
    continuation_sum: float = 0.0
    note_counts: Counter[str] = field(default_factory=Counter)
    probe_counts: Counter[int] = field(default_factory=Counter)

    def add(self, row: dict[str, Any]) -> None:
        frame_index = _safe_int(row.get("frame_index"), self.end_frame)
        time_sec = _safe_float(row.get("time_sec"), self.end_time_sec)
        seed_score = _safe_float(row.get("seed_score"), 0.0)
        energy = _safe_float(row.get("energy"), 0.0)
        rise = _safe_float(row.get("rise"), 0.0)
        continuation = _safe_float(row.get("continuation"), 0.0)
        note_token = str(row.get("note_token", "")).strip()
        probe_index = _safe_int(row.get("probe_index"), -1)
        frequency_hz = _safe_float(row.get("frequency_hz"), 0.0)

        self.end_frame = max(self.end_frame, frame_index)
        self.end_time_sec = max(self.end_time_sec, time_sec)
        self.seed_count += 1
        self.total_seed_score += seed_score
        self.max_energy = max(self.max_energy, energy)
        self.rise_sum += rise
        self.continuation_sum += continuation
        if note_token:
            self.note_counts[note_token] += 1
        if probe_index >= 0:
            self.probe_counts[probe_index] += 1

        if seed_score > self.peak_seed_score:
            self.peak_seed_score = seed_score
            self.peak_frame = frame_index
            self.peak_time_sec = time_sec
            self.peak_note_token = note_token
            self.peak_probe_index = probe_index
            self.peak_frequency_hz = frequency_hz

    def to_row(self) -> dict[str, Any]:
        duration_frames = self.end_frame - self.start_frame + 1
        mean_seed_score = self.total_seed_score / max(self.seed_count, 1)
        mean_rise = self.rise_sum / max(self.seed_count, 1)
        mean_continuation = self.continuation_sum / max(self.seed_count, 1)
        duration_bonus = max(0.0, 1.0 - abs(duration_frames - 3.0) / 4.0)
        rise_ratio = mean_rise / max(self.max_energy, 1e-9)
        peak_ratio = self.peak_seed_score / max(self.total_seed_score, 1e-9)
        exciter_confidence = max(
            0.0,
            min(
                1.0,
                0.40 * min(rise_ratio, 1.0)
                + 0.30 * min(mean_continuation, 1.0)
                + 0.20 * duration_bonus
                + 0.10 * min(peak_ratio * 2.0, 1.0),
            ),
        )
        dominant_note = self.note_counts.most_common(1)[0][0] if self.note_counts else ""
        return {
            "proto_exciter_id": self.proto_exciter_id,
            "coarse_note": self.coarse_note,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": duration_frames,
            "start_time_sec": f"{self.start_time_sec:.9f}",
            "end_time_sec": f"{self.end_time_sec:.9f}",
            "peak_frame": self.peak_frame,
            "peak_time_sec": f"{self.peak_time_sec:.9f}",
            "peak_note_token": self.peak_note_token,
            "peak_probe_index": self.peak_probe_index,
            "peak_frequency_hz": f"{self.peak_frequency_hz:.9f}",
            "peak_seed_score": f"{self.peak_seed_score:.9f}",
            "seed_count": self.seed_count,
            "total_seed_score": f"{self.total_seed_score:.9f}",
            "mean_seed_score": f"{mean_seed_score:.9f}",
            "max_energy": f"{self.max_energy:.9f}",
            "mean_rise": f"{mean_rise:.9f}",
            "mean_continuation": f"{mean_continuation:.9f}",
            "dominant_note_token": dominant_note,
            "member_note_tokens_json": json.dumps(dict(self.note_counts), ensure_ascii=False, sort_keys=True),
            "member_probe_indices_json": json.dumps(dict(self.probe_counts), ensure_ascii=False, sort_keys=True),
            "exciter_confidence": f"{exciter_confidence:.9f}",
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge short-lived excitation seeds into distributed proto-exciters."
    )
    ap.add_argument("--excitation-seeds-csv", required=True)
    ap.add_argument("--out-proto-exciters-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--max-gap-frames", type=int, default=2)
    args = ap.parse_args()

    seed_rows = _load_csv(Path(args.excitation_seeds_csv))
    seed_rows.sort(
        key=lambda r: (
            str(r.get("coarse_note", "")),
            _safe_int(r.get("frame_index"), 0),
            -_safe_float(r.get("seed_score"), 0.0),
        )
    )

    protos: list[ProtoExciter] = []
    active: dict[str, ProtoExciter] = {}
    next_id = 1

    for row in seed_rows:
        coarse = str(row.get("coarse_note", "")).strip()
        if not coarse:
            continue
        frame_index = _safe_int(row.get("frame_index"), 0)
        time_sec = _safe_float(row.get("time_sec"), 0.0)
        seed_score = _safe_float(row.get("seed_score"), 0.0)
        note_token = str(row.get("note_token", "")).strip()
        probe_index = _safe_int(row.get("probe_index"), -1)
        frequency_hz = _safe_float(row.get("frequency_hz"), 0.0)

        current = active.get(coarse)
        if current is None or frame_index - current.end_frame > int(args.max_gap_frames):
            current = ProtoExciter(
                proto_exciter_id=next_id,
                coarse_note=coarse,
                start_frame=frame_index,
                end_frame=frame_index,
                start_time_sec=time_sec,
                end_time_sec=time_sec,
                peak_frame=frame_index,
                peak_time_sec=time_sec,
                peak_note_token=note_token,
                peak_probe_index=probe_index,
                peak_frequency_hz=frequency_hz,
                peak_seed_score=seed_score,
            )
            protos.append(current)
            active[coarse] = current
            next_id += 1

        current.add(row)

    out_rows = [p.to_row() for p in protos]
    out_rows.sort(key=lambda r: float(r["start_time_sec"]))

    out_csv = Path(args.out_proto_exciters_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "proto_exciter_id",
        "coarse_note",
        "start_frame",
        "end_frame",
        "duration_frames",
        "start_time_sec",
        "end_time_sec",
        "peak_frame",
        "peak_time_sec",
        "peak_note_token",
        "peak_probe_index",
        "peak_frequency_hz",
        "peak_seed_score",
        "seed_count",
        "total_seed_score",
        "mean_seed_score",
        "max_energy",
        "mean_rise",
        "mean_continuation",
        "dominant_note_token",
        "member_note_tokens_json",
        "member_probe_indices_json",
        "exciter_confidence",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    confidence_values = [_safe_float(row.get("exciter_confidence"), 0.0) for row in out_rows]
    summary = {
        "stage": "proto_exciter_builder",
        "inputs": {"excitation_seeds_csv": args.excitation_seeds_csv},
        "parameters": {"max_gap_frames": int(args.max_gap_frames)},
        "result": {
            "proto_exciter_count": len(out_rows),
            "mean_confidence": sum(confidence_values) / max(len(confidence_values), 1),
        },
    }

    lines = [
        "PROTO EXCITER BUILD",
        "=" * 72,
        f"proto_exciter_count : {len(out_rows)}",
        f"mean_confidence     : {summary['result']['mean_confidence']:.6f}",
    ]

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
