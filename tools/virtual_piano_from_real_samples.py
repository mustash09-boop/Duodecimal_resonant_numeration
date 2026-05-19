from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


NOTE_RE = re.compile(r"piano_real_([0-9A-Z]+\.[0-9A-Z]+-)\.wav$", re.IGNORECASE)


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _normalize_note_token(token: str) -> str:
    return token.replace("'", "").replace('"', "").strip()


def _build_sample_index(sample_dir: Path, report_dir: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for wav_path in sorted(sample_dir.glob("*.wav")):
        m = NOTE_RE.search(wav_path.name)
        if not m:
            continue
        note = _normalize_note_token(m.group(1))
        prefix = wav_path.stem.split("_piano_real_", 1)[0]
        report_match = sorted(report_dir.glob(f"{prefix}_piano_real_{note.rstrip('-')}*"))
        report_path = report_match[0] if report_match else None
        info = sf.info(str(wav_path))
        index[note] = {
            "note12": note,
            "wav_path": str(wav_path),
            "report_dir": str(report_path) if report_path else "",
            "duration_sec": f"{info.duration:.9f}",
            "sample_rate": str(info.samplerate),
            "channels": str(info.channels),
        }
    return index


def _build_activity_masks(
    midi_rows: list[dict[str, str]],
    total_samples: int,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    piano_mask = np.zeros(total_samples, dtype=np.bool_)
    other_mask = np.zeros(total_samples, dtype=np.bool_)
    for row in midi_rows:
        track = str(row.get("track_name", "")).strip()
        start = max(0, int(round(_safe_float(row.get("start_sec")) * sr)))
        end = min(total_samples, int(round(_safe_float(row.get("end_sec")) * sr)))
        if end <= start:
            continue
        if track.startswith("Piano"):
            piano_mask[start:end] = True
        else:
            other_mask[start:end] = True
    return piano_mask, other_mask


def _velocity_gain(velocity: int, curve: str) -> float:
    norm = max(0.0, min(1.0, velocity / 127.0))
    if curve == "linear":
        return norm
    if curve == "sqrt":
        return math.sqrt(norm)
    if curve == "pow_0_4":
        return norm ** 0.4
    raise SystemExit(f"Unknown velocity curve: {curve}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a virtual piano track from RealPiano single-note WAVs using MIDI piano part timing.")
    ap.add_argument("--midi_parts_csv", required=True)
    ap.add_argument("--sample_dir", required=True)
    ap.add_argument("--report_dir", required=True)
    ap.add_argument("--source_audio_wav", required=True)
    ap.add_argument("--out_plan_csv", required=True)
    ap.add_argument("--out_virtual_piano_wav", required=True)
    ap.add_argument("--out_residual_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_fitted_virtual_piano_wav")
    ap.add_argument("--out_fitted_preview_wav")
    ap.add_argument("--release_pad_sec", type=float, default=0.180)
    ap.add_argument("--velocity_curve", choices=["linear", "sqrt", "pow_0_4"], default="sqrt")
    args = ap.parse_args()

    midi_rows = _read_csv(Path(args.midi_parts_csv))
    piano_rows = [r for r in midi_rows if str(r.get("track_name", "")).startswith("Piano")]
    sample_index = _build_sample_index(Path(args.sample_dir), Path(args.report_dir))

    source_audio, sr = sf.read(str(Path(args.source_audio_wav)))
    if source_audio.ndim > 1:
        source_audio = np.mean(source_audio, axis=1)
    source_audio = np.asarray(source_audio, dtype=np.float32)
    total_samples = len(source_audio)

    virtual = np.zeros(total_samples, dtype=np.float32)
    plan_rows: list[dict[str, str]] = []
    missing_notes: Counter[str] = Counter()
    used_notes: Counter[str] = Counter()
    used_event_count = 0

    for idx, row in enumerate(piano_rows, start=1):
        note = str(row.get("note12", "")).strip()
        normalized_note = _normalize_note_token(note)
        sample_meta = sample_index.get(normalized_note)
        if not sample_meta:
            missing_notes[normalized_note or note] += 1
            plan_rows.append(
                {
                    "row_id": str(idx),
                    "track_name": str(row.get("track_name", "")).strip(),
                    "note12": note,
                    "normalized_note12": normalized_note,
                    "start_sec": str(row.get("start_sec", "")),
                    "end_sec": str(row.get("end_sec", "")),
                    "velocity": str(row.get("velocity", "")),
                    "sample_found": "NO",
                    "sample_wav_path": "",
                    "sample_report_dir": "",
                    "sample_duration_sec": "",
                    "overlay_duration_sec": "",
                    "velocity_gain": "",
                }
            )
            continue

        sample_audio, sample_sr = sf.read(sample_meta["wav_path"])
        if sample_audio.ndim > 1:
            sample_audio = np.mean(sample_audio, axis=1)
        sample_audio = np.asarray(sample_audio, dtype=np.float32)
        if int(sample_sr) != int(sr):
            raise SystemExit(f"Sample SR mismatch for {sample_meta['wav_path']}: {sample_sr} != {sr}")

        start_sec = _safe_float(row.get("start_sec"))
        end_sec = _safe_float(row.get("end_sec"))
        midi_duration_sec = max(0.0, end_sec - start_sec)
        overlay_duration_sec = min(
            len(sample_audio) / sr,
            midi_duration_sec + args.release_pad_sec,
        )
        take_samples = max(1, min(len(sample_audio), int(round(overlay_duration_sec * sr))))
        start_sample = max(0, int(round(start_sec * sr)))
        end_sample = min(total_samples, start_sample + take_samples)
        if end_sample <= start_sample:
            continue

        vel = _safe_int(row.get("velocity"), 96)
        gain = _velocity_gain(vel, args.velocity_curve)
        chunk = sample_audio[: end_sample - start_sample] * np.float32(gain)
        virtual[start_sample:end_sample] += chunk
        used_notes[normalized_note] += 1
        used_event_count += 1

        plan_rows.append(
            {
                "row_id": str(idx),
                "track_name": str(row.get("track_name", "")).strip(),
                "note12": note,
                "normalized_note12": normalized_note,
                "start_sec": f"{start_sec:.9f}",
                "end_sec": f"{end_sec:.9f}",
                "velocity": str(vel),
                "sample_found": "YES",
                "sample_wav_path": sample_meta["wav_path"],
                "sample_report_dir": sample_meta["report_dir"],
                "sample_duration_sec": sample_meta["duration_sec"],
                "overlay_duration_sec": f"{overlay_duration_sec:.9f}",
                "velocity_gain": f"{gain:.9f}",
            }
        )

    piano_mask, other_mask = _build_activity_masks(midi_rows, total_samples, int(sr))
    piano_only_mask = piano_mask & (~other_mask)

    if np.count_nonzero(piano_only_mask) > 0 and float(np.sum(virtual[piano_only_mask] ** 2)) > 1e-9:
        alpha = float(np.dot(source_audio[piano_only_mask], virtual[piano_only_mask]) / np.dot(virtual[piano_only_mask], virtual[piano_only_mask]))
    else:
        alpha = 1.0
    alpha = max(0.0, min(4.0, alpha))

    raw_virtual = virtual.copy()
    raw_peak = float(max(np.max(np.abs(raw_virtual)), 1e-9))
    raw_virtual_for_listen = raw_virtual / np.float32(raw_peak) * np.float32(0.95)

    fitted_virtual = raw_virtual * np.float32(alpha)
    residual = source_audio - fitted_virtual
    pair_peak = float(max(np.max(np.abs(fitted_virtual)), np.max(np.abs(residual)), 1e-9))
    if pair_peak > 0.999:
        fitted_virtual = fitted_virtual / np.float32(pair_peak)
        residual = residual / np.float32(pair_peak)

    fitted_peak = float(max(np.max(np.abs(fitted_virtual)), 1e-9))
    fitted_preview = fitted_virtual / np.float32(fitted_peak) * np.float32(0.95)

    sf.write(str(Path(args.out_virtual_piano_wav)), raw_virtual_for_listen, int(sr), subtype="PCM_16")
    sf.write(str(Path(args.out_residual_wav)), residual, int(sr), subtype="PCM_16")
    if args.out_fitted_virtual_piano_wav:
        sf.write(str(Path(args.out_fitted_virtual_piano_wav)), fitted_virtual, int(sr), subtype="PCM_16")
    if args.out_fitted_preview_wav:
        sf.write(str(Path(args.out_fitted_preview_wav)), fitted_preview, int(sr), subtype="PCM_16")

    with Path(args.out_plan_csv).open("w", encoding="utf-8", newline="") as fh:
        fields = list(plan_rows[0].keys()) if plan_rows else [
            "row_id", "track_name", "note12", "start_sec", "end_sec", "velocity",
            "sample_found", "sample_wav_path", "sample_report_dir", "sample_duration_sec",
            "overlay_duration_sec", "velocity_gain",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plan_rows)

    summary_lines = [
        "VIRTUAL PIANO FROM REAL SAMPLES",
        "=" * 72,
        f"source_audio_wav: {args.source_audio_wav}",
        f"piano_midi_events: {len(piano_rows)}",
        f"sample_index_notes: {len(sample_index)}",
        f"used_note_events: {used_event_count}",
        f"missing_note_events: {sum(missing_notes.values())}",
        f"piano_only_audio_ratio: {float(np.count_nonzero(piano_only_mask)) / max(1, total_samples):.6f}",
        f"raw_virtual_peak_before_listen_norm: {raw_peak:.9f}",
        f"fit_gain_alpha: {alpha:.9f}",
        f"fitted_peak_before_preview_norm: {fitted_peak:.9f}",
        "",
        "top_used_notes:",
    ]
    for key, value in used_notes.most_common(12):
        summary_lines.append(f"  {key}: {value}")
    if missing_notes:
        summary_lines.append("")
        summary_lines.append("missing_notes:")
        for key, value in missing_notes.most_common():
            summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "piano_midi_events": len(piano_rows),
                "sample_index_notes": len(sample_index),
                "used_note_events": used_event_count,
                "missing_note_events": int(sum(missing_notes.values())),
                "piano_only_audio_ratio": float(np.count_nonzero(piano_only_mask)) / max(1, total_samples),
                "raw_virtual_peak_before_listen_norm": raw_peak,
                "fit_gain_alpha": alpha,
                "fitted_peak_before_preview_norm": fitted_peak,
                "top_used_notes": dict(used_notes.most_common(24)),
                "missing_notes": dict(missing_notes),
                "release_pad_sec": args.release_pad_sec,
                "velocity_curve": args.velocity_curve,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
