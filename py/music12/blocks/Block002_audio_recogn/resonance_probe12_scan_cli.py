from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from music12.core.notation12 import (
    bij12_to_int,
    int_to_base12_digit,
    int_to_bij12,
)
from music12.core.resonance_probe12_core import (
    ProbeBankConfig,
    ProbeFrequencyConfig,
    ProbeShapeConfig,
    ResonanceScanConfig,
    TimeGridConfig,
    scan_wav_with_resonance_probes,
)


def parse_duodecimal_octave_arg(value: str) -> int:
    s = str(value).strip().upper()
    try:
        return bij12_to_int(s)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid bijective base-12 octave '{value}'. Allowed symbols: 1..9,A,B,C (no zero)"
        )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _write_matrix_csv(matrix: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe_index"] + [f"frame_{i}" for i in range(matrix.shape[1])])

        for probe_idx in range(matrix.shape[0]):
            writer.writerow([probe_idx] + matrix[probe_idx, :].tolist())


def _write_times_csv(times: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "time_seconds"])

        for frame_idx, t in enumerate(times.tolist()):
            writer.writerow([frame_idx, t])


def _format_degree12_external(value: Any) -> str:
    iv = int(value)
    if not 0 <= iv < 12:
        raise ValueError(
            f"degree12 index out of allowed internal range 0..11: {value!r}"
        )
    return int_to_base12_digit(iv)


def _format_subdivisions_external(subdivisions: Any) -> list[str]:
    out: list[str] = []
    for raw in list(subdivisions):
        iv = int(raw)
        if not 0 <= iv < 12:
            raise ValueError(
                f"subdivision index out of allowed internal range 0..11: {raw!r}"
            )
        out.append(int_to_base12_digit(iv))
    return out


def _format_subdivisions_delta(subdivisions: Any) -> list[str]:
    center = 6
    out: list[str] = []

    for raw in list(subdivisions):
        iv = int(raw)
        if not 0 <= iv < 12:
            raise ValueError(
                f"subdivision index out of allowed internal range 0..11: {raw!r}"
            )

        delta = iv - center
        if delta == 0:
            continue

        sign = "i" if delta > 0 else "a"
        magnitude = abs(delta)
        digit = int_to_base12_digit(magnitude)
        out.append(f"{sign}{digit}")

    return out


def _build_note_token(
    octave_external: str,
    degree12_external: str,
    subdivisions_external: list[str],
) -> str:
    note_token = f"{octave_external}.{degree12_external}"
    if subdivisions_external:
        note_token += "'" + "".join(subdivisions_external)
    else:
        note_token += "'-"
    return note_token


def _write_coords_csv(
    *,
    coords,
    frequencies_hz: np.ndarray,
    global_indices: np.ndarray,
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "probe_index",
                "octave",
                "degree12",
                "subdivisions",
                "note_token",
                "frequency_hz",
                "global_index",
            ]
        )

        for probe_idx, coord in enumerate(coords):
            octave_external = int_to_bij12(coord.octave)
            degree12_external = _format_degree12_external(coord.degree12)
            subdivisions_external = _format_subdivisions_delta(coord.subdivisions)
            note_token = _build_note_token(
                octave_external=octave_external,
                degree12_external=degree12_external,
                subdivisions_external=subdivisions_external,
            )

            writer.writerow(
                [
                    probe_idx,
                    octave_external,
                    degree12_external,
                    json.dumps(subdivisions_external, ensure_ascii=False),
                    note_token,
                    float(frequencies_hz[probe_idx]),
                    int(global_indices[probe_idx]),
                ]
            )


def _slice_wav_if_needed(
    wav_path: Path,
    time_start: float | None,
    time_end: float | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None, dict[str, Any]]:
    info = sf.info(str(wav_path))
    sr = info.samplerate
    total_frames = info.frames
    total_duration = total_frames / sr if sr else 0.0

    if time_start is None and time_end is None:
        return wav_path, None, {
            "applied": False,
            "source_duration_seconds": float(total_duration),
            "start_seconds": None,
            "end_seconds": None,
            "effective_duration_seconds": float(total_duration),
        }

    t0 = 0.0 if time_start is None else float(time_start)
    t1 = total_duration if time_end is None else float(time_end)

    if t0 < 0:
        raise ValueError(f"time_start must be >= 0, got {t0}")
    if t1 <= t0:
        raise ValueError(f"time_end must be > time_start, got {t1} <= {t0}")
    if t0 >= total_duration:
        raise ValueError(
            f"time_start={t0} is outside WAV duration={total_duration:.6f}"
        )

    t1 = min(t1, total_duration)

    start_frame = int(round(t0 * sr))
    end_frame = int(round(t1 * sr))
    end_frame = max(end_frame, start_frame + 1)

    with sf.SoundFile(str(wav_path), "r") as src:
        src.seek(start_frame)
        audio = src.read(end_frame - start_frame)

    tmpdir = tempfile.TemporaryDirectory(prefix="music12_slice_")
    tmp_path = Path(tmpdir.name) / "segment.wav"

    sf.write(str(tmp_path), audio, sr)

    return tmp_path, tmpdir, {
        "applied": True,
        "source_duration_seconds": float(total_duration),
        "start_seconds": float(t0),
        "end_seconds": float(t1),
        "effective_duration_seconds": float(t1 - t0),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "samplerate": int(sr),
        "temporary_wav": str(tmp_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block002: scan WAV with internal resonance probes and save response matrix"
    )

    parser.add_argument("--wav", required=True, help="Path to input WAV")
    parser.add_argument("--out_matrix_csv", required=True, help="Output CSV for response matrix")
    parser.add_argument("--out_meta_json", required=True, help="Output JSON metadata")
    parser.add_argument("--out_times_csv", default="", help="Optional CSV for frame times")
    parser.add_argument("--out_coords_csv", default="", help="Optional CSV for probe coordinates")

    parser.add_argument("--octave_min", type=parse_duodecimal_octave_arg, required=True, help="Minimum octave for probe bank")
    parser.add_argument("--octave_max", type=parse_duodecimal_octave_arg, required=True, help="Maximum octave for probe bank")
    parser.add_argument("--detail_depth", type=int, default=1, help="0->12, 1->144, 2->1728, ...")
    parser.add_argument(
        "--projection_depth",
        type=int,
        default=-1,
        help="Projection detail depth; default = detail_depth",
    )

    parser.add_argument("--time_step_seconds", type=float, default=1.0 / 60.0, help="Analytical time grid step")
    parser.add_argument("--window_seconds", type=float, default=1.0 / 20.0, help="Analysis window length")

    parser.add_argument("--attack_portion", type=float, default=0.15, help="Probe envelope attack portion")
    parser.add_argument("--decay_portion", type=float, default=0.20, help="Probe envelope decay portion")
    parser.add_argument(
        "--harmonic_weights",
        default="1.0,0.45,0.22,0.10",
        help='Comma-separated harmonic weights, e.g. "1.0,0.45,0.22,0.10"',
    )
    parser.add_argument(
        "--window_type",
        default="hamming",
        choices=["hamming", "attack_decay"],
        help="Probe envelope window type",
    )

    parser.add_argument(
        "--time_start",
        type=float,
        default=None,
        help="Optional segment start in seconds",
    )
    parser.add_argument(
        "--time_end",
        type=float,
        default=None,
        help="Optional segment end in seconds",
    )

    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    out_matrix_csv = Path(args.out_matrix_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()
    out_times_csv = Path(args.out_times_csv).resolve() if args.out_times_csv else None
    out_coords_csv = Path(args.out_coords_csv).resolve() if args.out_coords_csv else None

    harmonic_weights = tuple(
        float(x.strip())
        for x in args.harmonic_weights.split(",")
        if x.strip()
    )

    projection_depth = args.detail_depth if args.projection_depth < 0 else args.projection_depth

    effective_wav_path, tmpdir, time_slice_meta = _slice_wav_if_needed(
        wav_path=wav_path,
        time_start=args.time_start,
        time_end=args.time_end,
    )

    config = ResonanceScanConfig(
        probe_bank=ProbeBankConfig(
            octave_min=int(args.octave_min),
            octave_max=int(args.octave_max),
            detail_depth=int(args.detail_depth),
        ),
        probe_frequency=ProbeFrequencyConfig(
            detail_depth_for_projection=int(projection_depth),
        ),
        time_grid=TimeGridConfig(
            step_seconds=float(args.time_step_seconds),
            window_seconds=float(args.window_seconds),
            center_frames=True,
        ),
        probe_shape=ProbeShapeConfig(
            attack_portion=float(args.attack_portion),
            decay_portion=float(args.decay_portion),
            harmonic_weights=harmonic_weights,
            normalize_input_segment=True,
            window_type=str(args.window_type),
        ),
    )

    print("=== SCAN CLI START ===", flush=True)
    print(f"WAV                : {wav_path}", flush=True)
    print(f"EFFECTIVE WAV      : {effective_wav_path}", flush=True)
    print(f"OCTAVE RANGE       : {args.octave_min} .. {args.octave_max}", flush=True)
    print(f"DETAIL DEPTH       : {args.detail_depth}", flush=True)
    print(f"PROJECTION DEPTH   : {projection_depth}", flush=True)
    print(f"TIME STEP SEC      : {args.time_step_seconds}", flush=True)
    print(f"WINDOW SEC         : {args.window_seconds}", flush=True)
    print(f"ATTACK / DECAY     : {args.attack_portion} / {args.decay_portion}", flush=True)
    print(f"HARMONIC WEIGHTS   : {list(harmonic_weights)}", flush=True)
    print(f"WINDOW TYPE        : {args.window_type}", flush=True)
    print(f"OUT MATRIX CSV     : {out_matrix_csv}", flush=True)
    print(f"OUT TIMES CSV      : {out_times_csv if out_times_csv else '(disabled)'}", flush=True)
    print(f"OUT COORDS CSV     : {out_coords_csv if out_coords_csv else '(disabled)'}", flush=True)
    print(f"OUT META JSON      : {out_meta_json}", flush=True)
    if time_slice_meta.get("applied"):
        print(f"TIME SLICE         : {time_slice_meta}", flush=True)

    try:
        print("SCAN CALL STARTED", flush=True)
        result = scan_wav_with_resonance_probes(
            wav_path=effective_wav_path,
            config=config,
        )
        print("SCAN CALL FINISHED", flush=True)

        print(
            f"RESULT MATRIX SHAPE: {list(result.matrix.shape)}; "
            f"frames={len(result.frame_times)}; "
            f"coords={len(result.coords)}; "
            f"detail_depth={result.detail_depth}",
            flush=True,
        )

        print(f"WRITING MATRIX CSV : {out_matrix_csv}", flush=True)
        _write_matrix_csv(result.matrix, out_matrix_csv)

        if out_times_csv is not None:
            print(f"WRITING TIMES CSV  : {out_times_csv}", flush=True)
            _write_times_csv(result.frame_times, out_times_csv)

        if out_coords_csv is not None:
            print(f"WRITING COORDS CSV : {out_coords_csv}", flush=True)
            _write_coords_csv(
                coords=result.coords,
                frequencies_hz=result.frequencies_hz,
                global_indices=result.global_indices,
                out_csv=out_coords_csv,
            )

        meta = {
            "wav": str(wav_path),
            "effective_wav": str(effective_wav_path),
            "matrix_shape": list(result.matrix.shape),
            "detail_depth": int(result.detail_depth),
            "octave_min": int(args.octave_min),
            "octave_max": int(args.octave_max),
            "formal_anchor": {
                "note": "9.A'-",
                "frequency_hz": 440.0,
                "role": "reference_only_not_used_for_inference",
            },
            "projection_depth": int(projection_depth),
            "time_grid": {
                "step_seconds": float(args.time_step_seconds),
                "window_seconds": float(args.window_seconds),
            },
            "time_slice": time_slice_meta,
            "probe_shape": {
                "attack_portion": float(args.attack_portion),
                "decay_portion": float(args.decay_portion),
                "harmonic_weights": list(harmonic_weights),
                "window_type": str(args.window_type),
            },
            "outputs": {
                "matrix_csv": str(out_matrix_csv),
                "times_csv": str(out_times_csv) if out_times_csv else "",
                "coords_csv": str(out_coords_csv) if out_coords_csv else "",
                "meta_json": str(out_meta_json),
            },
        }

        print(f"WRITING META JSON  : {out_meta_json}", flush=True)
        out_meta_json.parent.mkdir(parents=True, exist_ok=True)
        out_meta_json.write_text(
            json.dumps(_to_serializable(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("=== SCAN CLI DONE ===", flush=True)
        print("resonance probe scan complete", flush=True)
        print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


if __name__ == "__main__":
    main()