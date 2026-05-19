from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from music12.blocks.Block002_pipeline.resonance_candidate_inference_core import (
    infer_candidates,
    iter_frame_candidates,
    load_coords_csv,
    load_matrix_csv_memmap,
    load_times_csv,
)


SEMANTIC_NOTE = (
    "This stage does NOT produce true f0. "
    "It produces framewise candidate sets from resonance scan data. "
    "No strongest peak is used as a semantic decision. "
    "The output of this stage is only candidate material for later chain evaluation."
)


def _split_token_micro(token: str) -> tuple[str, str]:
    token = str(token or "").strip()
    if "'" not in token:
        return token, ""
    coarse, micro = token.split("'", 1)
    return coarse, micro or "-"


def _token_coarse(token: str) -> str:
    coarse, _micro = _split_token_micro(token)
    return coarse


def _support_to_json(s) -> dict:
    matched_note_micro = s.matched_note or ""
    matched_note_coarse = _token_coarse(matched_note_micro) if matched_note_micro else ""

    return {
        "harmonic_index": s.harmonic_index,
        "expected_hz": s.expected_hz,
        "matched_probe_index": s.matched_probe_index,
        "matched_hz": s.matched_hz,
        "matched_energy": s.matched_energy,
        "matched_note_micro": matched_note_micro,
        "matched_note_coarse": matched_note_coarse,
        "matched_note": matched_note_micro,
        "is_hit": s.is_hit,
    }


def _serialize_candidates_json(candidates) -> str:
    payload = []

    for c in candidates:
        note_token_micro = c.note_token
        note_token_coarse = _token_coarse(note_token_micro)
        support_hits = sum(1 for s in c.supports if s.is_hit)

        payload.append(
            {
                "probe_index": c.probe_index,
                "frequency_hz": c.frequency_hz,
                "note_token_micro": note_token_micro,
                "note_token_coarse": note_token_coarse,
                "note_token": note_token_micro,
                "energy": c.energy,
                "support_hit_count": support_hits,
                "supports": [_support_to_json(s) for s in c.supports],
            }
        )

    return json.dumps(payload, ensure_ascii=False)


def _serialize_candidates_compact_json(candidates) -> str:
    payload = []

    for c in candidates:
        note_token_micro = c.note_token
        payload.append(
            {
                "probe_index": c.probe_index,
                "frequency_hz": c.frequency_hz,
                "note_token_micro": note_token_micro,
                "note_token_coarse": _token_coarse(note_token_micro),
                "energy": c.energy,
                "support_hit_count": sum(1 for s in c.supports if s.is_hit),
            }
        )

    return json.dumps(payload, ensure_ascii=False)


def _format_candidate_notes_report(candidates) -> str:
    return " | ".join(c.note_token for c in candidates)


def _format_candidate_notes_coarse_report(candidates) -> str:
    return " | ".join(_token_coarse(c.note_token) for c in candidates)


def _format_candidate_scores_report(candidates) -> str:
    parts = []
    for c in candidates:
        support_hits = sum(1 for s in c.supports if s.is_hit)
        parts.append(f"{c.note_token}:hits={support_hits}:E={c.energy:.6f}")
    return " | ".join(parts)


def _write_meta_json(
    path: Path,
    *,
    matrix_csv: Path,
    times_csv: Path,
    coords_csv: Path,
    out_framewise_csv: Path,
    out_readable_csv: Path,
    energy_threshold: float,
    top_n_candidates: int,
    tolerance_ratio: float,
    analysis_min_hz: float,
    analysis_max_hz: float,
    max_polyphonic_candidates: int,
    result_count: int,
    matrix_cache_info,
    ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "inputs": {
            "matrix_csv": str(matrix_csv),
            "times_csv": str(times_csv),
            "coords_csv": str(coords_csv),
        },
        "candidate_inference": {
            "energy_threshold": energy_threshold,
            "top_n_candidates": top_n_candidates,
            "tolerance_ratio": tolerance_ratio,
            "analysis_min_hz": analysis_min_hz,
            "analysis_max_hz": analysis_max_hz,
            "max_polyphonic_candidates": max_polyphonic_candidates,
            "semantic_note": SEMANTIC_NOTE,
            "machine_readable_source": "selected_candidates_json",
            "compact_machine_readable_source": "selected_candidates_compact_json",
            "human_readable_fields": [
                "selected_notes_report",
                "selected_notes_coarse_report",
                "selected_scores_report",
            ],
            "ontology_note": (
                "Downstream stages must use selected_candidates_json or "
                "selected_candidates_compact_json. Human-readable report fields "
                "must not be parsed as semantic source."
            ),
        },
        "memory_architecture": {
            "mode": "streaming_plus_numpy_memmap",
            "matrix_cache": asdict(matrix_cache_info),
            "note": (
                "The probe matrix is not loaded into RAM as nested Python lists. "
                "It is converted/reused as a disk-backed numpy.memmap and processed frame-by-frame."
            ),
        },
        "derived": {
            "frame_count": result_count,
        },
        "outputs": {
            "framewise_csv": str(out_framewise_csv),
            "framewise_readable_csv": str(out_readable_csv),
            "meta_json": str(path),
        },
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_resume_frame(progress_json: Path, out_framewise_csv: Path) -> int:
    if progress_json.exists():
        try:
            data = json.loads(progress_json.read_text(encoding="utf-8"))
            frame_index = int(data.get("last_completed_frame_index", -1))
            if frame_index >= 0:
                return frame_index
        except Exception:
            pass

    if out_framewise_csv.exists() and out_framewise_csv.stat().st_size > 0:
        last_frame = -1
        try:
            with out_framewise_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_frame = int(row.get("frame_index", -1))
        except Exception:
            last_frame = -1
        return last_frame

    return -1


def _write_progress_json(
    path: Path,
    *,
    frame_index: int,
    result_count: int,
    start_frame: int,
    stop_frame: int | None,
    out_framewise_csv: Path,
    out_readable_csv: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_completed_frame_index": int(frame_index),
        "result_count": int(result_count),
        "start_frame": int(start_frame),
        "stop_frame": None if stop_frame is None else int(stop_frame),
        "out_framewise_csv": str(out_framewise_csv),
        "out_framewise_readable_csv": str(out_readable_csv),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Produce framewise candidate sets from resonance scan data. "
            "This stage does NOT produce true f0 and does NOT use strongest peak as semantic output. "
            "Large matrices are handled by disk-backed memmap and frame streaming."
        )
    )

    ap.add_argument("--matrix_csv", required=True)
    ap.add_argument("--times_csv", required=True)
    ap.add_argument("--coords_csv", required=True)
    ap.add_argument("--out_framewise_csv", required=True)
    ap.add_argument("--out_framewise_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--energy_threshold", type=float, default=0.01)
    ap.add_argument("--top_n_candidates", type=int, default=24)
    ap.add_argument("--tolerance_ratio", type=float, default=0.03)
    ap.add_argument("--analysis_min_hz", type=float, default=16.0)
    ap.add_argument("--analysis_max_hz", type=float, default=22000.0)
    ap.add_argument("--max_polyphonic_candidates", type=int, default=8)
    ap.add_argument("--matrix_cache_dir", default="")
    ap.add_argument("--force_rebuild_matrix_cache", action="store_true")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--stop_frame", type=int, default=-1)
    ap.add_argument("--resume_if_possible", action="store_true")
    ap.add_argument("--flush_every", type=int, default=50)
    ap.add_argument("--progress_json", default="")

    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv).resolve()
    times_csv = Path(args.times_csv).resolve()
    coords_csv = Path(args.coords_csv).resolve()
    out_framewise_csv = Path(args.out_framewise_csv).resolve()
    out_framewise_readable_csv = Path(args.out_framewise_readable_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()
    progress_json = (
        Path(args.progress_json).resolve()
        if args.progress_json
        else out_framewise_csv.with_name(out_framewise_csv.stem + "_progress.json")
    )

    cache_dir = Path(args.matrix_cache_dir).resolve() if args.matrix_cache_dir else None
    requested_start_frame = max(0, int(args.start_frame))
    requested_stop_frame = None if int(args.stop_frame) < 0 else int(args.stop_frame)

    if args.resume_if_possible:
        last_completed_frame = _read_resume_frame(progress_json, out_framewise_csv)
        if last_completed_frame >= requested_start_frame:
            requested_start_frame = last_completed_frame + 1

    matrix, matrix_cache_info = load_matrix_csv_memmap(
        matrix_csv,
        cache_dir=cache_dir,
        force_rebuild_cache=bool(args.force_rebuild_matrix_cache),
        dtype="float32",
    )
    times = load_times_csv(times_csv)
    coords = load_coords_csv(coords_csv)

    out_framewise_csv.parent.mkdir(parents=True, exist_ok=True)
    out_framewise_readable_csv.parent.mkdir(parents=True, exist_ok=True)

    result_count = 0
    write_mode = "a" if args.resume_if_possible and requested_start_frame > 0 else "w"

    with out_framewise_csv.open(write_mode, encoding="utf-8", newline="") as full_f, \
         out_framewise_readable_csv.open(write_mode, encoding="utf-8", newline="") as readable_f:

        full_writer = csv.DictWriter(
            full_f,
            fieldnames=[
                "frame_index",
                "time_sec",
                "candidate_count",
                "selected_notes_report",
                "selected_notes_coarse_report",
                "selected_scores_report",
                "selected_candidates_compact_json",
                "selected_candidates_json",
            ],
        )
        readable_writer = csv.DictWriter(
            readable_f,
            fieldnames=[
                "frame_index",
                "time_sec",
                "candidate_count",
                "selected_notes_report",
                "selected_notes_coarse_report",
                "selected_scores_report",
            ],
        )

        if write_mode == "w":
            full_writer.writeheader()
            readable_writer.writeheader()

        for frame_index, time_sec, candidates in iter_frame_candidates(
            matrix=matrix,
            times=times,
            coords=coords,
            energy_threshold=args.energy_threshold,
            top_n_candidates=args.top_n_candidates,
            tolerance_ratio=args.tolerance_ratio,
            analysis_min_hz=args.analysis_min_hz,
            analysis_max_hz=args.analysis_max_hz,
            max_polyphonic_candidates=args.max_polyphonic_candidates,
            start_frame=requested_start_frame,
            stop_frame=requested_stop_frame,
        ):
            inference = infer_candidates(candidates)

            row = {
                "frame_index": frame_index,
                "time_sec": float(time_sec),
                "candidate_count": len(inference.candidates),
                "selected_notes_report": _format_candidate_notes_report(inference.candidates),
                "selected_notes_coarse_report": _format_candidate_notes_coarse_report(inference.candidates),
                "selected_scores_report": _format_candidate_scores_report(inference.candidates),
                "selected_candidates_compact_json": _serialize_candidates_compact_json(inference.candidates),
                "selected_candidates_json": _serialize_candidates_json(inference.candidates),
            }

            full_writer.writerow(row)
            readable_writer.writerow(
                {
                    "frame_index": row["frame_index"],
                    "time_sec": row["time_sec"],
                    "candidate_count": row["candidate_count"],
                    "selected_notes_report": row["selected_notes_report"],
                    "selected_notes_coarse_report": row["selected_notes_coarse_report"],
                    "selected_scores_report": row["selected_scores_report"],
                }
            )

            result_count += 1

            if result_count % max(1, int(args.flush_every)) == 0:
                full_f.flush()
                readable_f.flush()
                os.fsync(full_f.fileno())
                os.fsync(readable_f.fileno())
                _write_progress_json(
                    progress_json,
                    frame_index=frame_index,
                    result_count=result_count,
                    start_frame=requested_start_frame,
                    stop_frame=requested_stop_frame,
                    out_framewise_csv=out_framewise_csv,
                    out_readable_csv=out_framewise_readable_csv,
                )
            if result_count % 250 == 0:
                print(f"candidate inference progress: frames={result_count}", flush=True)

        if result_count > 0:
            full_f.flush()
            readable_f.flush()
            os.fsync(full_f.fileno())
            os.fsync(readable_f.fileno())
            _write_progress_json(
                progress_json,
                frame_index=frame_index,
                result_count=result_count,
                start_frame=requested_start_frame,
                stop_frame=requested_stop_frame,
                out_framewise_csv=out_framewise_csv,
                out_readable_csv=out_framewise_readable_csv,
            )

    _write_meta_json(
        out_meta_json,
        matrix_csv=matrix_csv,
        times_csv=times_csv,
        coords_csv=coords_csv,
        out_framewise_csv=out_framewise_csv,
        out_readable_csv=out_framewise_readable_csv,
        energy_threshold=args.energy_threshold,
        top_n_candidates=args.top_n_candidates,
        tolerance_ratio=args.tolerance_ratio,
        analysis_min_hz=args.analysis_min_hz,
        analysis_max_hz=args.analysis_max_hz,
        max_polyphonic_candidates=args.max_polyphonic_candidates,
        result_count=result_count,
        matrix_cache_info=matrix_cache_info,
    )

    print("resonance candidate inference complete")
    print(json.dumps(
        {
            "frame_count": result_count,
            "analysis_min_hz": args.analysis_min_hz,
            "analysis_max_hz": args.analysis_max_hz,
            "max_polyphonic_candidates": args.max_polyphonic_candidates,
            "start_frame": requested_start_frame,
            "stop_frame": requested_stop_frame,
            "resume_if_possible": bool(args.resume_if_possible),
            "progress_json": str(progress_json),
            "matrix_cache": asdict(matrix_cache_info),
            "out_framewise_csv": str(out_framewise_csv),
            "out_framewise_readable_csv": str(out_framewise_readable_csv),
            "out_meta_json": str(out_meta_json),
            "machine_readable_source": "selected_candidates_json",
            "semantic_note": SEMANTIC_NOTE,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
