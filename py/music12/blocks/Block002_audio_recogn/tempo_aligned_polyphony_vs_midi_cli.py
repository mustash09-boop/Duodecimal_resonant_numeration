# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitch_class(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _active_reference_notes(
    reference: List[Dict[str, Any]],
    t_real: float,
    tempo_ratio: float,
) -> Set[str]:
    t_ref = t_real / max(tempo_ratio, 1e-9)

    active = set()

    for r in reference:
        note = _normalize_note(r.get("note_token", ""))
        if not note:
            continue

        s = _safe_float(r.get("time_start_sec"), 0.0)
        e = _safe_float(r.get("time_end_sec"), 0.0)

        if s <= t_ref <= e:
            active.add(note)

    return active


def _active_detected_notes_by_frame(rows: List[Dict[str, Any]]) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        note = _normalize_note(r.get("note_token", ""))

        if not note:
            continue

        out.setdefault(frame, set()).add(note)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare detected simultaneous polyphony against MIDI reference with tempo alignment."
    )

    ap.add_argument("--detected_frame_notes_csv", required=True)
    ap.add_argument("--reference_events_csv", required=True)

    ap.add_argument("--out_frame_compare_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--detected_duration_sec", type=float, required=True)
    ap.add_argument("--reference_duration_sec", type=float, required=True)
    ap.add_argument("--fps", type=float, default=60.0)

    args = ap.parse_args()

    detected_rows = _load_csv(Path(args.detected_frame_notes_csv))
    reference_rows = _load_csv(Path(args.reference_events_csv))

    detected_by_frame = _active_detected_notes_by_frame(detected_rows)

    tempo_ratio = args.detected_duration_sec / max(args.reference_duration_sec, 1e-9)

    max_frame = max(detected_by_frame.keys(), default=0)

    frame_rows = []

    exact_tp = 0
    exact_fp = 0
    exact_fn = 0

    pc_tp = 0
    pc_fp = 0
    pc_fn = 0

    polyphony_abs_error_sum = 0
    frames_total = 0

    for frame in range(0, max_frame + 1):
        t_real = frame / args.fps

        detected = detected_by_frame.get(frame, set())
        reference = _active_reference_notes(reference_rows, t_real, tempo_ratio)

        tp = detected & reference
        fp = detected - reference
        fn = reference - detected

        exact_tp += len(tp)
        exact_fp += len(fp)
        exact_fn += len(fn)

        detected_pc = set(_pitch_class(n) for n in detected if _pitch_class(n))
        reference_pc = set(_pitch_class(n) for n in reference if _pitch_class(n))

        pc_match = detected_pc & reference_pc
        pc_extra = detected_pc - reference_pc
        pc_miss = reference_pc - detected_pc

        pc_tp += len(pc_match)
        pc_fp += len(pc_extra)
        pc_fn += len(pc_miss)

        poly_err = abs(len(detected) - len(reference))
        polyphony_abs_error_sum += poly_err
        frames_total += 1

        frame_rows.append({
            "frame_index": frame,
            "time_real_sec": f"{t_real:.9f}",
            "time_reference_sec": f"{(t_real / max(tempo_ratio, 1e-9)):.9f}",
            "detected_count": len(detected),
            "reference_count": len(reference),
            "polyphony_abs_error": poly_err,
            "exact_tp": len(tp),
            "exact_fp": len(fp),
            "exact_fn": len(fn),
            "pitch_class_tp": len(pc_match),
            "pitch_class_fp": len(pc_extra),
            "pitch_class_fn": len(pc_miss),
            "detected_notes": " ".join(sorted(detected)),
            "reference_notes": " ".join(sorted(reference)),
            "exact_matched_notes": " ".join(sorted(tp)),
            "pitch_class_matched": " ".join(sorted(pc_match)),
        })

    exact_precision = exact_tp / max(exact_tp + exact_fp, 1)
    exact_recall = exact_tp / max(exact_tp + exact_fn, 1)

    pc_precision = pc_tp / max(pc_tp + pc_fp, 1)
    pc_recall = pc_tp / max(pc_tp + pc_fn, 1)

    mean_polyphony_abs_error = polyphony_abs_error_sum / max(frames_total, 1)

    out_frame = Path(args.out_frame_compare_csv)
    out_json = Path(args.out_summary_json)
    out_txt = Path(args.out_summary_txt)

    out_frame.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "frame_index",
        "time_real_sec",
        "time_reference_sec",
        "detected_count",
        "reference_count",
        "polyphony_abs_error",
        "exact_tp",
        "exact_fp",
        "exact_fn",
        "pitch_class_tp",
        "pitch_class_fp",
        "pitch_class_fn",
        "detected_notes",
        "reference_notes",
        "exact_matched_notes",
        "pitch_class_matched",
    ]

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)

    summary = {
        "stage": "tempo_aligned_polyphony_vs_midi",
        "inputs": {
            "detected_frame_notes_csv": args.detected_frame_notes_csv,
            "reference_events_csv": args.reference_events_csv,
        },
        "parameters": {
            "detected_duration_sec": args.detected_duration_sec,
            "reference_duration_sec": args.reference_duration_sec,
            "tempo_ratio": tempo_ratio,
            "fps": args.fps,
        },
        "result": {
            "frames_total": frames_total,
            "exact_tp": exact_tp,
            "exact_fp": exact_fp,
            "exact_fn": exact_fn,
            "exact_precision": exact_precision,
            "exact_recall": exact_recall,
            "pitch_class_tp": pc_tp,
            "pitch_class_fp": pc_fp,
            "pitch_class_fn": pc_fn,
            "pitch_class_precision": pc_precision,
            "pitch_class_recall": pc_recall,
            "mean_polyphony_abs_error": mean_polyphony_abs_error,
        },
    }

    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "TEMPO-ALIGNED POLYPHONY VS MIDI",
        "=" * 72,
        f"detected_frame_notes_csv : {args.detected_frame_notes_csv}",
        f"reference_events_csv     : {args.reference_events_csv}",
        "",
        f"detected_duration_sec    : {args.detected_duration_sec:.6f}",
        f"reference_duration_sec   : {args.reference_duration_sec:.6f}",
        f"tempo_ratio              : {tempo_ratio:.9f}",
        f"fps                      : {args.fps:.6f}",
        "",
        f"frames_total             : {frames_total}",
        "",
        "Exact note match:",
        f"  tp                     : {exact_tp}",
        f"  fp                     : {exact_fp}",
        f"  fn                     : {exact_fn}",
        f"  precision              : {exact_precision:.6f}",
        f"  recall                 : {exact_recall:.6f}",
        "",
        "Pitch-class match:",
        f"  tp                     : {pc_tp}",
        f"  fp                     : {pc_fp}",
        f"  fn                     : {pc_fn}",
        f"  precision              : {pc_precision:.6f}",
        f"  recall                 : {pc_recall:.6f}",
        "",
        f"mean_polyphony_abs_error : {mean_polyphony_abs_error:.6f}",
        "",
        "Principle:",
        "  Compare simultaneous detected causal notes against MIDI reference",
        "  after linear tempo alignment between MIDI-render duration and real performance duration.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("tempo-aligned polyphony comparison complete")
    print(json.dumps(summary["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()