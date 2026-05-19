# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _notes(raw: str) -> Set[str]:
    return {x.strip() for x in str(raw or "").split() if x.strip()}


def _pc(note: str) -> str:
    try:
        return note.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _pc_set(notes: Set[str]) -> Set[str]:
    return {_pc(n) for n in notes if _pc(n)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose polyphonic note recognition errors frame-by-frame."
    )

    ap.add_argument("--frame_compare_csv", required=True)

    ap.add_argument("--out_error_summary_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_problem_windows_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--problem_min_error", type=int, default=4)

    args = ap.parse_args()

    rows = _load_csv(Path(args.frame_compare_csv))

    missed_counter = Counter()
    extra_counter = Counter()
    pc_only_counter = Counter()
    overload_counter = Counter()
    underload_counter = Counter()

    readable_rows = []
    problem_rows = []

    total_exact_tp = 0
    total_exact_fp = 0
    total_exact_fn = 0
    total_pc_tp = 0
    total_pc_fp = 0
    total_pc_fn = 0

    for r in rows:
        frame = _safe_int(r.get("frame_index"), 0)
        t_real = _safe_float(r.get("time_real_sec"), 0.0)
        t_ref = _safe_float(r.get("time_reference_sec"), 0.0)

        detected = _notes(r.get("detected_notes", ""))
        reference = _notes(r.get("reference_notes", ""))

        exact_match = detected & reference
        extra = detected - reference
        missed = reference - detected

        detected_pc = _pc_set(detected)
        reference_pc = _pc_set(reference)
        pc_match = detected_pc & reference_pc

        pc_only_detected = []
        for d in detected:
            if d not in exact_match and _pc(d) in reference_pc:
                pc_only_detected.append(d)

        for n in missed:
            missed_counter[n] += 1

        for n in extra:
            extra_counter[n] += 1

        for n in pc_only_detected:
            pc_only_counter[n] += 1

        poly_error = abs(len(detected) - len(reference))

        if len(detected) > len(reference):
            overload_counter[poly_error] += 1
            poly_state = "OVERLOAD"
        elif len(detected) < len(reference):
            underload_counter[poly_error] += 1
            poly_state = "UNDERLOAD"
        else:
            poly_state = "BALANCED"

        exact_tp = len(exact_match)
        exact_fp = len(extra)
        exact_fn = len(missed)

        pc_tp = len(pc_match)
        pc_fp = len(detected_pc - reference_pc)
        pc_fn = len(reference_pc - detected_pc)

        total_exact_tp += exact_tp
        total_exact_fp += exact_fp
        total_exact_fn += exact_fn
        total_pc_tp += pc_tp
        total_pc_fp += pc_fp
        total_pc_fn += pc_fn

        row = {
            "frame_index": frame,
            "time_real_sec": f"{t_real:.6f}",
            "time_reference_sec": f"{t_ref:.6f}",
            "midi_should_sound": " ".join(sorted(reference)),
            "algorithm_detected": " ".join(sorted(detected)),
            "exact_matched": " ".join(sorted(exact_match)),
            "missed_from_midi": " ".join(sorted(missed)),
            "extra_from_algorithm": " ".join(sorted(extra)),
            "pitch_class_matched": " ".join(sorted(pc_match)),
            "pitch_class_only_detected": " ".join(sorted(pc_only_detected)),
            "midi_count": len(reference),
            "detected_count": len(detected),
            "polyphony_error": poly_error,
            "polyphony_state": poly_state,
            "exact_tp": exact_tp,
            "exact_fp": exact_fp,
            "exact_fn": exact_fn,
            "pc_tp": pc_tp,
            "pc_fp": pc_fp,
            "pc_fn": pc_fn,
        }

        readable_rows.append(row)

        if poly_error >= args.problem_min_error or exact_fn >= args.problem_min_error or exact_fp >= args.problem_min_error:
            problem_rows.append(row)

    summary_rows = []

    for note, count in missed_counter.most_common(40):
        summary_rows.append({
            "error_type": "MISSED_REFERENCE",
            "item": note,
            "count": count,
        })

    for note, count in extra_counter.most_common(40):
        summary_rows.append({
            "error_type": "EXTRA_RESONANCE_OR_FALSE_NOTE",
            "item": note,
            "count": count,
        })

    for note, count in pc_only_counter.most_common(40):
        summary_rows.append({
            "error_type": "PITCH_CLASS_ONLY_WRONG_OCTAVE_OR_ROOT",
            "item": note,
            "count": count,
        })

    out_summary = Path(args.out_error_summary_csv)
    out_readable = Path(args.out_readable_csv)
    out_problems = Path(args.out_problem_windows_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_summary.parent.mkdir(parents=True, exist_ok=True)

    with out_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["error_type", "item", "count"])
        w.writeheader()
        w.writerows(summary_rows)

    readable_fields = [
        "frame_index",
        "time_real_sec",
        "time_reference_sec",
        "midi_should_sound",
        "algorithm_detected",
        "exact_matched",
        "missed_from_midi",
        "extra_from_algorithm",
        "pitch_class_matched",
        "pitch_class_only_detected",
        "midi_count",
        "detected_count",
        "polyphony_error",
        "polyphony_state",
        "exact_tp",
        "exact_fp",
        "exact_fn",
        "pc_tp",
        "pc_fp",
        "pc_fn",
    ]

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=readable_fields)
        w.writeheader()
        w.writerows(readable_rows)

    with out_problems.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=readable_fields)
        w.writeheader()
        w.writerows(problem_rows)

    exact_precision = total_exact_tp / max(total_exact_tp + total_exact_fp, 1)
    exact_recall = total_exact_tp / max(total_exact_tp + total_exact_fn, 1)
    pc_precision = total_pc_tp / max(total_pc_tp + total_pc_fp, 1)
    pc_recall = total_pc_tp / max(total_pc_tp + total_pc_fn, 1)

    meta = {
        "stage": "polyphony_error_diagnostics",
        "inputs": {
            "frame_compare_csv": args.frame_compare_csv,
        },
        "outputs": {
            "error_summary_csv": args.out_error_summary_csv,
            "readable_csv": args.out_readable_csv,
            "problem_windows_csv": args.out_problem_windows_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "result": {
            "frames": len(rows),
            "problem_windows": len(problem_rows),
            "exact_precision": exact_precision,
            "exact_recall": exact_recall,
            "pitch_class_precision": pc_precision,
            "pitch_class_recall": pc_recall,
            "top_missed": missed_counter.most_common(10),
            "top_extra": extra_counter.most_common(10),
            "top_pitch_class_only": pc_only_counter.most_common(10),
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "POLYPHONY ERROR DIAGNOSTICS",
        "=" * 72,
        f"frame_compare_csv : {args.frame_compare_csv}",
        "",
        f"frames            : {len(rows)}",
        f"problem_windows   : {len(problem_rows)}",
        "",
        f"exact_precision   : {exact_precision:.6f}",
        f"exact_recall      : {exact_recall:.6f}",
        f"pc_precision      : {pc_precision:.6f}",
        f"pc_recall         : {pc_recall:.6f}",
        "",
        "Top missed MIDI notes:",
    ]

    for note, count in missed_counter.most_common(12):
        txt.append(f"  {note}: {count}")

    txt.append("")
    txt.append("Top extra algorithm notes:")
    for note, count in extra_counter.most_common(12):
        txt.append(f"  {note}: {count}")

    txt.append("")
    txt.append("Top pitch-class-only / likely octave-root errors:")
    for note, count in pc_only_counter.most_common(12):
        txt.append(f"  {note}: {count}")

    txt.extend([
        "",
        "Readable report columns:",
        "  midi_should_sound      — что должно звучать по MIDI",
        "  algorithm_detected     — что нашёл алгоритм",
        "  missed_from_midi       — что MIDI требует, но алгоритм не нашёл",
        "  extra_from_algorithm   — что алгоритм добавил сверх MIDI",
        "  pitch_class_only       — похожие ноты по классу, но с ошибкой октавы/корня",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("polyphony error diagnostics complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()