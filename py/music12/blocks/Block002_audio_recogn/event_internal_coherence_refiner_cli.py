# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


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


def _parse_path(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _coherence_score(row: Dict[str, Any]) -> Dict[str, Any]:
    note_path = [_normalize_note(x) for x in _parse_path(row.get("note_path", ""))]
    state_path = _parse_path(row.get("state_path", ""))

    unique_note_count = len(set(note_path))
    duration = _safe_int(row.get("duration_frames"), 0)
    segment_count = _safe_int(row.get("segment_count"), 1)

    re_count = _safe_int(row.get("re_excitation_count"), 0)
    active_count = _safe_int(row.get("active_body_count"), 0)
    sustain_count = _safe_int(row.get("sustain_body_count"), 0)
    response_count = _safe_int(row.get("response_trace_count"), 0)
    decay_count = _safe_int(row.get("decay_trace_count"), 0)

    mean_score = _safe_float(row.get("mean_score"), 0.0)
    max_score = _safe_float(row.get("max_score"), 0.0)
    birth_score = _safe_float(row.get("birth_score"), 0.0)
    final_score = _safe_float(row.get("final_score"), 0.0)

    same_note_ratio = 1.0
    if note_path:
        primary = _normalize_note(row.get("candidate_note", note_path[0]))
        same_note_ratio = sum(1 for n in note_path if n == primary) / max(len(note_path), 1)

    body_frames = active_count + sustain_count
    trace_frames = response_count + decay_count

    body_ratio = body_frames / max(duration, 1)
    trace_ratio = trace_frames / max(duration, 1)

    energy_span = max_score - min(birth_score, final_score)
    relative_energy_span = energy_span / max(max_score, 1e-9)

    # Чем выше, тем больше это единый устойчивый жизненный цикл.
    coherence = 0.0
    coherence += same_note_ratio * 0.35
    coherence += min(duration / 90.0, 1.0) * 0.18
    coherence += body_ratio * 0.18
    coherence += min(mean_score / 2.5, 1.0) * 0.14
    coherence -= min(segment_count - 1, 6) * 0.04
    coherence -= trace_ratio * 0.10

    # Внутренние волны энергии допустимы, если нота та же и тело события устойчиво.
    re_as_internal_wave = False
    if re_count > 0 and same_note_ratio >= 0.95 and body_ratio >= 0.30:
        re_as_internal_wave = True
        coherence += 0.12

    # Сильная пульсация без смены ноты — скорее внутренняя жизнь, а не новое событие.
    if relative_energy_span >= 0.25 and same_note_ratio >= 0.95:
        coherence += 0.06

    coherence = max(0.0, min(coherence, 1.0))

    if coherence >= 0.72 and re_as_internal_wave:
        refined_kind = "coherent_sustained_lifecycle_with_internal_waves"
    elif coherence >= 0.72:
        refined_kind = "coherent_sustained_lifecycle"
    elif trace_ratio > body_ratio:
        refined_kind = "resonance_trace_lifecycle"
    elif segment_count >= 3:
        refined_kind = "fragmented_lifecycle"
    else:
        refined_kind = "weak_or_short_lifecycle"

    return {
        "internal_coherence_score": coherence,
        "same_note_ratio": same_note_ratio,
        "body_ratio": body_ratio,
        "trace_ratio": trace_ratio,
        "relative_energy_span": relative_energy_span,
        "re_as_internal_wave": "YES" if re_as_internal_wave else "NO",
        "refined_lifecycle_kind": refined_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine event lifecycle labels by internal coherence instead of treating energy waves as new excitation."
    )

    ap.add_argument("--event_matches_csv", required=True)

    ap.add_argument("--out_refined_events_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_coherent_score", type=float, default=0.72)

    args = ap.parse_args()

    rows = _load_csv(Path(args.event_matches_csv))

    out_rows = []
    readable_rows = []

    refined_counts: Dict[str, int] = {}
    wave_count = 0
    coherent_count = 0

    for r in rows:
        c = _coherence_score(r)

        rr = dict(r)
        rr["internal_coherence_score"] = f"{c['internal_coherence_score']:.9f}"
        rr["same_note_ratio"] = f"{c['same_note_ratio']:.9f}"
        rr["body_ratio"] = f"{c['body_ratio']:.9f}"
        rr["trace_ratio"] = f"{c['trace_ratio']:.9f}"
        rr["relative_energy_span"] = f"{c['relative_energy_span']:.9f}"
        rr["re_as_internal_wave"] = c["re_as_internal_wave"]
        rr["refined_lifecycle_kind"] = c["refined_lifecycle_kind"]

        if c["re_as_internal_wave"] == "YES":
            wave_count += 1

        if c["internal_coherence_score"] >= args.min_coherent_score:
            coherent_count += 1

        refined_counts[c["refined_lifecycle_kind"]] = refined_counts.get(c["refined_lifecycle_kind"], 0) + 1

        out_rows.append(rr)

        readable_rows.append({
            "event_id": r.get("merged_event_id", r.get("event_id", "")),
            "candidate_note": r.get("candidate_note", ""),
            "passport_status": r.get("passport_status", ""),
            "passport_score": r.get("event_passport_score", ""),
            "old_lifecycle_kind": r.get("lifecycle_kind", ""),
            "refined_lifecycle_kind": c["refined_lifecycle_kind"],
            "internal_coherence_score": f"{c['internal_coherence_score']:.3f}",
            "same_note_ratio": f"{c['same_note_ratio']:.3f}",
            "body_ratio": f"{c['body_ratio']:.3f}",
            "trace_ratio": f"{c['trace_ratio']:.3f}",
            "re_as_internal_wave": c["re_as_internal_wave"],
            "birth_frame": r.get("birth_frame", ""),
            "end_frame": r.get("end_frame", ""),
            "duration_frames": r.get("duration_frames", ""),
        })

    out_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            -_safe_float(r.get("internal_coherence_score"), 0.0),
        )
    )

    out_csv = Path(args.out_refined_events_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys()) if out_rows else []

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)

    readable_fields = [
        "event_id",
        "candidate_note",
        "passport_status",
        "passport_score",
        "old_lifecycle_kind",
        "refined_lifecycle_kind",
        "internal_coherence_score",
        "same_note_ratio",
        "body_ratio",
        "trace_ratio",
        "re_as_internal_wave",
        "birth_frame",
        "end_frame",
        "duration_frames",
    ]

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=readable_fields)
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "event_internal_coherence_refiner",
        "inputs": {
            "event_matches_csv": args.event_matches_csv,
        },
        "outputs": {
            "refined_events_csv": args.out_refined_events_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_coherent_score": args.min_coherent_score,
        },
        "result": {
            "input_events": len(rows),
            "coherent_events": coherent_count,
            "re_excitation_reinterpreted_as_internal_wave": wave_count,
            "refined_counts": refined_counts,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "EVENT INTERNAL COHERENCE REFINER",
        "=" * 72,
        f"event_matches_csv : {args.event_matches_csv}",
        "",
        f"input_events       : {len(rows)}",
        f"coherent_events    : {coherent_count}",
        f"re_as_wave_events  : {wave_count}",
        "",
        "Refined lifecycle counts:",
    ]

    for k in sorted(refined_counts):
        txt.append(f"  {k}: {refined_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  A rise of energy inside the same stable topology is treated as internal life",
        "  of the event, not automatically as a new causal excitation.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("event internal coherence refiner complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()