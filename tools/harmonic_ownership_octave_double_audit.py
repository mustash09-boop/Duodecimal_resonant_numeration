from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_list(value: Any) -> list[Any]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return raw
    except Exception:
        return []
    return []


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


def _octave_label(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[0]
    except Exception:
        return ""


def _chain_points(chain_json_string: str) -> list[dict[str, Any]]:
    try:
        raw = json.loads(str(chain_json_string or "[]"))
        if isinstance(raw, list):
            out: list[dict[str, Any]] = []
            for row in raw:
                if isinstance(row, dict):
                    out.append(
                        {
                            "h": _safe_int(row.get("h"), 0),
                            "freq": _safe_float(row.get("freq"), 0.0),
                            "amp": _safe_float(row.get("amp"), 0.0),
                        }
                    )
            return out
    except Exception:
        return []
    return []


def _close_freq(a: float, b: float) -> bool:
    if a <= 0.0 or b <= 0.0:
        return False
    return abs(a - b) <= max(8.0, 0.015 * max(a, b))


def _find_fundamental(chain: list[dict[str, Any]]) -> float:
    for row in chain:
        if _safe_int(row.get("h"), 0) == 1:
            return _safe_float(row.get("freq"), 0.0)
    return 0.0


def _shared_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[int, float]:
    shared = 0
    amp_sum = 0.0
    for lrow in left:
        lf = _safe_float(lrow.get("freq"), 0.0)
        la = _safe_float(lrow.get("amp"), 0.0)
        for rrow in right:
            rf = _safe_float(rrow.get("freq"), 0.0)
            ra = _safe_float(rrow.get("amp"), 0.0)
            if _close_freq(lf, rf):
                shared += 1
                amp_sum += min(la, ra)
                break
    return shared, amp_sum


def _fundamental_in_other_harmonics(
    fund: float,
    other_chain: list[dict[str, Any]],
) -> int:
    if fund <= 0.0:
        return 0
    for row in other_chain:
        h = _safe_int(row.get("h"), 0)
        freq = _safe_float(row.get("freq"), 0.0)
        if h >= 2 and _close_freq(fund, freq):
            return h
    return 0


def _classify_mode(
    *,
    exact_present: bool,
    same_pc_candidates: list[dict[str, Any]],
    truth_rank: int,
    other_rank: int,
    truth_led: bool,
    shared_count: int,
    truth_in_other_h: int,
    other_in_truth_h: int,
) -> str:
    if not same_pc_candidates:
        return "NO_PITCHCLASS_TRACE"
    if len(same_pc_candidates) == 1:
        return "SINGLE_OCTAVE_PITCHCLASS"
    if not exact_present:
        return "OCTAVE_FAMILY_TRUTH_MISSING"
    if truth_in_other_h in {5, 7} or other_in_truth_h in {5, 7}:
        return "H5_H7_ALIASING"
    if shared_count >= 3:
        if truth_led:
            return "OCTAVE_DOUBLE_TRUTH_LED_SHARED"
        return "OCTAVE_DOUBLE_OTHER_LED_SHARED"
    if truth_rank > 0 and other_rank > 0 and abs(truth_rank - other_rank) <= 1:
        return "OCTAVE_DOUBLE_BALANCED"
    if truth_led:
        return "OCTAVE_DOUBLE_TRUTH_LED"
    return "OCTAVE_DOUBLE_OTHER_LED"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit harmonic ownership and octave doubles inside local DSP note candidates, "
            "to see whether wrong note identity is often an octave-family conflict."
        )
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--recognized-note-event-audit-csv", required=True)
    ap.add_argument("--dsp-group-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    recognized_rows = _load_csv(Path(args.recognized_note_event_audit_csv))
    dsp_rows = _load_csv(Path(args.dsp_group_audit_csv))

    rec_by_event = {
        str(row.get("event_index", "")).strip(): row
        for row in recognized_rows
    }
    dsp_by_group = {
        str(row.get("onset_group", "")).strip(): row
        for row in dsp_rows
    }

    audit_rows: list[dict[str, Any]] = []
    mode_counter: Counter[str] = Counter()
    breathing_counter: Counter[str] = Counter()
    octave_double_by_breathing: dict[str, Counter[str]] = defaultdict(Counter)

    for midi in midi_rows:
        event_index = str(midi.get("event_index", "")).strip()
        onset_group = str(midi.get("onset_group", "")).strip()
        midi_note = _normalize_note(midi.get("expected_note_token", midi.get("note_token", "")))
        midi_pc = _pitch_class(midi_note)
        rec = rec_by_event.get(event_index, {})
        breathing_status = str(rec.get("breathing_status", "")).strip()
        strict_status = str(rec.get("strict_status", "")).strip()
        dsp = dsp_by_group.get(onset_group, {})
        chain_rows = _json_list(dsp.get("dsp_chain_rows_json", "[]"))

        norm_candidates: list[dict[str, Any]] = []
        for rank, cand in enumerate(chain_rows, start=1):
            if not isinstance(cand, dict):
                continue
            note = _normalize_note(cand.get("note_token", ""))
            if not note:
                continue
            norm_candidates.append(
                {
                    "rank": rank,
                    "note_token": note,
                    "pitch_class": _pitch_class(note),
                    "octave_label": _octave_label(note),
                    "score": _safe_float(cand.get("score"), 0.0),
                    "harmonic_count": _safe_int(cand.get("harmonic_count"), 0),
                    "chain": _chain_points(cand.get("chain_json", "[]")),
                }
            )

        same_pc_candidates = [row for row in norm_candidates if row.get("pitch_class") == midi_pc]
        exact_row = next((row for row in same_pc_candidates if row.get("note_token") == midi_note), None)
        other_rows = [row for row in same_pc_candidates if row.get("note_token") != midi_note]
        other_rows.sort(key=lambda row: (_safe_int(row.get("rank"), 10**9), -_safe_float(row.get("score"), 0.0)))
        best_other = other_rows[0] if other_rows else None

        truth_rank = _safe_int(exact_row.get("rank"), 0) if exact_row else 0
        other_rank = _safe_int(best_other.get("rank"), 0) if best_other else 0
        exact_present = exact_row is not None
        octave_double_present = len({_octave_label(row.get("note_token", "")) for row in same_pc_candidates}) >= 2

        shared_count = 0
        shared_amp = 0.0
        truth_in_other_h = 0
        other_in_truth_h = 0
        if exact_row and best_other:
            shared_count, shared_amp = _shared_overlap(exact_row["chain"], best_other["chain"])
            truth_fund = _find_fundamental(exact_row["chain"])
            other_fund = _find_fundamental(best_other["chain"])
            truth_in_other_h = _fundamental_in_other_harmonics(truth_fund, best_other["chain"])
            other_in_truth_h = _fundamental_in_other_harmonics(other_fund, exact_row["chain"])

        truth_led = False
        if exact_row:
            if not best_other:
                truth_led = True
            elif truth_rank < other_rank:
                truth_led = True
            elif truth_rank == other_rank and _safe_float(exact_row.get("score"), 0.0) >= _safe_float(best_other.get("score"), 0.0):
                truth_led = True

        mode = _classify_mode(
            exact_present=exact_present,
            same_pc_candidates=same_pc_candidates,
            truth_rank=truth_rank,
            other_rank=other_rank,
            truth_led=truth_led,
            shared_count=shared_count,
            truth_in_other_h=truth_in_other_h,
            other_in_truth_h=other_in_truth_h,
        )
        mode_counter[mode] += 1
        breathing_counter[breathing_status] += 1
        if octave_double_present:
            octave_double_by_breathing[breathing_status]["octave_double"] += 1
        else:
            octave_double_by_breathing[breathing_status]["no_octave_double"] += 1

        audit_rows.append(
            {
                "event_index": event_index,
                "onset_group": onset_group,
                "midi_note_token": midi_note,
                "midi_pitch_class": midi_pc,
                "strict_status": strict_status,
                "breathing_status": breathing_status,
                "dsp_peak_count": _safe_int(dsp.get("peak_count"), 0),
                "dsp_candidate_count": len(norm_candidates),
                "same_pitchclass_candidate_count": len(same_pc_candidates),
                "octave_double_present": int(octave_double_present),
                "truth_present_in_dsp": int(exact_present),
                "truth_rank": truth_rank,
                "truth_score": f"{_safe_float(exact_row.get('score'), 0.0):.9f}" if exact_row else "",
                "best_other_octave_note": str(best_other.get("note_token", "")) if best_other else "",
                "best_other_rank": other_rank,
                "best_other_score": f"{_safe_float(best_other.get('score'), 0.0):.9f}" if best_other else "",
                "truth_led": int(truth_led),
                "shared_harmonic_count": shared_count,
                "shared_harmonic_amp": f"{shared_amp:.9f}",
                "truth_fund_in_other_harmonic": truth_in_other_h,
                "other_fund_in_truth_harmonic": other_in_truth_h,
                "mode": mode,
                "same_pitchclass_notes_json": json.dumps([row["note_token"] for row in same_pc_candidates], ensure_ascii=False),
                "dsp_notes_json": json.dumps([row["note_token"] for row in norm_candidates], ensure_ascii=False),
            }
        )

    out_csv = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "HARMONIC OWNERSHIP AND OCTAVE DOUBLES AUDIT",
        "=" * 72,
        f"midi_event_count               : {len(midi_rows)}",
        f"recognized_event_count         : {len(recognized_rows)}",
        f"dsp_group_count                : {len(dsp_rows)}",
        "",
        "MODE COUNTS",
        "-" * 72,
    ]
    for key in sorted(mode_counter):
        lines.append(f"{key:34s}: {mode_counter[key]}")
    lines.extend(["", "OCTAVE DOUBLE BY BREATHING STATUS", "-" * 72])
    for status in sorted(octave_double_by_breathing):
        lines.append(status or "<empty>")
        counters = octave_double_by_breathing[status]
        for key in sorted(counters):
            lines.append(f"  {key:30s}: {counters[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "harmonic_ownership_octave_double_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "recognized_note_event_audit_csv": args.recognized_note_event_audit_csv,
            "dsp_group_audit_csv": args.dsp_group_audit_csv,
        },
        "result": {
            "mode_counter": dict(mode_counter),
            "octave_double_by_breathing": {k: dict(v) for k, v in octave_double_by_breathing.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
