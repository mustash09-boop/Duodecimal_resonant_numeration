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


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
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


def _register_band(note: str) -> str:
    octave = str(_normalize_note(note).split(".", 1)[0])
    if octave in {"6", "7"}:
        return "low_zone"
    if octave in {"8", "9"}:
        return "mid_zone"
    if octave in {"A", "B", "C", "D", "E", "F"}:
        return "high_zone"
    return "other_zone"


def _ownership_status(note: str, candidates: list[str]) -> str:
    norm_note = _normalize_note(note)
    norm_candidates = [_normalize_note(x) for x in candidates if _normalize_note(x)]
    if norm_note in norm_candidates:
        return "OWN_EXACT"
    target_pc = _pitch_class(norm_note)
    if any(_pitch_class(x) == target_pc for x in norm_candidates):
        return "OWN_PITCHCLASS"
    if norm_candidates:
        return "OWN_WRONG"
    return "OWN_EMPTY"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Break down the two main failure layers after octave-double audit: "
            "NO_PITCHCLASS_TRACE and SINGLE_OCTAVE_PITCHCLASS."
        )
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--harmonic-ownership-audit-csv", required=True)
    ap.add_argument("--ownership-window-groups-csv", required=True)
    ap.add_argument("--recognized-note-event-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--focus-modes", default="NO_PITCHCLASS_TRACE,SINGLE_OCTAVE_PITCHCLASS")
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    harmonic_rows = _load_csv(Path(args.harmonic_ownership_audit_csv))
    ownership_rows = _load_csv(Path(args.ownership_window_groups_csv))
    recognized_rows = _load_csv(Path(args.recognized_note_event_audit_csv))

    ownership_by_group = {
        str(row.get("onset_group_id", "")).strip(): row
        for row in ownership_rows
    }
    recognized_by_event = {
        str(row.get("event_index", "")).strip(): row
        for row in recognized_rows
    }
    midi_by_event = {
        str(row.get("event_index", "")).strip(): row
        for row in midi_rows
    }

    focus_modes = {part.strip() for part in str(args.focus_modes).split(",") if part.strip()}
    audit_rows: list[dict[str, Any]] = []

    mode_counter: Counter[str] = Counter()
    breathing_counter: dict[str, Counter[str]] = defaultdict(Counter)
    ownership_counter: dict[str, Counter[str]] = defaultdict(Counter)
    register_counter: dict[str, Counter[str]] = defaultdict(Counter)
    pitchclass_counter: dict[str, Counter[str]] = defaultdict(Counter)
    polyphony_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for row in harmonic_rows:
        mode = str(row.get("mode", "")).strip()
        if mode not in focus_modes:
            continue

        event_index = str(row.get("event_index", "")).strip()
        onset_group = str(row.get("onset_group", "")).strip()
        midi = midi_by_event.get(event_index, {})
        rec = recognized_by_event.get(event_index, {})
        ownership = ownership_by_group.get(onset_group, {})

        midi_note = _normalize_note(row.get("midi_note_token", "") or midi.get("expected_note_token", midi.get("note_token", "")))
        breath = str(row.get("breathing_status", "") or rec.get("breathing_status", "")).strip()
        strict = str(row.get("strict_status", "") or rec.get("strict_status", "")).strip()
        polyphony = _safe_int(midi.get("onset_polyphony", rec.get("onset_polyphony", 0)), 0)
        own_candidates = _json_list(ownership.get("candidate_notes_json", "[]"))
        own_status = _ownership_status(midi_note, own_candidates)
        register = _register_band(midi_note)
        pitchclass = _pitch_class(midi_note)

        mode_counter[mode] += 1
        breathing_counter[mode][breath] += 1
        ownership_counter[mode][own_status] += 1
        register_counter[mode][register] += 1
        pitchclass_counter[mode][pitchclass] += 1
        polyphony_counter[mode][f"poly_{polyphony}"] += 1

        audit_rows.append(
            {
                "event_index": event_index,
                "onset_group": onset_group,
                "mode": mode,
                "midi_note_token": midi_note,
                "pitch_class": pitchclass,
                "register_band": register,
                "onset_polyphony": polyphony,
                "strict_status": strict,
                "breathing_status": breath,
                "ownership_status": own_status,
                "ownership_candidate_count": _safe_int(ownership.get("candidate_count"), 0),
                "ownership_top_note": _normalize_note(ownership.get("top_note_token", "")),
                "same_pitchclass_candidate_count": _safe_int(row.get("same_pitchclass_candidate_count"), 0),
                "truth_present_in_dsp": _safe_int(row.get("truth_present_in_dsp"), 0),
                "best_other_octave_note": _normalize_note(row.get("best_other_octave_note", "")),
                "dsp_notes_json": row.get("dsp_notes_json", "[]"),
                "ownership_notes_json": json.dumps([_normalize_note(x) for x in own_candidates], ensure_ascii=False),
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
        "MISSING PITCHCLASS / SINGLE OCTAVE IDENTITY AUDIT",
        "=" * 72,
        f"focus_modes                    : {', '.join(sorted(focus_modes))}",
        f"case_count                     : {len(audit_rows)}",
        "",
    ]
    for mode in sorted(mode_counter):
        lines.extend([mode, "-" * 72, f"count                          : {mode_counter[mode]}", "", "breathing_status"])
        for key in sorted(breathing_counter[mode]):
            lines.append(f"  {key:30s}: {breathing_counter[mode][key]}")
        lines.extend(["", "ownership_status"])
        for key in sorted(ownership_counter[mode]):
            lines.append(f"  {key:30s}: {ownership_counter[mode][key]}")
        lines.extend(["", "register_band"])
        for key in sorted(register_counter[mode]):
            lines.append(f"  {key:30s}: {register_counter[mode][key]}")
        lines.extend(["", "top pitch classes"])
        for key, value in pitchclass_counter[mode].most_common(12):
            lines.append(f"  {key:30s}: {value}")
        lines.extend(["", "onset_polyphony"])
        for key in sorted(polyphony_counter[mode]):
            lines.append(f"  {key:30s}: {polyphony_counter[mode][key]}")
        lines.append("")

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "missing_pitchclass_identity_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "harmonic_ownership_audit_csv": args.harmonic_ownership_audit_csv,
            "ownership_window_groups_csv": args.ownership_window_groups_csv,
            "recognized_note_event_audit_csv": args.recognized_note_event_audit_csv,
        },
        "focus_modes": sorted(focus_modes),
        "result": {
            "mode_counter": dict(mode_counter),
            "breathing_counter": {k: dict(v) for k, v in breathing_counter.items()},
            "ownership_counter": {k: dict(v) for k, v in ownership_counter.items()},
            "register_counter": {k: dict(v) for k, v in register_counter.items()},
            "pitchclass_counter": {k: dict(v) for k, v in pitchclass_counter.items()},
            "polyphony_counter": {k: dict(v) for k, v in polyphony_counter.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
