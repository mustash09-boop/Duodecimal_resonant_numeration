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


def _first_json_list(row: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        values = _json_list(row.get(key, "[]"))
        if values:
            return values
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


def _classify_topk(note: str, candidates: list[str], topk: int = 5) -> str:
    norm_note = _normalize_note(note)
    top = [_normalize_note(x) for x in candidates[:topk] if _normalize_note(x)]
    if not top:
        return "EMPTY_TOP5"
    if norm_note in top:
        return "EXACT_TOP5"
    target_pc = _pitch_class(norm_note)
    if any(_pitch_class(x) == target_pc for x in top):
        return "PITCHCLASS_ONLY_TOP5"
    return "NO_PITCHCLASS_TOP5"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit the residual failures of the best two-stage mode, separating cases where "
            "the pitch class survives in top-5 from cases where even the pitch class is gone."
        )
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--two-stage-csv", required=True)
    ap.add_argument("--recognized-note-event-audit-csv", required=True)
    ap.add_argument("--harmonic-ownership-audit-csv", required=True)
    ap.add_argument("--missing-pitchclass-identity-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    two_stage_rows = _load_csv(Path(args.two_stage_csv))
    recognized_rows = _load_csv(Path(args.recognized_note_event_audit_csv))
    harmonic_rows = _load_csv(Path(args.harmonic_ownership_audit_csv))
    missing_rows = _load_csv(Path(args.missing_pitchclass_identity_audit_csv))

    two_stage_by_group = {
        str(row.get("onset_group", "")).strip(): row
        for row in two_stage_rows
    }
    recognized_by_event = {
        str(row.get("event_index", "")).strip(): row
        for row in recognized_rows
    }
    harmonic_by_event = {
        str(row.get("event_index", "")).strip(): row
        for row in harmonic_rows
    }
    missing_by_event = {
        str(row.get("event_index", "")).strip(): row
        for row in missing_rows
    }

    audit_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    breathing_counter: dict[str, Counter[str]] = defaultdict(Counter)
    harmonic_mode_counter: dict[str, Counter[str]] = defaultdict(Counter)
    missing_mode_counter: dict[str, Counter[str]] = defaultdict(Counter)
    register_counter: dict[str, Counter[str]] = defaultdict(Counter)
    polyphony_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for midi in midi_rows:
        event_index = str(midi.get("event_index", "")).strip()
        onset_group = str(midi.get("onset_group", "")).strip()
        note = _normalize_note(midi.get("expected_note_token", midi.get("note_token", "")))
        polyphony = _safe_int(midi.get("onset_polyphony"), 0)
        two_stage = two_stage_by_group.get(onset_group, {})
        merged_notes = _first_json_list(
            two_stage,
            "final_notes_json",
            "merged_topk_json",
            "merged_notes_json",
        )
        top5_class = _classify_topk(note, merged_notes, 5)

        if top5_class == "EXACT_TOP5":
            continue

        rec = recognized_by_event.get(event_index, {})
        harmonic = harmonic_by_event.get(event_index, {})
        missing = missing_by_event.get(event_index, {})
        breathing = str(rec.get("breathing_status", "")).strip()
        harmonic_mode = str(harmonic.get("mode", "")).strip()
        missing_mode = str(missing.get("mode", "")).strip()
        register = _register_band(note)

        class_counter[top5_class] += 1
        breathing_counter[top5_class][breathing] += 1
        harmonic_mode_counter[top5_class][harmonic_mode] += 1
        missing_mode_counter[top5_class][missing_mode or "<none>"] += 1
        register_counter[top5_class][register] += 1
        polyphony_counter[top5_class][f"poly_{polyphony}"] += 1

        audit_rows.append(
            {
                "event_index": event_index,
                "onset_group": onset_group,
                "midi_note_token": note,
                "onset_polyphony": polyphony,
                "register_band": register,
                "top5_class": top5_class,
                "breathing_status": breathing,
                "harmonic_mode": harmonic_mode,
                "missing_mode": missing_mode,
                "main_note": _normalize_note(two_stage.get("main_note", "")),
                "cloud_top_note": _normalize_note(
                    two_stage.get("cloud_top_note", "") or two_stage.get("main_note", "")
                ),
                "merged_notes_json": json.dumps([_normalize_note(x) for x in merged_notes[:5]], ensure_ascii=False),
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
        "TWO STAGE RESIDUAL IDENTITY AUDIT",
        "=" * 72,
        f"case_count                     : {len(audit_rows)}",
        "",
    ]
    for top5_class in sorted(class_counter):
        lines.extend([top5_class, "-" * 72, f"count                          : {class_counter[top5_class]}", "", "breathing_status"])
        for key in sorted(breathing_counter[top5_class]):
            lines.append(f"  {key:30s}: {breathing_counter[top5_class][key]}")
        lines.extend(["", "harmonic_mode"])
        for key in sorted(harmonic_mode_counter[top5_class]):
            lines.append(f"  {key:30s}: {harmonic_mode_counter[top5_class][key]}")
        lines.extend(["", "missing_mode"])
        for key in sorted(missing_mode_counter[top5_class]):
            lines.append(f"  {key:30s}: {missing_mode_counter[top5_class][key]}")
        lines.extend(["", "register_band"])
        for key in sorted(register_counter[top5_class]):
            lines.append(f"  {key:30s}: {register_counter[top5_class][key]}")
        lines.extend(["", "onset_polyphony"])
        for key in sorted(polyphony_counter[top5_class]):
            lines.append(f"  {key:30s}: {polyphony_counter[top5_class][key]}")
        lines.append("")

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "two_stage_residual_identity_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "two_stage_csv": args.two_stage_csv,
            "recognized_note_event_audit_csv": args.recognized_note_event_audit_csv,
            "harmonic_ownership_audit_csv": args.harmonic_ownership_audit_csv,
            "missing_pitchclass_identity_audit_csv": args.missing_pitchclass_identity_audit_csv,
        },
        "result": {
            "class_counter": dict(class_counter),
            "breathing_counter": {k: dict(v) for k, v in breathing_counter.items()},
            "harmonic_mode_counter": {k: dict(v) for k, v in harmonic_mode_counter.items()},
            "missing_mode_counter": {k: dict(v) for k, v in missing_mode_counter.items()},
            "register_counter": {k: dict(v) for k, v in register_counter.items()},
            "polyphony_counter": {k: dict(v) for k, v in polyphony_counter.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
