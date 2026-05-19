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


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []
    return []


def _classify(
    note: str,
    candidates: list[str],
) -> str:
    norm_note = _normalize_note(note)
    note_pc = _pitch_class(norm_note)
    norm_candidates = [_normalize_note(x) for x in candidates if _normalize_note(x)]
    if norm_note in norm_candidates:
        return "EXACT_IN_OWNERSHIP_WINDOW"
    if any(_pitch_class(x) == note_pc for x in norm_candidates):
        return "PITCHCLASS_IN_OWNERSHIP_WINDOW"
    if norm_candidates:
        return "WRONG_SET_IN_OWNERSHIP_WINDOW"
    return "EMPTY_OWNERSHIP_WINDOW"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit how many MIDI note-events are covered by the early shared ownership window of their musical moment."
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--ownership-window-groups-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--window-frames", type=int, default=5)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    group_rows = _load_csv(Path(args.ownership_window_groups_csv))

    audit_rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    polyphony_counter: dict[int, Counter[str]] = defaultdict(Counter)

    for midi in midi_rows:
        note = _normalize_note(midi.get("expected_note_token", midi.get("note_token", "")))
        start_frame = _safe_int(midi.get("start_frame60"), 0)
        poly = _safe_int(midi.get("onset_polyphony"), 0)

        nearby = [
            row for row in group_rows
            if abs(_safe_int(row.get("anchor_frame"), 0) - start_frame) <= int(args.window_frames)
        ]
        nearby.sort(key=lambda r: abs(_safe_int(r.get("anchor_frame"), 0) - start_frame))
        best = nearby[0] if nearby else None
        candidates = _json_list(best.get("candidate_notes_json", "")) if best else []
        status = _classify(note, candidates)
        status_counter[status] += 1
        polyphony_counter[poly][status] += 1

        exact_rank = ""
        pitch_rank = ""
        if candidates:
            norm_candidates = [_normalize_note(x) for x in candidates]
            if note in norm_candidates:
                exact_rank = norm_candidates.index(note) + 1
            else:
                note_pc = _pitch_class(note)
                for idx, cand in enumerate(norm_candidates, start=1):
                    if _pitch_class(cand) == note_pc:
                        pitch_rank = idx
                        break

        audit_rows.append(
            {
                "event_index": midi.get("event_index", ""),
                "midi_note_token": note,
                "start_frame60": start_frame,
                "onset_polyphony": poly,
                "status": status,
                "matched_onset_group_id": str(best.get("onset_group_id", "")).strip() if best else "",
                "matched_anchor_frame": _safe_int(best.get("anchor_frame"), 0) if best else "",
                "candidate_count": _safe_int(best.get("candidate_count"), 0) if best else 0,
                "exact_rank": exact_rank,
                "pitchclass_rank": pitch_rank,
                "candidate_notes_json": json.dumps(candidates, ensure_ascii=False),
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
        "OWNERSHIP WINDOW NOTE EVENT AUDIT",
        "=" * 72,
        f"midi_event_count          : {len(midi_rows)}",
        "",
        "STATUS COUNTS",
        "-" * 72,
    ]
    for key in sorted(status_counter):
        lines.append(f"{key:30s}: {status_counter[key]}")
    lines.extend(["", "BY ONSET POLYPHONY", "-" * 72])
    for poly in sorted(polyphony_counter):
        lines.append(f"polyphony={poly}")
        for key in sorted(polyphony_counter[poly]):
            lines.append(f"  {key:28s}: {polyphony_counter[poly][key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "ownership_window_note_event_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "ownership_window_groups_csv": args.ownership_window_groups_csv,
        },
        "parameters": {
            "window_frames": int(args.window_frames),
        },
        "result": {
            "status_counter": dict(status_counter),
            "polyphony_counter": {str(k): dict(v) for k, v in polyphony_counter.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
