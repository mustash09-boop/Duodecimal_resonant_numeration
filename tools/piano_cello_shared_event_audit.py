from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _build_part_index(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        track = str(row.get("track_name", "")).strip()
        if track.startswith("Piano"):
            out["piano"].append(row)
        elif track.startswith("Cello"):
            out["cello"].append(row)
    for key in out:
        out[key].sort(key=lambda r: _safe_int(r.get("start_frame60")))
    return out


def _nearby(rows: list[dict[str, str]], start_frame: int, end_frame: int, onset_window: int) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    lo = start_frame - onset_window
    hi = start_frame + onset_window
    for row in rows:
        rs = _safe_int(row.get("start_frame60"))
        re = _safe_int(row.get("end_frame60"))
        if rs > hi + 24:
            break
        if rs < lo - 24:
            continue
        if abs(rs - start_frame) <= onset_window or not (re < start_frame or rs > end_frame):
            found.append(row)
    return found


def _match_note(rows: list[dict[str, str]], note: str) -> list[dict[str, str]]:
    return [r for r in rows if str(r.get("note12", "")).strip() == note]


def _same_pitchclass(note_a: str, note_b: str) -> bool:
    if "." not in note_a or "." not in note_b:
        return note_a == note_b
    return note_a.split(".", 1)[1] == note_b.split(".", 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit shared piano/cello events against MIDI part windows.")
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--midi_parts_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--onset_window", type=int, default=6)
    args = ap.parse_args()

    layered_rows = _read_csv(Path(args.layered_csv))
    event_rows = { _safe_int(r.get("merged_event_id")): r for r in _read_csv(Path(args.events_csv)) }
    midi_rows = _read_csv(Path(args.midi_parts_csv))
    midi_index = _build_part_index(midi_rows)

    audit_rows: list[dict[str, str]] = []
    mode_counts: Counter[str] = Counter()
    ownership_counts: Counter[str] = Counter()

    for row in layered_rows:
        dominant = str(row.get("dominant_instrument", "")).strip()
        support_combo = str(row.get("support_combo_key", "")).strip()
        if dominant not in {"piano", "cello"}:
            continue
        if "cello" not in support_combo and not (dominant == "cello" and "piano" in support_combo):
            continue

        event_id = _safe_int(row.get("merged_event_id"))
        ev = event_rows.get(event_id)
        if not ev:
            continue
        note = str(ev.get("candidate_note", "")).strip()
        birth = _safe_int(ev.get("birth_frame"))
        end = _safe_int(ev.get("end_frame"))
        piano_near = _nearby(midi_index.get("piano", []), birth, end, args.onset_window)
        cello_near = _nearby(midi_index.get("cello", []), birth, end, args.onset_window)
        piano_same = _match_note(piano_near, note)
        cello_same = _match_note(cello_near, note)

        piano_pc = [r for r in piano_near if _same_pitchclass(str(r.get("note12", "")).strip(), note)]
        cello_pc = [r for r in cello_near if _same_pitchclass(str(r.get("note12", "")).strip(), note)]

        if piano_same and cello_same:
            mode = "SAME_NOTE_IN_BOTH_PARTS"
        elif piano_same and cello_pc:
            mode = "PIANO_EXACT_CELLO_PITCHCLASS"
        elif cello_same and piano_pc:
            mode = "CELLO_EXACT_PIANO_PITCHCLASS"
        elif piano_same:
            mode = "PIANO_ONLY_EXACT"
        elif cello_same:
            mode = "CELLO_ONLY_EXACT"
        elif piano_pc and cello_pc:
            mode = "SHARED_PITCHCLASS_ONLY"
        elif piano_pc:
            mode = "PIANO_ONLY_PITCHCLASS"
        elif cello_pc:
            mode = "CELLO_ONLY_PITCHCLASS"
        else:
            mode = "NO_DIRECT_PART_MATCH"

        if piano_same and cello_same:
            piano_len = max(_safe_int(r.get("end_frame60")) - _safe_int(r.get("start_frame60")) for r in piano_same)
            cello_len = max(_safe_int(r.get("end_frame60")) - _safe_int(r.get("start_frame60")) for r in cello_same)
            if cello_len >= piano_len + 12:
                ownership = "PIANO_ATTACK_CELLO_SUSTAIN"
            elif piano_len >= cello_len + 12:
                ownership = "CELLO_ATTACK_PIANO_SUSTAIN"
            else:
                ownership = "SHARED_SIMILAR_DURATION"
        elif dominant == "piano" and "cello" in support_combo:
            ownership = "PIANO_DOMINANT_WITH_CELLO_SUPPORT"
        elif dominant == "cello" and "piano" in support_combo:
            ownership = "CELLO_DOMINANT_WITH_PIANO_SUPPORT"
        else:
            ownership = "UNRESOLVED_SHARED"

        mode_counts[mode] += 1
        ownership_counts[ownership] += 1
        audit_rows.append(
            {
                "merged_event_id": str(event_id),
                "candidate_note": note,
                "birth_frame": str(birth),
                "end_frame": str(end),
                "dominant_instrument": dominant,
                "support_combo_key": support_combo,
                "shared_mode": mode,
                "ownership_mode": ownership,
                "piano_near_count": str(len(piano_near)),
                "cello_near_count": str(len(cello_near)),
                "piano_exact_count": str(len(piano_same)),
                "cello_exact_count": str(len(cello_same)),
                "piano_pitchclass_count": str(len(piano_pc)),
                "cello_pitchclass_count": str(len(cello_pc)),
            }
        )

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as fh:
        fields = list(audit_rows[0].keys()) if audit_rows else [
            "merged_event_id", "candidate_note", "birth_frame", "end_frame",
            "dominant_instrument", "support_combo_key", "shared_mode", "ownership_mode"
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    lines = [
        "PIANO CELLO SHARED EVENT AUDIT",
        "=" * 72,
        f"input_shared_events: {len(audit_rows)}",
        "",
        "shared_mode_counts:",
    ]
    for key, value in mode_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("ownership_mode_counts:")
    for key, value in ownership_counts.most_common():
        lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
