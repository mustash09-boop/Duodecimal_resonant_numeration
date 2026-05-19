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


def _octave_token(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[0]
    except Exception:
        return ""


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _classify_relation(chain_note: str, midi_note: str) -> str:
    c = _normalize_note(chain_note)
    m = _normalize_note(midi_note)
    if not c or not m:
        return "NO_NOTE"
    if c == m:
        return "EXACT_IDENTITY"
    c_pc = _pitch_class(c)
    m_pc = _pitch_class(m)
    c_oct = _octave_token(c)
    m_oct = _octave_token(m)
    if c_pc == m_pc and c_oct != m_oct:
        return "OCTAVE_CONFUSION"
    if c_oct == m_oct and c_pc != m_pc:
        return "STEP_DRIFT"
    return "FOREIGN_CAPTURE"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare built note chains against MIDI event reality to expose where note identity survives or breaks."
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--controlled-sustain-chains-csv", required=True)
    ap.add_argument("--controlled-sustain-frames-csv", required=True)
    ap.add_argument("--primary-note-chains-csv", required=True)
    ap.add_argument("--branch-analysis-csv", required=True)
    ap.add_argument("--out-chain-audit-csv", required=True)
    ap.add_argument("--out-midi-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    sustain_chains = _load_csv(Path(args.controlled_sustain_chains_csv))
    sustain_frames = _load_csv(Path(args.controlled_sustain_frames_csv))
    primary_chains = _load_csv(Path(args.primary_note_chains_csv))
    branch_rows = _load_csv(Path(args.branch_analysis_csv))

    primary_by_proto = {str(row.get("proto_exciter_id", "")).strip(): row for row in primary_chains}
    branch_by_proto = {str(row.get("proto_exciter_id", "")).strip(): row for row in branch_rows}
    frames_by_proto: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sustain_frames:
        frames_by_proto[str(row.get("proto_exciter_id", "")).strip()].append(row)

    for rows in frames_by_proto.values():
        rows.sort(key=lambda r: _safe_int(r.get("frame_index"), 0))

    chain_audit_rows: list[dict[str, Any]] = []
    chain_status_counter: Counter[str] = Counter()

    # Chain -> MIDI
    for chain in sustain_chains:
        proto_id = str(chain.get("proto_exciter_id", "")).strip()
        chain_note = _normalize_note(chain.get("dominant_note_token", chain.get("anchor_note_token", "")))
        chain_start = _safe_int(chain.get("start_frame"), 0)
        chain_end = _safe_int(chain.get("end_frame"), chain_start)
        best_midi = None
        best_overlap = -1
        best_start_delta = 10**9
        for midi in midi_rows:
            midi_start = _safe_int(midi.get("start_frame60"), 0)
            midi_end = _safe_int(midi.get("end_frame60"), midi_start)
            ov = _overlap(chain_start, chain_end, midi_start, midi_end)
            if ov > best_overlap or (ov == best_overlap and abs(chain_start - midi_start) < best_start_delta):
                best_midi = midi
                best_overlap = ov
                best_start_delta = abs(chain_start - midi_start)

        matched_note = _normalize_note(best_midi.get("expected_note_token", best_midi.get("note_token", ""))) if best_midi else ""
        relation = _classify_relation(chain_note, matched_note)

        frame_rows = frames_by_proto.get(proto_id, [])
        selected_counter = Counter(_normalize_note(r.get("selected_note_token", "")) for r in frame_rows if _normalize_note(r.get("selected_note_token", "")))
        exact_frame_count = selected_counter.get(matched_note, 0)
        pitchclass_frame_count = sum(count for note, count in selected_counter.items() if _pitch_class(note) == _pitch_class(matched_note))
        foreign_frame_count = sum(count for note, count in selected_counter.items() if note and _pitch_class(note) != _pitch_class(matched_note))

        if relation == "EXACT_IDENTITY":
            status = "CHAIN_EXACT_IDENTITY"
        elif relation == "OCTAVE_CONFUSION":
            status = "CHAIN_OCTAVE_CONFUSION"
        elif relation == "STEP_DRIFT":
            status = "CHAIN_STEP_DRIFT"
        else:
            status = "CHAIN_FOREIGN_CAPTURE"
        chain_status_counter[status] += 1

        primary = primary_by_proto.get(proto_id, {})
        branch = branch_by_proto.get(proto_id, {})
        chain_audit_rows.append(
            {
                "proto_exciter_id": proto_id,
                "chain_start_frame": chain_start,
                "chain_end_frame": chain_end,
                "chain_note_token": chain_note,
                "matched_midi_event_index": best_midi.get("event_index", "") if best_midi else "",
                "matched_midi_onset_group": best_midi.get("onset_group", "") if best_midi else "",
                "matched_midi_note_token": matched_note,
                "frame_overlap_with_midi": best_overlap,
                "start_frame_delta": chain_start - _safe_int(best_midi.get("start_frame60"), 0) if best_midi else "",
                "identity_relation": relation,
                "status": status,
                "exact_frame_count": exact_frame_count,
                "pitchclass_frame_count": pitchclass_frame_count,
                "foreign_frame_count": foreign_frame_count,
                "selected_notes_json": json.dumps(dict(selected_counter), ensure_ascii=False, sort_keys=True),
                "phase_counts_json": str(chain.get("phase_counts_json", "")),
                "primary_bridge_resistance": primary.get("bridge_resistance", ""),
                "primary_mean_chain_score": primary.get("mean_chain_score", ""),
                "branch_label": branch.get("branch_label", ""),
                "route_label": branch.get("route_label", ""),
            }
        )

    # MIDI -> best chain
    midi_audit_rows: list[dict[str, Any]] = []
    midi_status_counter: Counter[str] = Counter()
    for midi in midi_rows:
        midi_note = _normalize_note(midi.get("expected_note_token", midi.get("note_token", "")))
        midi_start = _safe_int(midi.get("start_frame60"), 0)
        midi_end = _safe_int(midi.get("end_frame60"), midi_start)
        best_chain = None
        best_overlap = -1
        best_start_delta = 10**9
        for chain in sustain_chains:
            chain_start = _safe_int(chain.get("start_frame"), 0)
            chain_end = _safe_int(chain.get("end_frame"), chain_start)
            ov = _overlap(chain_start, chain_end, midi_start, midi_end)
            if ov > best_overlap or (ov == best_overlap and abs(chain_start - midi_start) < best_start_delta):
                best_chain = chain
                best_overlap = ov
                best_start_delta = abs(chain_start - midi_start)

        chain_note = _normalize_note(best_chain.get("dominant_note_token", best_chain.get("anchor_note_token", ""))) if best_chain else ""
        relation = _classify_relation(chain_note, midi_note)

        # visibility before final chain
        nearby_proto = [
            row for row in branch_rows
            if abs(_safe_int(row.get("start_frame"), 0) - midi_start) <= 3
        ]
        exact_proto = [row for row in nearby_proto if _normalize_note(row.get("coarse_note", "")) == midi_note]
        pitch_proto = [row for row in nearby_proto if _pitch_class(row.get("coarse_note", "")) == _pitch_class(midi_note)]

        if relation == "EXACT_IDENTITY" and best_overlap > 0:
            status = "MIDI_EXACT_CHAIN"
        elif relation == "OCTAVE_CONFUSION" and best_overlap > 0:
            status = "MIDI_OCTAVE_CONFUSION"
        elif relation == "STEP_DRIFT" and best_overlap > 0:
            status = "MIDI_STEP_DRIFT"
        elif relation == "FOREIGN_CAPTURE" and best_overlap > 0:
            status = "MIDI_FOREIGN_CAPTURE"
        elif exact_proto or pitch_proto:
            status = "MIDI_PROTO_VISIBLE_BUT_NOT_CHAINED"
        else:
            status = "MIDI_NOT_COLLECTED"
        midi_status_counter[status] += 1

        midi_audit_rows.append(
            {
                "event_index": midi.get("event_index", ""),
                "onset_group": midi.get("onset_group", ""),
                "start_frame60": midi_start,
                "end_frame60": midi_end,
                "expected_note_token": midi_note,
                "best_chain_proto_exciter_id": best_chain.get("proto_exciter_id", "") if best_chain else "",
                "best_chain_note_token": chain_note,
                "frame_overlap_with_chain": best_overlap,
                "chain_start_delta": _safe_int(best_chain.get("start_frame"), 0) - midi_start if best_chain else "",
                "identity_relation": relation,
                "status": status,
                "nearby_proto_count": len(nearby_proto),
                "exact_proto_count": len(exact_proto),
                "pitch_proto_count": len(pitch_proto),
                "nearby_proto_notes_json": json.dumps(sorted({_normalize_note(r.get('coarse_note', '')) for r in nearby_proto if _normalize_note(r.get('coarse_note', ''))}), ensure_ascii=False),
            }
        )

    out_chain = Path(args.out_chain_audit_csv)
    out_midi = Path(args.out_midi_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_chain.parent.mkdir(parents=True, exist_ok=True)

    if chain_audit_rows:
        with out_chain.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(chain_audit_rows[0].keys()))
            w.writeheader()
            w.writerows(chain_audit_rows)

    if midi_audit_rows:
        with out_midi.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(midi_audit_rows[0].keys()))
            w.writeheader()
            w.writerows(midi_audit_rows)

    lines = [
        "CHAIN VS MIDI IDENTITY AUDIT",
        "=" * 72,
        f"midi_event_count         : {len(midi_rows)}",
        f"sustain_chain_count      : {len(sustain_chains)}",
        "",
        "CHAIN STATUS COUNTS",
        "-" * 72,
    ]
    for key in sorted(chain_status_counter):
        lines.append(f"{key:<28}: {chain_status_counter[key]}")
    lines.extend(["", "MIDI STATUS COUNTS", "-" * 72])
    for key in sorted(midi_status_counter):
        lines.append(f"{key:<28}: {midi_status_counter[key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "chain_vs_midi_identity_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "controlled_sustain_chains_csv": args.controlled_sustain_chains_csv,
            "controlled_sustain_frames_csv": args.controlled_sustain_frames_csv,
            "primary_note_chains_csv": args.primary_note_chains_csv,
            "branch_analysis_csv": args.branch_analysis_csv,
        },
        "result": {
            "chain_status_counter": dict(chain_status_counter),
            "midi_status_counter": dict(midi_status_counter),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
