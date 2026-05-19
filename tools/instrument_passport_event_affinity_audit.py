from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


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


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _split_tokens(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split() if x.strip()]


def _note_root(token: str) -> str:
    t = str(token or "").strip()
    if not t:
        return ""
    apos = t.find("'")
    if apos >= 0:
        return t[: apos + 1] + "-"
    return t


def _extract_tokens(passport_source: Path) -> tuple[Set[str], Set[str], str]:
    if passport_source.suffix.lower() == ".csv":
        rows = _load_csv(passport_source)
        if not rows:
            raise SystemExit("Empty passport source csv")
        row0 = rows[0]
        body_tokens = set(_split_tokens(row0.get("body_tokens", "")))
        secondary_tokens = set(_split_tokens(row0.get("secondary_tokens", "")))
        instrument_name = str(row0.get("passport_name", passport_source.stem)).strip() or passport_source.stem
        return body_tokens, secondary_tokens, instrument_name

    data = json.loads(passport_source.read_text(encoding="utf-8"))
    instrument_name = str(data.get("instrument_name", passport_source.stem)).strip() or passport_source.stem
    if "body_tokens" in data or "secondary_tokens" in data:
        body_tokens = set(_split_tokens(data.get("body_tokens", "")))
        secondary_tokens = set(_split_tokens(data.get("secondary_tokens", "")))
        return body_tokens, secondary_tokens, instrument_name

    body_tokens = {
        str(x.get("cluster_token", "")).strip()
        for x in data.get("box_resonance_top", [])
        if str(x.get("cluster_token", "")).strip()
    }
    secondary_tokens = {
        str(x.get("cluster_token", "")).strip()
        for x in data.get("box_breath_top", [])
        if str(x.get("cluster_token", "")).strip()
    }
    return body_tokens, secondary_tokens, instrument_name


def _frame_target_window_map(
    midi_rows: List[Dict[str, Any]],
    target_prefixes: List[str],
    window: int,
) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = defaultdict(lambda: {"target": 0, "other": 0})
    prefixes = [p.strip() for p in target_prefixes if p.strip()]
    for r in midi_rows:
        frame = _safe_int(r.get("start_frame60"), -999999)
        if frame < 0:
            continue
        track_name = str(r.get("track_name", "")).strip()
        is_target = any(track_name.startswith(pref) for pref in prefixes)
        for f in range(frame - window, frame + window + 1):
            if is_target:
                out[f]["target"] += 1
            else:
                out[f]["other"] += 1
    return out


def _window_label(target_hits: int, other_hits: int) -> str:
    if target_hits > 0 and other_hits == 0:
        return "TARGET_ONLY_WINDOW"
    if target_hits > 0 and other_hits > 0:
        return "MIXED_WINDOW"
    if target_hits == 0 and other_hits > 0:
        return "OTHER_ONLY_WINDOW"
    return "EMPTY_WINDOW"


def _affinity_label(body_ratio: float, secondary_ratio: float, body_hits: int, secondary_hits: int) -> str:
    if body_ratio >= 0.35 or body_hits >= 6:
        return "HIGH_BODY_AFFINITY"
    if body_ratio >= 0.15 or body_hits >= 3:
        return "MEDIUM_BODY_AFFINITY"
    if secondary_ratio >= 0.25 or secondary_hits >= 4:
        return "SECONDARY_ONLY_AFFINITY"
    if body_hits > 0 or secondary_hits > 0:
        return "WEAK_INSTRUMENT_TRACE"
    return "NO_INSTRUMENT_TRACE"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit event affinity against a chosen instrument passport and a chosen target MIDI part window."
    )
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--passport-source", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--target-track-prefixes", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--midi-window-frames", type=int, default=3)
    args = ap.parse_args()

    events = _load_csv(Path(args.events_csv))
    families = _load_csv(Path(args.micro_families_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    body_tokens, secondary_tokens, instrument_name = _extract_tokens(Path(args.passport_source))
    body_root_notes = {_note_root(t) for t in body_tokens}
    secondary_root_notes = {_note_root(t) for t in secondary_tokens}
    target_prefixes = [x.strip() for x in str(args.target_track_prefixes).split(",") if x.strip()]

    family_rows_by_frame_and_note: Dict[tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in families:
        frame = _safe_int(r.get("frame_index"), -1)
        note = str(r.get("family_root_note_micro", "")).strip()
        if frame >= 0 and note:
            family_rows_by_frame_and_note[(frame, note)].append(r)

    midi_window_map = _frame_target_window_map(midi_rows, target_prefixes, args.midi_window_frames)

    out_rows: List[Dict[str, Any]] = []
    affinity_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    joint_counts: Counter[tuple[str, str]] = Counter()

    for ev in events:
        candidate_note = str(ev.get("candidate_note", "")).strip()
        birth = _safe_int(ev.get("birth_frame"), 0)
        end = _safe_int(ev.get("end_frame"), birth)
        frame_count = max(_safe_int(ev.get("frame_count"), 0), 1)

        exact_family_frames = 0
        body_support_frames = 0
        secondary_support_frames = 0
        total_body_hits = 0
        total_secondary_hits = 0
        max_family_score = 0.0
        body_root_frame_hits = 0
        secondary_root_frame_hits = 0

        for frame in range(birth, end + 1):
            rows = family_rows_by_frame_and_note.get((frame, candidate_note), [])
            if not rows:
                continue
            best = max(rows, key=lambda r: _safe_float(r.get("family_score"), 0.0))
            exact_family_frames += 1
            max_family_score = max(max_family_score, _safe_float(best.get("family_score"), 0.0))
            members = _split_tokens(best.get("root_micro_members", ""))
            body_hits = sum(1 for m in members if m in body_tokens)
            secondary_hits = sum(1 for m in members if m in secondary_tokens)
            total_body_hits += body_hits
            total_secondary_hits += secondary_hits
            if body_hits > 0:
                body_support_frames += 1
            if secondary_hits > 0:
                secondary_support_frames += 1
            if any(_note_root(m) in body_root_notes for m in members):
                body_root_frame_hits += 1
            if any(_note_root(m) in secondary_root_notes for m in members):
                secondary_root_frame_hits += 1

        target_hits = 0
        other_hits = 0
        for frame in range(birth, end + 1):
            w = midi_window_map.get(frame)
            if not w:
                continue
            target_hits += int(w["target"] > 0)
            other_hits += int(w["other"] > 0)

        window_label = _window_label(target_hits, other_hits)
        body_ratio = body_support_frames / frame_count
        secondary_ratio = secondary_support_frames / frame_count
        affinity_label = _affinity_label(body_ratio, secondary_ratio, total_body_hits, total_secondary_hits)

        rr = dict(ev)
        rr["affinity_instrument"] = instrument_name
        rr["target_track_prefixes"] = ",".join(target_prefixes)
        rr["instrument_exact_family_frames"] = exact_family_frames
        rr["instrument_body_support_frames"] = body_support_frames
        rr["instrument_secondary_support_frames"] = secondary_support_frames
        rr["instrument_body_root_frame_hits"] = body_root_frame_hits
        rr["instrument_secondary_root_frame_hits"] = secondary_root_frame_hits
        rr["instrument_total_body_hits"] = total_body_hits
        rr["instrument_total_secondary_hits"] = total_secondary_hits
        rr["instrument_body_support_ratio"] = f"{body_ratio:.9f}"
        rr["instrument_secondary_support_ratio"] = f"{secondary_ratio:.9f}"
        rr["instrument_max_family_score"] = f"{max_family_score:.9f}"
        rr["instrument_affinity_class"] = affinity_label
        rr["target_window_class"] = window_label
        out_rows.append(rr)

        affinity_counts[affinity_label] += 1
        window_counts[window_label] += 1
        joint_counts[(affinity_label, window_label)] += 1

    out_rows.sort(
        key=lambda r: (
            {"HIGH_BODY_AFFINITY": 0, "MEDIUM_BODY_AFFINITY": 1, "SECONDARY_ONLY_AFFINITY": 2, "WEAK_INSTRUMENT_TRACE": 3, "NO_INSTRUMENT_TRACE": 4}.get(str(r.get("instrument_affinity_class", "")), 9),
            -_safe_float(r.get("instrument_body_support_ratio"), 0.0),
            _safe_int(r.get("birth_frame"), 0),
        )
    )

    _write_csv(Path(args.out_audit_csv), out_rows, out_rows[0].keys())

    lines = [
        "INSTRUMENT PASSPORT EVENT AFFINITY AUDIT",
        "=" * 72,
        f"instrument_name      : {instrument_name}",
        f"events_csv           : {args.events_csv}",
        f"micro_families_csv   : {args.micro_families_csv}",
        f"passport_source      : {args.passport_source}",
        f"midi_events_csv      : {args.midi_events_csv}",
        f"target_track_prefixes: {','.join(target_prefixes)}",
        f"input_events         : {len(events)}",
        "",
        "instrument_affinity_counts:",
    ]
    for k, v in sorted(affinity_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("target_window_counts:")
    for k, v in sorted(window_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("affinity_vs_target_window:")
    for (aff, win), v in sorted(joint_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        lines.append(f"  {aff} | {win}: {v}")

    high_or_med = [
        r for r in out_rows
        if str(r.get("instrument_affinity_class")) in {"HIGH_BODY_AFFINITY", "MEDIUM_BODY_AFFINITY"}
    ]
    target_friendly = [
        r for r in high_or_med
        if str(r.get("target_window_class")) in {"TARGET_ONLY_WINDOW", "MIXED_WINDOW"}
    ]
    lines.extend(
        [
            "",
            f"high_or_medium_affinity        : {len(high_or_med)}",
            f"high_or_medium_in_target_window: {len(target_friendly)}",
        ]
    )

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "events_csv": args.events_csv,
                    "micro_families_csv": args.micro_families_csv,
                    "passport_source": args.passport_source,
                    "midi_events_csv": args.midi_events_csv,
                    "target_track_prefixes": target_prefixes,
                },
                "parameters": {
                    "midi_window_frames": args.midi_window_frames,
                    "body_token_count": len(body_tokens),
                    "secondary_token_count": len(secondary_tokens),
                    "instrument_name": instrument_name,
                },
                "result": {
                    "input_events": len(events),
                    "instrument_affinity_counts": dict(sorted(affinity_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "target_window_counts": dict(sorted(window_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "high_or_medium_affinity": len(high_or_med),
                    "high_or_medium_in_target_window": len(target_friendly),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
