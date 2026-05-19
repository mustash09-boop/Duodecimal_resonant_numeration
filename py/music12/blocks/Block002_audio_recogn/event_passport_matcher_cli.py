# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _degree(token: str) -> str:
    try:
        return _normalize_note(token).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _note_from_profile_filename(path: Path) -> str:
    left = path.name.split("__note_box_profile", 1)[0]
    raw = left.split("_")[-1]
    return _normalize_note(raw)


def _load_passports(folder: Path) -> Dict[str, Dict[str, Any]]:
    passports = {}

    for p in sorted(folder.glob("*__note_box_profile.csv")):
        note = _note_from_profile_filename(p)
        rows = _load_csv(p)

        strong = set()
        persistent = set()
        echo = set()

        amp_sum = 0.0
        presence_sum = 0.0

        for r in rows:
            token = _normalize_note(r.get("token", ""))
            amp = _safe_float(r.get("mean_amp"), 0.0)
            presence = _safe_float(r.get("presence_ratio"), 0.0)

            amp_sum += amp
            presence_sum += presence

            if amp >= 0.16 and presence >= 0.06:
                strong.add(token)
            if presence >= 0.16:
                persistent.add(token)
            if presence >= 0.10 and amp < 0.05:
                echo.add(token)

        passports[note] = {
            "note": note,
            "degree": _degree(note),
            "strong": strong,
            "persistent": persistent,
            "echo": echo,
            "mean_amp": amp_sum / max(len(rows), 1),
            "mean_presence": presence_sum / max(len(rows), 1),
        }

    return passports


def _find_passport(note: str, passports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n = _normalize_note(note)
    if n in passports:
        return passports[n]

    d = _degree(n)
    for pnote, p in passports.items():
        if _degree(pnote) == d:
            return p

    return {
        "note": n,
        "degree": d,
        "strong": set(),
        "persistent": set(),
        "echo": set(),
        "mean_amp": 0.0,
        "mean_presence": 0.0,
    }


def _parse_path(raw: str) -> List[str]:
    return [_normalize_note(x) for x in str(raw or "").split() if x.strip()]


def _match_event(event: Dict[str, Any], passport: Dict[str, Any]) -> Dict[str, Any]:
    note_path = _parse_path(event.get("note_path", ""))
    states = str(event.get("state_path", "")).split()

    strong_hits = len(set(note_path) & passport["strong"])
    persistent_hits = len(set(note_path) & passport["persistent"])
    echo_hits = len(set(note_path) & passport["echo"])

    duration = _safe_int(event.get("duration_frames"), 0)
    mean_score = _safe_float(event.get("mean_score"), 0.0)
    max_score = _safe_float(event.get("max_score"), 0.0)

    birth_count = _safe_int(event.get("birth_count"), 0)
    active_count = _safe_int(event.get("active_body_count"), 0)
    sustain_count = _safe_int(event.get("sustain_body_count"), 0)
    re_count = _safe_int(event.get("re_excitation_count"), 0)

    score = 0.0
    score += mean_score * 0.35
    score += max_score * 0.15
    score += min(duration / 90.0, 1.0) * 0.20
    score += strong_hits * 0.25
    score += persistent_hits * 0.06
    score -= echo_hits * 0.18
    score += min(active_count + sustain_count, 40) * 0.01
    score += min(birth_count, 2) * 0.08
    score -= max(0, re_count - 2) * 0.03

    if event.get("candidate_note", "") == passport["note"]:
        score += 0.35
    elif _degree(event.get("candidate_note", "")) == passport["degree"]:
        score += 0.12

    if not states:
        lifecycle_shape = "unknown"
    elif re_count >= 2:
        lifecycle_shape = "re_excited"
    elif sustain_count + active_count >= 6:
        lifecycle_shape = "sustained"
    else:
        lifecycle_shape = "short"

    return {
        "event_passport_score": max(score, 0.0),
        "passport_note": passport["note"],
        "strong_hits": strong_hits,
        "persistent_hits": persistent_hits,
        "echo_hits": echo_hits,
        "lifecycle_shape": lifecycle_shape,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Match merged resonance lifecycle events against single-note passports."
    )

    ap.add_argument("--merged_events_csv", required=True)
    ap.add_argument("--passport_folder", required=True)

    ap.add_argument("--out_event_matches_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_match_score", type=float, default=0.75)

    args = ap.parse_args()

    events = _load_csv(Path(args.merged_events_csv))
    passports = _load_passports(Path(args.passport_folder))

    out_rows = []
    readable_rows = []

    matched = 0
    weak = 0

    for ev in events:
        note = _normalize_note(ev.get("candidate_note", ""))
        passport = _find_passport(note, passports)

        m = _match_event(ev, passport)

        status = "MATCH" if m["event_passport_score"] >= args.min_match_score else "WEAK"
        if status == "MATCH":
            matched += 1
        else:
            weak += 1

        row = dict(ev)
        row["matched_passport_note"] = m["passport_note"]
        row["event_passport_score"] = f"{m['event_passport_score']:.9f}"
        row["passport_status"] = status
        row["strong_hits"] = m["strong_hits"]
        row["persistent_hits"] = m["persistent_hits"]
        row["echo_hits"] = m["echo_hits"]
        row["lifecycle_shape"] = m["lifecycle_shape"]

        out_rows.append(row)

        readable_rows.append({
            "event_id": ev.get("merged_event_id", ev.get("event_id", "")),
            "candidate_note": note,
            "matched_passport_note": m["passport_note"],
            "status": status,
            "score": f"{m['event_passport_score']:.3f}",
            "birth_frame": ev.get("birth_frame", ""),
            "end_frame": ev.get("end_frame", ""),
            "duration_frames": ev.get("duration_frames", ""),
            "shape": m["lifecycle_shape"],
            "hits": f"S{m['strong_hits']}/P{m['persistent_hits']}/E{m['echo_hits']}",
        })

    out_rows.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            -_safe_float(r.get("event_passport_score"), 0.0),
        )
    )

    out_csv = Path(args.out_event_matches_csv)
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

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "candidate_note",
                "matched_passport_note",
                "status",
                "score",
                "birth_frame",
                "end_frame",
                "duration_frames",
                "shape",
                "hits",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "event_passport_matcher",
        "inputs": {
            "merged_events_csv": args.merged_events_csv,
            "passport_folder": args.passport_folder,
        },
        "outputs": {
            "event_matches_csv": args.out_event_matches_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_match_score": args.min_match_score,
        },
        "result": {
            "events": len(events),
            "passports_loaded": len(passports),
            "matched_events": matched,
            "weak_events": weak,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "EVENT PASSPORT MATCHER",
        "=" * 72,
        f"merged_events_csv : {args.merged_events_csv}",
        f"passport_folder   : {args.passport_folder}",
        "",
        f"events            : {len(events)}",
        f"passports_loaded  : {len(passports)}",
        f"matched_events    : {matched}",
        f"weak_events       : {weak}",
        "",
        "Principle:",
        "  Compare full resonance event lifecycle against single-note passports.",
        "  Analysis unit is now an event, not a frame.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("event passport matcher complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()