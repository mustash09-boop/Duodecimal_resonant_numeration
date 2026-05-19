from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
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


def _note_zone(note: str) -> str:
    token = str(note or "").strip()
    if not token:
        return "UNKNOWN"
    octave = token.split(".")[0]
    if octave in {"5", "6", "7"}:
        return "LOW_ZONE"
    if octave in {"8", "9"}:
        return "MID_ZONE"
    if octave in {"A", "B", "C", "11", "12", "13"}:
        return "HIGH_ZONE"
    return "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser(description="Focused audit for piano+organ mixed layer and unresolved field.")
    ap.add_argument("--layered-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    rows = _load_csv(Path(args.layered_csv))
    focus_rows: List[Dict[str, Any]] = []

    dominant_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    zone_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    focus_kind_counts: Counter[str] = Counter()
    piano_window_counts: Counter[str] = Counter()
    organ_window_counts: Counter[str] = Counter()

    for r in rows:
        support = str(r.get("support_instruments", "")).strip()
        dominant = str(r.get("dominant_instrument", "")).strip()
        kind = ""
        if dominant == "UNRESOLVED_FIELD":
            kind = "UNRESOLVED_FIELD"
        elif dominant == "piano" and ("organ" in support.split()):
            kind = "PIANO_WITH_ORGAN_SUPPORT"
        elif dominant == "organ" and ("piano" in support.split()):
            kind = "ORGAN_WITH_PIANO_SUPPORT"
        elif support == "piano+organ" or support == "organ+piano":
            kind = "PIANO_ORGAN_DOUBLE_SUPPORT"
        if not kind:
            continue

        rr = dict(r)
        rr["focus_kind"] = kind
        rr["note_zone"] = _note_zone(r.get("candidate_note", ""))
        focus_rows.append(rr)

        focus_kind_counts[kind] += 1
        dominant_counts[dominant] += 1
        state_counts[str(r.get("dominant_state_layered", ""))] += 1
        zone_counts[rr["note_zone"]] += 1
        support_counts[support or "<NONE>"] += 1
        piano_window_counts[str(r.get("piano_window", ""))] += 1
        organ_window_counts[str(r.get("organ_window", ""))] += 1

    focus_rows.sort(
        key=lambda r: (
            {"PIANO_WITH_ORGAN_SUPPORT": 0, "ORGAN_WITH_PIANO_SUPPORT": 1, "PIANO_ORGAN_DOUBLE_SUPPORT": 2, "UNRESOLVED_FIELD": 3}.get(str(r.get("focus_kind", "")), 9),
            _safe_int(r.get("birth_frame"), 0),
            str(r.get("candidate_note", "")),
        )
    )

    _write_csv(Path(args.out_audit_csv), focus_rows, focus_rows[0].keys() if focus_rows else ["focus_kind"])

    lines = [
        "LAYERED ASSIGNMENT FOCUS AUDIT",
        "=" * 72,
        f"layered_csv  : {args.layered_csv}",
        f"focus_events : {len(focus_rows)}",
        "",
        "focus_kind_counts:",
    ]
    for k, v in sorted(focus_kind_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("dominant_counts:")
    for k, v in sorted(dominant_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("dominant_state_counts:")
    for k, v in sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("note_zone_counts:")
    for k, v in sorted(zone_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("support_counts:")
    for k, v in sorted(support_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("piano_window_counts:")
    for k, v in sorted(piano_window_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("organ_window_counts:")
    for k, v in sorted(organ_window_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {"layered_csv": args.layered_csv},
                "result": {
                    "focus_events": len(focus_rows),
                    "focus_kind_counts": dict(sorted(focus_kind_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "dominant_counts": dict(sorted(dominant_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "dominant_state_counts": dict(sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "note_zone_counts": dict(sorted(zone_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "piano_window_counts": dict(sorted(piano_window_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "organ_window_counts": dict(sorted(organ_window_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
