from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def _pitch_class(note: str) -> str:
    s = str(note or "").strip()
    try:
        return s.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _counter_lines(counter: Counter[tuple[str, str]], limit: int = 12) -> list[str]:
    lines: list[str] = []
    for (src, dst), count in counter.most_common(limit):
        lines.append(f"  {src:>2s} -> {dst:<2s}: {count}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze which foreign pitch classes most often replace the correct pitch class "
            "in residual NO_PITCHCLASS_TOP5 note events."
        )
    )
    ap.add_argument("--residual-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    rows = _load_csv(Path(args.residual_audit_csv))
    rows = [row for row in rows if str(row.get("top5_class", "")).strip() == "NO_PITCHCLASS_TOP5"]

    audit_rows: list[dict[str, Any]] = []
    pair_counter: Counter[tuple[str, str]] = Counter()
    zone_counter: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    poly_counter: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    breathing_counter: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    harmonic_counter: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    for row in rows:
        truth_pc = _pitch_class(row.get("midi_note_token", ""))
        main_note = str(row.get("main_note", "")).strip()
        main_pc = _pitch_class(main_note)
        merged_notes = _json_list(row.get("merged_notes_json", "[]"))
        merged_pcs = []
        for note in merged_notes:
            pc = _pitch_class(note)
            if pc and pc not in merged_pcs:
                merged_pcs.append(pc)

        if not truth_pc or not main_pc or truth_pc == main_pc:
            continue

        pair = (truth_pc, main_pc)
        pair_counter[pair] += 1
        zone_counter[str(row.get("register_band", "")).strip()][pair] += 1
        poly_counter[f"poly_{str(row.get('onset_polyphony', '')).strip()}"][pair] += 1
        breathing_counter[str(row.get("breathing_status", "")).strip()][pair] += 1
        harmonic_counter[str(row.get("harmonic_mode", "")).strip()][pair] += 1

        audit_rows.append(
            {
                "event_index": str(row.get("event_index", "")).strip(),
                "onset_group": str(row.get("onset_group", "")).strip(),
                "truth_pitch_class": truth_pc,
                "captured_pitch_class": main_pc,
                "register_band": str(row.get("register_band", "")).strip(),
                "onset_polyphony": str(row.get("onset_polyphony", "")).strip(),
                "breathing_status": str(row.get("breathing_status", "")).strip(),
                "harmonic_mode": str(row.get("harmonic_mode", "")).strip(),
                "main_note": main_note,
                "merged_pitch_classes_json": json.dumps(merged_pcs, ensure_ascii=False),
                "merged_notes_json": json.dumps(merged_notes, ensure_ascii=False),
            }
        )

    out_audit = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_audit.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_audit.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "RESIDUAL PITCHCLASS SUBSTITUTION AUDIT",
        "=" * 72,
        f"case_count                     : {len(audit_rows)}",
        "",
        "TOP MAIN SUBSTITUTIONS",
        "-" * 72,
        *_counter_lines(pair_counter, 16),
        "",
    ]

    for title, counters in [
        ("BY REGISTER BAND", zone_counter),
        ("BY POLYPHONY", poly_counter),
        ("BY BREATHING STATUS", breathing_counter),
        ("BY HARMONIC MODE", harmonic_counter),
    ]:
        lines.extend([title, "-" * 72])
        for key in sorted(counters):
            lines.append(key or "<none>")
            lines.extend(_counter_lines(counters[key], 8))
            lines.append("")

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "residual_pitchclass_substitution_audit",
        "inputs": {
            "residual_audit_csv": args.residual_audit_csv,
        },
        "result": {
            "pair_counter": {f"{a}->{b}": c for (a, b), c in pair_counter.items()},
            "zone_counter": {
                zone: {f"{a}->{b}": c for (a, b), c in counter.items()}
                for zone, counter in zone_counter.items()
            },
            "poly_counter": {
                poly: {f"{a}->{b}": c for (a, b), c in counter.items()}
                for poly, counter in poly_counter.items()
            },
            "breathing_counter": {
                key: {f"{a}->{b}": c for (a, b), c in counter.items()}
                for key, counter in breathing_counter.items()
            },
            "harmonic_counter": {
                key: {f"{a}->{b}": c for (a, b), c in counter.items()}
                for key, counter in harmonic_counter.items()
            },
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
