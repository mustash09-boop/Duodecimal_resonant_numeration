from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _score(row: Dict[str, Any]) -> float:
    body_ratio = _safe_float(row.get("instrument_body_support_ratio"), 0.0)
    sec_ratio = _safe_float(row.get("instrument_secondary_support_ratio"), 0.0)
    body_hits = _safe_int(row.get("instrument_total_body_hits"), 0)
    exact_frames = _safe_int(row.get("instrument_exact_family_frames"), 0)
    return body_ratio * 10.0 + sec_ratio * 2.5 + min(body_hits, 12) * 0.2 + min(exact_frames, 12) * 0.1


def _window_bonus(row: Dict[str, Any]) -> float:
    label = str(row.get("target_window_class", ""))
    if label == "TARGET_ONLY_WINDOW":
        return 1.0
    if label == "MIXED_WINDOW":
        return 0.4
    if label == "OTHER_ONLY_WINDOW":
        return -0.2
    return 0.0


def _winner_state(best: Tuple[str, float], second: Tuple[str, float]) -> str:
    gap = best[1] - second[1]
    if gap >= 2.5:
        return "CLEAR_WINNER"
    if gap >= 1.0:
        return "LEANING_WINNER"
    return "AMBIGUOUS"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple instrument affinity audits and pick per-event winners.")
    ap.add_argument("--piano-audit-csv", required=True)
    ap.add_argument("--violin-audit-csv", required=True)
    ap.add_argument("--cello-audit-csv", required=True)
    ap.add_argument("--organ-audit-csv", default="")
    ap.add_argument("--out-compare-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    sources = {
        "piano": _load_csv(Path(args.piano_audit_csv)),
        "violin": _load_csv(Path(args.violin_audit_csv)),
        "cello": _load_csv(Path(args.cello_audit_csv)),
    }
    if str(args.organ_audit_csv).strip():
        sources["organ"] = _load_csv(Path(args.organ_audit_csv))

    by_event: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for name, rows in sources.items():
        for r in rows:
            by_event[str(r.get("merged_event_id", ""))][name] = r

    out_rows: List[Dict[str, Any]] = []
    winner_counts: Counter[str] = Counter()
    winner_state_counts: Counter[str] = Counter()
    window_alignment_counts: Counter[str] = Counter()

    for event_id, bag in sorted(by_event.items(), key=lambda kv: _safe_int(kv[0], 0)):
        scores: List[Tuple[str, float]] = []
        base_row = next(iter(bag.values()))
        ordered_names = [name for name in ("piano", "violin", "cello", "organ") if name in sources]
        for name in ordered_names:
            row = bag.get(name)
            if row is None:
                scores.append((name, -9999.0))
                continue
            scores.append((name, _score(row) + _window_bonus(row)))

        scores.sort(key=lambda x: (-x[1], x[0]))
        best = scores[0]
        second = scores[1]
        state = _winner_state(best, second)

        best_row = bag.get(best[0], base_row)
        best_window = str(best_row.get("target_window_class", ""))
        if best_window == "TARGET_ONLY_WINDOW":
            window_alignment = "WINNER_IN_OWN_TARGET_WINDOW"
        elif best_window == "MIXED_WINDOW":
            window_alignment = "WINNER_IN_MIXED_WINDOW"
        elif best_window == "OTHER_ONLY_WINDOW":
            window_alignment = "WINNER_OUTSIDE_TARGET_WINDOW"
        else:
            window_alignment = "WINNER_IN_EMPTY_WINDOW"

        out = {
            "merged_event_id": event_id,
            "candidate_note": str(base_row.get("candidate_note", "")),
            "birth_frame": str(base_row.get("birth_frame", "")),
            "winner_instrument": best[0],
            "winner_score": f"{best[1]:.9f}",
            "runner_up_instrument": second[0],
            "runner_up_score": f"{second[1]:.9f}",
            "winner_state": state,
            "winner_window_alignment": window_alignment,
        }
        for name in ordered_names:
            row = bag.get(name, {})
            out[f"{name}_score"] = f"{next((s for n, s in scores if n == name), -9999.0):.9f}"
            out[f"{name}_affinity"] = str(row.get("instrument_affinity_class", ""))
            out[f"{name}_window"] = str(row.get("target_window_class", ""))
            out[f"{name}_body_ratio"] = str(row.get("instrument_body_support_ratio", "0"))
            out[f"{name}_secondary_ratio"] = str(row.get("instrument_secondary_support_ratio", "0"))
        out_rows.append(out)

        winner_counts[best[0]] += 1
        winner_state_counts[state] += 1
        window_alignment_counts[window_alignment] += 1

    _write_csv(Path(args.out_compare_csv), out_rows, out_rows[0].keys())

    lines = [
        "MULTI INSTRUMENT AFFINITY COMPARE",
        "=" * 72,
        f"input_events: {len(out_rows)}",
        "",
        "winner_counts:",
    ]
    for k, v in sorted(winner_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("winner_state_counts:")
    for k, v in sorted(winner_state_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("winner_window_alignment_counts:")
    for k, v in sorted(window_alignment_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "piano_audit_csv": args.piano_audit_csv,
                    "violin_audit_csv": args.violin_audit_csv,
                    "cello_audit_csv": args.cello_audit_csv,
                    "organ_audit_csv": args.organ_audit_csv,
                },
                "result": {
                    "input_events": len(out_rows),
                    "winner_counts": dict(sorted(winner_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "winner_state_counts": dict(sorted(winner_state_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "winner_window_alignment_counts": dict(sorted(window_alignment_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
