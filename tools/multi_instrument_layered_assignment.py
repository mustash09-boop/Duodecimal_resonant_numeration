from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _has_structural_presence(row: Dict[str, Any], name: str) -> bool:
    affinity = str(row.get(f"{name}_affinity", ""))
    score = _safe_float(row.get(f"{name}_score"), 0.0)
    body = _safe_float(row.get(f"{name}_body_ratio"), 0.0)
    sec = _safe_float(row.get(f"{name}_secondary_ratio"), 0.0)
    if affinity in {"HIGH_BODY_AFFINITY", "MEDIUM_BODY_AFFINITY", "SECONDARY_ONLY_AFFINITY"}:
        return True
    if body > 0.0 or sec > 0.0:
        return True
    if affinity == "WEAK_INSTRUMENT_TRACE" and score >= 1.2:
        return True
    return False


def _candidate_tuple(row: Dict[str, Any], name: str) -> Tuple[str, float, str, str]:
    return (
        name,
        _safe_float(row.get(f"{name}_score"), 0.0),
        str(row.get(f"{name}_affinity", "")),
        str(row.get(f"{name}_window", "")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build layered instrument assignment: dominant + support instruments.")
    ap.add_argument("--compare-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    args = ap.parse_args()

    rows = _load_csv(Path(args.compare_csv))
    instruments = [name for name in ("piano", "violin", "cello", "organ") if f"{name}_score" in rows[0]]

    out_rows: List[Dict[str, Any]] = []
    dominant_counts: Counter[str] = Counter()
    support_count_hist: Counter[int] = Counter()
    support_combo_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()

    for row in rows:
        candidates = [_candidate_tuple(row, name) for name in instruments]
        candidates.sort(key=lambda x: (-x[1], x[0]))

        structurals = [c for c in candidates if _has_structural_presence(row, c[0])]
        if not structurals:
            dominant = "UNRESOLVED_FIELD"
            dominant_score = candidates[0][1]
            dominant_state = "NO_STRUCTURAL_OWNER"
            supports: List[str] = []
        else:
            top = structurals[0]
            second_struct = structurals[1] if len(structurals) > 1 else ("", -9999.0, "", "")
            dominant = top[0]
            dominant_score = top[1]
            gap = top[1] - second_struct[1]

            top_window = top[3]
            if top_window == "TARGET_ONLY_WINDOW" and gap >= 1.0:
                dominant_state = "OWN_WINDOW_DOMINANT"
            elif gap >= 2.0:
                dominant_state = "CLEAR_DOMINANT"
            elif gap >= 0.8:
                dominant_state = "LEANING_DOMINANT"
            else:
                dominant_state = "MIXED_DOMINANT"

            supports = []
            for cand in structurals[1:]:
                name, score, affinity, window = cand
                if score >= top[1] - 1.2:
                    supports.append(name)
                    continue
                if window in {"TARGET_ONLY_WINDOW", "MIXED_WINDOW"} and score >= 1.2:
                    supports.append(name)
                    continue
                if affinity == "SECONDARY_ONLY_AFFINITY" and score >= 1.5:
                    supports.append(name)
                    continue

        rr = dict(row)
        rr["dominant_instrument"] = dominant
        rr["dominant_score"] = f"{dominant_score:.9f}"
        rr["dominant_state_layered"] = dominant_state
        rr["support_instruments"] = " ".join(supports)
        rr["support_count"] = str(len(supports))
        rr["support_combo_key"] = "+".join(supports) if supports else "<NONE>"
        out_rows.append(rr)

        dominant_counts[dominant] += 1
        support_count_hist[len(supports)] += 1
        support_combo_counts[rr["support_combo_key"]] += 1
        state_counts[dominant_state] += 1

    _write_csv(Path(args.out_csv), out_rows, out_rows[0].keys())

    lines = [
        "MULTI INSTRUMENT LAYERED ASSIGNMENT",
        "=" * 72,
        f"input_events: {len(rows)}",
        "",
        "dominant_counts:",
    ]
    for k, v in sorted(dominant_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("dominant_state_counts:")
    for k, v in sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("support_count_histogram:")
    for k, v in sorted(support_count_hist.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("top_support_combos:")
    for k, v in sorted(support_combo_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
        lines.append(f"  {k}: {v}")

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {"compare_csv": args.compare_csv},
                "result": {
                    "input_events": len(rows),
                    "dominant_counts": dict(sorted(dominant_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "dominant_state_counts": dict(sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                    "support_count_histogram": dict(sorted(support_count_hist.items())),
                    "top_support_combos": dict(sorted(support_combo_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
