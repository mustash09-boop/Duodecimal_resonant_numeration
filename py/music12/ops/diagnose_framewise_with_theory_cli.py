from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def safe_str(v):
    if v is None:
        return ""
    return str(v).strip()


def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    path = Path(args.in_csv)
    out_txt = Path(args.out_txt)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    root_counter = Counter()
    verdict_counter = Counter()
    chosen_counter = Counter()

    support_vals = []
    consistency_vals = []

    for row in rows:
        root = safe_str(row.get("best_theoretical_root_token"))
        verdict = safe_str(row.get("theoretical_chain_verdict"))
        chosen = safe_str(row.get("chosen_rc_note"))

        if root:
            root_counter[root] += 1

        if verdict:
            verdict_counter[verdict] += 1

        if chosen:
            chosen_counter[chosen] += 1

        support_vals.append(safe_float(row.get("support_hits")))
        consistency_vals.append(safe_float(row.get("spiral_consistency_score")))

    lines = []

    lines.append("ADAPTIVE INPUT DIAGNOSTIC")
    lines.append("=" * 60)
    lines.append(f"rows: {len(rows)}")
    lines.append(f"mean_support: {sum(support_vals)/len(support_vals):.6f}")
    lines.append(f"mean_consistency: {sum(consistency_vals)/len(consistency_vals):.6f}")
    lines.append("")

    lines.append("TOP ROOTS")
    for k, v in root_counter.most_common(10):
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("TOP RC")
    for k, v in chosen_counter.most_common(10):
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("VERDICTS")
    for k, v in verdict_counter.most_common():
        lines.append(f"{k}: {v}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("DONE")


if __name__ == "__main__":
    main()