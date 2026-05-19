from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from music12.core.theoretical_rc_chains12 import (
    THEORETICAL_RC_CHAIN_TABLE_5A_TO_11_1,
    rank_possible_roots_for_observed_supports,
)

from music12.core.spiral12_geometry import (
    parse_token_to_spiral,
    chain_consistency_score,
)


# ============================================================
# DATA
# ============================================================

@dataclass
class Candidate:
    note_token: str
    supports: List[str]


# ============================================================
# LOAD
# ============================================================

def load_candidates(row: dict) -> List[Candidate]:
    raw = (row.get("selected_candidates_json") or "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    out = []

    for item in data:
        note = str(item.get("note_token", "")).strip()
        if not note:
            continue

        supports = [
            str(s.get("matched_note", "")).strip()
            for s in item.get("supports", [])
            if s.get("matched_note")
        ]

        out.append(Candidate(note_token=note, supports=supports))

    return out


# ============================================================
# CORE LOGIC
# ============================================================

def evaluate_candidate(candidate: Candidate):
    table = THEORETICAL_RC_CHAIN_TABLE_5A_TO_11_1

    matches = rank_possible_roots_for_observed_supports(
        chosen_rc_note=candidate.note_token,
        support_tokens=candidate.supports,
        table=table,
        top_k=5,
    )

    results = []

    for m in matches:
        chain = table[m.root_token]
        chain_tokens = list(chain.chain_tokens)

        # берём только supports кандидата
        spirals = [
            parse_token_to_spiral(t)
            for t in candidate.supports
            if parse_token_to_spiral(t)
        ]

        # считаем только по этим точкам
        consistency = chain_consistency_score(spirals)

        match_count = sum(
            1 for t in candidate.supports
            if t in chain_tokens
        )

        verdict = "CHAIN_UNCERTAIN"

        if match_count >= 4 and consistency > 0.7:
            verdict = "CHAIN_CONFIRMED"
        elif match_count >= 3 and consistency > 0.5:
            verdict = "CHAIN_PARTIAL"
        elif match_count >= 1:
            verdict = "CHAIN_WEAK"

        results.append({
            "root": m.root_token,
            "match_count": match_count,
            "consistency": consistency,
            "verdict": verdict,
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framewise_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []

    with open(args.framewise_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out = []

    for r in rows:
        candidates = load_candidates(r)

        frame_results = []

        for c in candidates:
            frame_results.extend(evaluate_candidate(c))

        out.append({
            "frame_index": r.get("frame_index"),
            "results_json": json.dumps(frame_results, ensure_ascii=False)
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "results_json"])
        writer.writeheader()
        writer.writerows(out)

    print("DONE (clean chain mode)")


if __name__ == "__main__":
    main()