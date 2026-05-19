# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _tokens(raw: Any) -> Set[str]:
    return {
        x.strip()
        for x in str(raw or "").replace("|", " ").replace(",", " ").split()
        if x.strip()
    }


def _parse_distribution(raw: Any) -> Counter:
    c = Counter()

    for part in str(raw or "").split("|"):
        part = part.strip()
        if ":" not in part:
            continue

        k, v = part.split(":", 1)

        try:
            c[k.strip()] += float(v.strip())
        except Exception:
            pass

    return c


def _counter_similarity(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 0.0

    keys = set(a) | set(b)

    overlap = 0.0
    total = 0.0

    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))

        overlap += min(av, bv)
        total += max(av, bv)

    return overlap / max(total, 1e-9)


def _token_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0

    return len(a & b) / max(len(a | b), 1)


def _passport_score(
    observed: Dict[str, Any],
    passport: Dict[str, Any],
) -> Dict[str, Any]:

    observed_ranges = _parse_distribution(
        observed.get("range_distribution", "")
    )

    passport_ranges = _parse_distribution(
        passport.get("range_distribution", "")
    )

    observed_behaviors = _parse_distribution(
        observed.get("field_behavior_distribution", "")
    )

    passport_behaviors = _parse_distribution(
        passport.get("field_behavior_distribution", "")
    )

    observed_clusters = _parse_distribution(
        observed.get("cluster_kind_distribution", "")
    )

    passport_clusters = _parse_distribution(
        passport.get("cluster_kind_distribution", "")
    )

    observed_body = _tokens(
        observed.get("body_tokens", "")
    )

    passport_body = _tokens(
        passport.get("body_tokens", "")
    )

    observed_secondary = _tokens(
        observed.get("secondary_tokens", "")
    )

    passport_secondary = _tokens(
        passport.get("secondary_tokens", "")
    )

    range_sim = _counter_similarity(
        observed_ranges,
        passport_ranges,
    )

    behavior_sim = _counter_similarity(
        observed_behaviors,
        passport_behaviors,
    )

    cluster_sim = _counter_similarity(
        observed_clusters,
        passport_clusters,
    )

    body_sim = _token_similarity(
        observed_body,
        passport_body,
    )

    secondary_sim = _token_similarity(
        observed_secondary,
        passport_secondary,
    )

    observed_strength = _safe_float(
        observed.get(
            "fingerprint_strength",
            0.0,
        )
    )

    passport_strength = _safe_float(
        passport.get(
            "fingerprint_strength",
            0.0,
        )
    )

    strength_similarity = (
        1.0 -
        min(
            abs(
                observed_strength -
                passport_strength
            ),
            1.0,
        )
    )

    score = 0.0

    score += range_sim * 0.16
    score += behavior_sim * 0.28
    score += cluster_sim * 0.18
    score += body_sim * 0.22
    score += secondary_sim * 0.08
    score += strength_similarity * 0.08

    score = max(
        0.0,
        min(score, 1.0),
    )

    return {
        "score": score,

        "range_similarity":
            range_sim,

        "behavior_similarity":
            behavior_sim,

        "cluster_similarity":
            cluster_sim,

        "body_similarity":
            body_sim,

        "secondary_similarity":
            secondary_sim,

        "strength_similarity":
            strength_similarity,
    }


def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Match observed instrument-body "
            "fingerprints against "
            "Block004 resonance passports."
        )
    )

    ap.add_argument(
        "--observed_fingerprint_csv",
        required=True,
    )

    ap.add_argument(
        "--passport_fingerprint_csv",
        required=True,
    )

    ap.add_argument(
        "--passport_name",
        required=True,
    )

    ap.add_argument(
        "--out_match_csv",
        required=True,
    )

    ap.add_argument(
        "--out_summary_txt",
        required=True,
    )

    args = ap.parse_args()

    observed_rows = _load_csv(
        Path(args.observed_fingerprint_csv)
    )

    passport_rows = _load_csv(
        Path(args.passport_fingerprint_csv)
    )

    matches = []

    status_counts = Counter()

    for observed in observed_rows:

        for passport in passport_rows:

            score = _passport_score(
                observed,
                passport,
            )

            final_score = score["score"]

            if final_score >= 0.74:
                status = "STRONG_PASSPORT_MATCH"

            elif final_score >= 0.52:
                status = "PARTIAL_PASSPORT_MATCH"

            else:
                status = "WEAK_PASSPORT_MATCH"

            status_counts[status] += 1

            matches.append({
                "passport_name":
                    args.passport_name,

                "match_status":
                    status,

                "match_score":
                    f"{final_score:.9f}",

                "range_similarity":
                    f"{score['range_similarity']:.9f}",

                "behavior_similarity":
                    f"{score['behavior_similarity']:.9f}",

                "cluster_similarity":
                    f"{score['cluster_similarity']:.9f}",

                "body_similarity":
                    f"{score['body_similarity']:.9f}",

                "secondary_similarity":
                    f"{score['secondary_similarity']:.9f}",

                "strength_similarity":
                    f"{score['strength_similarity']:.9f}",
            })

    matches.sort(
        key=lambda r: (
            -_safe_float(
                r["match_score"]
            ),
            r["passport_name"],
        )
    )

    _write_csv(
        Path(args.out_match_csv),
        matches,
        [
            "passport_name",

            "match_status",

            "match_score",

            "range_similarity",
            "behavior_similarity",
            "cluster_similarity",

            "body_similarity",
            "secondary_similarity",

            "strength_similarity",
        ]
    )

    summary = {
        "passport_name":
            args.passport_name,

        "matches":
            len(matches),

        "status_counts":
            dict(status_counts),

        "best_match":
            matches[0]
            if matches
            else None,
    }

    Path(args.out_summary_txt).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.out_summary_txt).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()