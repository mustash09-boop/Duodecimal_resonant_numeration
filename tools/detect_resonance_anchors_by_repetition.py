from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# HELPERS
# ============================================================

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CORE
# ============================================================

def collect_observed_token_stats(edge_rows: list[dict[str, str]]) -> list[dict]:
    by_token_roots: dict[str, set[str]] = defaultdict(set)
    by_token_expected: dict[str, set[str]] = defaultdict(set)

    edge_count_by_token: dict[str, int] = defaultdict(int)
    observed_count_sum_by_token: dict[str, int] = defaultdict(int)
    amplitude_sum_by_token: dict[str, float] = defaultdict(float)
    match_score_sum_by_token: dict[str, float] = defaultdict(float)

    for r in edge_rows:
        tok = (r.get("observed_token", "") or "").strip()
        root = (r.get("root", "") or "").strip()
        expected = (r.get("expected_token", "") or "").strip()

        if not tok:
            continue

        by_token_roots[tok].add(root)
        if expected:
            by_token_expected[tok].add(expected)

        edge_count_by_token[tok] += 1
        observed_count_sum_by_token[tok] += safe_int(r.get("observed_count", 0))
        amplitude_sum_by_token[tok] += safe_float(r.get("observed_mean_amplitude", 0.0))
        match_score_sum_by_token[tok] += safe_float(r.get("match_score", 0.0))

    rows = []
    for tok in sorted(edge_count_by_token.keys()):
        edge_count = edge_count_by_token[tok]
        root_coverage = len(by_token_roots[tok])
        expected_coverage = len(by_token_expected[tok])

        observed_count_sum = observed_count_sum_by_token[tok]
        mean_amplitude = amplitude_sum_by_token[tok] / edge_count if edge_count else 0.0
        mean_match_score = match_score_sum_by_token[tok] / edge_count if edge_count else 0.0

        rows.append({
            "observed_token": tok,
            "root_coverage": root_coverage,
            "expected_coverage": expected_coverage,
            "edge_count": edge_count,
            "observed_count_sum": observed_count_sum,
            "mean_amplitude": mean_amplitude,
            "mean_match_score": mean_match_score,
        })

    rows.sort(
        key=lambda x: (
            -safe_int(x["root_coverage"]),
            -safe_int(x["observed_count_sum"]),
            -safe_int(x["edge_count"]),
            -safe_float(x["mean_amplitude"]),
            x["observed_token"],
        )
    )
    return rows


def select_anchor_tokens(
    token_rows: list[dict],
    *,
    min_root_coverage: int = 20,
    min_observed_count_sum: int = 150,
    min_edge_count: int = 40,
) -> list[dict]:
    """
    True resonance anchors:
    repeated across many roots with high total presence.
    No hard dependence on geometric match score.
    """
    out = []

    for r in token_rows:
        root_coverage = safe_int(r["root_coverage"], 0)
        observed_count_sum = safe_int(r["observed_count_sum"], 0)
        edge_count = safe_int(r["edge_count"], 0)

        if (
            root_coverage >= min_root_coverage
            and observed_count_sum >= min_observed_count_sum
            and edge_count >= min_edge_count
        ):
            out.append(dict(r))

    out.sort(
        key=lambda x: (
            -safe_int(x["root_coverage"]),
            -safe_int(x["observed_count_sum"]),
            -safe_int(x["edge_count"]),
            x["observed_token"],
        )
    )
    return out


# ============================================================
# WRITE TXT / META
# ============================================================

def write_txt(
    path: Path,
    *,
    all_rows: list[dict],
    anchor_rows: list[dict],
) -> None:
    ensure_parent(path)

    with path.open("w", encoding="utf-8") as f:
        f.write("RESONANCE ANCHORS BY REPETITION\n")
        f.write("=" * 80 + "\n")
        f.write(f"observed_token_count: {len(all_rows)}\n")
        f.write(f"anchor_token_count: {len(anchor_rows)}\n\n")

        f.write("TOP OBSERVED TOKENS BY REPETITION\n")
        for r in all_rows[:40]:
            f.write(
                f"  {r['observed_token']} | "
                f"root_coverage={r['root_coverage']} | "
                f"observed_count_sum={r['observed_count_sum']} | "
                f"edge_count={r['edge_count']} | "
                f"amp={float(r['mean_amplitude']):.2f} | "
                f"mean_match={float(r['mean_match_score']):.3f}\n"
            )

        f.write("\nSELECTED RESONANCE ANCHORS\n")
        for r in anchor_rows:
            f.write(
                f"  {r['observed_token']} | "
                f"root_coverage={r['root_coverage']} | "
                f"observed_count_sum={r['observed_count_sum']} | "
                f"edge_count={r['edge_count']} | "
                f"amp={float(r['mean_amplitude']):.2f} | "
                f"mean_match={float(r['mean_match_score']):.3f}\n"
            )

        f.write("\nINTERPRETATION\n")
        f.write("  - anchors are selected by repetition across many roots.\n")
        f.write("  - they are not required to match any ideal chain strongly.\n")
        f.write("  - these tokens behave as stable instrument resonance nodes.\n")


def write_meta_json(
    path: Path,
    *,
    input_edges_csv: Path,
    outputs: dict,
    min_root_coverage: int,
    min_observed_count_sum: int,
    min_edge_count: int,
) -> None:
    ensure_parent(path)
    data = {
        "inputs": {
            "input_edges_csv": str(input_edges_csv),
        },
        "outputs": outputs,
        "params": {
            "min_root_coverage": min_root_coverage,
            "min_observed_count_sum": min_observed_count_sum,
            "min_edge_count": min_edge_count,
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Detect resonance anchors by repetition across many roots."
    )
    ap.add_argument("--input_edges_csv", required=True)
    ap.add_argument("--out_all_tokens_csv", required=True)
    ap.add_argument("--out_anchor_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--min_root_coverage", type=int, default=20)
    ap.add_argument("--min_observed_count_sum", type=int, default=150)
    ap.add_argument("--min_edge_count", type=int, default=40)
    args = ap.parse_args()

    input_edges_csv = Path(args.input_edges_csv).resolve()
    edge_rows = load_csv(input_edges_csv)

    all_rows = collect_observed_token_stats(edge_rows)
    anchor_rows = select_anchor_tokens(
        all_rows,
        min_root_coverage=args.min_root_coverage,
        min_observed_count_sum=args.min_observed_count_sum,
        min_edge_count=args.min_edge_count,
    )

    write_csv(
        Path(args.out_all_tokens_csv),
        [
            "observed_token",
            "root_coverage",
            "expected_coverage",
            "edge_count",
            "observed_count_sum",
            "mean_amplitude",
            "mean_match_score",
        ],
        all_rows,
    )

    write_csv(
        Path(args.out_anchor_csv),
        [
            "observed_token",
            "root_coverage",
            "expected_coverage",
            "edge_count",
            "observed_count_sum",
            "mean_amplitude",
            "mean_match_score",
        ],
        anchor_rows,
    )

    write_txt(
        Path(args.out_txt),
        all_rows=all_rows,
        anchor_rows=anchor_rows,
    )

    write_meta_json(
        Path(args.out_meta_json),
        input_edges_csv=input_edges_csv,
        outputs={
            "all_tokens_csv": str(Path(args.out_all_tokens_csv).resolve()),
            "anchor_csv": str(Path(args.out_anchor_csv).resolve()),
            "txt": str(Path(args.out_txt).resolve()),
        },
        min_root_coverage=args.min_root_coverage,
        min_observed_count_sum=args.min_observed_count_sum,
        min_edge_count=args.min_edge_count,
    )

    print("detect resonance anchors by repetition complete")
    print(f"observed_token_count={len(all_rows)}")
    print(f"anchor_token_count={len(anchor_rows)}")


if __name__ == "__main__":
    main()