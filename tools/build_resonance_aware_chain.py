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
# STEP 1. DETECT INSTRUMENT ANCHORS
# ============================================================

def detect_anchor_tokens(
    edge_rows: list[dict[str, str]],
    *,
    min_root_coverage: int = 12,
    min_mean_match_score: float = 0.72,
) -> list[dict]:
    """
    Anchor token = observed token that appears across many different roots
    with strong geometric match.
    """
    by_token_roots: dict[str, set[str]] = defaultdict(set)
    by_token_scores: dict[str, list[float]] = defaultdict(list)
    by_token_counts: dict[str, int] = defaultdict(int)

    for r in edge_rows:
        tok = (r.get("observed_token", "") or "").strip()
        root = (r.get("root", "") or "").strip()
        score = safe_float(r.get("match_score", 0.0))

        if not tok or not root:
            continue

        by_token_roots[tok].add(root)
        by_token_scores[tok].append(score)
        by_token_counts[tok] += 1

    out = []
    for tok in sorted(by_token_roots.keys()):
        root_coverage = len(by_token_roots[tok])
        scores = by_token_scores[tok]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        edge_count = by_token_counts[tok]

        is_anchor = (
            root_coverage >= min_root_coverage
            and mean_score >= min_mean_match_score
        )

        if is_anchor:
            out.append({
                "anchor_token": tok,
                "root_coverage": root_coverage,
                "edge_count": edge_count,
                "mean_match_score": mean_score,
            })

    out.sort(key=lambda x: (-x["root_coverage"], -x["mean_match_score"], x["anchor_token"]))
    return out


# ============================================================
# STEP 2. LABEL EDGES
# ============================================================

def label_edge(row: dict[str, str], anchor_tokens: set[str]) -> str:
    expected = (row.get("expected_token", "") or "").strip()
    observed = (row.get("observed_token", "") or "").strip()
    score = safe_float(row.get("match_score", 0.0))

    exact_like = expected == observed
    anchor_like = observed in anchor_tokens

    if exact_like and not anchor_like:
        return "intrinsic"

    if anchor_like and not exact_like:
        return "resonance_anchor"

    if exact_like and anchor_like:
        return "mixed"

    # soft intrinsic if score is high and token is not anchor
    if (score >= 0.82) and not anchor_like:
        return "intrinsic"

    if anchor_like:
        return "resonance_anchor"

    return "mixed"


def build_resonance_aware_edges(
    edge_rows: list[dict[str, str]],
    anchor_tokens: set[str],
) -> list[dict]:
    out = []

    for r in edge_rows:
        label = label_edge(r, anchor_tokens)

        new_row = dict(r)
        new_row["edge_role"] = label
        new_row["is_anchor_token"] = int((r.get("observed_token", "") or "").strip() in anchor_tokens)
        out.append(new_row)

    return out


# ============================================================
# STEP 3. ROOT SUMMARIES
# ============================================================

def summarize_resonance_aware_chain(rows: list[dict[str, str]]) -> list[dict]:
    by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        root = (r.get("root", "") or "").strip()
        if root:
            by_root[root].append(r)

    out = []

    for root in sorted(by_root.keys()):
        group = by_root[root]

        intrinsic = [r for r in group if r["edge_role"] == "intrinsic"]
        anchors = [r for r in group if r["edge_role"] == "resonance_anchor"]
        mixed = [r for r in group if r["edge_role"] == "mixed"]

        def avg_score(items):
            if not items:
                return 0.0
            vals = [safe_float(x.get("match_score", 0.0)) for x in items]
            return sum(vals) / len(vals)

        intrinsic_sorted = sorted(
            intrinsic,
            key=lambda x: (-safe_float(x.get("match_score", 0.0)), x.get("expected_token", "")),
        )
        anchors_sorted = sorted(
            anchors,
            key=lambda x: (-safe_float(x.get("match_score", 0.0)), x.get("observed_token", "")),
        )
        mixed_sorted = sorted(
            mixed,
            key=lambda x: (-safe_float(x.get("match_score", 0.0)), x.get("observed_token", "")),
        )

        out.append({
            "root": root,
            "edge_count_total": len(group),
            "intrinsic_edge_count": len(intrinsic),
            "anchor_edge_count": len(anchors),
            "mixed_edge_count": len(mixed),
            "intrinsic_avg_score": avg_score(intrinsic),
            "anchor_avg_score": avg_score(anchors),
            "mixed_avg_score": avg_score(mixed),

            "top_intrinsic_expected": intrinsic_sorted[0]["expected_token"] if intrinsic_sorted else "",
            "top_intrinsic_observed": intrinsic_sorted[0]["observed_token"] if intrinsic_sorted else "",
            "top_intrinsic_score": safe_float(intrinsic_sorted[0]["match_score"], 0.0) if intrinsic_sorted else 0.0,

            "top_anchor_expected": anchors_sorted[0]["expected_token"] if anchors_sorted else "",
            "top_anchor_observed": anchors_sorted[0]["observed_token"] if anchors_sorted else "",
            "top_anchor_score": safe_float(anchors_sorted[0]["match_score"], 0.0) if anchors_sorted else 0.0,

            "top_mixed_expected": mixed_sorted[0]["expected_token"] if mixed_sorted else "",
            "top_mixed_observed": mixed_sorted[0]["observed_token"] if mixed_sorted else "",
            "top_mixed_score": safe_float(mixed_sorted[0]["match_score"], 0.0) if mixed_sorted else 0.0,
        })

    return out


# ============================================================
# WRITE TXT / META
# ============================================================

def write_txt(
    path: Path,
    *,
    anchor_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    ensure_parent(path)

    strongest_intrinsic = sorted(
        summary_rows,
        key=lambda x: (-safe_float(x["intrinsic_avg_score"]), x["root"])
    )
    strongest_anchors = sorted(
        summary_rows,
        key=lambda x: (-safe_float(x["anchor_avg_score"]), x["root"])
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("RESONANCE-AWARE CHAIN\n")
        f.write("=" * 80 + "\n")
        f.write(f"anchor_token_count: {len(anchor_rows)}\n")
        f.write(f"root_count: {len(summary_rows)}\n\n")

        f.write("DETECTED RESONANCE ANCHORS\n")
        for row in anchor_rows[:40]:
            f.write(
                f"  {row['anchor_token']} | "
                f"root_coverage={row['root_coverage']} | "
                f"edge_count={row['edge_count']} | "
                f"mean_match={float(row['mean_match_score']):.3f}\n"
            )

        f.write("\nROOTS WITH STRONGEST INTRINSIC COMPONENT\n")
        for row in strongest_intrinsic[:25]:
            f.write(
                f"  {row['root']} | "
                f"intrinsic_edges={row['intrinsic_edge_count']} | "
                f"intrinsic_avg={float(row['intrinsic_avg_score']):.3f} | "
                f"top={row['top_intrinsic_expected']}<-{row['top_intrinsic_observed']}({float(row['top_intrinsic_score']):.3f})\n"
            )

        f.write("\nROOTS WITH STRONGEST RESONANCE-ANCHOR COMPONENT\n")
        for row in strongest_anchors[:25]:
            f.write(
                f"  {row['root']} | "
                f"anchor_edges={row['anchor_edge_count']} | "
                f"anchor_avg={float(row['anchor_avg_score']):.3f} | "
                f"top={row['top_anchor_expected']}<-{row['top_anchor_observed']}({float(row['top_anchor_score']):.3f})\n"
            )

        f.write("\nFULL ROOT SUMMARY\n")
        for row in sorted(summary_rows, key=lambda x: x["root"]):
            f.write(f"\n[{row['root']}]\n")
            f.write(
                f"intrinsic_edges={row['intrinsic_edge_count']} "
                f"(avg={float(row['intrinsic_avg_score']):.3f})\n"
            )
            f.write(
                f"anchor_edges={row['anchor_edge_count']} "
                f"(avg={float(row['anchor_avg_score']):.3f})\n"
            )
            f.write(
                f"mixed_edges={row['mixed_edge_count']} "
                f"(avg={float(row['mixed_avg_score']):.3f})\n"
            )
            f.write(
                f"top_intrinsic: {row['top_intrinsic_expected']} <- {row['top_intrinsic_observed']} "
                f"({float(row['top_intrinsic_score']):.3f})\n"
            )
            f.write(
                f"top_anchor: {row['top_anchor_expected']} <- {row['top_anchor_observed']} "
                f"({float(row['top_anchor_score']):.3f})\n"
            )
            f.write(
                f"top_mixed: {row['top_mixed_expected']} <- {row['top_mixed_observed']} "
                f"({float(row['top_mixed_score']):.3f})\n"
            )


def write_meta_json(
    path: Path,
    *,
    input_edges_csv: Path,
    outputs: dict,
    min_root_coverage: int,
    min_mean_match_score: float,
) -> None:
    ensure_parent(path)
    data = {
        "inputs": {
            "input_edges_csv": str(input_edges_csv),
        },
        "outputs": outputs,
        "params": {
            "min_root_coverage": min_root_coverage,
            "min_mean_match_score": min_mean_match_score,
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build resonance-aware chain by separating intrinsic and instrument-anchor geometry."
    )
    ap.add_argument("--input_edges_csv", required=True)
    ap.add_argument("--out_anchor_csv", required=True)
    ap.add_argument("--out_edges_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--min_root_coverage", type=int, default=12)
    ap.add_argument("--min_mean_match_score", type=float, default=0.72)
    args = ap.parse_args()

    input_edges_csv = Path(args.input_edges_csv).resolve()
    edge_rows = load_csv(input_edges_csv)

    anchor_rows = detect_anchor_tokens(
        edge_rows,
        min_root_coverage=args.min_root_coverage,
        min_mean_match_score=args.min_mean_match_score,
    )
    anchor_tokens = {r["anchor_token"] for r in anchor_rows}

    resonance_aware_edges = build_resonance_aware_edges(edge_rows, anchor_tokens)
    summary_rows = summarize_resonance_aware_chain(resonance_aware_edges)

    write_csv(
        Path(args.out_anchor_csv),
        ["anchor_token", "root_coverage", "edge_count", "mean_match_score"],
        anchor_rows,
    )

    write_csv(
        Path(args.out_edges_csv),
        list(resonance_aware_edges[0].keys()) if resonance_aware_edges else [
            "root", "chain_dir", "harmonic_index", "expected_token", "expected_phase_deg",
            "expected_radial_level", "observed_token", "observed_count",
            "observed_mean_amplitude", "observed_mean_phase_deg",
            "observed_mean_radial_level", "match_score", "phase_diff_deg",
            "angle_diff_deg", "radial_diff", "phase_score", "angle_score",
            "radial_score", "amp_score", "count_score", "edge_role", "is_anchor_token"
        ],
        resonance_aware_edges,
    )

    write_csv(
        Path(args.out_summary_csv),
        [
            "root",
            "edge_count_total",
            "intrinsic_edge_count",
            "anchor_edge_count",
            "mixed_edge_count",
            "intrinsic_avg_score",
            "anchor_avg_score",
            "mixed_avg_score",
            "top_intrinsic_expected",
            "top_intrinsic_observed",
            "top_intrinsic_score",
            "top_anchor_expected",
            "top_anchor_observed",
            "top_anchor_score",
            "top_mixed_expected",
            "top_mixed_observed",
            "top_mixed_score",
        ],
        summary_rows,
    )

    write_txt(
        Path(args.out_txt),
        anchor_rows=anchor_rows,
        summary_rows=summary_rows,
    )

    write_meta_json(
        Path(args.out_meta_json),
        input_edges_csv=input_edges_csv,
        outputs={
            "anchor_csv": str(Path(args.out_anchor_csv).resolve()),
            "edges_csv": str(Path(args.out_edges_csv).resolve()),
            "summary_csv": str(Path(args.out_summary_csv).resolve()),
            "txt": str(Path(args.out_txt).resolve()),
        },
        min_root_coverage=args.min_root_coverage,
        min_mean_match_score=args.min_mean_match_score,
    )

    print("build resonance-aware chain complete")
    print(f"anchor_token_count={len(anchor_rows)}")
    print(f"root_count={len(summary_rows)}")


if __name__ == "__main__":
    main()