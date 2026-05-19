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
# LOAD ANCHORS
# ============================================================

def load_anchor_tokens(anchor_csv: Path) -> set[str]:
    rows = load_csv(anchor_csv)
    out = set()
    for r in rows:
        tok = (r.get("observed_token", "") or "").strip()
        if tok:
            out.add(tok)
    return out


# ============================================================
# CORE
# ============================================================

def compute_adjusted_score(
    row: dict[str, str],
    *,
    anchor_tokens: set[str],
    anchor_weight: float,
) -> tuple[float, str]:
    observed = (row.get("observed_token", "") or "").strip()

    match_score = safe_float(row.get("match_score", 0.0))
    observed_count = safe_int(row.get("observed_count", 0))
    amp = safe_float(row.get("observed_mean_amplitude", 0.0))

    count_term = min(observed_count / 25.0, 1.0)
    amp_term = min(amp / 50.0, 1.0)

    base_score = (
        0.60 * match_score +
        0.20 * count_term +
        0.20 * amp_term
    )

    if observed in anchor_tokens:
        return base_score * anchor_weight, "anchor_penalized"

    return base_score, "intrinsic_kept"


def rebuild_root_without_resonance(
    edge_rows: list[dict[str, str]],
    *,
    anchor_tokens: set[str],
    anchor_weight: float,
):
    rescored_rows = []
    by_root = defaultdict(list)

    for r in edge_rows:
        adjusted_score, score_role = compute_adjusted_score(
            r,
            anchor_tokens=anchor_tokens,
            anchor_weight=anchor_weight,
        )

        new_row = dict(r)
        new_row["adjusted_score"] = adjusted_score
        new_row["score_role"] = score_role
        new_row["is_anchor_token"] = int((r.get("observed_token", "") or "").strip() in anchor_tokens)

        rescored_rows.append(new_row)

        root = (r.get("root", "") or "").strip()
        if root:
            by_root[root].append(new_row)

    summary_rows = []

    for root in sorted(by_root.keys()):
        rows = by_root[root]

        rows_sorted = sorted(
            rows,
            key=lambda x: (
                -safe_float(x.get("adjusted_score", 0.0)),
                -safe_float(x.get("match_score", 0.0)),
                -safe_int(x.get("observed_count", 0)),
                x.get("observed_token", ""),
            )
        )

        top = rows_sorted[:10]

        intrinsic_rows = [r for r in rows if safe_int(r.get("is_anchor_token", 0)) == 0]
        anchor_rows = [r for r in rows if safe_int(r.get("is_anchor_token", 0)) == 1]

        def avg_adjusted(items):
            if not items:
                return 0.0
            vals = [safe_float(x.get("adjusted_score", 0.0)) for x in items]
            return sum(vals) / len(vals)

        summary_rows.append({
            "root": root,
            "edge_count_total": len(rows),
            "intrinsic_edge_count": len(intrinsic_rows),
            "anchor_edge_count": len(anchor_rows),
            "intrinsic_avg_adjusted": avg_adjusted(intrinsic_rows),
            "anchor_avg_adjusted": avg_adjusted(anchor_rows),

            "top1_expected": top[0]["expected_token"] if len(top) > 0 else "",
            "top1_observed": top[0]["observed_token"] if len(top) > 0 else "",
            "top1_adjusted_score": safe_float(top[0]["adjusted_score"], 0.0) if len(top) > 0 else 0.0,
            "top1_role": top[0]["score_role"] if len(top) > 0 else "",

            "top2_expected": top[1]["expected_token"] if len(top) > 1 else "",
            "top2_observed": top[1]["observed_token"] if len(top) > 1 else "",
            "top2_adjusted_score": safe_float(top[1]["adjusted_score"], 0.0) if len(top) > 1 else 0.0,
            "top2_role": top[1]["score_role"] if len(top) > 1 else "",

            "top3_expected": top[2]["expected_token"] if len(top) > 2 else "",
            "top3_observed": top[2]["observed_token"] if len(top) > 2 else "",
            "top3_adjusted_score": safe_float(top[2]["adjusted_score"], 0.0) if len(top) > 2 else 0.0,
            "top3_role": top[2]["score_role"] if len(top) > 2 else "",
        })

    return rescored_rows, summary_rows


# ============================================================
# WRITE TXT / META
# ============================================================

def write_txt(path: Path, *, summary_rows: list[dict], anchor_weight: float) -> None:
    ensure_parent(path)

    strongest = sorted(
        summary_rows,
        key=lambda x: (
            -safe_float(x["intrinsic_avg_adjusted"]),
            x["root"],
        )
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("ROOT WITHOUT RESONANCE\n")
        f.write("=" * 80 + "\n")
        f.write(f"anchor_weight: {anchor_weight}\n")
        f.write(f"root_count: {len(summary_rows)}\n\n")

        f.write("ROOTS WITH STRONGEST POST-ANCHOR INTRINSIC SIGNAL\n")
        for row in strongest[:30]:
            f.write(
                f"  {row['root']} | "
                f"intrinsic_avg={float(row['intrinsic_avg_adjusted']):.3f} | "
                f"anchor_avg={float(row['anchor_avg_adjusted']):.3f} | "
                f"top1={row['top1_expected']}<-{row['top1_observed']}({float(row['top1_adjusted_score']):.3f},{row['top1_role']}) | "
                f"top2={row['top2_expected']}<-{row['top2_observed']}({float(row['top2_adjusted_score']):.3f},{row['top2_role']}) | "
                f"top3={row['top3_expected']}<-{row['top3_observed']}({float(row['top3_adjusted_score']):.3f},{row['top3_role']})\n"
            )

        f.write("\nFULL ROOT SUMMARY\n")
        for row in sorted(summary_rows, key=lambda x: x["root"]):
            f.write(f"\n[{row['root']}]\n")
            f.write(f"intrinsic_edge_count: {row['intrinsic_edge_count']}\n")
            f.write(f"anchor_edge_count: {row['anchor_edge_count']}\n")
            f.write(f"intrinsic_avg_adjusted: {float(row['intrinsic_avg_adjusted']):.3f}\n")
            f.write(f"anchor_avg_adjusted: {float(row['anchor_avg_adjusted']):.3f}\n")
            f.write(
                f"top1: {row['top1_expected']} <- {row['top1_observed']} "
                f"({float(row['top1_adjusted_score']):.3f}, {row['top1_role']})\n"
            )
            f.write(
                f"top2: {row['top2_expected']} <- {row['top2_observed']} "
                f"({float(row['top2_adjusted_score']):.3f}, {row['top2_role']})\n"
            )
            f.write(
                f"top3: {row['top3_expected']} <- {row['top3_observed']} "
                f"({float(row['top3_adjusted_score']):.3f}, {row['top3_role']})\n"
            )


def write_meta_json(
    path: Path,
    *,
    input_edges_csv: Path,
    anchor_csv: Path,
    outputs: dict,
    anchor_weight: float,
) -> None:
    ensure_parent(path)
    data = {
        "inputs": {
            "input_edges_csv": str(input_edges_csv),
            "anchor_csv": str(anchor_csv),
        },
        "outputs": outputs,
        "params": {
            "anchor_weight": anchor_weight,
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Downweight resonance anchors and rebuild root-dominant chain."
    )
    ap.add_argument("--input_edges_csv", required=True)
    ap.add_argument("--anchor_csv", required=True)
    ap.add_argument("--out_edges_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--anchor_weight", type=float, default=0.20)
    args = ap.parse_args()

    input_edges_csv = Path(args.input_edges_csv).resolve()
    anchor_csv = Path(args.anchor_csv).resolve()

    edge_rows = load_csv(input_edges_csv)
    anchor_tokens = load_anchor_tokens(anchor_csv)

    rescored_rows, summary_rows = rebuild_root_without_resonance(
        edge_rows,
        anchor_tokens=anchor_tokens,
        anchor_weight=args.anchor_weight,
    )

    write_csv(
        Path(args.out_edges_csv),
        list(rescored_rows[0].keys()) if rescored_rows else [
            "root", "chain_dir", "harmonic_index", "expected_token",
            "expected_phase_deg", "expected_radial_level",
            "observed_token", "observed_count", "observed_mean_amplitude",
            "observed_mean_phase_deg", "observed_mean_radial_level",
            "match_score", "phase_diff_deg", "angle_diff_deg", "radial_diff",
            "phase_score", "angle_score", "radial_score", "amp_score", "count_score",
            "adjusted_score", "score_role", "is_anchor_token"
        ],
        rescored_rows,
    )

    write_csv(
        Path(args.out_summary_csv),
        [
            "root",
            "edge_count_total",
            "intrinsic_edge_count",
            "anchor_edge_count",
            "intrinsic_avg_adjusted",
            "anchor_avg_adjusted",
            "top1_expected",
            "top1_observed",
            "top1_adjusted_score",
            "top1_role",
            "top2_expected",
            "top2_observed",
            "top2_adjusted_score",
            "top2_role",
            "top3_expected",
            "top3_observed",
            "top3_adjusted_score",
            "top3_role",
        ],
        summary_rows,
    )

    write_txt(
        Path(args.out_txt),
        summary_rows=summary_rows,
        anchor_weight=args.anchor_weight,
    )

    write_meta_json(
        Path(args.out_meta_json),
        input_edges_csv=input_edges_csv,
        anchor_csv=anchor_csv,
        outputs={
            "edges_csv": str(Path(args.out_edges_csv).resolve()),
            "summary_csv": str(Path(args.out_summary_csv).resolve()),
            "txt": str(Path(args.out_txt).resolve()),
        },
        anchor_weight=args.anchor_weight,
    )

    print("build root without resonance complete")
    print(f"root_count={len(summary_rows)}")
    print(f"anchor_token_count={len(anchor_tokens)}")


if __name__ == "__main__":
    main()