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


def split_token(token: str) -> tuple[str, str]:
    tok = (token or "").strip()
    if "." not in tok:
        return tok, ""
    left, right = tok.split(".", 1)
    step = ""
    for ch in right:
        if ch.upper() in "123456789ABC":
            step += ch.upper()
        else:
            break
    return left, step


# ============================================================
# CORE
# ============================================================

def collect_note_candidates(edge_rows: list[dict[str, str]]) -> list[dict]:
    """
    Build note candidates from intrinsic_kept rows only.
    Group by root and observed_token.
    """
    by_root_obs = defaultdict(list)

    for r in edge_rows:
        role = (r.get("score_role", "") or "").strip()
        if role != "intrinsic_kept":
            continue

        root = (r.get("root", "") or "").strip()
        observed = (r.get("observed_token", "") or "").strip()
        expected = (r.get("expected_token", "") or "").strip()
        if not root or not observed or not expected:
            continue

        by_root_obs[(root, observed)].append(r)

    out = []

    for (root, observed), rows in sorted(by_root_obs.items()):
        expected_tokens = []
        expected_steps = set()
        expected_octaves = []
        adjusted_scores = []
        match_scores = []
        counts = []
        amplitudes = []

        for r in rows:
            exp = (r.get("expected_token", "") or "").strip()
            expected_tokens.append(exp)

            octv, step = split_token(exp)
            if step:
                expected_steps.add(step)
            if octv:
                expected_octaves.append(octv)

            adjusted_scores.append(safe_float(r.get("adjusted_score", 0.0)))
            match_scores.append(safe_float(r.get("match_score", 0.0)))
            counts.append(safe_int(r.get("observed_count", 0)))
            amplitudes.append(safe_float(r.get("observed_mean_amplitude", 0.0)))

        unique_expected = sorted(set(expected_tokens))
        octave_span = len(set(expected_octaves))
        step_consistency = 1 if len(expected_steps) == 1 else 0

        avg_adjusted = sum(adjusted_scores) / len(adjusted_scores) if adjusted_scores else 0.0
        avg_match = sum(match_scores) / len(match_scores) if match_scores else 0.0
        avg_count = sum(counts) / len(counts) if counts else 0.0
        avg_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else 0.0

        # verticality: same step across multiple octave levels
        verticality_bonus = min(octave_span / 4.0, 1.0) * (1.0 if step_consistency else 0.5)

        note_score = (
            0.45 * avg_adjusted +
            0.20 * avg_match +
            0.15 * min(avg_count / 20.0, 1.0) +
            0.10 * min(avg_amplitude / 40.0, 1.0) +
            0.10 * verticality_bonus
        )

        out.append({
            "root": root,
            "observed_token": observed,
            "note_score": note_score,
            "avg_adjusted_score": avg_adjusted,
            "avg_match_score": avg_match,
            "avg_count": avg_count,
            "avg_amplitude": avg_amplitude,
            "expected_support_count": len(unique_expected),
            "octave_span": octave_span,
            "step_consistency": step_consistency,
            "supported_expected_tokens": " ".join(unique_expected),
        })

    out.sort(
        key=lambda x: (
            x["root"],
            -safe_float(x["note_score"]),
            -safe_int(x["expected_support_count"]),
            x["observed_token"],
        )
    )
    return out


def summarize_notes(candidate_rows: list[dict]) -> list[dict]:
    by_root = defaultdict(list)
    for r in candidate_rows:
        root = (r.get("root", "") or "").strip()
        if root:
            by_root[root].append(r)

    summary = []

    for root in sorted(by_root.keys()):
        rows = sorted(
            by_root[root],
            key=lambda x: (-safe_float(x["note_score"]), x["observed_token"])
        )

        top = rows[:8]

        summary.append({
            "root": root,
            "candidate_count": len(rows),

            "top1_observed": top[0]["observed_token"] if len(top) > 0 else "",
            "top1_score": safe_float(top[0]["note_score"], 0.0) if len(top) > 0 else 0.0,
            "top1_supports": top[0]["supported_expected_tokens"] if len(top) > 0 else "",

            "top2_observed": top[1]["observed_token"] if len(top) > 1 else "",
            "top2_score": safe_float(top[1]["note_score"], 0.0) if len(top) > 1 else 0.0,
            "top2_supports": top[1]["supported_expected_tokens"] if len(top) > 1 else "",

            "top3_observed": top[2]["observed_token"] if len(top) > 2 else "",
            "top3_score": safe_float(top[2]["note_score"], 0.0) if len(top) > 2 else 0.0,
            "top3_supports": top[2]["supported_expected_tokens"] if len(top) > 2 else "",
        })

    return summary


# ============================================================
# WRITE
# ============================================================

def write_txt(path: Path, summary_rows: list[dict]) -> None:
    ensure_parent(path)

    strongest = sorted(
        summary_rows,
        key=lambda x: (-safe_float(x["top1_score"]), x["root"])
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("DETECTED NOTES V1\n")
        f.write("=" * 80 + "\n")
        f.write(f"root_count: {len(summary_rows)}\n\n")

        f.write("ROOTS WITH STRONGEST NOTE CANDIDATES\n")
        for row in strongest[:30]:
            f.write(
                f"  {row['root']} | "
                f"top1={row['top1_observed']}({float(row['top1_score']):.3f}) | "
                f"supports={row['top1_supports']} | "
                f"top2={row['top2_observed']}({float(row['top2_score']):.3f}) | "
                f"top3={row['top3_observed']}({float(row['top3_score']):.3f})\n"
            )

        f.write("\nFULL ROOT SUMMARY\n")
        for row in sorted(summary_rows, key=lambda x: x["root"]):
            f.write(f"\n[{row['root']}]\n")
            f.write(f"candidate_count: {row['candidate_count']}\n")
            f.write(
                f"top1: {row['top1_observed']} ({float(row['top1_score']):.3f}) | "
                f"supports: {row['top1_supports']}\n"
            )
            f.write(
                f"top2: {row['top2_observed']} ({float(row['top2_score']):.3f}) | "
                f"supports: {row['top2_supports']}\n"
            )
            f.write(
                f"top3: {row['top3_observed']} ({float(row['top3_score']):.3f}) | "
                f"supports: {row['top3_supports']}\n"
            )


def write_meta_json(path: Path, *, input_edges_csv: Path, outputs: dict) -> None:
    ensure_parent(path)
    data = {
        "inputs": {
            "input_edges_csv": str(input_edges_csv),
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build note candidates from root_without_resonance intrinsic edges."
    )
    ap.add_argument("--input_edges_csv", required=True)
    ap.add_argument("--out_candidates_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    input_edges_csv = Path(args.input_edges_csv).resolve()
    edge_rows = load_csv(input_edges_csv)

    candidate_rows = collect_note_candidates(edge_rows)
    summary_rows = summarize_notes(candidate_rows)

    write_csv(
        Path(args.out_candidates_csv),
        [
            "root",
            "observed_token",
            "note_score",
            "avg_adjusted_score",
            "avg_match_score",
            "avg_count",
            "avg_amplitude",
            "expected_support_count",
            "octave_span",
            "step_consistency",
            "supported_expected_tokens",
        ],
        candidate_rows,
    )

    write_csv(
        Path(args.out_summary_csv),
        [
            "root",
            "candidate_count",
            "top1_observed",
            "top1_score",
            "top1_supports",
            "top2_observed",
            "top2_score",
            "top2_supports",
            "top3_observed",
            "top3_score",
            "top3_supports",
        ],
        summary_rows,
    )

    write_txt(Path(args.out_txt), summary_rows)
    write_meta_json(
        Path(args.out_meta_json),
        input_edges_csv=input_edges_csv,
        outputs={
            "candidates_csv": str(Path(args.out_candidates_csv).resolve()),
            "summary_csv": str(Path(args.out_summary_csv).resolve()),
            "txt": str(Path(args.out_txt).resolve()),
        },
    )

    print("build root detector v1 complete")
    print(f"root_count={len(summary_rows)}")
    print(f"candidate_count={len(candidate_rows)}")


if __name__ == "__main__":
    main()
	