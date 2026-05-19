from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from music12.core.harmonic_alphabet12 import harmonic_token_from_root
from music12.core.notation12 import normalize_token


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


def safe_norm_token(tok: str) -> str:
    tok = (tok or "").strip()
    if not tok:
        return tok
    try:
        return normalize_token(tok)
    except Exception:
        return tok


def circular_distance_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# ============================================================
# TOKEN -> GEOMETRY
# ============================================================

DUO_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
}


def duo_str_to_int(s: str) -> int:
    s = (s or "").strip().upper()
    value = 0
    for ch in s:
        if ch not in DUO_MAP:
            raise ValueError(f"Unsupported duodecimal digit: {ch}")
        value = value * 12 + DUO_MAP[ch]
    return value


def token_geometry(token: str) -> dict:
    """
    Lightweight spiral geometry fallback:
    - angle from step position
    - radial from octave + step fraction
    """
    tok = safe_norm_token(token)

    if "." not in tok:
        raise ValueError(f"Bad token: {token!r}")

    left, right = tok.split(".", 1)

    octave = duo_str_to_int(left)

    step = ""
    for ch in right:
        ch = ch.upper()
        if ch in DUO_MAP:
            step += ch
        else:
            break

    if not step:
        raise ValueError(f"Bad token step: {token!r}")

    step_val = DUO_MAP[step] - 1  # 0..11
    angle = step_val * 30.0
    radial = float(octave) + (step_val / 12.0)

    return {
        "token": tok,
        "angle_deg": angle,
        "phase_deg": angle,
        "radial_level": radial,
    }


# ============================================================
# IDEAL CHAIN
# ============================================================

def build_upward_chain(root: str, max_h_up: int) -> list[dict]:
    root = safe_norm_token(root)
    out = []
    seen = set()

    # root itself
    try:
        g = token_geometry(root)
    except Exception:
        g = {
            "token": root,
            "angle_deg": 0.0,
            "phase_deg": 0.0,
            "radial_level": 0.0,
        }

    out.append({
        "chain_dir": "root",
        "harmonic_index": 0,
        "expected_token": root,
        **g,
    })
    seen.add(root)

    for h in range(1, max_h_up + 1):
        try:
            tok = harmonic_token_from_root(root, h)
            tok = safe_norm_token(tok)
        except Exception:
            continue

        if tok in seen:
            continue
        seen.add(tok)

        try:
            g = token_geometry(tok)
        except Exception:
            g = {
                "token": tok,
                "angle_deg": 0.0,
                "phase_deg": 0.0,
                "radial_level": 0.0,
            }

        out.append({
            "chain_dir": "up",
            "harmonic_index": h,
            "expected_token": tok,
            **g,
        })

    return out


def build_downward_octave_chain(root: str, max_octaves_down: int) -> list[dict]:
    """
    Downward chain by repeated octave division only.
    This is the first correct minimal step.
    """
    root = safe_norm_token(root)

    if "." not in root:
        return []

    left, right = root.split(".", 1)
    try:
        octave = duo_str_to_int(left)
    except Exception:
        return []

    out = []
    for k in range(1, max_octaves_down + 1):
        new_oct = octave - k
        if new_oct < 1:
            break

        tok = f"{int_to_duo_str(new_oct)}.{right}"
        tok = safe_norm_token(tok)

        try:
            g = token_geometry(tok)
        except Exception:
            g = {
                "token": tok,
                "angle_deg": 0.0,
                "phase_deg": 0.0,
                "radial_level": 0.0,
            }

        out.append({
            "chain_dir": "down_octave",
            "harmonic_index": -k,
            "expected_token": tok,
            **g,
        })

    return out


def int_to_duo_str(n: int) -> str:
    if n <= 0:
        raise ValueError("n must be positive")
    rev = []
    while n > 0:
        rem = n % 12
        n //= 12
        if rem == 0:
            rem = 12
            n -= 1
        for k, v in DUO_MAP.items():
            if v == rem:
                rev.append(k)
                break
    return "".join(reversed(rev))


def build_ideal_chain_full(root: str, max_h_up: int = 16, max_octaves_down: int = 4) -> list[dict]:
    up = build_upward_chain(root, max_h_up=max_h_up)
    down = build_downward_octave_chain(root, max_octaves_down=max_octaves_down)

    all_items = down + up

    # de-duplicate by expected token while preserving order
    out = []
    seen = set()
    for item in all_items:
        tok = item["expected_token"]
        if tok in seen:
            continue
        seen.add(tok)
        out.append(item)
    return out


# ============================================================
# OBSERVED DATA
# ============================================================

def load_per_root_geometry(per_root_csv: Path) -> dict[str, list[dict]]:
    rows = load_csv(per_root_csv)
    out = defaultdict(list)

    for r in rows:
        root = safe_norm_token(r.get("root", ""))
        note = safe_norm_token(r.get("response_note", ""))
        if not root or not note:
            continue

        out[root].append({
            "root": root,
            "response_note": note,
            "count": safe_int(r.get("count", 0)),
            "mean_amplitude": safe_float(r.get("mean_amplitude", 0.0)),
            "mean_phase_deg": safe_float(r.get("mean_phase_deg", 0.0)),
            "mean_radial_level": safe_float(r.get("mean_radial_level", 0.0)),
        })

    return out


# ============================================================
# GRAPH SCORING
# ============================================================

def geometry_match_components(obs: dict, exp: dict) -> dict:
    phase_diff = circular_distance_deg(
        float(obs["mean_phase_deg"]),
        float(exp["phase_deg"]),
    )
    angle_diff = circular_distance_deg(
        float(obs["mean_phase_deg"]),
        float(exp["angle_deg"]),
    )
    radial_diff = abs(
        float(obs["mean_radial_level"]) - float(exp["radial_level"])
    )

    phase_score = max(0.0, 1.0 - phase_diff / 180.0)
    angle_score = max(0.0, 1.0 - angle_diff / 180.0)
    radial_score = max(0.0, 1.0 - min(radial_diff, 3.0) / 3.0)
    amp_score = min(float(obs["mean_amplitude"]) / 50.0, 1.0)
    count_score = min(float(obs["count"]) / 25.0, 1.0)

    total = (
        0.25 * phase_score +
        0.20 * angle_score +
        0.25 * radial_score +
        0.15 * amp_score +
        0.15 * count_score
    )

    return {
        "match_score": total,
        "phase_diff_deg": phase_diff,
        "angle_diff_deg": angle_diff,
        "radial_diff": radial_diff,
        "phase_score": phase_score,
        "angle_score": angle_score,
        "radial_score": radial_score,
        "amp_score": amp_score,
        "count_score": count_score,
    }


def build_chain_graph(
    per_root_csv: Path,
    *,
    max_h_up: int = 16,
    max_octaves_down: int = 4,
    min_match_score: float = 0.35,
):
    observed = load_per_root_geometry(per_root_csv)

    edge_rows = []
    summary_rows = []

    for root in sorted(observed.keys()):
        ideal_chain = build_ideal_chain_full(
            root,
            max_h_up=max_h_up,
            max_octaves_down=max_octaves_down,
        )
        obs_rows = observed[root]

        root_edge_count = 0
        root_score_sum = 0.0
        per_expected_best = []

        for exp in ideal_chain:
            local_matches = []

            for obs in obs_rows:
                comps = geometry_match_components(obs, exp)
                score = comps["match_score"]
                if score < min_match_score:
                    continue

                row = {
                    "root": root,
                    "chain_dir": exp["chain_dir"],
                    "harmonic_index": exp["harmonic_index"],
                    "expected_token": exp["expected_token"],
                    "expected_phase_deg": exp["phase_deg"],
                    "expected_radial_level": exp["radial_level"],
                    "observed_token": obs["response_note"],
                    "observed_count": obs["count"],
                    "observed_mean_amplitude": obs["mean_amplitude"],
                    "observed_mean_phase_deg": obs["mean_phase_deg"],
                    "observed_mean_radial_level": obs["mean_radial_level"],
                    "match_score": score,
                    "phase_diff_deg": comps["phase_diff_deg"],
                    "angle_diff_deg": comps["angle_diff_deg"],
                    "radial_diff": comps["radial_diff"],
                    "phase_score": comps["phase_score"],
                    "angle_score": comps["angle_score"],
                    "radial_score": comps["radial_score"],
                    "amp_score": comps["amp_score"],
                    "count_score": comps["count_score"],
                }
                edge_rows.append(row)
                local_matches.append(row)

            local_matches.sort(key=lambda x: (-x["match_score"], -x["observed_count"], x["observed_token"]))

            if local_matches:
                best = local_matches[0]
                per_expected_best.append(best)
                root_edge_count += len(local_matches)
                root_score_sum += best["match_score"]
            else:
                per_expected_best.append({
                    "expected_token": exp["expected_token"],
                    "harmonic_index": exp["harmonic_index"],
                    "chain_dir": exp["chain_dir"],
                    "observed_token": "",
                    "match_score": 0.0,
                })

        avg_best_score = (
            root_score_sum / len(ideal_chain)
            if ideal_chain else 0.0
        )

        strongest = sorted(
            per_expected_best,
            key=lambda x: (-float(x["match_score"]), x["harmonic_index"])
        )

        summary_rows.append({
            "root": root,
            "ideal_node_count": len(ideal_chain),
            "graph_edge_count": root_edge_count,
            "average_best_score": avg_best_score,
            "best_link_1_expected": strongest[0]["expected_token"] if len(strongest) > 0 else "",
            "best_link_1_observed": strongest[0].get("observed_token", "") if len(strongest) > 0 else "",
            "best_link_1_score": strongest[0]["match_score"] if len(strongest) > 0 else 0.0,
            "best_link_2_expected": strongest[1]["expected_token"] if len(strongest) > 1 else "",
            "best_link_2_observed": strongest[1].get("observed_token", "") if len(strongest) > 1 else "",
            "best_link_2_score": strongest[1]["match_score"] if len(strongest) > 1 else 0.0,
            "best_link_3_expected": strongest[2]["expected_token"] if len(strongest) > 2 else "",
            "best_link_3_observed": strongest[2].get("observed_token", "") if len(strongest) > 2 else "",
            "best_link_3_score": strongest[2]["match_score"] if len(strongest) > 2 else 0.0,
        })

    return edge_rows, summary_rows


# ============================================================
# WRITE
# ============================================================

def write_edges_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "root",
        "chain_dir",
        "harmonic_index",
        "expected_token",
        "expected_phase_deg",
        "expected_radial_level",
        "observed_token",
        "observed_count",
        "observed_mean_amplitude",
        "observed_mean_phase_deg",
        "observed_mean_radial_level",
        "match_score",
        "phase_diff_deg",
        "angle_diff_deg",
        "radial_diff",
        "phase_score",
        "angle_score",
        "radial_score",
        "amp_score",
        "count_score",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "root",
        "ideal_node_count",
        "graph_edge_count",
        "average_best_score",
        "best_link_1_expected",
        "best_link_1_observed",
        "best_link_1_score",
        "best_link_2_expected",
        "best_link_2_observed",
        "best_link_2_score",
        "best_link_3_expected",
        "best_link_3_observed",
        "best_link_3_score",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, summary_rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("GEOMETRIC CHAIN GRAPH\n")
        f.write("=" * 80 + "\n\n")

        best = sorted(summary_rows, key=lambda r: (-float(r["average_best_score"]), r["root"]))

        f.write("ROOTS WITH STRONGEST CHAIN GEOMETRY\n")
        for row in best[:25]:
            f.write(
                f"{row['root']} | "
                f"avg_best={float(row['average_best_score']):.3f} | "
                f"edges={row['graph_edge_count']} | "
                f"best1={row['best_link_1_expected']}<-{row['best_link_1_observed']}({float(row['best_link_1_score']):.3f}) | "
                f"best2={row['best_link_2_expected']}<-{row['best_link_2_observed']}({float(row['best_link_2_score']):.3f}) | "
                f"best3={row['best_link_3_expected']}<-{row['best_link_3_observed']}({float(row['best_link_3_score']):.3f})\n"
            )

        f.write("\nFULL ROOT SUMMARY\n")
        for row in sorted(summary_rows, key=lambda r: r["root"]):
            f.write(f"\n[{row['root']}]\n")
            f.write(f"ideal_node_count: {row['ideal_node_count']}\n")
            f.write(f"graph_edge_count: {row['graph_edge_count']}\n")
            f.write(f"average_best_score: {float(row['average_best_score']):.3f}\n")
            f.write(
                f"best1: {row['best_link_1_expected']} <- {row['best_link_1_observed']} "
                f"({float(row['best_link_1_score']):.3f})\n"
            )
            f.write(
                f"best2: {row['best_link_2_expected']} <- {row['best_link_2_observed']} "
                f"({float(row['best_link_2_score']):.3f})\n"
            )
            f.write(
                f"best3: {row['best_link_3_expected']} <- {row['best_link_3_observed']} "
                f"({float(row['best_link_3_score']):.3f})\n"
            )


def write_meta_json(
    path: Path,
    *,
    per_root_csv: Path,
    edges_csv: Path,
    summary_csv: Path,
    txt_path: Path,
    max_h_up: int,
    max_octaves_down: int,
    min_match_score: float,
) -> None:
    ensure_parent(path)
    data = {
        "inputs": {
            "per_root_csv": str(per_root_csv),
        },
        "outputs": {
            "edges_csv": str(edges_csv),
            "summary_csv": str(summary_csv),
            "txt": str(txt_path),
        },
        "params": {
            "max_h_up": max_h_up,
            "max_octaves_down": max_octaves_down,
            "min_match_score": min_match_score,
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build full geometric chain graph with upward harmonics and downward octave links."
    )
    ap.add_argument("--per_root_csv", required=True)
    ap.add_argument("--out_edges_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_h_up", type=int, default=16)
    ap.add_argument("--max_octaves_down", type=int, default=4)
    ap.add_argument("--min_match_score", type=float, default=0.35)
    args = ap.parse_args()

    per_root_csv = Path(args.per_root_csv).resolve()
    out_edges_csv = Path(args.out_edges_csv).resolve()
    out_summary_csv = Path(args.out_summary_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    edge_rows, summary_rows = build_chain_graph(
        per_root_csv,
        max_h_up=args.max_h_up,
        max_octaves_down=args.max_octaves_down,
        min_match_score=args.min_match_score,
    )

    write_edges_csv(out_edges_csv, edge_rows)
    write_summary_csv(out_summary_csv, summary_rows)
    write_txt(out_txt, summary_rows)
    write_meta_json(
        out_meta_json,
        per_root_csv=per_root_csv,
        edges_csv=out_edges_csv,
        summary_csv=out_summary_csv,
        txt_path=out_txt,
        max_h_up=args.max_h_up,
        max_octaves_down=args.max_octaves_down,
        min_match_score=args.min_match_score,
    )

    print("build geometric chain graph complete")
    print(f"root_count={len(summary_rows)}")
    print(f"edge_rows={len(edge_rows)}")


if __name__ == "__main__":
    main()