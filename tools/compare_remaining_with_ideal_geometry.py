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
    Lightweight geometry fallback:
    - angle from step
    - radial from octave + step fraction
    This is not the full spiral engine, but it is consistent and stable.
    """
    tok = safe_norm_token(token)

    if "." not in tok:
        raise ValueError(f"Bad token: {token}")

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
        raise ValueError(f"Bad token step: {token}")

    step_val = DUO_MAP[step] - 1  # 0..11
    angle = step_val * 30.0
    radial = float(octave) + (step_val / 12.0)

    return {
        "token": tok,
        "angle_deg": angle,
        "phase_deg": angle,
        "radial_level": radial,
    }


def build_ideal_chain_geometry(root: str, max_h: int = 8) -> list[dict]:
    root = safe_norm_token(root)
    out = []
    seen = set()

    if root:
        try:
            g = token_geometry(root)
            out.append({
                "harmonic_no": 0,
                "expected_token": root,
                **g,
            })
            seen.add(root)
        except Exception:
            out.append({
                "harmonic_no": 0,
                "expected_token": root,
                "angle_deg": 0.0,
                "phase_deg": 0.0,
                "radial_level": 0.0,
            })

    for h in range(1, max_h + 1):
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
            "harmonic_no": h,
            "expected_token": tok,
            **g,
        })

    return out


# ============================================================
# LOAD OBSERVED REMAINING TOKENS WITH GEOMETRY
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
# GEOMETRIC MATCH
# ============================================================

def geometry_match_score(obs: dict, exp: dict) -> float:
    phase_diff = circular_distance_deg(
        float(obs["mean_phase_deg"]),
        float(exp["phase_deg"]),
    )
    radial_diff = abs(
        float(obs["mean_radial_level"]) - float(exp["radial_level"])
    )
    angle_diff = circular_distance_deg(
        float(obs["mean_phase_deg"]),
        float(exp["angle_deg"]),
    )

    phase_score = max(0.0, 1.0 - phase_diff / 180.0)
    angle_score = max(0.0, 1.0 - angle_diff / 180.0)
    radial_score = max(0.0, 1.0 - min(radial_diff, 2.0) / 2.0)

    # observed strength matters, but softly
    amp_term = min(float(obs["mean_amplitude"]) / 50.0, 1.0)
    count_term = min(float(obs["count"]) / 25.0, 1.0)

    score = (
        0.25 * phase_score +
        0.20 * angle_score +
        0.25 * radial_score +
        0.15 * amp_term +
        0.15 * count_term
    )
    return score


def compare_geometry(per_root_csv: Path, *, max_h: int = 8):
    observed = load_per_root_geometry(per_root_csv)

    detail_rows = []
    summary_rows = []

    for root in sorted(observed.keys()):
        ideal = build_ideal_chain_geometry(root, max_h=max_h)
        obs_rows = observed[root]

        total_obs_count = sum(x["count"] for x in obs_rows)
        used_obs = set()
        matched_rows = []

        total_match_score = 0.0

        for exp in ideal:
            best_idx = None
            best_score = -1.0

            for idx, obs in enumerate(obs_rows):
                if idx in used_obs:
                    continue
                s = geometry_match_score(obs, exp)
                if s > best_score:
                    best_score = s
                    best_idx = idx

            if best_idx is not None:
                used_obs.add(best_idx)
                obs = obs_rows[best_idx]
                total_match_score += best_score

                matched_rows.append({
                    "root": root,
                    "harmonic_no": exp["harmonic_no"],
                    "expected_token": exp["expected_token"],
                    "matched_response_note": obs["response_note"],
                    "match_score": best_score,
                    "count": obs["count"],
                    "mean_amplitude": obs["mean_amplitude"],
                    "observed_phase_deg": obs["mean_phase_deg"],
                    "observed_radial_level": obs["mean_radial_level"],
                    "expected_phase_deg": exp["phase_deg"],
                    "expected_radial_level": exp["radial_level"],
                })

        matched_rows.sort(key=lambda x: (x["harmonic_no"], -x["match_score"]))
        detail_rows.extend(matched_rows)

        avg_match_score = (total_match_score / len(ideal)) if ideal else 0.0
        strongest = sorted(matched_rows, key=lambda x: (-x["match_score"], x["harmonic_no"]))

        summary_rows.append({
            "root": root,
            "ideal_chain_size": len(ideal),
            "observed_remaining_count": total_obs_count,
            "average_match_score": avg_match_score,
            "best_match_1_expected": strongest[0]["expected_token"] if len(strongest) > 0 else "",
            "best_match_1_observed": strongest[0]["matched_response_note"] if len(strongest) > 0 else "",
            "best_match_1_score": strongest[0]["match_score"] if len(strongest) > 0 else 0.0,
            "best_match_2_expected": strongest[1]["expected_token"] if len(strongest) > 1 else "",
            "best_match_2_observed": strongest[1]["matched_response_note"] if len(strongest) > 1 else "",
            "best_match_2_score": strongest[1]["match_score"] if len(strongest) > 1 else 0.0,
            "best_match_3_expected": strongest[2]["expected_token"] if len(strongest) > 2 else "",
            "best_match_3_observed": strongest[2]["matched_response_note"] if len(strongest) > 2 else "",
            "best_match_3_score": strongest[2]["match_score"] if len(strongest) > 2 else 0.0,
        })

    return detail_rows, summary_rows


# ============================================================
# WRITE
# ============================================================

def write_detail_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "root",
        "harmonic_no",
        "expected_token",
        "matched_response_note",
        "match_score",
        "count",
        "mean_amplitude",
        "observed_phase_deg",
        "observed_radial_level",
        "expected_phase_deg",
        "expected_radial_level",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "root",
        "ideal_chain_size",
        "observed_remaining_count",
        "average_match_score",
        "best_match_1_expected",
        "best_match_1_observed",
        "best_match_1_score",
        "best_match_2_expected",
        "best_match_2_observed",
        "best_match_2_score",
        "best_match_3_expected",
        "best_match_3_observed",
        "best_match_3_score",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, summary_rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("REMAINING TOKENS VS IDEAL GEOMETRY\n")
        f.write("=" * 80 + "\n\n")

        best = sorted(summary_rows, key=lambda r: (-float(r["average_match_score"]), r["root"]))

        f.write("ROOTS WITH STRONGEST GEOMETRIC ALIGNMENT\n")
        for row in best[:25]:
            f.write(
                f"{row['root']} | "
                f"avg_match={float(row['average_match_score']):.3f} | "
                f"best1={row['best_match_1_expected']}<-{row['best_match_1_observed']}({float(row['best_match_1_score']):.3f}) | "
                f"best2={row['best_match_2_expected']}<-{row['best_match_2_observed']}({float(row['best_match_2_score']):.3f}) | "
                f"best3={row['best_match_3_expected']}<-{row['best_match_3_observed']}({float(row['best_match_3_score']):.3f})\n"
            )

        f.write("\nFULL ROOT SUMMARY\n")
        for row in sorted(summary_rows, key=lambda r: r["root"]):
            f.write(f"\n[{row['root']}]\n")
            f.write(f"average_match_score: {float(row['average_match_score']):.3f}\n")
            f.write(
                f"best1: {row['best_match_1_expected']} <- {row['best_match_1_observed']} "
                f"({float(row['best_match_1_score']):.3f})\n"
            )
            f.write(
                f"best2: {row['best_match_2_expected']} <- {row['best_match_2_observed']} "
                f"({float(row['best_match_2_score']):.3f})\n"
            )
            f.write(
                f"best3: {row['best_match_3_expected']} <- {row['best_match_3_observed']} "
                f"({float(row['best_match_3_score']):.3f})\n"
            )


def write_meta_json(path: Path, *, per_root_csv: Path, detail_csv: Path, summary_csv: Path, txt_path: Path, max_h: int):
    ensure_parent(path)
    data = {
        "inputs": {
            "per_root_csv": str(per_root_csv),
            "max_h": max_h,
        },
        "outputs": {
            "detail_csv": str(detail_csv),
            "summary_csv": str(summary_csv),
            "txt": str(txt_path),
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Compare filtered remaining tokens with ideal chain using phase/radial/angle geometry."
    )
    ap.add_argument("--per_root_csv", required=True)
    ap.add_argument("--out_detail_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_h", type=int, default=8)
    args = ap.parse_args()

    per_root_csv = Path(args.per_root_csv).resolve()
    out_detail_csv = Path(args.out_detail_csv).resolve()
    out_summary_csv = Path(args.out_summary_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    detail_rows, summary_rows = compare_geometry(
        per_root_csv,
        max_h=args.max_h,
    )

    write_detail_csv(out_detail_csv, detail_rows)
    write_summary_csv(out_summary_csv, summary_rows)
    write_txt(out_txt, summary_rows)
    write_meta_json(
        out_meta_json,
        per_root_csv=per_root_csv,
        detail_csv=out_detail_csv,
        summary_csv=out_summary_csv,
        txt_path=out_txt,
        max_h=args.max_h,
    )

    print("compare remaining with ideal geometry complete")
    print(f"root_count={len(summary_rows)}")
    print(f"detail_rows={len(detail_rows)}")


if __name__ == "__main__":
    main()