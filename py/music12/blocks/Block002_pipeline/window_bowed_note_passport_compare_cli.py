# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(path_str: str, payload: dict[str, Any]) -> None:
    if not str(path_str).strip():
        return
    Path(path_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _get_candidate_summary(label: str, chain_json_path: Path) -> dict[str, Any]:
    data = _load_json(chain_json_path)
    best = dict(data.get("best_track", {}) or {})
    hits = best.get("representative_hits", []) or []
    hit_by_index = {int(h.get("harmonic_index", 0)): h for h in hits}
    amp1 = _safe_float((hit_by_index.get(1) or {}).get("matched_amplitude"), 0.0)
    amp2 = _safe_float((hit_by_index.get(2) or {}).get("matched_amplitude"), 0.0)
    return {
        "label": label,
        "root_note_token": str(best.get("root_note_token", "")),
        "root_hz_mean": _safe_float(best.get("root_hz_mean"), 0.0),
        "frame_count": int(best.get("frame_count", 0)),
        "harmonic_presence_profile": dict(best.get("harmonic_presence_profile", {}) or {}),
        "amp_h1": amp1,
        "amp_h2": amp2,
        "h2_over_h1_ratio": (amp2 / amp1) if amp1 > 0.0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare the current window bowed-layer evidence against violin and cello isolated-note passport scenarios."
    )
    ap.add_argument("--data-grounded-owner-csv", required=True)
    ap.add_argument("--violin-b5-chain-json", required=True)
    ap.add_argument("--violin-a5-chain-json", required=True)
    ap.add_argument("--cello-a5-chain-json", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    _write_progress(
        args.progress_json,
        {"status": "running", "phase": "loading_inputs"},
    )

    rows = _load_csv(Path(args.data_grounded_owner_csv))
    upper = [r for r in rows if str(r.get("data_grounded_support_role", "")).strip() == "UPPER_DIRECT_SUPPORT"]
    lower = [r for r in rows if str(r.get("data_grounded_support_role", "")).strip() == "LOWER_OCTAVE_SUPPORT"]

    observed_upper_hz = _mean([_safe_float(r.get("frequency_hz"), 0.0) for r in upper])
    observed_lower_hz = _mean([_safe_float(r.get("frequency_hz"), 0.0) for r in lower])
    observed_upper_energy = _mean([_safe_float(r.get("energy"), 0.0) for r in upper])
    observed_lower_energy = _mean([_safe_float(r.get("energy"), 0.0) for r in lower])
    observed_ratio = observed_upper_hz / observed_lower_hz if observed_lower_hz > 0.0 else 0.0
    observed_upper_token = "B.5'-"
    observed_lower_token = "A.5'-"

    candidates = [
        _get_candidate_summary("violin_B5_root", Path(args.violin_b5_chain_json)),
        _get_candidate_summary("violin_A5_root", Path(args.violin_a5_chain_json)),
        _get_candidate_summary("cello_A5_root", Path(args.cello_a5_chain_json)),
    ]

    observed_h1_count = len(lower)
    observed_h2_count = len(upper)
    observed_h2_over_h1_count_ratio = (float(observed_h2_count) / float(observed_h1_count)) if observed_h1_count else 0.0
    observed_h2_over_h1_energy_ratio = (observed_upper_energy / observed_lower_energy) if observed_lower_energy > 0.0 else 0.0

    compare_rows = []
    for candidate in candidates:
        root_hz = float(candidate["root_hz_mean"])
        amp_ratio = float(candidate["h2_over_h1_ratio"])
        root_match = 1.0 - min(1.0, abs(root_hz - observed_lower_hz) / 120.0)
        octave_match = 1.0 - min(1.0, abs((root_hz * 2.0) - observed_upper_hz) / 180.0)
        ratio_match = 1.0 - min(1.0, abs(amp_ratio - observed_h2_over_h1_energy_ratio) / 1.2)
        structural_score = 0.45 * root_match + 0.35 * octave_match + 0.20 * ratio_match
        compare_rows.append(
            {
                "label": candidate["label"],
                "root_note_token": candidate["root_note_token"],
                "root_hz_mean": candidate["root_hz_mean"],
                "frame_count": candidate["frame_count"],
                "amp_h1": candidate["amp_h1"],
                "amp_h2": candidate["amp_h2"],
                "h2_over_h1_ratio": candidate["h2_over_h1_ratio"],
                "root_match_score": root_match,
                "octave_match_score": octave_match,
                "ratio_match_score": ratio_match,
                "structural_score": structural_score,
            }
        )

    compare_rows.sort(key=lambda item: float(item["structural_score"]), reverse=True)
    best_label = compare_rows[0]["label"] if compare_rows else ""

    summary_lines = [
        "WINDOW BOWED NOTE PASSPORT COMPARE",
        "=" * 72,
        f"observed_upper_token                : {observed_upper_token}",
        f"observed_lower_token                : {observed_lower_token}",
        f"observed_upper_count                : {len(upper)}",
        f"observed_lower_count                : {len(lower)}",
        f"observed_upper_mean_hz              : {observed_upper_hz:.6f}",
        f"observed_lower_mean_hz              : {observed_lower_hz:.6f}",
        f"observed_upper_lower_ratio          : {observed_ratio:.6f}",
        f"observed_h2_over_h1_count_ratio     : {observed_h2_over_h1_count_ratio:.6f}",
        f"observed_h2_over_h1_energy_ratio    : {observed_h2_over_h1_energy_ratio:.6f}",
        "",
        "candidate_scores:",
    ]
    for row in compare_rows:
        summary_lines.append(
            "  "
            f"{row['label']}: score={row['structural_score']:.6f} "
            f"root={row['root_note_token']} h2/h1={row['h2_over_h1_ratio']:.6f}"
        )
    summary_lines.extend(
        [
            "",
            f"best_structural_match              : {best_label}",
            "",
            "interpretation:",
            "  If A.5-root candidates outrank violin_B5_root, the window behaves more like",
            "  a bowed note with hidden fundamental and exposed second harmonic than like",
            "  a direct B.5-root note.",
        ]
    )
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "window_bowed_note_passport_compare",
                "observed": {
                    "upper_token": observed_upper_token,
                    "lower_token": observed_lower_token,
                    "upper_count": len(upper),
                    "lower_count": len(lower),
                    "upper_mean_hz": observed_upper_hz,
                    "lower_mean_hz": observed_lower_hz,
                    "upper_lower_ratio": observed_ratio,
                    "h2_over_h1_count_ratio": observed_h2_over_h1_count_ratio,
                    "h2_over_h1_energy_ratio": observed_h2_over_h1_energy_ratio,
                },
                "candidate_scores": compare_rows,
                "best_structural_match": best_label,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    _write_progress(
        args.progress_json,
        {"status": "done", "phase": "complete", "best_structural_match": best_label},
    )


if __name__ == "__main__":
    main()
