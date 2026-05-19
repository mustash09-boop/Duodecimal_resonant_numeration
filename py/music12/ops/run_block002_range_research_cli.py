from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        return default if s == "" else float(s)
    except Exception:
        return default


def expected_note_from_dirname(name: str) -> str:
    m = re.match(r"^\d+__[^_]+(?:_[^_]+)*__(.+)$", name)
    if m:
        return m.group(1)
    m = re.match(r"^\d+_(.+)$", name)
    if m:
        return m.group(1)
    return ""


def detect_prefix(note_dir: Path) -> str:
    for p in sorted(note_dir.glob("*__probe_matrix.csv")):
        return p.stem.replace("__probe_matrix", "")
    return ""


def run_cmd(cmd: list[str], env: dict[str, str], log_txt: Path) -> int:
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    with log_txt.open("a", encoding="utf-8") as f:
        f.write("\n=== RUNNING ===\n")
        f.write(" ".join(cmd) + "\n")
        if cp.stdout:
            f.write("\n--- STDOUT ---\n")
            f.write(cp.stdout)
            if not cp.stdout.endswith("\n"):
                f.write("\n")
        if cp.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(cp.stderr)
            if not cp.stderr.endswith("\n"):
                f.write("\n")
        f.write(f"\nRETURN CODE: {cp.returncode}\n")
    return cp.returncode


def load_framewise_readable(path: Path) -> tuple[str, float, int]:
    if not path.exists():
        return "", 0.0, 0

    votes: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected = _safe_str(row.get("selected_notes", ""))
            if not selected:
                continue
            notes = [x.strip() for x in selected.split("|") if x.strip()]
            if not notes:
                continue
            total += 1
            votes[notes[0]] += 1
            for extra in notes[1:]:
                votes[extra] += 0.25

    if not votes:
        return "", 0.0, total

    dominant = votes.most_common(1)[0][0]
    ratio = float(votes[dominant]) / max(total, 1)
    return dominant, ratio, total


def load_framewise_with_theory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "dominant_root": "",
            "dominant_root_ratio": 0.0,
            "mean_support_hits": 0.0,
            "mean_spiral_consistency": 0.0,
            "verdict_mode": "",
            "row_count": 0,
            "confirmed_count": 0,
            "uncertain_count": 0,
            "chosen_rc_mode": "",
            "chosen_rc_hz_mean": 0.0,
        }

    root_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()
    rc_counter: Counter[str] = Counter()
    support_hits = []
    consistency = []
    rc_hz = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        root = _safe_str(row.get("best_theoretical_root_token", ""))
        if root:
            root_counter[root] += 1

        verdict = _safe_str(row.get("theoretical_chain_verdict", ""))
        if verdict:
            verdict_counter[verdict] += 1

        chosen = _safe_str(row.get("chosen_rc_note", ""))
        if chosen:
            rc_counter[chosen] += 1

        support_hits.append(_safe_float(row.get("support_hits", 0.0), 0.0))
        consistency.append(_safe_float(row.get("spiral_consistency_score", 0.0), 0.0))
        hz = _safe_float(row.get("chosen_rc_hz", 0.0), 0.0)
        if hz > 0:
            rc_hz.append(hz)

    dominant_root = root_counter.most_common(1)[0][0] if root_counter else ""
    dominant_root_ratio = float(root_counter[dominant_root]) / max(len(rows), 1) if dominant_root else 0.0
    verdict_mode = verdict_counter.most_common(1)[0][0] if verdict_counter else ""
    chosen_rc_mode = rc_counter.most_common(1)[0][0] if rc_counter else ""

    return {
        "dominant_root": dominant_root,
        "dominant_root_ratio": dominant_root_ratio,
        "mean_support_hits": sum(support_hits) / max(len(support_hits), 1),
        "mean_spiral_consistency": sum(consistency) / max(len(consistency), 1),
        "verdict_mode": verdict_mode,
        "row_count": len(rows),
        "confirmed_count": verdict_counter.get("CHAIN_CONFIRMED", 0),
        "uncertain_count": verdict_counter.get("CHAIN_UNCERTAIN", 0),
        "chosen_rc_mode": chosen_rc_mode,
        "chosen_rc_hz_mean": sum(rc_hz) / max(len(rc_hz), 1),
    }


def confidence_label(expected_note: str, chosen_rc_mode: str, dominant_root: str, verdict_mode: str, dominant_root_ratio: float) -> str:
    if dominant_root and expected_note and dominant_root == expected_note and verdict_mode == "CHAIN_CONFIRMED":
        return "HIGH"
    if dominant_root and expected_note and dominant_root == expected_note and dominant_root_ratio >= 0.20:
        return "MEDIUM"
    if chosen_rc_mode or dominant_root:
        return "LOW"
    return "UNRESOLVED"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run research chain over all note directories and keep going even when some stages are doubtful or fail."
    )
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--bridge_script", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--plot_top_k_probes", type=int, default=120)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    bridge_script = Path(args.bridge_script).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "py")
    py = sys.executable

    note_dirs = sorted([p for p in reports_root.iterdir() if p.is_dir()])
    rows_out: list[dict[str, Any]] = []
    txt_lines: list[str] = []
    txt_lines.append("BLOCK002 RANGE RESEARCH SUMMARY")
    txt_lines.append("=" * 100)
    txt_lines.append(f"reports_root: {reports_root}")
    txt_lines.append("")

    for note_dir in note_dirs:
        prefix = detect_prefix(note_dir)
        if not prefix:
            continue

        expected = expected_note_from_dirname(note_dir.name)
        stage_log = note_dir / f"{prefix}__research_chain_log.txt"
        stage_log.write_text("", encoding="utf-8")

        matrix_csv = note_dir / f"{prefix}__probe_matrix.csv"
        times_csv = note_dir / f"{prefix}__probe_times.csv"
        coords_csv = note_dir / f"{prefix}__probe_coords.csv"

        framewise_csv = note_dir / f"{prefix}__framewise.csv"
        framewise_readable_csv = note_dir / f"{prefix}__framewise_readable.csv"
        candidate_meta_json = note_dir / f"{prefix}__candidate_meta.json"

        theory_csv = note_dir / f"{prefix}__theory_match.csv"
        bridge_csv = note_dir / f"{prefix}__framewise_with_theory.csv"
        bridge_meta_json = note_dir / f"{prefix}__framewise_with_theory_meta.json"

        adaptive_csv = note_dir / f"{prefix}__adaptive_root.csv"
        adaptive_meta_json = note_dir / f"{prefix}__adaptive_root_meta.json"

        regime_csv = note_dir / f"{prefix}__regime_confirmation.csv"
        regime_meta_json = note_dir / f"{prefix}__regime_confirmation_meta.json"

        stabilized_csv = note_dir / f"{prefix}__stabilized.csv"
        stabilized_meta_json = note_dir / f"{prefix}__stabilized_meta.json"

        spiral_png = note_dir / f"{prefix}__field_spiral.png"
        field3d_png = note_dir / f"{prefix}__field_3d.png"

        rc_candidate = run_cmd([
            py, "-m", "music12.blocks.Block002_audio_recogn.resonance_candidate_inference_cli",
            "--matrix_csv", str(matrix_csv),
            "--times_csv", str(times_csv),
            "--coords_csv", str(coords_csv),
            "--out_framewise_csv", str(framewise_csv),
            "--out_framewise_readable_csv", str(framewise_readable_csv),
            "--out_meta_json", str(candidate_meta_json),
            "--energy_threshold", "0.01",
            "--top_n_candidates", "24",
            "--tolerance_ratio", "0.03",
            "--analysis_min_hz", "16",
            "--analysis_max_hz", "22000",
            "--max_polyphonic_candidates", "8",
        ], env, stage_log)

        rc_theory = -999
        if rc_candidate == 0:
            rc_theory = run_cmd([
                py, "-m", "music12.blocks.Block002_audio_recogn.theoretical_chain_window_match_cli",
                "--framewise_csv", str(framewise_csv),
                "--out_csv", str(theory_csv),
            ], env, stage_log)

        rc_bridge = -999
        if rc_theory == 0:
            rc_bridge = run_cmd([
                py, str(bridge_script),
                "--framewise_csv", str(framewise_csv),
                "--theory_match_csv", str(theory_csv),
                "--out_csv", str(bridge_csv),
                "--out_meta_json", str(bridge_meta_json),
            ], env, stage_log)

        rc_adaptive = -999
        if rc_bridge == 0:
            rc_adaptive = run_cmd([
                py, "-m", "music12.blocks.Block002_audio_recogn.adaptive_root_selection_cli",
                "--in_csv", str(bridge_csv),
                "--out_csv", str(adaptive_csv),
                "--out_meta_json", str(adaptive_meta_json),
            ], env, stage_log)

        rc_regime = -999
        if rc_adaptive == 0:
            rc_regime = run_cmd([
                py, "-m", "music12.blocks.Block002_audio_recogn.regime_harmonic_confirmation_cli",
                "--in_csv", str(adaptive_csv),
                "--out_csv", str(regime_csv),
                "--out_meta_json", str(regime_meta_json),
            ], env, stage_log)

        rc_stabilize = -999
        if rc_bridge == 0:
            rc_stabilize = run_cmd([
                py, "-m", "music12.blocks.Block002_audio_recogn.stabilize_chain_candidates_cli",
                "--framewise_with_theory_csv", str(bridge_csv),
                "--out_csv", str(stabilized_csv),
                "--out_meta_json", str(stabilized_meta_json),
            ], env, stage_log)

        rc_plot2d = run_cmd([
            py, "-m", "music12.blocks.Block002_audio_recogn.resonance_probe12_plot_cli",
            "--matrix_csv", str(matrix_csv),
            "--times_csv", str(times_csv),
            "--coords_csv", str(coords_csv),
            "--out_png", str(spiral_png),
            "--plot_mode", "spiral",
            "--display_mode", "log",
            "--top_k_probes", str(args.plot_top_k_probes),
            "--title", f"{prefix} spiral field",
        ], env, stage_log)

        rc_plot3d = run_cmd([
            py, "-m", "music12.blocks.Block002_audio_recogn.resonance_probe12_plot3d_cli",
            "--matrix_csv", str(matrix_csv),
            "--times_csv", str(times_csv),
            "--coords_csv", str(coords_csv),
            "--out_png", str(field3d_png),
            "--display_mode", "log",
            "--top_k_probes", str(args.plot_top_k_probes),
            "--title", f"{prefix} 3D field",
        ], env, stage_log)

        dominant_framewise_note, dominant_framewise_ratio, framewise_rows = load_framewise_readable(framewise_readable_csv)
        bridge_stats = load_framewise_with_theory(bridge_csv)

        final_detected = bridge_stats["dominant_root"] or dominant_framewise_note or bridge_stats["chosen_rc_mode"]
        confidence = confidence_label(
            expected_note=expected,
            chosen_rc_mode=bridge_stats["chosen_rc_mode"],
            dominant_root=bridge_stats["dominant_root"],
            verdict_mode=bridge_stats["verdict_mode"],
            dominant_root_ratio=bridge_stats["dominant_root_ratio"],
        )

        row = {
            "note_dir": note_dir.name,
            "prefix": prefix,
            "expected_note": expected,
            "dominant_framewise_note": dominant_framewise_note,
            "dominant_framewise_ratio": round(dominant_framewise_ratio, 6),
            "framewise_row_count": framewise_rows,
            "dominant_root": bridge_stats["dominant_root"],
            "dominant_root_ratio": round(bridge_stats["dominant_root_ratio"], 6),
            "chosen_rc_mode": bridge_stats["chosen_rc_mode"],
            "chosen_rc_hz_mean": round(bridge_stats["chosen_rc_hz_mean"], 6),
            "mean_support_hits": round(bridge_stats["mean_support_hits"], 6),
            "mean_spiral_consistency": round(bridge_stats["mean_spiral_consistency"], 6),
            "verdict_mode": bridge_stats["verdict_mode"],
            "confirmed_count": bridge_stats["confirmed_count"],
            "uncertain_count": bridge_stats["uncertain_count"],
            "final_detected_note": final_detected,
            "confidence": confidence,
            "rc_candidate": rc_candidate,
            "rc_theory": rc_theory,
            "rc_bridge": rc_bridge,
            "rc_adaptive": rc_adaptive,
            "rc_regime": rc_regime,
            "rc_stabilize": rc_stabilize,
            "rc_plot2d": rc_plot2d,
            "rc_plot3d": rc_plot3d,
            "stage_log": str(stage_log),
        }
        rows_out.append(row)

        txt_lines.append(f"[{note_dir.name}]")
        txt_lines.append(f"expected_note            : {expected}")
        txt_lines.append(f"dominant_framewise_note  : {dominant_framewise_note}")
        txt_lines.append(f"dominant_root            : {bridge_stats['dominant_root']}")
        txt_lines.append(f"chosen_rc_mode           : {bridge_stats['chosen_rc_mode']}")
        txt_lines.append(f"final_detected_note      : {final_detected}")
        txt_lines.append(f"confidence               : {confidence}")
        txt_lines.append(f"verdict_mode             : {bridge_stats['verdict_mode']}")
        txt_lines.append(f"mean_support_hits        : {bridge_stats['mean_support_hits']:.3f}")
        txt_lines.append(f"mean_spiral_consistency  : {bridge_stats['mean_spiral_consistency']:.3f}")
        txt_lines.append(
            "stages rc                : "
            f"candidate={rc_candidate}, theory={rc_theory}, bridge={rc_bridge}, "
            f"adaptive={rc_adaptive}, regime={rc_regime}, stabilize={rc_stabilize}, "
            f"plot2d={rc_plot2d}, plot3d={rc_plot3d}"
        )
        txt_lines.append("")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    else:
        out_csv.write_text("", encoding="utf-8")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
