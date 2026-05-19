from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n=== RUNNING ===")
    print(" ".join(cmd))
    cp = subprocess.run(cmd, check=False)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Block002 5-stage chain sequentially for one note directory.")
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--note_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--plot_top_k_probes", type=int, default=120)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    note_dir = Path(args.note_dir).resolve()
    prefix = args.prefix

    py = sys.executable
    env_pythonpath = str(project_root / "py")

    def mod(module: str, extra: list[str]) -> list[str]:
        return [py, "-m", module] + extra

    common_env = dict()

    matrix_csv = note_dir / f"{prefix}__probe_matrix.csv"
    times_csv = note_dir / f"{prefix}__probe_times.csv"
    coords_csv = note_dir / f"{prefix}__probe_coords.csv"

    framewise_csv = note_dir / f"{prefix}__framewise.csv"
    framewise_readable_csv = note_dir / f"{prefix}__framewise_readable.csv"
    candidate_meta_json = note_dir / f"{prefix}__candidate_meta.json"

    theory_csv = note_dir / f"{prefix}__theory_match.csv"
    bridge_csv = note_dir / f"{prefix}__framewise_with_theory.csv"
    bridge_meta = note_dir / f"{prefix}__framewise_with_theory_meta.json"

    adaptive_csv = note_dir / f"{prefix}__adaptive_root.csv"
    adaptive_meta = note_dir / f"{prefix}__adaptive_root_meta.json"

    regime_csv = note_dir / f"{prefix}__regime_confirmation.csv"
    regime_meta = note_dir / f"{prefix}__regime_confirmation_meta.json"

    stabilized_csv = note_dir / f"{prefix}__stabilized.csv"
    stabilized_meta = note_dir / f"{prefix}__stabilized_meta.json"

    spiral_png = note_dir / f"{prefix}__field_spiral.png"
    field3d_png = note_dir / f"{prefix}__field_3d.png"

    cmds = [
        mod("music12.blocks.Block002_audio_recogn.resonance_candidate_inference_cli", [
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
        ]),
        mod("music12.blocks.Block002_audio_recogn.theoretical_chain_window_match_cli", [
            "--framewise_csv", str(framewise_csv),
            "--out_csv", str(theory_csv),
        ]),
        [py, str(Path(__file__).resolve().parent / "framewise_theory_bridge_cli.py"),
            "--framewise_csv", str(framewise_csv),
            "--theory_match_csv", str(theory_csv),
            "--out_csv", str(bridge_csv),
            "--out_meta_json", str(bridge_meta),
        ],
        mod("music12.blocks.Block002_audio_recogn.adaptive_root_selection_cli", [
            "--in_csv", str(bridge_csv),
            "--out_csv", str(adaptive_csv),
            "--out_meta_json", str(adaptive_meta),
        ]),
        mod("music12.blocks.Block002_audio_recogn.regime_harmonic_confirmation_cli", [
            "--in_csv", str(adaptive_csv),
            "--out_csv", str(regime_csv),
            "--out_meta_json", str(regime_meta),
        ]),
        mod("music12.blocks.Block002_audio_recogn.stabilize_chain_candidates_cli", [
            "--framewise_with_theory_csv", str(bridge_csv),
            "--out_csv", str(stabilized_csv),
            "--out_meta_json", str(stabilized_meta),
        ]),
        mod("music12.blocks.Block002_audio_recogn.resonance_probe12_plot_cli", [
            "--matrix_csv", str(matrix_csv),
            "--times_csv", str(times_csv),
            "--coords_csv", str(coords_csv),
            "--out_png", str(spiral_png),
            "--plot_mode", "spiral",
            "--display_mode", "log",
            "--top_k_probes", str(args.plot_top_k_probes),
            "--title", f"{prefix} spiral field",
        ]),
        mod("music12.blocks.Block002_audio_recogn.resonance_probe12_plot3d_cli", [
            "--matrix_csv", str(matrix_csv),
            "--times_csv", str(times_csv),
            "--coords_csv", str(coords_csv),
            "--out_png", str(field3d_png),
            "--display_mode", "log",
            "--top_k_probes", str(args.plot_top_k_probes),
            "--title", f"{prefix} 3D field",
        ]),
    ]

    # run with PYTHONPATH set to project_root/py
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = env_pythonpath

    for cmd in cmds:
        print("\n=== RUNNING ===")
        print(" ".join(cmd))
        cp = subprocess.run(cmd, env=env, check=False)
        if cp.returncode != 0:
            raise SystemExit(cp.returncode)

    print("\n=== DONE ===")
    print(json.dumps({
        "note_dir": str(note_dir),
        "prefix": prefix,
        "outputs": {
            "framewise_csv": str(framewise_csv),
            "theory_csv": str(theory_csv),
            "bridge_csv": str(bridge_csv),
            "adaptive_csv": str(adaptive_csv),
            "regime_csv": str(regime_csv),
            "stabilized_csv": str(stabilized_csv),
            "spiral_png": str(spiral_png),
            "field_3d_png": str(field3d_png),
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
