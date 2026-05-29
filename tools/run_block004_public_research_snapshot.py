from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
RUNNER = PROJECT_ROOT / "py" / "music12" / "blocks" / "Block004_real_instruments" / "instrument_pipeline_runner_cli.py"
LINEAGE_AUGMENTER = PROJECT_ROOT / "tools" / "augment_block004_passports_with_harmonic_lineage.py"


def find_manifest(dataset_dir: Path) -> Path:
    candidates = []
    for pattern in [
        "01_manifest12/*__manifest12.csv",
        "01_manifest12/*.csv",
        "20_manifest/*__manifest12.csv",
        "20_manifest/*.csv",
    ]:
        candidates.extend(sorted(dataset_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"Could not find manifest CSV under {dataset_dir}")
    return candidates[0]


def find_audio_dir(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "00_sources" / "audio_notes_wav",
        dataset_dir / "00_sources",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find audio directory under {dataset_dir}")


def run(cmd: list[str]) -> None:
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Public wrapper for reproducing the Block004 isolated-note research pipeline."
    )
    ap.add_argument("--instrument-name", required=True)
    ap.add_argument("--dataset-dir", required=True, help="Block004_data/<instrument> directory")
    ap.add_argument(
        "--stages",
        default="dense,chain,root,box,box_split,clean_box,dense_vs_theory,spiral12,note_box_profile,spiral3d,harmonic_chain_spiral3d,relation,passport",
    )
    ap.add_argument("--layer", default="01_core_notes")
    ap.add_argument("--harmonic-morphology-html", default="")
    ap.add_argument("--harmonic-morphology-out-dir", default="")
    ap.add_argument("--refresh-lineage-passport", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    manifest_csv = find_manifest(dataset_dir)
    audio_dir = find_audio_dir(dataset_dir)
    reports_root = dataset_dir / "10_reports"

    cmd = [
        sys.executable,
        str(RUNNER),
        "--instrument_name",
        args.instrument_name,
        "--audio_dir",
        str(audio_dir),
        "--manifest_csv",
        str(manifest_csv),
        "--reports_root",
        str(reports_root),
        "--layer",
        args.layer,
        "--stages",
        args.stages,
    ]

    if args.harmonic_morphology_html:
        cmd.extend(["--harmonic_morphology_html", args.harmonic_morphology_html])
    if args.harmonic_morphology_out_dir:
        cmd.extend(["--harmonic_morphology_out_dir", args.harmonic_morphology_out_dir])

    run(cmd)

    if args.refresh_lineage_passport:
        run([sys.executable, str(LINEAGE_AUGMENTER)])

    print("\nDONE")
    print("Instrument :", args.instrument_name)
    print("Dataset    :", dataset_dir)
    print("Manifest   :", manifest_csv)
    print("Audio dir  :", audio_dir)
    print("Reports    :", reports_root)


if __name__ == "__main__":
    main()

