from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")


def read_manifest(path: Path, layer: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("parse_status", "OK") != "OK":
                continue
            if layer != "all" and row.get("semantic_layer", "") != layer:
                continue
            rows.append(row)
    return rows


def stem_of(name: str) -> str:
    """
    Имя папки/отчёта. Апостроф убираем только из имени отчёта,
    но НЕ из имени исходного WAV. WAV берётся отдельно из original_filename.
    """
    return Path(name).stem.replace("'", "").replace('"', "")


def note_for_theory(note12: str) -> str:
    """
    В именах файлов чистая нота может быть записана как 8.8'-,
    а в эталонной таблице теории обычно как 8.8-.
    Для сравнения с theoretical CSV убираем только маркер чистой ноты.
    Микро-альтерации вида 'i2 / 'a3 не трогаем.
    """
    n = (note12 or "").strip()
    if n.endswith("'-"):
        return n.replace("'-", "-")
    return n


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def safe_tag(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    text = text.strip("._-")
    return text or "stage"


def run_module(module: str, module_args: list[str], args, stage_tag: str) -> None:
    if getattr(args, "use_maxwell", False):
        tag_parts = [
            getattr(args, "maxwell_tag_prefix", ""),
            getattr(args, "instrument_name", ""),
            stage_tag,
        ]
        tag = "__".join(safe_tag(x) for x in tag_parts if str(x).strip())
        cmd = [
            sys.executable, "-m", "music12.demons.demon_maxwell_cli",
            "-m", module,
            "--task-class", args.maxwell_task_class,
            "--project-root", str(PROJECT_ROOT),
            "--logdir", args.maxwell_logdir,
            "--tag", tag,
            "--",
            *module_args,
        ]
    else:
        cmd = [sys.executable, "-m", module, *module_args]

    run_cmd(cmd)


def out_dir(args) -> Path:
    return Path(args.box_out_dir)


def passport_notes_csv(args) -> Path:
    return Path(args.passport_out_dir) / f"{args.instrument_name}__instrument_passport_notes.csv"


def passport_json(args) -> Path:
    return Path(args.passport_out_dir) / f"{args.instrument_name}__instrument_passport.json"


def passport_md(args) -> Path:
    return Path(args.passport_out_dir) / f"{args.instrument_name}__instrument_passport.md"


def box_breath_csv(args) -> Path:
    return out_dir(args) / f"{args.instrument_name}__box_breath.csv"


def box_resonance_csv(args) -> Path:
    return out_dir(args) / f"{args.instrument_name}__box_resonance.csv"


def box_relation_csv(args) -> Path:
    return out_dir(args) / f"{args.instrument_name}__box_harmonic_relation.csv"


def run_dense(audio_path: Path, report_dir: Path, stem: str, args) -> Path:
    out_csv = report_dir / f"{stem}__dense.csv"
    run_module(
        "music12.blocks.Block002_audio_recogn.dense_spectral_observer_cli",
        [
            "--wav", str(audio_path),
            "--out_csv", str(out_csv),
            "--window_sec", args.window_sec,
            "--step_sec", args.step_sec,
            "--peak_threshold", args.peak_threshold,
        ],
        args,
        f"dense__{stem}",
    )
    return out_csv


def run_chain(dense_csv: Path, report_dir: Path, stem: str, args) -> None:
    run_module(
        "music12.blocks.Block002_audio_recogn.dense_harmonic_chain_builder_cli",
        [
            "--dense_csv", str(dense_csv),
            "--out_chain_candidates_csv", str(report_dir / f"{stem}__dense_chain_candidates.csv"),
            "--out_chain_summary_json", str(report_dir / f"{stem}__dense_chain_summary.json"),
            "--out_chain_summary_txt", str(report_dir / f"{stem}__dense_chain_summary.txt"),
            "--max_harmonic", args.max_harmonic,
            "--tolerance_cents", args.tolerance_cents,
            "--cluster_cents", args.cluster_cents,
            "--root_min_hz", args.root_min_hz,
            "--root_max_hz", args.root_max_hz,
            "--anchor_token", args.anchor_token,
            "--anchor_hz", args.anchor_hz,
            "--top_n_per_frame", args.top_n_per_frame,
            "--max_gap_frames", args.max_gap_frames,
            "--max_root_jump_cents", args.max_root_jump_cents,
            "--min_link_score", args.min_link_score,
        ],
        args,
        f"chain__{stem}",
    )


def run_root(dense_csv: Path, report_dir: Path, stem: str, expected_note: str, args) -> None:
    run_module(
        "music12.blocks.Block002_audio_recogn.root_from_harmonic_consensus_cli",
        [
            "--dense_csv", str(dense_csv),
            "--expected_note", expected_note,
            "--out_root_candidates_csv", str(report_dir / f"{stem}__root_consensus_candidates.csv"),
            "--out_cluster_summary_csv", str(report_dir / f"{stem}__root_consensus_clusters.csv"),
            "--out_summary_txt", str(report_dir / f"{stem}__root_consensus_summary.txt"),
            "--out_meta_json", str(report_dir / f"{stem}__root_consensus_meta.json"),
            "--anchor_token", args.anchor_token,
            "--anchor_hz", args.anchor_hz,
            "--harmonic_min", args.harmonic_min,
            "--harmonic_max", args.max_harmonic,
            "--root_min_hz", args.root_min_hz,
            "--root_max_hz", args.root_max_hz,
            "--min_amplitude", args.min_amplitude,
            "--cluster_cents", args.cluster_cents,
            "--expected_root_tolerance_cents", args.expected_root_tolerance_cents,
        ],
        args,
        f"root__{stem}",
    )


def run_box(args) -> None:
    out_root = out_dir(args)
    out_root.mkdir(parents=True, exist_ok=True)

    run_module(
        "music12.blocks.Block004_real_instruments.instrument_box_from_dense_cli",
        [
            "--reports_root", args.reports_root,
            "--out_dense_global_presence_csv", str(out_root / f"{args.instrument_name}__dense_global_presence.csv"),
            "--out_dense_range_presence_csv", str(out_root / f"{args.instrument_name}__dense_range_presence.csv"),
            "--out_dense_frequency_clusters_csv", str(out_root / f"{args.instrument_name}__dense_frequency_clusters.csv"),
            "--out_dense_summary_txt", str(out_root / f"{args.instrument_name}__dense_box_summary.txt"),
            "--anchor_token", args.anchor_token,
            "--anchor_hz", args.anchor_hz,
            "--cluster_cents", args.cluster_cents,
        ],
        args,
        "box",
    )


def run_box_split(args) -> None:
    out_root = out_dir(args)
    out_root.mkdir(parents=True, exist_ok=True)

    run_module(
        "music12.blocks.Block004_real_instruments.split_box_layers_cli",
        [
            "--box_csv", args.box_csv,
            "--out_breath_csv", str(box_breath_csv(args)),
            "--out_resonance_csv", str(box_resonance_csv(args)),
        ],
        args,
        "box_split",
    )


def run_dense_vs_theory(report_dir: Path, stem: str, expected_note: str, args) -> None:
    expected_for_theory = note_for_theory(expected_note)

    run_module(
        "music12.blocks.Block002_audio_recogn.compare_dense_chain_vs_theoretical_cli",
        [
            "--dense_chain_summary_json", str(report_dir / f"{stem}__dense_chain_summary.json"),
            "--theoretical_csv", args.theoretical_csv,
            "--expected_note", expected_for_theory,
            "--out_csv", str(report_dir / f"{stem}__dense_vs_theory.csv"),
            "--out_txt", str(report_dir / f"{stem}__dense_vs_theory.txt"),
        ],
        args,
        f"dense_vs_theory__{stem}",
    )


def run_clean_box(dense_csv: Path, report_dir: Path, stem: str, expected_note: str, args) -> Path:
    out_clean = report_dir / f"{stem}__dense_unified_clean.csv"
    clean_box_csv = args.clean_box_csv or str(
        box_resonance_csv(args) if box_resonance_csv(args).exists() else Path(args.box_csv)
    )

    run_module(
        "music12.blocks.Block004_real_instruments.subtract_box_from_dense_cli",
        [
            "--dense_csv", str(dense_csv),
            "--box_csv", clean_box_csv,
            "--out_clean_csv", str(out_clean),
            "--out_removed_csv", str(report_dir / f"{stem}__dense_unified_removed_box.csv"),
            "--out_summary_txt", str(report_dir / f"{stem}__dense_unified_clean_summary.txt"),
            "--expected_note", expected_note,
            "--anchor_token", args.anchor_token,
            "--anchor_hz", args.anchor_hz,
            "--max_harmonic", args.max_harmonic,
            "--harmonic_tolerance_cents", args.tolerance_cents,
            "--box_tolerance_hz", args.box_tolerance_hz,
            "--min_box_percent_notes", args.min_box_percent_notes,
            "--min_box_amp", args.min_box_amp,
            "--max_box_hz", args.root_max_hz,
        ],
        args,
        f"clean_box__{stem}",
    )
    return out_clean


def run_spiral12(dense_csv: Path, report_dir: Path, stem: str, args) -> None:
    run_module(
        "music12.blocks.Block002_audio_recogn.spiral12_from_dense_clean_cli",
        [
            "--dense_csv", str(dense_csv),
            "--out_csv", str(report_dir / f"{stem}__spiral12_clean_points.csv"),
            "--out_png", str(report_dir / f"{stem}__spiral12_clean.png"),
            "--anchor_token", args.anchor_token,
            "--anchor_hz", args.anchor_hz,
            "--title", f"{stem} 12-radix spiral",
        ],
        args,
        f"spiral12__{stem}",
    )


def run_passport(args) -> None:
    out_root = Path(args.passport_out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "--instrument_name", args.instrument_name,
        "--reports_root", args.reports_root,
        "--box_csv", args.box_csv,
        "--out_notes_csv", str(passport_notes_csv(args)),
        "--out_json", str(passport_json(args)),
        "--out_md", str(passport_md(args)),
        "--box_top_n", args.box_top_n,
    ]

    if box_breath_csv(args).exists():
        cmd.extend(["--box_breath_csv", str(box_breath_csv(args))])
    if box_resonance_csv(args).exists():
        cmd.extend(["--box_resonance_csv", str(box_resonance_csv(args))])
    if box_relation_csv(args).exists():
        cmd.extend(["--box_relation_csv", str(box_relation_csv(args))])

    run_module(
        "music12.blocks.Block004_real_instruments.instrument_passport_builder_cli",
        cmd,
        args,
        "passport",
    )


def run_box_relation(args) -> None:
    if not box_resonance_csv(args).exists() and not Path(args.box_csv).exists():
        print(f"[MISSING BOX] no resonance box and no box_csv: {args.box_csv}")
        return

    if not passport_notes_csv(args).exists():
        print("\n[INFO] passport notes CSV not found; creating preliminary notes/passport before relation.")
        run_passport(args)

    if not passport_notes_csv(args).exists():
        print(f"[MISSING PASSPORT NOTES] {passport_notes_csv(args)}")
        return

    run_module(
        "music12.blocks.Block004_real_instruments.box_harmonic_relation_cli",
        [
            "--box_csv", str(box_resonance_csv(args) if box_resonance_csv(args).exists() else Path(args.box_csv)),
            "--notes_csv", str(passport_notes_csv(args)),
            "--out_csv", str(box_relation_csv(args)),
        ],
        args,
        "relation",
    )


def note_box_out_dir(args) -> Path:
    return Path(args.note_box_out_dir)


def spiral3d_out_dir(args) -> Path:
    return Path(args.spiral3d_out_dir)


def run_note_box_profile(args) -> None:
    out_root = note_box_out_dir(args)
    out_root.mkdir(parents=True, exist_ok=True)

    run_module(
        "music12.blocks.Block004_real_instruments.note_box_profile_builder_cli",
        [
            "--instrument_name", args.instrument_name,
            "--reports_root", args.reports_root,
            "--out_dir", str(out_root),
            "--harmonic_tolerance_cents", args.note_box_harmonic_tolerance_cents,
            "--min_presence_ratio", args.note_box_min_presence_ratio,
            "--min_frame_count", args.note_box_min_frame_count,
        ],
        args,
        "note_box_profile",
    )


def run_spiral3d(args) -> None:
    out_root = spiral3d_out_dir(args)
    out_root.mkdir(parents=True, exist_ok=True)

    run_module(
        "music12.blocks.Block004_real_instruments.note_box_spiral3d_builder_cli",
        [
            "--instrument_name", args.instrument_name,
            "--reports_root", args.reports_root,
            "--note_box_dir", str(note_box_out_dir(args)),
            "--out_dir", str(out_root),
            "--harmonic_tolerance_cents", args.note_box_harmonic_tolerance_cents,
        ],
        args,
        "spiral3d",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal phased instrument pipeline runner.")

    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--reports_root", required=True)

    ap.add_argument("--layer", default="01_core_notes")
    ap.add_argument(
        "--stages",
        default="dense,chain,root,box,box_split,clean_box,dense_vs_theory,spiral12,note_box_profile,spiral3d,relation,passport",
    )

    ap.add_argument("--box_csv", default="")
    ap.add_argument("--clean_box_csv", default="")
    ap.add_argument("--box_out_dir", default="")
    ap.add_argument("--passport_out_dir", default="")
    ap.add_argument("--note_box_out_dir", default="")
    ap.add_argument("--spiral3d_out_dir", default="")
    ap.add_argument(
        "--theoretical_csv",
        default=r"E:\Duodecimal_resonant_numeration\py\music12\core\reference_theoretical_harmonics12.csv",
    )

    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", default="440")

    ap.add_argument("--window_sec", default="0.05")
    ap.add_argument("--step_sec", default="0.0166666667")
    ap.add_argument("--peak_threshold", default="0.01")

    ap.add_argument("--max_harmonic", default="12")
    ap.add_argument("--harmonic_min", default="1")
    ap.add_argument("--tolerance_cents", default="45")
    ap.add_argument("--cluster_cents", default="35")

    ap.add_argument("--root_min_hz", default="20")
    ap.add_argument("--root_max_hz", default="18000")
    ap.add_argument("--min_amplitude", default="0")
    ap.add_argument("--expected_root_tolerance_cents", default="120")

    ap.add_argument("--top_n_per_frame", default="12")
    ap.add_argument("--max_gap_frames", default="4")
    ap.add_argument("--max_root_jump_cents", default="80")
    ap.add_argument("--min_link_score", default="0.05")

    ap.add_argument("--box_tolerance_hz", default="5")
    ap.add_argument("--min_box_percent_notes", default="40")
    ap.add_argument("--min_box_amp", default="0")
    ap.add_argument("--box_top_n", default="120")
    ap.add_argument("--note_box_harmonic_tolerance_cents", default="18.0")
    ap.add_argument("--note_box_min_presence_ratio", default="0.05")
    ap.add_argument("--note_box_min_frame_count", default="2")

    ap.add_argument("--use_maxwell", action="store_true")
    ap.add_argument("--maxwell_logdir", default="_demon_logs")
    ap.add_argument("--maxwell_task_class", default="instrument_analysis")
    ap.add_argument("--maxwell_tag_prefix", default="")

    args = ap.parse_args()

    reports_root = Path(args.reports_root)
    if not args.box_out_dir:
        args.box_out_dir = str(reports_root.parent / "20_range_research")
    if not args.passport_out_dir:
        args.passport_out_dir = str(reports_root.parent / "20_range_research")
    if not args.note_box_out_dir:
        args.note_box_out_dir = str(reports_root.parent / "30_note_box_profiles")
    if not args.spiral3d_out_dir:
        args.spiral3d_out_dir = str(reports_root.parent / "50_spiral3d")
    if not args.box_csv:
        args.box_csv = str(Path(args.box_out_dir) / f"{args.instrument_name}__dense_frequency_clusters.csv")

    audio_dir = Path(args.audio_dir)
    rows = read_manifest(Path(args.manifest_csv), args.layer)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}

    print(f"Instrument: {args.instrument_name}")
    print(f"Layer     : {args.layer}")
    print(f"Files     : {len(rows)}")
    print(f"Stages    : {sorted(stages)}")
    print(f"Box CSV   : {args.box_csv}")
    print(f"Box out   : {args.box_out_dir}")
    print(f"Passport  : {args.passport_out_dir}")
    print(f"Note box  : {args.note_box_out_dir}")
    print(f"Spiral3D  : {args.spiral3d_out_dir}")
    print(f"Maxwell   : {bool(args.use_maxwell)}")

    # PHASE 1: dense + chain
    if "dense" in stages or "chain" in stages:
        print("\n=== PHASE 1: DENSE / CHAIN ===")
        for row in rows:
            stem = stem_of(row["original_filename"])
            audio_path = audio_dir / row["original_filename"]
            report_dir = reports_root / stem
            report_dir.mkdir(parents=True, exist_ok=True)
            dense_csv = report_dir / f"{stem}__dense.csv"

            print(f"\n=== {stem} ===", flush=True)

            if "dense" in stages:
                if not audio_path.exists():
                    print(f"[MISSING AUDIO] {audio_path}")
                else:
                    dense_csv = run_dense(audio_path, report_dir, stem, args)

            if "chain" in stages:
                if not dense_csv.exists():
                    print(f"[MISSING DENSE] {dense_csv}")
                else:
                    run_chain(dense_csv, report_dir, stem, args)

    # PHASE 2: root consensus
    if "root" in stages:
        print("\n=== PHASE 2: ROOT ===")
        for row in rows:
            stem = stem_of(row["original_filename"])
            note12 = row.get("note12", "")
            report_dir = reports_root / stem
            dense_csv = report_dir / f"{stem}__dense.csv"

            print(f"\n=== {stem} | {note12} ===", flush=True)

            if not note12:
                print("[MISSING NOTE12]")
            elif not dense_csv.exists():
                print(f"[MISSING DENSE] {dense_csv}")
            else:
                run_root(dense_csv, report_dir, stem, note12, args)

    # PHASE 3: instrument box from raw dense
    if "box" in stages:
        print("\n=== PHASE 3: BOX FROM RAW DENSE ===")
        run_box(args)

    # PHASE 4: split box into breath/resonance
    if "box_split" in stages:
        print("\n=== PHASE 4: BOX SPLIT ===")
        if not Path(args.box_csv).exists():
            print(f"[MISSING BOX CSV] {args.box_csv}")
        else:
            run_box_split(args)

    # PHASE 5: clean + theory + spiral
    derived_stages = {"clean_box", "dense_vs_theory", "spiral12"}
    if stages & derived_stages:
        print("\n=== PHASE 5: CLEAN / THEORY / SPIRAL ===")
        for row in rows:
            stem = stem_of(row["original_filename"])
            note12 = row.get("note12", "")
            report_dir = reports_root / stem
            raw_dense_csv = report_dir / f"{stem}__dense.csv"
            dense_csv = raw_dense_csv

            print(f"\n=== {stem} | {note12} ===", flush=True)

            if "clean_box" in stages and note12:
                if not raw_dense_csv.exists():
                    print(f"[MISSING DENSE FOR CLEAN] {raw_dense_csv}")
                else:
                    dense_csv = run_clean_box(raw_dense_csv, report_dir, stem, note12, args)

            if "dense_vs_theory" in stages and note12:
                chain_json = report_dir / f"{stem}__dense_chain_summary.json"
                if not chain_json.exists():
                    print(f"[MISSING CHAIN SUMMARY] {chain_json}")
                else:
                    run_dense_vs_theory(report_dir, stem, note12, args)

            if "spiral12" in stages:
                spiral_input = report_dir / f"{stem}__dense_unified_clean.csv"
                if spiral_input.exists():
                    dense_csv = spiral_input
                elif raw_dense_csv.exists():
                    dense_csv = raw_dense_csv

                if not dense_csv.exists():
                    print(f"[MISSING SPIRAL INPUT] {dense_csv}")
                else:
                    run_spiral12(dense_csv, report_dir, stem, args)

    # PHASE 6: resonance ↔ harmonic relation
    if "note_box_profile" in stages:
        print("\n=== PHASE 6: NOTE BOX PROFILE ===")
        run_note_box_profile(args)

    if "spiral3d" in stages:
        print("\n=== PHASE 7: SPIRAL 3D ===")
        run_spiral3d(args)

    if "relation" in stages:
        print("\n=== PHASE 8: BOX HARMONIC RELATION ===")
        run_box_relation(args)

    if "passport" in stages:
        print("\n=== PHASE 9: PASSPORT ===")
        run_passport(args)


if __name__ == "__main__":
    main()
