from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
BLOCK004_ROOT = PROJECT_ROOT / "Block004_data"
COMPARE_ROOT = BLOCK004_ROOT / "_multi_instrument_compare" / "91_harmonic_morphology_batch"
AMP_COMPARE_SCRIPT = PROJECT_ROOT / "tools" / "multi_instrument_harmonic_amplitude_compare.py"
MORPH_COMPARE_SCRIPT = PROJECT_ROOT / "py" / "music12" / "blocks" / "Block004_real_instruments" / "harmonic_morphology_compare_cli.py"
NOTE_INDEX_CSV = COMPARE_ROOT / "instrument_note_file_index.csv"
SHARED_NOTES_CSV = COMPARE_ROOT / "shared_notes_manifest.csv"
ALL_FEATURES_CSV = COMPARE_ROOT / "all_morphology_features.csv"
ALL_PAIRWISE_SUMMARY_CSV = COMPARE_ROOT / "all_pairwise_morphology_summary.csv"
ALL_PAIRWISE_DISTANCES_CSV = COMPARE_ROOT / "all_pairwise_harmonic_curve_distances.csv"
META_JSON = COMPARE_ROOT / "batch_meta.json"

NOTE12_RE = re.compile(r"(?P<note>[1-9ABC]+[.][1-9ABC](?:'[-ia0-9ABC]*)?-?)", re.IGNORECASE)
WESTERN_RE = re.compile(r"(?P<note>[A-G][#b]?\d)")
WESTERN_TO_DEGREE = {
    "C": "1",
    "C#": "2",
    "Db": "2",
    "D": "3",
    "D#": "4",
    "Eb": "4",
    "E": "5",
    "F": "6",
    "F#": "7",
    "Gb": "7",
    "G": "8",
    "G#": "9",
    "Ab": "9",
    "A": "A",
    "A#": "B",
    "Bb": "B",
    "B": "C",
}


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace("'", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def western_to_note12(western: str) -> str:
    m = re.match(r"^([A-G][#b]?)(\d)$", western)
    if not m:
        return ""
    name = m.group(1)
    octave = int(m.group(2))
    degree = WESTERN_TO_DEGREE.get(name)
    if not degree:
        return ""
    return f"{octave + 5}.{degree}-"


def extract_note_from_name(name: str) -> str:
    s = str(name)
    m = NOTE12_RE.search(s)
    if m:
        return m.group("note").replace("'", "")
    m = WESTERN_RE.search(s)
    if m:
        return western_to_note12(m.group("note"))
    return ""


def build_note_index() -> pd.DataFrame:
    rows: list[dict] = []
    for dataset_dir in sorted(BLOCK004_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name.startswith("_") or dataset_dir.name == "percussion":
            continue
        spiral_dir = dataset_dir / "50_spiral3d"
        if not spiral_dir.is_dir():
            continue

        for csv_path in sorted(spiral_dir.glob("*__spiral3d_points.csv")):
            stem = csv_path.name.replace("__spiral3d_points.csv", "")
            note12 = extract_note_from_name(stem)
            png_path = spiral_dir / f"{stem}__spiral3d.png"
            html_path = spiral_dir / f"{stem}__spiral3d.html"
            rows.append(
                {
                    "instrument": dataset_dir.name,
                    "source_note_name": stem,
                    "canonical_note12": note12,
                    "spiral3d_csv": str(csv_path),
                    "spiral3d_png": str(png_path) if png_path.exists() else "",
                    "spiral3d_html": str(html_path) if html_path.exists() else "",
                    "parse_status": "OK" if note12 else "FAIL",
                }
            )

    df = pd.DataFrame(rows)
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(NOTE_INDEX_CSV, index=False, encoding="utf-8-sig")
    return df


def shared_note_manifest(index_df: pd.DataFrame) -> pd.DataFrame:
    ok = index_df[index_df["parse_status"] == "OK"].copy()
    grouped = (
        ok.groupby("canonical_note12", as_index=False)
        .agg(
            instrument_count=("instrument", "nunique"),
            instruments=("instrument", lambda s: ",".join(sorted(set(str(x) for x in s)))),
            source_rows=("instrument", "count"),
        )
        .sort_values(["instrument_count", "canonical_note12"], ascending=[False, True])
    )
    grouped = grouped[grouped["instrument_count"] >= 2].reset_index(drop=True)
    grouped.to_csv(SHARED_NOTES_CSV, index=False, encoding="utf-8-sig")
    return grouped


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def build_note_outputs(note: str, instruments: list[str], skip_existing: bool) -> dict:
    note_tag = safe_name(note)
    note_dir = COMPARE_ROOT / note_tag
    note_dir.mkdir(parents=True, exist_ok=True)

    amp_html = note_dir / f"harmonic_amplitude_compare__{note_tag}__3d.html"
    morph_dir = note_dir / "harmonic_morphology"
    morph_summary_csv = morph_dir / "05_pairwise_instrument_morphology_summary.csv"

    if not (skip_existing and amp_html.exists()):
        cmd = [
            sys.executable,
            str(AMP_COMPARE_SCRIPT),
            "--note_index_csv",
            str(NOTE_INDEX_CSV),
            "--note",
            note,
            "--out_dir",
            str(note_dir),
            "--instruments",
            *instruments,
        ]
        run_cmd(cmd)

    if not (skip_existing and morph_summary_csv.exists()):
        cmd = [
            sys.executable,
            str(MORPH_COMPARE_SCRIPT),
            "--html",
            str(amp_html),
            "--outdir",
            str(morph_dir),
        ]
        run_cmd(cmd)

    return {
        "note": note,
        "note_tag": note_tag,
        "note_dir": str(note_dir),
        "instrument_count": len(instruments),
        "instruments": ",".join(instruments),
        "amplitude_html": str(amp_html),
        "morphology_dir": str(morph_dir),
        "morphology_summary_csv": str(morph_summary_csv),
    }


def main() -> None:
    skip_existing = True

    index_df = build_note_index()
    shared_df = shared_note_manifest(index_df)

    feature_frames: list[pd.DataFrame] = []
    pairwise_summary_frames: list[pd.DataFrame] = []
    pairwise_distance_frames: list[pd.DataFrame] = []
    note_rows: list[dict] = []

    ok_df = index_df[index_df["parse_status"] == "OK"].copy()

    for _, row in shared_df.iterrows():
        note = str(row["canonical_note12"])
        instruments = sorted(set(str(x) for x in ok_df[ok_df["canonical_note12"] == note]["instrument"].tolist()))
        note_info = build_note_outputs(note=note, instruments=instruments, skip_existing=skip_existing)
        note_rows.append(note_info)

        morph_dir = Path(note_info["morphology_dir"])

        feature_csv = morph_dir / "03_harmonic_morphology_features.csv"
        pairwise_summary_csv = morph_dir / "05_pairwise_instrument_morphology_summary.csv"
        pairwise_distance_csv = morph_dir / "04_pairwise_harmonic_curve_distances.csv"

        if feature_csv.exists():
            df = pd.read_csv(feature_csv)
            df["note"] = note
            feature_frames.append(df)
        if pairwise_summary_csv.exists():
            df = pd.read_csv(pairwise_summary_csv)
            df["note"] = note
            pairwise_summary_frames.append(df)
        if pairwise_distance_csv.exists():
            df = pd.read_csv(pairwise_distance_csv)
            df["note"] = note
            pairwise_distance_frames.append(df)

        print(f"DONE {note} ({len(instruments)} instruments)")

    notes_manifest_df = pd.DataFrame(note_rows)
    notes_manifest_df.to_csv(COMPARE_ROOT / "note_run_manifest.csv", index=False, encoding="utf-8-sig")

    all_features_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    all_pairwise_summary_df = pd.concat(pairwise_summary_frames, ignore_index=True) if pairwise_summary_frames else pd.DataFrame()
    all_pairwise_distances_df = pd.concat(pairwise_distance_frames, ignore_index=True) if pairwise_distance_frames else pd.DataFrame()

    all_features_df.to_csv(ALL_FEATURES_CSV, index=False, encoding="utf-8-sig")
    all_pairwise_summary_df.to_csv(ALL_PAIRWISE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    all_pairwise_distances_df.to_csv(ALL_PAIRWISE_DISTANCES_CSV, index=False, encoding="utf-8-sig")

    meta = {
        "note_index_csv": str(NOTE_INDEX_CSV),
        "shared_notes_csv": str(SHARED_NOTES_CSV),
        "notes_total": int(len(shared_df)),
        "instruments_total": int(index_df[index_df["parse_status"] == "OK"]["instrument"].nunique()),
        "all_features_csv": str(ALL_FEATURES_CSV),
        "all_pairwise_summary_csv": str(ALL_PAIRWISE_SUMMARY_CSV),
        "all_pairwise_distances_csv": str(ALL_PAIRWISE_DISTANCES_CSV),
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(f"shared_notes: {len(shared_df)}")
    print(f"features    : {len(all_features_df)}")
    print(f"pair_summary: {len(all_pairwise_summary_df)}")
    print(f"out_root    : {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
