# -*- coding: utf-8 -*-
"""
BLOCK004 SOURCE / MANIFEST / REPORT AUDIT

Проверяет по всем инструментам, кроме percussion:

1. Какие WAV есть в 00_sources/audio_notes_wav или 00_sources/audio_notes
2. Какие строки есть в 20_manifest/*.csv
3. Какие папки есть в 10_reports
4. Какие файлы есть в 50_spiral3d

Цель:
найти, на каком этапе исчезают ноты:
- исходник есть, manifest нет
- manifest FAIL
- manifest OK, но 10_reports нет
- 10_reports есть, но нет root/clean/spiral
- 50_spiral3d нет

Также помечает возможные кириллические символы в имени.
"""

import os
import re
import csv
import argparse
import unicodedata
import pandas as pd


NOTE12_RE = re.compile(
    r"(?P<note>[1-9ABCАВС]+[.][1-9ABCАВС](?:'[-ia0-9ABCАВС]*)?-?)",
    re.IGNORECASE,
)

CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "С": "C",
    "а": "A", "в": "B", "с": "C",
})


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def has_cyrillic(s):
    return any("CYRILLIC" in unicodedata.name(ch, "") for ch in str(s))


def normalize_abc(s):
    return str(s).translate(CYR_TO_LAT)


def normalize_note_token(s):
    s = normalize_abc(str(s)).strip()
    s = s.replace("'", "")
    return s


def extract_note12_from_name(name):
    s = normalize_abc(str(name))
    m = NOTE12_RE.search(s)
    if not m:
        return ""
    return normalize_note_token(m.group("note"))


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_audio_dir(instrument_root):
    candidates = [
        os.path.join(instrument_root, "00_sources", "audio_notes_wav"),
        os.path.join(instrument_root, "00_sources", "audio_notes"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def find_manifest_csvs(instrument_root):
    mdir = os.path.join(instrument_root, "20_manifest")
    if not os.path.isdir(mdir):
        return []
    return [
        os.path.join(mdir, f)
        for f in os.listdir(mdir)
        if f.lower().endswith(".csv")
    ]


def report_dirs_map(instrument_root):
    root = os.path.join(instrument_root, "10_reports")
    out = {}
    if not os.path.isdir(root):
        return out

    for d in os.listdir(root):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        note = extract_note12_from_name(d)
        out.setdefault(note, []).append(d)

    return out


def spiral3d_map(instrument_root):
    root = os.path.join(instrument_root, "50_spiral3d")
    out = {}
    if not os.path.isdir(root):
        return out

    for f in os.listdir(root):
        if not f.endswith("__spiral3d_points.csv"):
            continue
        note = extract_note12_from_name(f)
        out.setdefault(note, []).append(f)

    return out


def report_file_presence(instrument_root, report_folder):
    report_dir = os.path.join(instrument_root, "10_reports", report_folder)

    wanted_suffixes = {
        "dense": "__dense.csv",
        "chain_candidates": "__dense_chain_candidates.csv",
        "chain_summary_json": "__dense_chain_summary.json",
        "chain_summary_txt": "__dense_chain_summary.txt",
        "root_candidates": "__root_consensus_candidates.csv",
        "root_clusters": "__root_consensus_clusters.csv",
        "root_meta": "__root_consensus_meta.json",
        "root_summary": "__root_consensus_summary.txt",
        "clean": "__dense_unified_clean.csv",
        "clean_summary": "__dense_unified_clean_summary.txt",
        "removed_box": "__dense_unified_removed_box.csv",
        "spiral_png": "__spiral12_clean.png",
        "spiral_points": "__spiral12_clean_points.csv",
    }

    files = os.listdir(report_dir) if os.path.isdir(report_dir) else []

    result = {}
    for key, suffix in wanted_suffixes.items():
        result[key] = int(any(f.endswith(suffix) for f in files))

    return result


def load_manifest_rows(instrument_root):
    rows = []

    for path in find_manifest_csvs(instrument_root):
        df = safe_read_csv(path)
        if df is None or len(df) == 0:
            continue

        for _, r in df.iterrows():
            original = str(
                r.get("original_filename")
                or r.get("filename")
                or os.path.basename(str(r.get("wav_path", "")))
                or ""
            )

            note = (
                r.get("note12")
                or r.get("expected_note")
                or r.get("expected_note_token")
                or r.get("note_token")
                or ""
            )

            note = normalize_note_token(note) if note else extract_note12_from_name(original)

            rows.append({
                "manifest_csv": path,
                "original_filename": original,
                "manifest_note12": note,
                "parse_status": str(r.get("parse_status", "")),
                "reason": str(r.get("reason", "")),
            })

    return rows


def audit_instrument(block_root, instrument):
    instrument_root = os.path.join(block_root, instrument)

    audio_dir = find_audio_dir(instrument_root)
    manifests = load_manifest_rows(instrument_root)
    reports = report_dirs_map(instrument_root)
    spirals = spiral3d_map(instrument_root)

    manifest_by_file = {}
    manifest_by_note = {}

    for r in manifests:
        manifest_by_file.setdefault(r["original_filename"], []).append(r)
        manifest_by_note.setdefault(r["manifest_note12"], []).append(r)

    rows = []

    if audio_dir:
        wavs = [
            f for f in os.listdir(audio_dir)
            if f.lower().endswith(".wav")
        ]
    else:
        wavs = []

    for wav in sorted(wavs):
        source_note = extract_note12_from_name(wav)
        mf = manifest_by_file.get(wav, [])
        mf_by_note = manifest_by_note.get(source_note, [])

        manifest_hit = int(bool(mf or mf_by_note))
        manifest_statuses = ";".join(sorted(set(r["parse_status"] for r in (mf or mf_by_note))))
        manifest_reasons = ";".join(sorted(set(r["reason"] for r in (mf or mf_by_note) if r["reason"] and r["reason"] != "nan")))

        report_folders = reports.get(source_note, [])
        report_hit = int(bool(report_folders))

        spiral_files = spirals.get(source_note, [])
        spiral_hit = int(bool(spiral_files))

        presence = {}
        if report_folders:
            presence = report_file_presence(instrument_root, report_folders[0])
        else:
            presence = {
                "dense": 0,
                "chain_candidates": 0,
                "chain_summary_json": 0,
                "chain_summary_txt": 0,
                "root_candidates": 0,
                "root_clusters": 0,
                "root_meta": 0,
                "root_summary": 0,
                "clean": 0,
                "clean_summary": 0,
                "removed_box": 0,
                "spiral_png": 0,
                "spiral_points": 0,
            }

        rows.append({
            "instrument": instrument,
            "audio_dir": audio_dir,
            "source_wav": wav,
            "source_note12": source_note,
            "has_cyrillic": int(has_cyrillic(wav)),
            "normalized_filename": normalize_abc(wav),

            "manifest_hit": manifest_hit,
            "manifest_statuses": manifest_statuses,
            "manifest_reasons": manifest_reasons,

            "report_hit": report_hit,
            "report_folders": "|".join(report_folders),

            "spiral3d_hit": spiral_hit,
            "spiral3d_files": "|".join(spiral_files),

            **presence,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block004_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    all_rows = []

    for instrument in sorted(os.listdir(args.block004_root)):
        if instrument.lower() == "percussion":
            continue
        if instrument.startswith("_"):
            continue

        instrument_root = os.path.join(args.block004_root, instrument)
        if not os.path.isdir(instrument_root):
            continue

        rows = audit_instrument(args.block004_root, instrument)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    out_csv = os.path.join(args.out_dir, "block004_audit_source_manifest_reports.csv")
    df.to_csv(out_csv, index=False)

    if len(df) > 0:
        summary = (
            df.groupby("instrument")
            .agg(
                source_wavs=("source_wav", "count"),
                cyrillic_names=("has_cyrillic", "sum"),
                manifest_missing=("manifest_hit", lambda x: int((x == 0).sum())),
                reports_missing=("report_hit", lambda x: int((x == 0).sum())),
                spiral3d_missing=("spiral3d_hit", lambda x: int((x == 0).sum())),
                root_missing=("root_summary", lambda x: int((x == 0).sum())),
                clean_missing=("clean", lambda x: int((x == 0).sum())),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()

    out_summary = os.path.join(args.out_dir, "block004_audit_missing_summary.csv")
    summary.to_csv(out_summary, index=False)

    print("BLOCK004 AUDIT DONE")
    print(f"rows        : {len(df)}")
    print(f"out_csv     : {out_csv}")
    print(f"out_summary : {out_summary}")


if __name__ == "__main__":
    main()