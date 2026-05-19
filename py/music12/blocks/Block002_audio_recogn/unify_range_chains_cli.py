from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any


def sf(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def si(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def extract_note(folder_name: str) -> str:
    m = re.search(r"([1-9ABC]+\.[1-9ABC]+-)$", folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract note from folder: {folder_name}")
    return m.group(1).upper()


def load_compare_modes(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            note = row.get("expected_note", "").strip().upper()
            if note:
                out[note] = row
    return out


def parse_dense_vs_theory(path: Path, expected_note: str, folder_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    root_note = ""
    root_hz = 0.0

    for line in lines:
        s = line.strip()
        if s.startswith("detected_root_note"):
            root_note = s.split(":", 1)[1].strip()
        elif s.startswith("detected_root_hz"):
            root_hz = sf(s.split(":", 1)[1].strip())

    # ожидаемый формат строк: hN ... theory=... obs=... status=...
    for line in lines:
        s = line.strip()
        if not s.startswith("h"):
            continue

        mh = re.match(r"h\s*(\d+)", s)
        if not mh:
            continue

        h = si(mh.group(1))

        status = ""
        if "status=" in s:
            status = s.split("status=", 1)[1].split()[0].strip()

        theory_hz = 0.0
        observed_hz = 0.0
        theory_token = ""
        observed_token = ""

        mt = re.search(r"theory=([^\s]+)\s*\(([-+0-9.]+)\)", s)
        if mt:
            theory_token = mt.group(1)
            theory_hz = sf(mt.group(2))

        mo = re.search(r"(?:obs|observed|matched)=([^\s]+)\s*\(([-+0-9.]+)\)", s)
        if mo:
            observed_token = mo.group(1)
            observed_hz = sf(mo.group(2))

        if status and status not in {"MATCH", "SHIFTED_UP", "SHIFTED_DOWN"}:
            continue

        rows.append({
            "folder_name": folder_name,
            "expected_note": expected_note,
            "range_mode": "MID",
            "selected_algorithm": "dense_vs_theory",
            "root_token": root_note,
            "root_hz": root_hz,
            "harmonic_index": h,
            "theoretical_hz": theory_hz,
            "theoretical_token": theory_token,
            "observed_hz": observed_hz,
            "observed_token": observed_token,
            "amplitude": "",
            "phase_rad": "",
            "present_percent_frames": "",
            "confidence": "",
            "source_status": status,
            "source_file": str(path),
        })

    return rows


def parse_high_presence(path: Path, expected_note: str, folder_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if si(row.get("is_stable")) != 1:
                continue

            rows.append({
                "folder_name": folder_name,
                "expected_note": expected_note,
                "range_mode": "HIGH",
                "selected_algorithm": "high_presence",
                "root_token": row.get("root_token", ""),
                "root_hz": row.get("root_hz", ""),
                "harmonic_index": row.get("harmonic_index", ""),
                "theoretical_hz": row.get("theoretical_hz", ""),
                "theoretical_token": row.get("theoretical_token", ""),
                "observed_hz": row.get("median_matched_hz") or row.get("mean_matched_hz", ""),
                "observed_token": row.get("median_matched_token") or row.get("mean_matched_token", ""),
                "amplitude": row.get("median_amplitude") or row.get("mean_amplitude", ""),
                "phase_rad": row.get("mean_phase_rad", ""),
                "present_percent_frames": row.get("present_percent_frames", ""),
                "confidence": row.get("present_percent_frames", ""),
                "source_status": "STABLE",
                "source_file": str(path),
            })

    return rows


def parse_root_consensus(path: Path, expected_note: str, folder_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        best = next(r, None)

    if not best:
        return []

    root_token = best.get("consensus_root_token", "")
    root_hz = sf(best.get("consensus_root_hz"))

    present_raw = best.get("present_harmonics", "[]")
    try:
        present_h = ast.literal_eval(present_raw)
    except Exception:
        present_h = []

    counts_raw = best.get("harmonic_index_counts", "{}")
    try:
        counts = json.loads(counts_raw)
    except Exception:
        try:
            counts = ast.literal_eval(counts_raw)
        except Exception:
            counts = {}

    for h in present_h:
        h_int = si(h)
        rows.append({
            "folder_name": folder_name,
            "expected_note": expected_note,
            "range_mode": "LOW_OR_MIXED",
            "selected_algorithm": "root_consensus",
            "root_token": root_token,
            "root_hz": root_hz,
            "harmonic_index": h_int,
            "theoretical_hz": root_hz * h_int if root_hz > 0 else "",
            "theoretical_token": "",
            "observed_hz": "",
            "observed_token": "",
            "amplitude": best.get("mean_observed_amplitude", ""),
            "phase_rad": "",
            "present_percent_frames": "",
            "confidence": best.get("tuner_confidence", ""),
            "source_status": f"CONSENSUS_COUNT={counts.get(str(h_int), counts.get(h_int, ''))}",
            "source_file": str(path),
        })

    return rows


def choose_source_rows(
    folder: Path,
    compare_row: dict[str, Any],
) -> list[dict[str, Any]]:
    folder_name = folder.name
    expected_note = extract_note(folder_name)
    mode = compare_row.get("recommended_mode", "")

    dense_path = folder / f"{folder_name}__dense_vs_theory.txt"
    root_path = folder / f"{folder_name}__root_consensus_clusters.csv"
    high_path = folder / f"{folder_name}__high_presence.csv"

    if mode == "USE_ROOT_CONSENSUS":
        return parse_root_consensus(root_path, expected_note, folder_name)

    if mode == "USE_DENSE_CHAIN":
        dense_rows = parse_dense_vs_theory(dense_path, expected_note, folder_name)
        if dense_rows:
            return dense_rows

        # Для верхних нот dense может быть бедным, но high_presence уже надёжнее.
        high_rows = parse_high_presence(high_path, expected_note, folder_name)
        return high_rows

    if mode == "USE_MIXED_MODE":
        # MIXED: берём consensus root + dense/high подтверждения.
        out = []
        out.extend(parse_root_consensus(root_path, expected_note, folder_name))
        dense_rows = parse_dense_vs_theory(dense_path, expected_note, folder_name)
        high_rows = parse_high_presence(high_path, expected_note, folder_name)

        if high_rows:
            out.extend(high_rows)
        else:
            out.extend(dense_rows)

        return out

    if mode == "REQUIRES_HIGH_REGISTER_MODE":
        return parse_high_presence(high_path, expected_note, folder_name)

    return []


def collect_unified(reports_root: Path, compare_csv: Path) -> list[dict[str, Any]]:
    compare = load_compare_modes(compare_csv)

    rows: list[dict[str, Any]] = []
    for folder in sorted([p for p in reports_root.iterdir() if p.is_dir()]):
        note = extract_note(folder.name)
        c = compare.get(note, {})
        if not c:
            continue
        rows.extend(choose_source_rows(folder, c))

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "folder_name",
        "expected_note",
        "range_mode",
        "selected_algorithm",
        "root_token",
        "root_hz",
        "harmonic_index",
        "theoretical_hz",
        "theoretical_token",
        "observed_hz",
        "observed_token",
        "amplitude",
        "phase_rad",
        "present_percent_frames",
        "confidence",
        "source_status",
        "source_file",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_note: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_note.setdefault(r["expected_note"], []).append(r)

    lines = []
    lines.append("UNIFIED NOTE CHAINS")
    lines.append("=" * 100)
    lines.append(f"total_rows : {len(rows)}")
    lines.append(f"total_notes: {len(by_note)}")
    lines.append("")

    for note, note_rows in by_note.items():
        algs = sorted(set(r["selected_algorithm"] for r in note_rows))
        hs = ",".join(str(r["harmonic_index"]) for r in note_rows)
        lines.append(f"{note:8s} rows={len(note_rows):3d} alg={'+'.join(algs):35s} h={hs}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unify LOW/MID/HIGH note chain reports into one CSV.")
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--compare_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    rows = collect_unified(
        Path(args.reports_root).resolve(),
        Path(args.compare_csv).resolve(),
    )

    write_csv(Path(args.out_csv).resolve(), rows)
    write_txt(Path(args.out_txt).resolve(), rows)

    print(json.dumps({
        "rows": len(rows),
        "out_csv": args.out_csv,
        "out_txt": args.out_txt,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()