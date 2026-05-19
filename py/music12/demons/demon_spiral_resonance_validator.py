from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate(per_root_path: Path, per_band_path: Path):
    per_root = load_csv(per_root_path)
    per_band = load_csv(per_band_path)

    report = {
        "status": "OK",
        "errors": [],
        "warnings": [],
        "insights": [],
    }

    # ============================================================
    # 1. Проверка наличия данных
    # ============================================================

    if not per_root:
        report["status"] = "FAIL"
        report["errors"].append("per_root.csv is empty")
        return report

    # ============================================================
    # 2. Проверка количества нот
    # ============================================================

    roots = set(r["root"] for r in per_root)
    if len(roots) < 20:
        report["warnings"].append(f"too few roots detected: {len(roots)}")
    else:
        report["insights"].append(f"roots_detected={len(roots)}")

    # ============================================================
    # 3. Проверка амплитуд
    # ============================================================

    zero_amp = [r for r in per_root if float(r["mean_amplitude"]) == 0.0]
    if zero_amp:
        report["warnings"].append(f"{len(zero_amp)} rows with zero amplitude")

    # ============================================================
    # 4. Проверка фаз
    # ============================================================

    invalid_phase = [
        r for r in per_root
        if not (0.0 <= float(r["mean_phase_deg"]) <= 360.0)
    ]

    if invalid_phase:
        report["warnings"].append(f"{len(invalid_phase)} rows with invalid phase")

    # ============================================================
    # 5. Проверка концентрации (есть ли лидеры)
    # ============================================================

    counts = defaultdict(int)
    for r in per_root:
        counts[r["response_note"]] += int(r["count"])

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    if not sorted_counts:
        report["status"] = "FAIL"
        report["errors"].append("no harmonic responses found")
        return report

    top = sorted_counts[:5]

    report["insights"].append("top_response_notes:")
    for note, c in top:
        report["insights"].append(f"  {note} -> {c}")

    # ============================================================
    # 6. Проверка различий по диапазонам
    # ============================================================

    band_map = defaultdict(list)
    for r in per_band:
        band_map[r["band"]].append(r)

    if len(band_map) < 2:
        report["warnings"].append("no band differentiation detected")
    else:
        report["insights"].append(f"bands_detected={list(band_map.keys())}")

        # сравним top для каждой зоны
        for band, rows in band_map.items():
            local_counts = defaultdict(int)
            for r in rows:
                local_counts[r["response_note"]] += int(r["count"])

            top_local = sorted(local_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            report["insights"].append(f"[{band}] top responses:")
            for note, c in top_local:
                report["insights"].append(f"  {note} -> {c}")

    # ============================================================
    # 7. Итоговая оценка
    # ============================================================

    if report["errors"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "WARN"
    else:
        report["status"] = "OK"

    return report


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("SPIRAL RESONANCE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"STATUS: {report['status']}\n\n")

        if report["errors"]:
            f.write("ERRORS:\n")
            for e in report["errors"]:
                f.write(f"  - {e}\n")

        if report["warnings"]:
            f.write("\nWARNINGS:\n")
            for w in report["warnings"]:
                f.write(f"  - {w}\n")

        if report["insights"]:
            f.write("\nINSIGHTS:\n")
            for i in report["insights"]:
                f.write(f"  {i}\n")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per_root_csv", required=True)
    ap.add_argument("--per_band_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    report = validate(
        Path(args.per_root_csv),
        Path(args.per_band_csv),
    )

    write_report(Path(args.out_txt), report)

    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("validation complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()