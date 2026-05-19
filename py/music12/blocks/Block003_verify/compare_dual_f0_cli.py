from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


COMPARE_NOTE = (
    "This stage compares Block001 etalon f0 with Block002 stabilized representative rc "
    "and strongest peak note. Representative rc is not true f0."
)


def load_etalon(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "start": float(r["time_start_sec"]),
                    "end": float(r["time_end_sec"]),
                    "f0": (r["f0_note"] or "").strip(),
                }
            )
    return rows


def load_detected(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "t": float(r["chosen_time_sec"]),
                    "rc": (r.get("representative_rc_note", "") or "").strip(),
                    "peak": (r.get("strongest_peak_note", "") or "").strip(),
                    "stabilization_role": (r.get("stabilization_role", "") or "").strip(),
                    "support_hits": int(r.get("support_hits", "0") or 0),
                    "rc_chain_score": float(r.get("rc_chain_score", "0") or 0.0),
                }
            )
    return rows


def find_etalon(etalon: list[dict], t: float) -> dict | None:
    for e in etalon:
        if e["start"] <= t <= e["end"]:
            return e
    return None


def match(det_note: str, et_note: str) -> str:
    if not det_note or not et_note:
        return "NO_DATA"
    if det_note == et_note:
        return "MATCH"
    return "NO_MATCH"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, stat_rc: Counter, stat_peak: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for k, v in stat_rc.items():
            writer.writerow([f"representative_rc_{k}", v])

        for k, v in stat_peak.items():
            writer.writerow([f"strongest_peak_{k}", v])


def write_meta_json(path: Path, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "row_count": row_count,
                "semantic_note": COMPARE_NOTE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare Block001 etalon f0 with Block002 stabilized representative rc "
            "and strongest peak note."
        )
    )
    ap.add_argument("--etalon_csv", required=True)
    ap.add_argument("--detected_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    etalon = load_etalon(Path(args.etalon_csv).resolve())
    detected = load_detected(Path(args.detected_csv).resolve())

    out: list[dict] = []
    stat_rc: Counter = Counter()
    stat_peak: Counter = Counter()

    for d in detected:
        et = find_etalon(etalon, d["t"])
        et_note = et["f0"] if et else ""

        rc_match = match(d["rc"], et_note)
        peak_match = match(d["peak"], et_note)

        stat_rc[rc_match] += 1
        stat_peak[peak_match] += 1

        out.append(
            {
                "time": d["t"],
                "et_f0": et_note,
                "representative_rc": d["rc"],
                "peak_note": d["peak"],
                "representative_rc_match": rc_match,
                "peak_match": peak_match,
                "stabilization_role": d["stabilization_role"],
                "support_hits": d["support_hits"],
                "rc_chain_score": d["rc_chain_score"],
            }
        )

    write_csv(Path(args.out_csv).resolve(), out)
    write_summary_csv(Path(args.out_summary_csv).resolve(), stat_rc, stat_peak)
    write_meta_json(Path(args.out_meta_json).resolve(), len(out))

    print("compare stabilized rc vs strongest peak complete")
    print(
        json.dumps(
            {
                "row_count": len(out),
                "out_csv": str(Path(args.out_csv).resolve()),
                "out_summary_csv": str(Path(args.out_summary_csv).resolve()),
                "out_meta_json": str(Path(args.out_meta_json).resolve()),
                "semantic_note": COMPARE_NOTE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()