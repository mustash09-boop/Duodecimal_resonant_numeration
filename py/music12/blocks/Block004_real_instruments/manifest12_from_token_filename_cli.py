from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TOKEN_RE = re.compile(
    r"^(?P<index>\d+)_"
    r"(?P<note12>[1-9ABC]+\.[1-9ABC]+'?(?:[ia][1-9ABC]+)?-?)_"
    r"(?P<label>.+?)\.wav$",
    re.IGNORECASE,
)


def parse_filename(path: Path) -> dict:
    m = TOKEN_RE.match(path.name)
    if not m:
        return {
            "parse_status": "FAIL",
            "original_filename": path.name,
            "index": "",
            "note12": "",
            "semantic_layer": "",
            "label": "",
            "reason": "filename does not match NNN_NOTE12_label.wav",
        }

    return {
        "parse_status": "OK",
        "original_filename": path.name,
        "index": m.group("index"),
        "note12": normalize_note12(m.group("note12")),
        "semantic_layer": "01_core_notes",
        "label": m.group("label"),
        "reason": "",
    }

def normalize_note12(token: str) -> str:
    t = token.strip().upper()

    if t.endswith("'-"):
        return t

    if t.endswith("'"):
        return t[:-1] + "-"

    if not t.endswith("-"):
        return t + "-"

    return t

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build manifest from filenames like 001_8.8'-_violin2_4string.wav"
    )
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [parse_filename(p) for p in sorted(audio_dir.glob("*.wav"))]

    fields = [
        "parse_status",
        "original_filename",
        "index",
        "note12",
        "semantic_layer",
        "label",
        "reason",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["parse_status"] == "OK")
    fail = len(rows) - ok
    print(f"Manifest written: {out_csv}")
    print(f"OK: {ok}")
    print(f"FAIL: {fail}")


if __name__ == "__main__":
    main()