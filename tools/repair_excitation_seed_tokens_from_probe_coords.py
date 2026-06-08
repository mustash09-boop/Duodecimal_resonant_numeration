import argparse
import csv
from pathlib import Path


def normalize_note(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def main():
    ap = argparse.ArgumentParser(
        description="Repair excitation seed note_token/coarse_note fields by joining with valid probe_coords tokens."
    )
    ap.add_argument("--probe-coords-csv", required=True)
    ap.add_argument("--seeds-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    coords_by_probe = {}
    with Path(args.probe_coords_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            probe_index = int(row["probe_index"])
            note_token = str(row.get("note_token", "")).strip()
            coords_by_probe[probe_index] = {
                "note_token": note_token,
                "coarse_note": normalize_note(note_token),
            }

    repaired_rows = []
    changed = 0
    total = 0
    with Path(args.seeds_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            total += 1
            probe_index = int(row["probe_index"])
            fixed = coords_by_probe.get(probe_index)
            if fixed:
                if row.get("note_token", "") != fixed["note_token"] or row.get("coarse_note", "") != fixed["coarse_note"]:
                    changed += 1
                row["note_token"] = fixed["note_token"]
                row["coarse_note"] = fixed["coarse_note"]
            repaired_rows.append(row)

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repaired_rows)

    print(f"rows={total}")
    print(f"changed={changed}")
    print(f"out={args.out_csv}")


if __name__ == "__main__":
    main()
