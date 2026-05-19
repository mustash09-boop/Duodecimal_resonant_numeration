from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_harmonic_list_field(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [x.strip() for x in value.split() if x.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_statistics(input_dir: Path):
    global_counter = Counter()
    per_note_counter = defaultdict(Counter)
    per_state_counter = defaultdict(Counter)
    per_note_state_counter = defaultdict(Counter)

    files = sorted(input_dir.glob("*__target_root_convergence.csv"))

    for f in files:
        prefix = f.name.replace("__target_root_convergence.csv", "")
        source_note = prefix.split("__")[2] if "__" in prefix else prefix

        with f.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                target_state = (row.get("target_state", "") or "").strip()
                harmonics = parse_harmonic_list_field(row.get("matched_harmonics_window", ""))

                for h in harmonics:
                    global_counter[h] += 1
                    per_note_counter[source_note][h] += 1
                    per_state_counter[target_state][h] += 1
                    per_note_state_counter[(source_note, target_state)][h] += 1

    return global_counter, per_note_counter, per_state_counter, per_note_state_counter, files


def write_global(path: Path, counter: Counter):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["harmonic", "count"])
        for harmonic, count in counter.most_common():
            writer.writerow([harmonic, count])


def write_per_note(path: Path, per_note):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note", "harmonic", "count"])
        for note in sorted(per_note.keys()):
            for harmonic, count in per_note[note].most_common():
                writer.writerow([note, harmonic, count])


def write_per_state(path: Path, per_state):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_state", "harmonic", "count"])
        for state in sorted(per_state.keys()):
            for harmonic, count in per_state[state].most_common():
                writer.writerow([state, harmonic, count])


def write_per_note_state(path: Path, per_note_state):
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note", "target_state", "harmonic", "count"])
        keys = sorted(per_note_state.keys(), key=lambda x: (x[0], x[1]))
        for note, state in keys:
            for harmonic, count in per_note_state[(note, state)].most_common():
                writer.writerow([note, state, harmonic, count])


def write_summary_txt(path: Path, *, files, global_counter, per_note_counter, per_state_counter):
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("HARMONIC PRESENCE STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"input_file_count: {len(files)}\n")
        f.write(f"note_count: {len(per_note_counter)}\n")
        f.write("\nGLOBAL TOP HARMONICS\n")
        for harmonic, count in global_counter.most_common(20):
            f.write(f"  {harmonic}: {count}\n")

        f.write("\nTOP HARMONICS BY TARGET STATE\n")
        for state in sorted(per_state_counter.keys()):
            f.write(f"  [{state}]\n")
            for harmonic, count in per_state_counter[state].most_common(10):
                f.write(f"    {harmonic}: {count}\n")

        f.write("\nINTERPRETATION\n")
        f.write("  - statistics are built from convergence layer, not adaptive root layer.\n")
        f.write("  - matched_harmonics_window is treated as the direct observed harmonic presence.\n")
        f.write("  - later this can be compared with regime-confirmed chains and Block004 templates.\n")


def write_meta_json(path: Path, *, input_dir: Path, outputs: dict):
    ensure_parent(path)
    data = {
        "inputs": {
            "input_dir": str(input_dir),
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Build harmonic presence statistics from convergence CSV files"
    )
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_global", required=True)
    ap.add_argument("--out_per_note", required=True)
    ap.add_argument("--out_per_state", required=True)
    ap.add_argument("--out_per_note_state", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()

    global_counter, per_note_counter, per_state_counter, per_note_state_counter, files = build_statistics(input_dir)

    write_global(Path(args.out_global), global_counter)
    write_per_note(Path(args.out_per_note), per_note_counter)
    write_per_state(Path(args.out_per_state), per_state_counter)
    write_per_note_state(Path(args.out_per_note_state), per_note_state_counter)
    write_summary_txt(
        Path(args.out_txt),
        files=files,
        global_counter=global_counter,
        per_note_counter=per_note_counter,
        per_state_counter=per_state_counter,
    )
    write_meta_json(
        Path(args.out_meta_json),
        input_dir=input_dir,
        outputs={
            "global_csv": str(Path(args.out_global).resolve()),
            "per_note_csv": str(Path(args.out_per_note).resolve()),
            "per_state_csv": str(Path(args.out_per_state).resolve()),
            "per_note_state_csv": str(Path(args.out_per_note_state).resolve()),
            "summary_txt": str(Path(args.out_txt).resolve()),
        },
    )

    print("harmonic presence statistics complete")


if __name__ == "__main__":
    main()