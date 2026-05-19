from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        return default if s == "" else float(s)
    except Exception:
        return default


def expected_note_from_dirname(name: str) -> str:
    base = name.strip()

    # RealPiano
    m = re.match(r"^\d+__[^_]+(?:_[^_]+)*__(.+)$", base)
    if m:
        return m.group(1)

    # piano_midi
    m = re.match(r"^\d+_piano_midi_(.+)$", base)
    if m:
        val = m.group(1)
        if val.lower().endswith(".wav"):
            val = val[:-4]
        return val

    # generic
    m = re.match(r"^\d+_(.+)$", base)
    if m:
        val = m.group(1)
        if val.lower().endswith(".wav"):
            val = val[:-4]
        return val

    return ""


def detect_prefix(note_dir: Path) -> str:
    for p in sorted(note_dir.glob("*__probe_matrix.csv")):
        return p.stem.replace("__probe_matrix", "")
    return ""


def run_cmd(cmd: list[str], env: dict[str, str], log_txt: Path) -> int:
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)

    log_txt.parent.mkdir(parents=True, exist_ok=True)

    with log_txt.open("a", encoding="utf-8") as f:
        f.write("\n=== RUNNING ===\n")
        f.write(" ".join(cmd) + "\n")

        if cp.stdout:
            f.write("\n--- STDOUT ---\n")
            f.write(cp.stdout)
            if not cp.stdout.endswith("\n"):
                f.write("\n")

        if cp.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(cp.stderr)
            if not cp.stderr.endswith("\n"):
                f.write("\n")

        f.write(f"\nRETURN CODE: {cp.returncode}\n")

    return cp.returncode


def load_framewise_readable(path: Path) -> tuple[str, float, int]:
    if not path.exists():
        return "", 0.0, 0

    votes: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected = _safe_str(row.get("selected_notes", ""))
            if not selected:
                continue

            notes = [x.strip() for x in selected.split("|") if x.strip()]
            if not notes:
                continue

            total += 1
            votes[notes[0]] += 1

            for extra in notes[1:]:
                votes[extra] += 0.25

    if not votes:
        return "", 0.0, total

    dominant = votes.most_common(1)[0][0]
    ratio = float(votes[dominant]) / max(total, 1)

    return dominant, ratio, total


def load_framewise_with_theory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "dominant_root": "",
            "dominant_root_ratio": 0.0,
            "mean_support_hits": 0.0,
            "mean_spiral_consistency": 0.0,
            "verdict_mode": "",
            "row_count": 0,
            "confirmed_count": 0,
            "uncertain_count": 0,
            "chosen_rc_mode": "",
            "chosen_rc_hz_mean": 0.0,
        }

    root_counter = Counter()
    verdict_counter = Counter()
    rc_counter = Counter()
    support_hits = []
    consistency = []
    rc_hz = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        root = _safe_str(row.get("best_theoretical_root_token", ""))
        if root:
            root_counter[root] += 1

        verdict = _safe_str(row.get("theoretical_chain_verdict", ""))
        if verdict:
            verdict_counter[verdict] += 1

        chosen = _safe_str(row.get("chosen_rc_note", ""))
        if chosen:
            rc_counter[chosen] += 1

        support_hits.append(_safe_float(row.get("support_hits", 0.0)))
        consistency.append(_safe_float(row.get("spiral_consistency_score", 0.0)))

        hz = _safe_float(row.get("chosen_rc_hz", 0.0))
        if hz > 0:
            rc_hz.append(hz)

    dominant_root = root_counter.most_common(1)[0][0] if root_counter else ""
    dominant_root_ratio = float(root_counter[dominant_root]) / max(len(rows), 1) if dominant_root else 0.0

    verdict_mode = verdict_counter.most_common(1)[0][0] if verdict_counter else ""
    chosen_rc_mode = rc_counter.most_common(1)[0][0] if rc_counter else ""

    return {
        "dominant_root": dominant_root,
        "dominant_root_ratio": dominant_root_ratio,
        "mean_support_hits": sum(support_hits) / max(len(support_hits), 1),
        "mean_spiral_consistency": sum(consistency) / max(len(consistency), 1),
        "verdict_mode": verdict_mode,
        "row_count": len(rows),
        "confirmed_count": verdict_counter.get("CHAIN_CONFIRMED", 0),
        "uncertain_count": verdict_counter.get("CHAIN_UNCERTAIN", 0),
        "chosen_rc_mode": chosen_rc_mode,
        "chosen_rc_hz_mean": sum(rc_hz) / max(len(rc_hz), 1),
    }


def confidence_label(expected_note, chosen_rc_mode, dominant_root, verdict_mode, dominant_root_ratio):
    if dominant_root and expected_note and dominant_root == expected_note and verdict_mode == "CHAIN_CONFIRMED":
        return "HIGH"
    if dominant_root and expected_note and dominant_root == expected_note and dominant_root_ratio >= 0.2:
        return "MEDIUM"
    if chosen_rc_mode or dominant_root:
        return "LOW"
    return "UNRESOLVED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--bridge_script", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    reports_root = Path(args.reports_root)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "py")
    py = sys.executable

    rows_out = []

    for note_dir in sorted(reports_root.iterdir()):
        if not note_dir.is_dir():
            continue

        prefix = detect_prefix(note_dir)
        if not prefix:
            continue

        expected = expected_note_from_dirname(note_dir.name)

        bridge_csv = note_dir / f"{prefix}__framewise_with_theory.csv"

        stats = load_framewise_with_theory(bridge_csv)

        final_detected = stats["dominant_root"] or stats["chosen_rc_mode"]

        confidence = confidence_label(
            expected,
            stats["chosen_rc_mode"],
            stats["dominant_root"],
            stats["verdict_mode"],
            stats["dominant_root_ratio"]
        )

        rows_out.append({
            "note_dir": note_dir.name,
            "expected_note": expected,
            "final_detected_note": final_detected,
            "confidence": confidence
        })

        print(f"{note_dir.name} -> {final_detected} ({confidence})")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    main()