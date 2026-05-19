from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.median(xs))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def _note_from_folder_name(folder_name: str) -> str:
    """
    Examples:
      001_5.A- -> 5.A-
      025_7.A- -> 7.A-
      001_piano_midi_5.A- -> 5.A-
    """
    s = _safe_str(folder_name)

    if "_piano_midi_" in s:
        return s.split("_piano_midi_", 1)[1]

    parts = s.split("_", 1)
    if len(parts) == 2:
        return parts[1]

    return s


def _token_to_hz_approx(token: str, anchor_token: str = "9.A-", anchor_hz: float = 440.0) -> float:
    """
    Rough converter only for range zoning if needed from root token.
    Uses 12-TET base with duodecimal note tokens.
    """
    token = _safe_str(token).upper().replace("’-", "'-")
    if not token or "." not in token:
        return 0.0

    alphabet = "123456789ABC"
    degree_map = {ch: i for i, ch in enumerate(alphabet)}

    # strip micro
    main = token.split("'")[0]
    if "." not in main:
        return 0.0

    oct_s, deg_s = main.split(".", 1)
    if not oct_s or not deg_s:
        return 0.0

    def bij12_to_int(s: str) -> int:
        val_map = {ch: i + 1 for i, ch in enumerate(alphabet)}
        n = 0
        for ch in s:
            if ch not in val_map:
                raise ValueError
            n = n * 12 + val_map[ch]
        return n

    try:
        octave = bij12_to_int(oct_s)
        degree = degree_map[deg_s[0]]
        anchor_oct_s, anchor_deg_s = anchor_token.split(".", 1)
        anchor_deg_s = anchor_deg_s.split("'")[0].split("-")[0]
        anchor_oct = bij12_to_int(anchor_oct_s)
        anchor_deg = degree_map[anchor_deg_s[0]]
    except Exception:
        return 0.0

    semitone_delta = (octave - anchor_oct) * 12 + (degree - anchor_deg)
    return float(anchor_hz * (2.0 ** (semitone_delta / 12.0)))


def _range_zone(root_hz: float) -> str:
    if root_hz <= 0:
        return "unknown"
    if root_hz < 110.0:
        return "low"
    if root_hz < 880.0:
        return "mid"
    return "high"


def _scan_dense_chain_summaries(reports_root: Path) -> list[Path]:
    return sorted(reports_root.glob("*/*__dense_chain_summary.json"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_passport_data(
    reports_root: Path,
) -> tuple[
    int,
    dict[str, dict[str, list[float] | set[str]]],
    dict[int, list[float]],
    dict[tuple[str, int], list[float]],
    dict[str, dict[str, list[float] | set[str]]],
]:
    """
    Returns:
      total_notes
      harmonic_presence
      harmonic_profile
      range_profile
      resonance_box
    """

    summary_paths = _scan_dense_chain_summaries(reports_root)
    total_notes = 0

    # token -> stats
    harmonic_presence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "notes": set(),
            "amplitudes": [],
            "relative_amplitudes": [],
            "harmonic_indices": [],
            "delta_cents": [],
        }
    )

    # harmonic_index -> relative amplitudes across notes
    harmonic_profile: dict[int, list[float]] = defaultdict(list)

    # (zone, harmonic_index) -> relative amplitudes
    range_profile: dict[tuple[str, int], list[float]] = defaultdict(list)

    # candidates for resonance box:
    # token repeated across many different notes/harmonic positions
    resonance_box: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "notes": set(),
            "amplitudes": [],
            "relative_amplitudes": [],
            "harmonic_indices": [],
            "delta_cents": [],
        }
    )

    for path in summary_paths:
        folder_name = path.parent.name
        expected_note = _note_from_folder_name(folder_name)

        summary = _load_json(path)
        best = summary.get("best_chain")
        if not best:
            continue

        hits = list(best.get("hits", []))
        if not hits:
            continue

        total_notes += 1

        root_hz = _safe_float(best.get("root_hz", 0.0), 0.0)
        if root_hz <= 0:
            root_hz = _token_to_hz_approx(expected_note)

        zone = _range_zone(root_hz)

        amps = [_safe_float(h.get("matched_amplitude", 0.0), 0.0) for h in hits]
        max_amp = max(amps) if amps else 1.0
        if max_amp <= 0:
            max_amp = 1.0

        for h in hits:
            token = _safe_str(h.get("matched_token", ""))
            if not token:
                continue

            harmonic_index = _safe_int(h.get("harmonic_index", 0), 0)
            amp = _safe_float(h.get("matched_amplitude", 0.0), 0.0)
            rel_amp = amp / max_amp if max_amp > 0 else 0.0
            delta_cents = _safe_float(h.get("delta_cents", 0.0), 0.0)

            hp = harmonic_presence[token]
            hp["notes"].add(expected_note)
            hp["amplitudes"].append(amp)
            hp["relative_amplitudes"].append(rel_amp)
            hp["harmonic_indices"].append(float(harmonic_index))
            hp["delta_cents"].append(delta_cents)

            harmonic_profile[harmonic_index].append(rel_amp)
            range_profile[(zone, harmonic_index)].append(rel_amp)

            # resonance_box collects repeated tokens irrespective of harmonic order
            rb = resonance_box[token]
            rb["notes"].add(expected_note)
            rb["amplitudes"].append(amp)
            rb["relative_amplitudes"].append(rel_amp)
            rb["harmonic_indices"].append(float(harmonic_index))
            rb["delta_cents"].append(delta_cents)

    return total_notes, harmonic_presence, harmonic_profile, range_profile, resonance_box


def _write_harmonic_presence_csv(
    out_csv: Path,
    total_notes: int,
    harmonic_presence: dict[str, dict[str, Any]],
) -> None:
    rows = []

    for token, data in harmonic_presence.items():
        note_count = len(data["notes"])
        percent_notes = (100.0 * note_count / total_notes) if total_notes > 0 else 0.0

        rows.append(
            {
                "harmonic_token": token,
                "count_notes": note_count,
                "percent_notes": percent_notes,
                "mean_amplitude": _mean(data["amplitudes"]),
                "median_amplitude": _median(data["amplitudes"]),
                "std_amplitude": _std(data["amplitudes"]),
                "mean_relative_amplitude": _mean(data["relative_amplitudes"]),
                "median_relative_amplitude": _median(data["relative_amplitudes"]),
                "std_relative_amplitude": _std(data["relative_amplitudes"]),
                "mean_harmonic_index": _mean(data["harmonic_indices"]),
                "mean_delta_cents": _mean(data["delta_cents"]),
                "note_examples": " | ".join(sorted(list(data["notes"]))[:10]),
            }
        )

    rows.sort(
        key=lambda r: (
            -r["count_notes"],
            -r["mean_relative_amplitude"],
            r["harmonic_token"],
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "harmonic_token",
                "count_notes",
                "percent_notes",
                "mean_amplitude",
                "median_amplitude",
                "std_amplitude",
                "mean_relative_amplitude",
                "median_relative_amplitude",
                "std_relative_amplitude",
                "mean_harmonic_index",
                "mean_delta_cents",
                "note_examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_harmonic_profile_csv(
    out_csv: Path,
    total_notes: int,
    harmonic_profile: dict[int, list[float]],
) -> None:
    rows = []

    for harmonic_index, rels in sorted(harmonic_profile.items()):
        rows.append(
            {
                "harmonic_index": harmonic_index,
                "count_notes": len(rels),
                "percent_notes": (100.0 * len(rels) / total_notes) if total_notes > 0 else 0.0,
                "mean_relative_amplitude": _mean(rels),
                "median_relative_amplitude": _median(rels),
                "std_relative_amplitude": _std(rels),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "harmonic_index",
                "count_notes",
                "percent_notes",
                "mean_relative_amplitude",
                "median_relative_amplitude",
                "std_relative_amplitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_range_profile_csv(
    out_csv: Path,
    total_notes: int,
    range_profile: dict[tuple[str, int], list[float]],
) -> None:
    rows = []

    for (zone, harmonic_index), rels in sorted(range_profile.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "range_zone": zone,
                "harmonic_index": harmonic_index,
                "count_notes": len(rels),
                "mean_relative_amplitude": _mean(rels),
                "median_relative_amplitude": _median(rels),
                "std_relative_amplitude": _std(rels),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "range_zone",
                "harmonic_index",
                "count_notes",
                "mean_relative_amplitude",
                "median_relative_amplitude",
                "std_relative_amplitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_resonance_box_csv(
    out_csv: Path,
    total_notes: int,
    resonance_box: dict[str, dict[str, Any]],
    *,
    min_percent_notes: float,
) -> None:
    rows = []

    for token, data in resonance_box.items():
        note_count = len(data["notes"])
        percent_notes = (100.0 * note_count / total_notes) if total_notes > 0 else 0.0
        if percent_notes < min_percent_notes:
            continue

        rows.append(
            {
                "candidate_token": token,
                "count_notes": note_count,
                "percent_notes": percent_notes,
                "mean_amplitude": _mean(data["amplitudes"]),
                "median_amplitude": _median(data["amplitudes"]),
                "std_amplitude": _std(data["amplitudes"]),
                "mean_relative_amplitude": _mean(data["relative_amplitudes"]),
                "mean_harmonic_index": _mean(data["harmonic_indices"]),
                "mean_delta_cents": _mean(data["delta_cents"]),
                "note_examples": " | ".join(sorted(list(data["notes"]))[:12]),
            }
        )

    rows.sort(
        key=lambda r: (
            -r["percent_notes"],
            -r["mean_relative_amplitude"],
            r["candidate_token"],
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_token",
                "count_notes",
                "percent_notes",
                "mean_amplitude",
                "median_amplitude",
                "std_amplitude",
                "mean_relative_amplitude",
                "mean_harmonic_index",
                "mean_delta_cents",
                "note_examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_txt(
    out_txt: Path,
    *,
    reports_root: Path,
    total_notes: int,
    harmonic_presence: dict[str, dict[str, Any]],
    harmonic_profile: dict[int, list[float]],
    resonance_box: dict[str, dict[str, Any]],
    min_percent_notes_for_box: float,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("INSTRUMENT PASSPORT FROM DENSE CHAINS")
    lines.append("=" * 100)
    lines.append(f"reports_root: {reports_root}")
    lines.append(f"total_notes : {total_notes}")
    lines.append("")

    # Top repeated harmonic tokens
    repeated = []
    for token, data in harmonic_presence.items():
        repeated.append(
            (
                len(data["notes"]),
                _mean(data["relative_amplitudes"]),
                token,
            )
        )
    repeated.sort(reverse=True)

    lines.append("TOP REPEATED HARMONIC TOKENS")
    lines.append("-" * 100)
    for count_notes, mean_rel_amp, token in repeated[:20]:
        percent = (100.0 * count_notes / total_notes) if total_notes > 0 else 0.0
        lines.append(
            f"{token:12s}  notes={count_notes:4d}  percent={percent:7.2f}  mean_rel_amp={mean_rel_amp:.6f}"
        )

    lines.append("")
    lines.append("HARMONIC PROFILE BY INDEX")
    lines.append("-" * 100)
    for harmonic_index in sorted(harmonic_profile):
        rels = harmonic_profile[harmonic_index]
        percent = (100.0 * len(rels) / total_notes) if total_notes > 0 else 0.0
        lines.append(
            f"h{harmonic_index:>2d}  count={len(rels):4d}  percent={percent:7.2f}  "
            f"mean_rel_amp={_mean(rels):.6f}  median_rel_amp={_median(rels):.6f}"
        )

    lines.append("")
    lines.append("RESONANCE BOX CANDIDATES")
    lines.append("-" * 100)
    box_rows = []
    for token, data in resonance_box.items():
        percent = (100.0 * len(data["notes"]) / total_notes) if total_notes > 0 else 0.0
        if percent < min_percent_notes_for_box:
            continue
        box_rows.append(
            (
                percent,
                _mean(data["relative_amplitudes"]),
                token,
                len(data["notes"]),
            )
        )
    box_rows.sort(reverse=True)

    for percent, mean_rel_amp, token, note_count in box_rows[:20]:
        lines.append(
            f"{token:12s}  notes={note_count:4d}  percent={percent:7.2f}  mean_rel_amp={mean_rel_amp:.6f}"
        )

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build instrument passport from all __dense_chain_summary.json files in reports root."
    )
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--out_harmonic_presence_csv", required=True)
    ap.add_argument("--out_harmonic_profile_csv", required=True)
    ap.add_argument("--out_range_profile_csv", required=True)
    ap.add_argument("--out_resonance_box_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--min_percent_notes_for_box", type=float, default=70.0)
    args = ap.parse_args()

    reports_root = Path(args.reports_root).resolve()

    (
        total_notes,
        harmonic_presence,
        harmonic_profile,
        range_profile,
        resonance_box,
    ) = _collect_passport_data(reports_root)

    _write_harmonic_presence_csv(
        Path(args.out_harmonic_presence_csv).resolve(),
        total_notes,
        harmonic_presence,
    )
    _write_harmonic_profile_csv(
        Path(args.out_harmonic_profile_csv).resolve(),
        total_notes,
        harmonic_profile,
    )
    _write_range_profile_csv(
        Path(args.out_range_profile_csv).resolve(),
        total_notes,
        range_profile,
    )
    _write_resonance_box_csv(
        Path(args.out_resonance_box_csv).resolve(),
        total_notes,
        resonance_box,
        min_percent_notes=float(args.min_percent_notes_for_box),
    )
    _write_summary_txt(
        Path(args.out_summary_txt).resolve(),
        reports_root=reports_root,
        total_notes=total_notes,
        harmonic_presence=harmonic_presence,
        harmonic_profile=harmonic_profile,
        resonance_box=resonance_box,
        min_percent_notes_for_box=float(args.min_percent_notes_for_box),
    )

    print(json.dumps(
        {
            "reports_root": str(reports_root),
            "total_notes": total_notes,
            "out_harmonic_presence_csv": str(Path(args.out_harmonic_presence_csv).resolve()),
            "out_harmonic_profile_csv": str(Path(args.out_harmonic_profile_csv).resolve()),
            "out_range_profile_csv": str(Path(args.out_range_profile_csv).resolve()),
            "out_resonance_box_csv": str(Path(args.out_resonance_box_csv).resolve()),
            "out_summary_txt": str(Path(args.out_summary_txt).resolve()),
            "min_percent_notes_for_box": float(args.min_percent_notes_for_box),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()