from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _analyze_pitched_library(library_dir: Path) -> dict[str, Any] | None:
    range_dir = library_dir / "20_range_research"
    box_dir = library_dir / "30_note_box_profiles"
    if not range_dir.exists() or not box_dir.exists():
        return None

    passport_paths = sorted(range_dir.glob("*__instrument_passport.json"))
    notes_paths = sorted(range_dir.glob("*__instrument_passport_notes.csv"))
    profile_paths = sorted(box_dir.glob("*__note_box_profile.csv"))
    if not passport_paths or not notes_paths or not profile_paths:
        return None

    passport = _load_json(passport_paths[0])
    note_rows = _load_csv_rows(notes_paths[0])

    top_presence_values: list[float] = []
    top_frame_values: list[int] = []
    component_counts: list[int] = []
    persistent_component_counts: list[int] = []

    for profile_path in profile_paths:
        rows = _load_csv_rows(profile_path)
        if not rows:
            continue
        component_counts.append(len(rows))
        rows_sorted = sorted(rows, key=lambda r: _safe_float(r.get("presence_ratio"), 0.0), reverse=True)
        top_presence_values.append(_safe_float(rows_sorted[0].get("presence_ratio"), 0.0))
        top_frame_values.append(_safe_int(rows_sorted[0].get("frame_count"), 0))
        persistent_component_counts.append(
            sum(1 for row in rows if _safe_float(row.get("presence_ratio"), 0.0) >= 0.10)
        )

    removed_box_percent = [_safe_float(row.get("dense_removed_percent"), 0.0) for row in note_rows]
    root_frame_counts = [_safe_int(row.get("root_frame_count"), 0) for row in note_rows]
    clean_rows = [_safe_int(row.get("clean_rows"), 0) for row in note_rows]
    dense_rows = [_safe_int(row.get("dense_rows"), 0) for row in note_rows]

    summary = passport.get("summary", {})
    return {
        "library": library_dir.name,
        "instrument_name": passport.get("instrument_name", library_dir.name),
        "total_notes": _safe_int(summary.get("total_notes"), 0),
        "box_breath_components": _safe_int(summary.get("box_breath_components"), 0),
        "box_resonance_components": _safe_int(summary.get("box_resonance_components"), 0),
        "avg_profile_components": _mean([float(x) for x in component_counts]),
        "median_profile_components": _median([float(x) for x in component_counts]),
        "avg_top_presence_ratio": _mean(top_presence_values),
        "median_top_presence_ratio": _median(top_presence_values),
        "avg_top_frame_count": _mean([float(x) for x in top_frame_values]),
        "avg_persistent_components_ge_10pct": _mean([float(x) for x in persistent_component_counts]),
        "avg_root_frame_count": _mean([float(x) for x in root_frame_counts]),
        "avg_dense_rows": _mean([float(x) for x in dense_rows]),
        "avg_clean_rows": _mean([float(x) for x in clean_rows]),
        "avg_removed_box_percent": _mean(removed_box_percent),
        "top_box_tokens": [
            f"{item.get('cluster_token', '')} ({item.get('percent_notes', 0):.2f}% notes)"
            for item in passport.get("box_all_top", [])[:5]
        ],
    }


def _analyze_percussion_family(percussion_dir: Path) -> dict[str, Any] | None:
    passports_dir = percussion_dir / "40_passports"
    family_path = passports_dir / "percussion__family_passport.json"
    if not family_path.exists():
        return None

    family = _load_json(family_path)
    instrument_passports = []
    for path in sorted(passports_dir.glob("*__percussion_passport.json")):
        if path.name == "percussion__family_passport.json":
            continue
        instrument_passports.append(_load_json(path))

    attack_ratio_values: list[float] = []
    duration_values: list[float] = []
    centroid_values: list[float] = []
    spread_values: list[float] = []
    gesture_counter: Counter[str] = Counter()
    top_token_counter: Counter[str] = Counter()

    sustained_like: list[dict[str, Any]] = []
    impulsive_like: list[dict[str, Any]] = []

    for passport in instrument_passports:
        summary = passport.get("summary", {})
        duration = _safe_float(summary.get("avg_duration_sec"), 0.0)
        attack = _safe_float(summary.get("avg_attack_time_sec"), 0.0)
        centroid = _safe_float(summary.get("avg_spectral_centroid_hz"), 0.0)
        spread = _safe_float(summary.get("avg_spectral_spread_hz"), 0.0)
        ratio = attack / duration if duration > 0 else 0.0

        duration_values.append(duration)
        attack_ratio_values.append(ratio)
        centroid_values.append(centroid)
        spread_values.append(spread)

        gestures = [str(x) for x in summary.get("gesture_types", [])]
        for gesture in gestures:
            gesture_counter[gesture] += 1
        for item in passport.get("top_resonances", [])[:5]:
            token = str(item.get("token", "")).strip()
            if token:
                top_token_counter[token] += 1

        row = {
            "instrument_name": passport.get("instrument_name", ""),
            "attack_ratio": ratio,
            "duration_sec": duration,
            "centroid_hz": centroid,
            "gestures": ",".join(gestures),
        }
        if ratio <= 0.20:
            impulsive_like.append(row)
        if ratio >= 0.45:
            sustained_like.append(row)

    return {
        "instrument_family": family.get("instrument_family", "percussion"),
        "instrument_count": _safe_int(family.get("instrument_count"), len(instrument_passports)),
        "avg_attack_ratio": _mean(attack_ratio_values),
        "median_attack_ratio": _median(attack_ratio_values),
        "avg_duration_sec": _mean(duration_values),
        "avg_centroid_hz": _mean(centroid_values),
        "avg_spread_hz": _mean(spread_values),
        "gesture_counter": gesture_counter,
        "top_token_counter": top_token_counter,
        "impulsive_like": sorted(impulsive_like, key=lambda x: x["attack_ratio"])[:8],
        "sustained_like": sorted(sustained_like, key=lambda x: x["attack_ratio"], reverse=True)[:8],
    }


def _build_markdown(pitched: list[dict[str, Any]], percussion: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Block004 Transfer Pattern Research")
    lines.append("")
    lines.append("## Main conclusion")
    lines.append("")
    lines.append("Pitched libraries and percussion do not share one universal `note -> box transfer` law in the same explicit form.")
    lines.append("Pitched instruments expose a stable post-note residual layer through `note_box_profile` and instrument `box_*` passports.")
    lines.append("Percussion is described in the project by `event -> resonance field` passports instead: attack, duration, gesture and top resonance tokens.")
    lines.append("")
    lines.append("That means a universal streaming algorithm probably needs at least two branches:")
    lines.append("- `pitched: exciter -> primary note chain -> controlled sustain -> box transfer -> secondary resonance`")
    lines.append("- `percussion: exciter/event -> gesture field -> resonance field persistence`, with no requirement that a stable note-chain exists")
    lines.append("")
    lines.append("## Pitched libraries")
    lines.append("")
    lines.append("| library | notes | avg box comps / note | avg top presence | avg persistent comps >=10% | avg root frames | box breath comps | box resonance comps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in pitched:
        lines.append(
            f"| {row['library']} | {row['total_notes']} | {row['avg_profile_components']:.1f} | "
            f"{row['avg_top_presence_ratio']:.3f} | {row['avg_persistent_components_ge_10pct']:.1f} | "
            f"{row['avg_root_frame_count']:.1f} | {row['box_breath_components']} | {row['box_resonance_components']} |"
        )
    lines.append("")
    for row in pitched:
        lines.append(f"### {row['library']}")
        lines.append("")
        lines.append(
            f"- Avg top `note_box` presence: `{row['avg_top_presence_ratio']:.3f}` over `{row['avg_top_frame_count']:.1f}` frames."
        )
        lines.append(
            f"- Avg persistent `note_box` components with presence >=10%: `{row['avg_persistent_components_ge_10pct']:.1f}`."
        )
        lines.append(
            f"- Avg root-support frames: `{row['avg_root_frame_count']:.1f}`. This is the closest in-project proxy for how long the primary note-chain stays explicit before residual structure matters."
        )
        lines.append(f"- Top instrument-box tokens: {', '.join(row['top_box_tokens'])}.")
        lines.append("")
    lines.append("### Reading of pitched data")
    lines.append("")
    lines.append("- `piano_midi1` shows many long-lived residual components per note-box profile. This matches the intuition that after excitation, the note hands energy to a rich corpus/body layer.")
    lines.append("- `violin` still has note-box structure, but its box signature is different: many more `box_breath` components and shorter note windows, which suggests tighter coupling between note production and body/air/bow activity instead of a clean hammer-then-box split.")
    lines.append("- So even inside pitched instruments there is not one single timing law; there is a common *architecture* of residual body layers, but the transition shape differs by instrument class.")
    lines.append("")
    lines.append("## Percussion family")
    lines.append("")
    lines.append(f"- Instrument count: `{percussion['instrument_count']}`")
    lines.append(f"- Avg attack/duration ratio: `{percussion['avg_attack_ratio']:.3f}`")
    lines.append(f"- Median attack/duration ratio: `{percussion['median_attack_ratio']:.3f}`")
    lines.append(f"- Avg event duration: `{percussion['avg_duration_sec']:.3f}` sec")
    lines.append(f"- Avg spectral centroid: `{percussion['avg_centroid_hz']:.2f}` Hz")
    lines.append(f"- Avg spectral spread: `{percussion['avg_spread_hz']:.2f}` Hz")
    lines.append("")
    lines.append("### Gesture distribution")
    lines.append("")
    for gesture, count in percussion["gesture_counter"].most_common():
        lines.append(f"- `{gesture}`: {count}")
    lines.append("")
    lines.append("### Most recurring resonance tokens across percussion passports")
    lines.append("")
    for token, count in percussion["top_token_counter"].most_common(12):
        lines.append(f"- `{token}`: {count} passports")
    lines.append("")
    lines.append("### Impulsive-like percussion examples")
    lines.append("")
    for row in percussion["impulsive_like"]:
        lines.append(
            f"- `{row['instrument_name']}` attack ratio `{row['attack_ratio']:.3f}`, duration `{row['duration_sec']:.3f}` sec, gestures `{row['gestures']}`"
        )
    lines.append("")
    lines.append("### Sustained / phrase-like percussion examples")
    lines.append("")
    for row in percussion["sustained_like"]:
        lines.append(
            f"- `{row['instrument_name']}` attack ratio `{row['attack_ratio']:.3f}`, duration `{row['duration_sec']:.3f}` sec, gestures `{row['gestures']}`"
        )
    lines.append("")
    lines.append("### Reading of percussion data")
    lines.append("")
    lines.append("- Percussion passports do not preserve a `note_box_profile` language. They preserve `event_count`, `gesture_types`, `attack_time`, `duration`, and recurring resonance tokens.")
    lines.append("- Some percussion is quasi-pitched or resonance-rich (`Thai-gong`, `triangle`, cymbals), but even there the project treats them as event-driven resonance fields, not as stable note roots with a later box-transfer stage.")
    lines.append("- This suggests that the absence of a universal `note -> box transfer` law is itself a meaningful distinction between classes: pitched sound often has an identifiable note-chain before body dominance, while percussion often starts as an event-field where attack and resonance are the primary ontology.")
    lines.append("")
    lines.append("## Practical implication for Block002")
    lines.append("")
    lines.append("- Do not tie the universal pipeline to `piano_midi1` note-box timings.")
    lines.append("- Keep a universal `excitation-first` entrypoint from `_audio_probe`.")
    lines.append("- Branch after early causality:")
    lines.append("  - if a stable root-bearing note-chain emerges, use the pitched branch with `box transfer`")
    lines.append("  - if the signal is better described by attack/gesture/resonance persistence, use the percussion/event branch")
    lines.append("")
    lines.append("So the useful generalization is not one shared transfer curve, but one shared early-causality entrance followed by class-specific continuation laws.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Block004 transfer patterns across pitched libraries and percussion.")
    ap.add_argument("--block004-root", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    root = Path(args.block004_root)
    pitched_results: list[dict[str, Any]] = []
    for library_dir in sorted(root.iterdir()):
        if not library_dir.is_dir():
            continue
        if library_dir.name in {"percussion", "_multi_instrument_compare"}:
            continue
        result = _analyze_pitched_library(library_dir)
        if result is not None:
            pitched_results.append(result)

    percussion_result = _analyze_percussion_family(root / "percussion")
    if percussion_result is None:
        raise SystemExit("percussion family passport not found")

    report = _build_markdown(pitched_results, percussion_result)
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
