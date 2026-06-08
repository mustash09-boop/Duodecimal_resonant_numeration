# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _zscore(x: float, mean: float, std: float) -> float:
    if std <= 1e-12:
        return 0.0
    return (x - mean) / std


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _guess_family(name: str) -> str:
    x = name.lower().replace("_", " ").replace("-", " ")
    if "piano" in x:
        return "keyboard"
    if x in {"cello", "cello2", "violin", "violin2", "viola", "double bass", "double bass2", "bass guitar", "guitar", "guitar2", "banjo", "mandolin"}:
        return "strings_plucked_or_bowed"
    if any(t in x for t in ["violin", "cello", "viola", "double bass"]):
        return "strings_bowed"
    if any(t in x for t in ["guitar", "banjo", "mandolin", "bass guitar"]):
        return "strings_plucked"
    if any(t in x for t in ["flute", "oboe", "clarinet", "bassoon", "saxophone", "cor anglais", "contrabassoon", "bass clarinet"]):
        return "winds"
    if any(t in x for t in ["horn"]):
        return "brass_or_horn"
    return "other_tonal"


def _extract_passport_vector(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    morph = data.get("harmonic_morphology_compare") or {}
    chain = data.get("harmonic_chain_spiral3d") or {}

    attack_h = (morph.get("top_harmonics_by_attack_energy") or [])[:3]
    sustain_h = (morph.get("top_harmonics_by_sustain_energy") or [])[:3]
    attack_hs = [int(row.get("harmonic", 0)) for row in attack_h]
    sustain_hs = [int(row.get("harmonic", 0)) for row in sustain_h]

    name = str(data.get("instrument_name") or path.stem).strip()
    return {
        "instrument_name": name,
        "family": _guess_family(name),
        "path": str(path),
        "attack_energy": float(summary.get("harmonic_morphology_mean_attack_energy", morph.get("mean_attack_energy", 0.0)) or 0.0),
        "sustain_energy": float(summary.get("harmonic_morphology_mean_sustain_energy", morph.get("mean_sustain_energy", 0.0)) or 0.0),
        "tail_energy": float(summary.get("harmonic_morphology_mean_tail_energy", morph.get("mean_tail_energy", 0.0)) or 0.0),
        "active_ratio": float(summary.get("harmonic_morphology_mean_active_ratio", morph.get("mean_active_ratio", 0.0)) or 0.0),
        "roughness": float(summary.get("harmonic_morphology_mean_roughness", morph.get("mean_roughness", 0.0)) or 0.0),
        "peak_time": float(morph.get("mean_peak_time", 0.0) or 0.0),
        "unassigned_ratio": float(summary.get("harmonic_chain_unassigned_ratio", chain.get("unassigned_ratio", 0.0)) or 0.0),
        "box_all_components": float(summary.get("box_all_components", 0.0) or 0.0),
        "box_breath_components": float(summary.get("box_breath_components", 0.0) or 0.0),
        "box_resonance_components": float(summary.get("box_resonance_components", 0.0) or 0.0),
        "box_relation_components": float(summary.get("box_relation_components", 0.0) or 0.0),
        "notes_total": float(summary.get("total_notes", 0.0) or 0.0),
        "attack_top_harmonics": attack_hs,
        "sustain_top_harmonics": sustain_hs,
        "attack_h57_bias": 1.0 if 5 in attack_hs or 7 in attack_hs else 0.0,
        "sustain_h57_bias": 1.0 if 5 in sustain_hs or 7 in sustain_hs else 0.0,
    }


def _normalize_passport_vectors(vectors: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        "attack_energy",
        "sustain_energy",
        "tail_energy",
        "active_ratio",
        "roughness",
        "peak_time",
        "unassigned_ratio",
        "box_all_components",
        "box_breath_components",
        "box_resonance_components",
        "box_relation_components",
        "notes_total",
    ]
    stats: dict[str, tuple[float, float]] = {}
    for key in numeric_keys:
        vals = [float(v[key]) for v in vectors]
        mean = sum(vals) / max(1, len(vals))
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
        stats[key] = (mean, math.sqrt(var))
    out: list[dict[str, object]] = []
    for row in vectors:
        item = dict(row)
        for key in numeric_keys:
            mean, std = stats[key]
            item[f"z_{key}"] = _zscore(float(row[key]), mean, std)
        out.append(item)
    return out


def _event_feature_vector(row: dict[str, str]) -> dict[str, float]:
    frame_count = max(1, _safe_int(row.get("frame_count"), 1))
    duration_frames = max(1, _safe_int(row.get("duration_frames"), frame_count))
    birth_score = _safe_float(row.get("birth_score"))
    final_score = _safe_float(row.get("final_score"))
    mean_score = _safe_float(row.get("mean_score"))
    max_score = max(_safe_float(row.get("max_score")), 1e-9)
    attack_peak_frame = _safe_int(row.get("attack_peak_frame"), _safe_int(row.get("birth_frame")))
    birth_frame = _safe_int(row.get("birth_frame"))
    re_exc = _safe_int(row.get("re_excitation_count"))
    active_body = _safe_int(row.get("active_body_count"))
    response = _safe_int(row.get("response_trace_count"))
    decay = _safe_int(row.get("decay_trace_count"))
    overlap = _safe_int(row.get("active_overlap_count_at_birth"))
    concurrency = _safe_int(row.get("concurrency_count_at_birth"), 1)

    event_type = str(row.get("initial_event_type", "")).strip()
    harmonic_guess = str(row.get("initial_harmonic_hypothesis", "")).strip()
    resonance_guess = str(row.get("initial_resonance_hypothesis", "")).strip()

    attack_index = birth_score / max_score
    sustain_index = active_body / frame_count
    tail_index = (response + decay) / frame_count
    roughness_proxy = (re_exc + response + decay) / frame_count
    peak_pos = max(0.0, min(1.0, (attack_peak_frame - birth_frame) / max(1, duration_frames)))
    unassigned_proxy = 0.0
    if "UNRESOLVED" in resonance_guess:
        unassigned_proxy += 0.65
    if "SHARED" in resonance_guess:
        unassigned_proxy += 0.25
    if "TAIL" in resonance_guess or "FIELD" in resonance_guess:
        unassigned_proxy += 0.15
    unassigned_proxy = min(1.0, unassigned_proxy)
    h57_proxy = 1.0 if "STABLE" in harmonic_guess else 0.35
    if "MIGRATING" in harmonic_guess or "SHARED" in harmonic_guess:
        h57_proxy = 0.15

    return {
        "attack_index": attack_index,
        "sustain_index": sustain_index,
        "tail_index": tail_index,
        "active_ratio_proxy": frame_count / duration_frames,
        "roughness_proxy": roughness_proxy,
        "peak_pos_proxy": peak_pos,
        "unassigned_proxy": unassigned_proxy,
        "box_complexity_proxy": min(1.0, overlap / 8.0 + concurrency / 16.0),
        "h57_proxy": h57_proxy,
        "mean_score_proxy": mean_score / max_score,
        "discrete_attack_flag": 1.0 if event_type == "DISCRETE_PIANO_EXCITATION" else 0.0,
        "body_return_flag": 1.0 if event_type == "BODY_RETURN_EVENT" else 0.0,
        "field_flag": 1.0 if event_type == "FIELD_TRACE_EVENT" else 0.0,
        "mixed_flag": 1.0 if event_type == "MIXED_EVENT" else 0.0,
        "short_flag": 1.0 if event_type == "VERY_SHORT_EVENT" else 0.0,
    }


def _score_event_against_passport(event_vec: dict[str, float], passport: dict[str, object]) -> tuple[float, dict[str, float]]:
    p_attack = float(passport["attack_energy"])
    p_sustain = float(passport["sustain_energy"])
    p_tail = float(passport["tail_energy"])
    p_active = float(passport["active_ratio"])
    p_rough = float(passport["roughness"])
    p_peak = float(passport["peak_time"])
    p_unassigned = float(passport["unassigned_ratio"])

    # Compare event proxies to passport morphology using relative differences and logits.
    sim_attack = 1.0 - min(1.0, abs(event_vec["attack_index"] - _sigmoid(float(passport["z_attack_energy"]))) )
    sim_sustain = 1.0 - min(1.0, abs(event_vec["sustain_index"] - _sigmoid(float(passport["z_sustain_energy"]))) )
    sim_tail = 1.0 - min(1.0, abs(event_vec["tail_index"] - _sigmoid(float(passport["z_tail_energy"]))) )
    sim_active = 1.0 - min(1.0, abs(event_vec["active_ratio_proxy"] - p_active))
    sim_rough = 1.0 - min(1.0, abs(event_vec["roughness_proxy"] - _sigmoid(float(passport["z_roughness"]))) )
    sim_peak = 1.0 - min(1.0, abs(event_vec["peak_pos_proxy"] - p_peak))
    sim_unassigned = 1.0 - min(1.0, abs(event_vec["unassigned_proxy"] - p_unassigned))

    box_density = float(passport["box_all_components"]) / max(1.0, float(passport["notes_total"]))
    box_res_density = float(passport["box_resonance_components"]) / max(1.0, float(passport["notes_total"]))
    box_norm = min(1.0, (box_density + box_res_density) / 6.0)
    sim_box = 1.0 - min(1.0, abs(event_vec["box_complexity_proxy"] - box_norm))

    h57_target = 0.5 * float(passport["attack_h57_bias"]) + 0.5 * float(passport["sustain_h57_bias"])
    sim_h57 = 1.0 - min(1.0, abs(event_vec["h57_proxy"] - h57_target))

    bias = 0.0
    family = str(passport["family"])
    if event_vec["discrete_attack_flag"] > 0.5 and family == "keyboard":
        bias += 0.08
    if event_vec["body_return_flag"] > 0.5 and family in {"strings_bowed", "winds", "brass_or_horn"}:
        bias += 0.04
    if event_vec["field_flag"] > 0.5 and family in {"winds", "brass_or_horn"}:
        bias += 0.03
    if event_vec["short_flag"] > 0.5 and family == "keyboard":
        bias += 0.04

    weighted = (
        0.16 * sim_attack +
        0.16 * sim_sustain +
        0.10 * sim_tail +
        0.12 * sim_active +
        0.10 * sim_rough +
        0.08 * sim_peak +
        0.12 * sim_unassigned +
        0.08 * sim_box +
        0.08 * sim_h57 +
        bias
    )
    weighted = max(0.0, min(1.0, weighted))
    parts = {
        "sim_attack": sim_attack,
        "sim_sustain": sim_sustain,
        "sim_tail": sim_tail,
        "sim_active": sim_active,
        "sim_rough": sim_rough,
        "sim_peak": sim_peak,
        "sim_unassigned": sim_unassigned,
        "sim_box": sim_box,
        "sim_h57": sim_h57,
        "bias": bias,
    }
    return weighted, parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Late refinement for event slots by comparing each event against all available tonal passports.")
    ap.add_argument("--event_slots_csv", required=True)
    ap.add_argument("--passports_root", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    event_rows = _load_csv(Path(args.event_slots_csv))
    passport_paths = sorted(Path(args.passports_root).rglob("*__instrument_passport.json"))
    passports = _normalize_passport_vectors([_extract_passport_vector(p) for p in passport_paths])

    out_rows: list[dict[str, str]] = []
    top1_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    family_mass_counter: Counter[str] = Counter()
    ambiguity_counter: Counter[str] = Counter()

    for row in event_rows:
        event_vec = _event_feature_vector(row)
        scored: list[tuple[float, dict[str, object], dict[str, float]]] = []
        for passport in passports:
            score, parts = _score_event_against_passport(event_vec, passport)
            scored.append((score, passport, parts))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, int(args.top_k))]
        best_score = top[0][0]
        second_score = top[1][0] if len(top) > 1 else 0.0
        score_gap = best_score - second_score
        top1_name = str(top[0][1]["instrument_name"])
        top1_family = str(top[0][1]["family"])
        top1_counter[top1_name] += 1
        family_counter[top1_family] += 1

        family_scores: dict[str, float] = defaultdict(float)
        for score, passport, _parts in top:
            family_scores[str(passport["family"])] += score
        best_family = max(family_scores.items(), key=lambda kv: kv[1])[0]
        family_mass_counter[best_family] += 1

        ambiguity = "CLEAR"
        if score_gap < 0.035:
            ambiguity = "HIGH_AMBIGUITY"
        elif score_gap < 0.08:
            ambiguity = "MEDIUM_AMBIGUITY"
        ambiguity_counter[ambiguity] += 1

        top_matches = []
        for rank, (score, passport, parts) in enumerate(top, start=1):
            top_matches.append(
                {
                    "rank": rank,
                    "instrument": passport["instrument_name"],
                    "family": passport["family"],
                    "score": round(score, 9),
                    "parts": {k: round(v, 6) for k, v in parts.items()},
                }
            )

        out = dict(row)
        out["passport_top1_instrument"] = top1_name
        out["passport_top1_family"] = top1_family
        out["passport_top1_score"] = f"{best_score:.9f}"
        out["passport_top2_score"] = f"{second_score:.9f}"
        out["passport_top1_gap"] = f"{score_gap:.9f}"
        out["passport_family_mass_winner"] = best_family
        out["passport_refinement_ambiguity"] = ambiguity
        out["passport_top_matches_json"] = json.dumps(top_matches, ensure_ascii=False)
        out_rows.append(out)

    base_fields = list(event_rows[0].keys()) if event_rows else []
    extra_fields = [
        "passport_top1_instrument",
        "passport_top1_family",
        "passport_top1_score",
        "passport_top2_score",
        "passport_top1_gap",
        "passport_family_mass_winner",
        "passport_refinement_ambiguity",
        "passport_top_matches_json",
    ]
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in base_fields + extra_fields})

    summary_lines = [
        "EVENT PASSPORT REFINEMENT RESOLVER",
        "=" * 72,
        f"input_events: {len(event_rows)}",
        f"passports_loaded: {len(passports)}",
        f"top_k: {int(args.top_k)}",
        "",
        "top1_instrument_counts:",
    ]
    for key, value in top1_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "top1_family_counts:"])
    for key, value in family_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "family_mass_winner_counts:"])
    for key, value in family_mass_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.extend(["", "ambiguity_counts:"])
    for key, value in ambiguity_counter.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "input_events": len(event_rows),
                "passports_loaded": len(passports),
                "top_k": int(args.top_k),
                "top1_instrument_counts": dict(top1_counter),
                "top1_family_counts": dict(family_counter),
                "family_mass_winner_counts": dict(family_mass_counter),
                "ambiguity_counts": dict(ambiguity_counter),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
