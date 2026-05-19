# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# Safe helpers
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_tokens(raw: Any) -> Set[str]:
    return {x.strip() for x in str(raw or "").replace(",", " ").split() if x.strip()}


def _join_preview(tokens: Iterable[str], limit: int = 48) -> str:
    return " ".join(sorted(set(tokens))[:limit])


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ============================================================
# 12-token helpers
# ============================================================

_DEGREE_TO_INDEX = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "A": 9,
    "B": 10,
    "C": 11,
}

_TOKEN_RE = re.compile(r"^\s*([1-9A-C]+)\.([1-9A-C])(?:'(.*))?\s*$")


def _parse_token(token: str) -> Optional[Tuple[int, int, str]]:
    m = _TOKEN_RE.match(str(token or "").strip())
    if not m:
        return None

    octave_s, degree_s, micro = m.groups()
    if degree_s not in _DEGREE_TO_INDEX:
        return None

    octave_value = 0
    for ch in octave_s:
        if ch not in _DEGREE_TO_INDEX:
            return None
        octave_value = octave_value * 12 + (_DEGREE_TO_INDEX[ch] + 1)

    return octave_value, _DEGREE_TO_INDEX[degree_s], micro or "-"


def _token_coarse(token: str) -> str:
    s = str(token or "").strip()
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if "." in s:
        return s + "'-"
    return s


def _pitch_index12(token: str) -> Optional[int]:
    p = _parse_token(token)
    if not p:
        return None
    octave, degree, _micro = p
    return octave * 12 + degree


# Approximate equal-tempered harmonic offsets in semitones.
# 2 -> 12, 3 -> 19, 4 -> 24, 5 -> 28, 6 -> 31, 7 -> 34, 8 -> 36
_HARMONIC_OFFSETS_12 = {
    2: 12,
    3: 19,
    4: 24,
    5: 28,
    6: 31,
    7: 34,
    8: 36,
}


def _expected_harmonic_degrees(root_token: str) -> Dict[int, int]:
    root_idx = _pitch_index12(root_token)
    if root_idx is None:
        return {}
    return {h: (root_idx + offset) % 12 for h, offset in _HARMONIC_OFFSETS_12.items()}


def _harmonic_support_for_root(root_token: str, observed_tokens: Set[str]) -> Dict[str, Any]:
    """
    Harmonic support with special emphasis on 5th and 7th harmonics.

    We do not require exact octave equality. For polyphony, octave-local evidence
    is unstable; degree-class and micro-neighborhood evidence is safer.
    """
    expected = _expected_harmonic_degrees(root_token)
    if not expected:
        return {
            "harmonic_support_score": 0.0,
            "harmonic_5_7_score": 0.0,
            "present_harmonics": "",
            "missing_harmonics": "",
            "harmonic_tokens": "",
        }

    observed_by_degree: Dict[int, List[str]] = defaultdict(list)
    for t in observed_tokens:
        p = _parse_token(t)
        if not p:
            continue
        _oct, degree, _micro = p
        observed_by_degree[degree].append(t)

    # 5 and 7 are intentionally stronger. They help compensate low/high register
    # ambiguity and are important for polyphonic chain evidence.
    weights = {
        2: 0.10,
        3: 0.15,
        4: 0.08,
        5: 0.26,
        6: 0.10,
        7: 0.24,
        8: 0.07,
    }

    present = []
    missing = []
    harmonic_tokens = []
    score = 0.0
    total = sum(weights.values())

    for h, degree in expected.items():
        toks = observed_by_degree.get(degree, [])
        if toks:
            present.append(str(h))
            harmonic_tokens.extend(toks[:8])
            score += weights.get(h, 0.0)
        else:
            missing.append(str(h))

    harmonic_support_score = score / max(total, 1e-9)

    h57_total = weights[5] + weights[7]
    h57_score = 0.0
    if observed_by_degree.get(expected.get(5, -1)):
        h57_score += weights[5]
    if observed_by_degree.get(expected.get(7, -1)):
        h57_score += weights[7]

    return {
        "harmonic_support_score": harmonic_support_score,
        "harmonic_5_7_score": h57_score / max(h57_total, 1e-9),
        "present_harmonics": " ".join(present),
        "missing_harmonics": " ".join(missing),
        "harmonic_tokens": _join_preview(harmonic_tokens, 48),
    }


def _register_compensation(root_token: str, harmonic_support: float, h57_score: float) -> Tuple[float, str]:
    """
    Low/high compensation learned from single-note behavior.

    Low register:
      fundamental/root can be unstable or swallowed by body response.
      Harmonic chain evidence matters more.

    High register:
      lower harmonics can be weak or absent; 5/7 and local micro topology
      become stronger evidence than broad low harmonics.
    """
    p = _parse_token(root_token)
    if not p:
        return 0.0, "unknown_register"

    octave, _degree, _micro = p

    if octave <= 7:
        return 0.18 * harmonic_support + 0.12 * h57_score, "low_register_compensation"

    if octave >= 11:
        return 0.10 * harmonic_support + 0.20 * h57_score, "high_register_compensation"

    return 0.12 * harmonic_support + 0.16 * h57_score, "mid_register_compensation"


# ============================================================
# Scene extraction helpers
# ============================================================

def _candidate_root(row: Dict[str, Any]) -> str:
    for k in (
        "source_root_hint_micro",
        "target_root_hint_micro",
        "root_hint_micro_not_identity",
        "root_hint_not_identity",
    ):
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def _candidate_tokens(row: Dict[str, Any]) -> Set[str]:
    toks = set()
    for k in (
        "source_micro_preview",
        "target_micro_preview",
        "micro_token_preview",
    ):
        toks.update(_split_tokens(row.get(k, "")))
    return toks


def _box_residual_tokens(note_root: str, tokens: Set[str], harmonic_tokens: Set[str]) -> Set[str]:
    """
    Residuals are tokens not explained as root/coarse-root/harmonic-chain evidence.

    These are not automatically instrument-box matches; they are box candidates /
    unexplained resonances until matched later against passports.
    """
    coarse_root = _token_coarse(note_root)
    residual = set()

    for t in tokens:
        if not t:
            continue
        if _token_coarse(t) == coarse_root:
            continue
        if t in harmonic_tokens:
            continue
        residual.add(t)

    return residual


def _box_signature_score(
    residual_tokens: Set[str],
    source_micro_count: int,
    target_micro_count: int,
    field_micro_richness: int,
) -> float:
    residual_density = min(len(residual_tokens) / 64.0, 1.0)
    entity_micro_density = min((source_micro_count + target_micro_count) / 140.0, 1.0)
    field_density = min(field_micro_richness / 160.0, 1.0)

    return residual_density * 0.46 + entity_micro_density * 0.28 + field_density * 0.26


def _note_candidate_score(
    *,
    episode_confidence: float,
    ownership_stability: float,
    continuity_support: float,
    harmonic_support: float,
    harmonic_5_7_score: float,
    register_compensation: float,
    masking_penalty: float,
) -> float:
    return _clamp(
        episode_confidence * 0.24
        + ownership_stability * 0.18
        + continuity_support * 0.18
        + harmonic_support * 0.16
        + harmonic_5_7_score * 0.16
        + register_compensation
        - masking_penalty
    )


def _scene_label(note_score: float, box_score: float, masking_warning: str) -> str:
    if masking_warning:
        return "NOTE_WITH_MASKING_RISK"
    if note_score >= 0.62 and box_score >= 0.42:
        return "NOTE_WITH_BOX_SIGNATURE"
    if note_score >= 0.62:
        return "NOTE_DOMINANT_SCENE"
    if box_score >= 0.48:
        return "BOX_DOMINANT_RESIDUAL_SCENE"
    if note_score >= 0.38:
        return "NOTE_CANDIDATE_SCENE"
    return "AMBIGUOUS_RESONANCE_SCENE"


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build composition-level note + box scenes without using instrument passports. "
            "The module infers note candidates, box residual signatures and harmonic support "
            "from ownership episodes, entity timelines and field windows."
        )
    )

    ap.add_argument("--ownership_episodes_csv", required=True)
    ap.add_argument("--entity_timeline_csv", required=True)
    ap.add_argument("--field_windows_csv", required=True)

    ap.add_argument("--out_scene_csv", required=True)
    ap.add_argument("--out_note_candidates_csv", required=True)
    ap.add_argument("--out_box_residuals_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_note_candidate_score", type=float, default=0.20)
    ap.add_argument("--window_frames", type=int, default=120)
    ap.add_argument("--step_frames", type=int, default=30)

    args = ap.parse_args()

    episodes = _load_csv(Path(args.ownership_episodes_csv))
    timelines = _load_csv(Path(args.entity_timeline_csv))
    fields = _load_csv(Path(args.field_windows_csv))

    timeline_by_entity = {str(r.get("entity_id", "")).strip(): r for r in timelines}

    field_windows = []
    for r in fields:
        field_windows.append({
            "start": _safe_int(r.get("window_start_frame"), 0),
            "end": _safe_int(r.get("window_end_frame"), 0),
            "field_state": str(r.get("field_state", "")).strip(),
            "micro_field_texture": str(r.get("micro_field_texture", "")).strip(),
            "continuity_texture": str(r.get("continuity_texture", "")).strip(),
            "micro_token_richness": _safe_int(r.get("micro_token_richness"), 0),
            "micro_token_preview": str(r.get("micro_token_preview", "")).strip(),
            "mean_continuity_support": _safe_float(r.get("mean_continuity_support"), 0.0),
            "mean_causal_confidence": _safe_float(r.get("mean_causal_confidence"), 0.0),
        })

    def _field_for_frame(frame: int) -> Dict[str, Any]:
        for fw in field_windows:
            if fw["start"] <= frame < fw["end"]:
                return fw
        return {
            "field_state": "",
            "micro_field_texture": "",
            "continuity_texture": "",
            "micro_token_richness": 0,
            "micro_token_preview": "",
            "mean_continuity_support": 0.0,
            "mean_causal_confidence": 0.0,
        }

    scene_rows = []
    note_rows = []
    box_rows = []
    readable_rows = []

    scene_counts = defaultdict(int)
    compensation_counts = defaultdict(int)

    for ep in episodes:
        frame = _safe_int(ep.get("source_birth_frame"), 0)
        src = str(ep.get("source_entity", "")).strip()
        dst = str(ep.get("target_entity", "")).strip()

        root = _candidate_root(ep)
        if not root:
            continue

        source_timeline = timeline_by_entity.get(src, {})
        target_timeline = timeline_by_entity.get(dst, {})

        ownership_stability = max(
            _safe_float(source_timeline.get("ownership_stability"), 0.0),
            _safe_float(target_timeline.get("ownership_stability"), 0.0),
        )

        episode_confidence = _safe_float(ep.get("episode_confidence"), 0.0)
        continuity_support = _safe_float(ep.get("continuity_support"), 0.0)
        masking_warning = str(ep.get("masking_warning", "")).strip()

        source_micro_count = _safe_int(ep.get("source_micro_count"), 0)
        target_micro_count = _safe_int(ep.get("target_micro_count"), 0)

        fw = _field_for_frame(frame)

        tokens = _candidate_tokens(ep)
        tokens.update(_split_tokens(fw.get("micro_token_preview", "")))

        harmonic_info = _harmonic_support_for_root(root, tokens)
        harmonic_tokens = _split_tokens(harmonic_info["harmonic_tokens"])

        harmonic_support = _safe_float(harmonic_info["harmonic_support_score"], 0.0)
        h57_score = _safe_float(harmonic_info["harmonic_5_7_score"], 0.0)

        register_compensation, compensation_basis = _register_compensation(root, harmonic_support, h57_score)
        compensation_counts[compensation_basis] += 1

        masking_penalty = 0.10 if masking_warning else 0.0

        note_score = _note_candidate_score(
            episode_confidence=episode_confidence,
            ownership_stability=ownership_stability,
            continuity_support=continuity_support,
            harmonic_support=harmonic_support,
            harmonic_5_7_score=h57_score,
            register_compensation=register_compensation,
            masking_penalty=masking_penalty,
        )

        residual_tokens = _box_residual_tokens(root, tokens, harmonic_tokens)

        box_score = _box_signature_score(
            residual_tokens=residual_tokens,
            source_micro_count=source_micro_count,
            target_micro_count=target_micro_count,
            field_micro_richness=_safe_int(fw.get("micro_token_richness"), 0),
        )

        scene_label = _scene_label(note_score, box_score, masking_warning)
        scene_counts[scene_label] += 1

        if note_score < args.min_note_candidate_score and box_score < 0.16:
            continue

        row_common = {
            "frame": frame,
            "source_entity": src,
            "target_entity": dst,
            "note_candidate": root,
            "scene_label": scene_label,
            "note_candidate_score": f"{note_score:.9f}",
            "box_signature_score": f"{box_score:.9f}",
            "episode_role": str(ep.get("episode_role", "")).strip(),
            "episode_confidence": f"{episode_confidence:.9f}",
            "ownership_stability": f"{ownership_stability:.9f}",
            "continuity_support": f"{continuity_support:.9f}",
            "harmonic_support_score": f"{harmonic_support:.9f}",
            "harmonic_5_7_score": f"{h57_score:.9f}",
            "present_harmonics": harmonic_info["present_harmonics"],
            "missing_harmonics": harmonic_info["missing_harmonics"],
            "harmonic_tokens": harmonic_info["harmonic_tokens"],
            "register_compensation": f"{register_compensation:.9f}",
            "compensation_basis": compensation_basis,
            "box_residual_count": len(residual_tokens),
            "box_residual_preview": _join_preview(residual_tokens, 72),
            "source_micro_count": source_micro_count,
            "target_micro_count": target_micro_count,
            "source_micro_preview": str(ep.get("source_micro_preview", "")).strip(),
            "target_micro_preview": str(ep.get("target_micro_preview", "")).strip(),
            "field_state": str(fw.get("field_state", "")).strip(),
            "micro_field_texture": str(fw.get("micro_field_texture", "")).strip(),
            "continuity_texture": str(fw.get("continuity_texture", "")).strip(),
            "field_micro_token_richness": _safe_int(fw.get("micro_token_richness"), 0),
            "field_mean_continuity_support": f"{_safe_float(fw.get('mean_continuity_support'), 0.0):.9f}",
            "field_mean_causal_confidence": f"{_safe_float(fw.get('mean_causal_confidence'), 0.0):.9f}",
            "masking_warning": masking_warning,
            "carrier_transition": str(ep.get("carrier_transition", "")).strip(),
            "confidence_reason": str(ep.get("confidence_reason", "")).strip(),
        }

        scene_rows.append(row_common)

        note_rows.append({
            "frame": frame,
            "note_candidate": root,
            "note_candidate_score": f"{note_score:.9f}",
            "harmonic_support_score": f"{harmonic_support:.9f}",
            "harmonic_5_7_score": f"{h57_score:.9f}",
            "ownership_stability": f"{ownership_stability:.9f}",
            "register_compensation": f"{register_compensation:.9f}",
            "compensation_basis": compensation_basis,
            "present_harmonics": harmonic_info["present_harmonics"],
            "missing_harmonics": harmonic_info["missing_harmonics"],
            "source_entity": src,
            "target_entity": dst,
            "scene_label": scene_label,
        })

        box_rows.append({
            "frame": frame,
            "note_candidate": root,
            "box_signature_score": f"{box_score:.9f}",
            "box_residual_count": len(residual_tokens),
            "box_residual_preview": _join_preview(residual_tokens, 120),
            "source_entity": src,
            "target_entity": dst,
            "scene_label": scene_label,
            "field_state": str(fw.get("field_state", "")).strip(),
            "micro_field_texture": str(fw.get("micro_field_texture", "")).strip(),
        })

        readable_rows.append({
            "frame": frame,
            "scene": f"{root} / {scene_label}",
            "note_score": f"{note_score:.3f}",
            "box_score": f"{box_score:.3f}",
            "h57": f"{h57_score:.3f}",
            "harmonics": harmonic_info["present_harmonics"],
            "compensation": compensation_basis,
            "box_residuals": _join_preview(residual_tokens, 18),
        })

    scene_rows.sort(key=lambda r: (_safe_int(r.get("frame"), 0), -_safe_float(r.get("note_candidate_score"), 0.0), -_safe_float(r.get("box_signature_score"), 0.0)))
    note_rows.sort(key=lambda r: (_safe_int(r.get("frame"), 0), -_safe_float(r.get("note_candidate_score"), 0.0)))
    box_rows.sort(key=lambda r: (_safe_int(r.get("frame"), 0), -_safe_float(r.get("box_signature_score"), 0.0)))

    out_scene = Path(args.out_scene_csv)
    out_note = Path(args.out_note_candidates_csv)
    out_box = Path(args.out_box_residuals_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_scene.parent.mkdir(parents=True, exist_ok=True)

    scene_fields = [
        "frame", "source_entity", "target_entity", "note_candidate", "scene_label",
        "note_candidate_score", "box_signature_score", "episode_role", "episode_confidence",
        "ownership_stability", "continuity_support", "harmonic_support_score", "harmonic_5_7_score",
        "present_harmonics", "missing_harmonics", "harmonic_tokens", "register_compensation",
        "compensation_basis", "box_residual_count", "box_residual_preview", "source_micro_count",
        "target_micro_count", "source_micro_preview", "target_micro_preview", "field_state",
        "micro_field_texture", "continuity_texture", "field_micro_token_richness",
        "field_mean_continuity_support", "field_mean_causal_confidence", "masking_warning",
        "carrier_transition", "confidence_reason",
    ]

    with out_scene.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scene_fields)
        w.writeheader()
        w.writerows(scene_rows)

    note_fields = [
        "frame", "note_candidate", "note_candidate_score", "harmonic_support_score",
        "harmonic_5_7_score", "ownership_stability", "register_compensation",
        "compensation_basis", "present_harmonics", "missing_harmonics", "source_entity",
        "target_entity", "scene_label",
    ]

    with out_note.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=note_fields)
        w.writeheader()
        w.writerows(note_rows)

    box_fields = [
        "frame", "note_candidate", "box_signature_score", "box_residual_count",
        "box_residual_preview", "source_entity", "target_entity", "scene_label",
        "field_state", "micro_field_texture",
    ]

    with out_box.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=box_fields)
        w.writeheader()
        w.writerows(box_rows)

    readable_fields = [
        "frame", "scene", "note_score", "box_score", "h57", "harmonics",
        "compensation", "box_residuals",
    ]

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=readable_fields)
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "composition_note_box_scene_builder",
        "semantic_version": "note_box_scene_builder_v1",
        "inputs": {
            "ownership_episodes_csv": args.ownership_episodes_csv,
            "entity_timeline_csv": args.entity_timeline_csv,
            "field_windows_csv": args.field_windows_csv,
        },
        "outputs": {
            "scene_csv": args.out_scene_csv,
            "note_candidates_csv": args.out_note_candidates_csv,
            "box_residuals_csv": args.out_box_residuals_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_note_candidate_score": args.min_note_candidate_score,
            "window_frames": args.window_frames,
            "step_frames": args.step_frames,
        },
        "result": {
            "episodes_in": len(episodes),
            "scene_rows": len(scene_rows),
            "note_candidate_rows": len(note_rows),
            "box_residual_rows": len(box_rows),
            "scene_counts": dict(scene_counts),
            "compensation_counts": dict(compensation_counts),
        },
        "ontology_note": (
            "This layer does not use instrument passports. It first builds an internal "
            "composition-level scene: note candidate + box residual signature + harmonic "
            "support. The 5th and 7th harmonics are weighted strongly because they help "
            "compensate low/high register instability and are important in polyphony."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "COMPOSITION NOTE + BOX SCENE BUILDER",
        "=" * 72,
        f"ownership_episodes_csv : {args.ownership_episodes_csv}",
        f"entity_timeline_csv     : {args.entity_timeline_csv}",
        f"field_windows_csv       : {args.field_windows_csv}",
        "",
        f"episodes_in             : {len(episodes)}",
        f"scene_rows              : {len(scene_rows)}",
        f"note_candidate_rows     : {len(note_rows)}",
        f"box_residual_rows       : {len(box_rows)}",
        "",
        "Scene counts:",
    ]

    for k in sorted(scene_counts):
        txt.append(f"  {k}: {scene_counts[k]}")

    txt.append("")
    txt.append("Register compensation counts:")
    for k in sorted(compensation_counts):
        txt.append(f"  {k}: {compensation_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  First build the internal scene without passport matching.",
        "  A note is treated as an exciter candidate.",
        "  A box signature is treated as residual resonance not explained by",
        "  root/coarse-root/harmonic-chain support.",
        "  Low/high register compensation uses harmonic evidence, with emphasis",
        "  on the 5th and 7th harmonics for polyphonic stability.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("composition note box scene builder complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
