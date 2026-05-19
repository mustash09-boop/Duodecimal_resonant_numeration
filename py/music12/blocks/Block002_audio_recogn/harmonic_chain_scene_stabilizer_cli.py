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


def _join_preview(tokens: Iterable[str], limit: int = 72) -> str:
    return " ".join(sorted(set(tokens))[:limit])


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


# ============================================================
# 12-token helpers
# ============================================================

_DEGREE_TO_INDEX = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
    "7": 6, "8": 7, "9": 8, "A": 9, "B": 10, "C": 11,
}

_TOKEN_RE = re.compile(r"^\s*([1-9A-C]+)\.([1-9A-C])(?:'(.*))?\s*$")


def _octave_label_value(octave_s: str) -> Optional[int]:
    value = 0
    for ch in str(octave_s or "").strip():
        if ch not in _DEGREE_TO_INDEX:
            return None
        value = value * 12 + (_DEGREE_TO_INDEX[ch] + 1)
    return value


def _parse_token(token: str) -> Optional[Tuple[int, int, str, str]]:
    m = _TOKEN_RE.match(str(token or "").strip())
    if not m:
        return None

    octave_s, degree_s, micro = m.groups()
    if degree_s not in _DEGREE_TO_INDEX:
        return None

    octave_value = _octave_label_value(octave_s)
    if octave_value is None:
        return None

    return octave_value, _DEGREE_TO_INDEX[degree_s], micro or "-", octave_s


def _pitch_index12(token: str) -> Optional[int]:
    p = _parse_token(token)
    if not p:
        return None
    octave_value, degree, _micro, _octave_raw = p
    return octave_value * 12 + degree


def _degree_index(token: str) -> Optional[int]:
    p = _parse_token(token)
    if not p:
        return None
    return p[1]


def _register_class(root_token: str) -> str:
    p = _parse_token(root_token)
    if not p:
        return "UNKNOWN_REGISTER"

    _oct_value, _degree, _micro, octave_raw = p
    first = octave_raw[0] if octave_raw else ""

    if first in {"5", "6", "7"}:
        return "LOW_REGISTER"
    if first in {"8", "9", "A"}:
        return "MID_REGISTER"
    if first in {"B", "C"} or len(octave_raw) > 1:
        return "HIGH_REGISTER"
    return "UNKNOWN_REGISTER"


_HARMONIC_OFFSETS_12 = {
    2: 12,
    3: 19,
    4: 24,
    5: 28,
    6: 31,
    7: 34,
    8: 36,
    9: 38,
    10: 40,
    11: 42,
    12: 43,
}

_HARMONIC_WEIGHTS = {
    2: 0.07,
    3: 0.12,
    4: 0.06,
    5: 0.26,
    6: 0.07,
    7: 0.24,
    8: 0.05,
    9: 0.04,
    10: 0.04,
    11: 0.03,
    12: 0.02,
}


def _expected_harmonics(root_token: str) -> Dict[int, int]:
    root_idx = _pitch_index12(root_token)
    if root_idx is None:
        return {}
    return {h: root_idx + off for h, off in _HARMONIC_OFFSETS_12.items()}


def _bucket_tokens(tokens: Set[str]) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    by_pitch: Dict[int, List[str]] = defaultdict(list)
    by_degree: Dict[int, List[str]] = defaultdict(list)

    for t in tokens:
        pi = _pitch_index12(t)
        di = _degree_index(t)
        if pi is not None:
            by_pitch[pi].append(t)
        if di is not None:
            by_degree[di].append(t)

    return by_pitch, by_degree


def _match_harmonic(
    *,
    expected_pitch: int,
    note_by_pitch: Dict[int, List[str]],
    note_by_degree: Dict[int, List[str]],
    field_by_pitch: Dict[int, List[str]],
    field_by_degree: Dict[int, List[str]],
) -> Dict[str, Any]:
    expected_degree = expected_pitch % 12

    tokens: List[str] = []
    support = 0.0
    basis = "missing"

    for delta, s in ((0, 1.00), (-1, 0.84), (1, 0.84), (-2, 0.62), (2, 0.62)):
        hits = note_by_pitch.get(expected_pitch + delta, [])
        if hits:
            tokens.extend(hits[:12])
            support = max(support, s)
            basis = "note_pitch_neighborhood"

    if support <= 0.0:
        hits = note_by_degree.get(expected_degree, [])
        if hits:
            tokens.extend(hits[:12])
            support = 0.72
            basis = "note_degree_class"

    if support <= 0.0:
        for delta, s in ((0, 0.56), (-1, 0.44), (1, 0.44), (-2, 0.32), (2, 0.32)):
            hits = field_by_pitch.get(expected_pitch + delta, [])
            if hits:
                tokens.extend(hits[:12])
                support = max(support, s)
                basis = "field_pitch_neighborhood"

    if support <= 0.0:
        hits = field_by_degree.get(expected_degree, [])
        if hits:
            tokens.extend(hits[:12])
            support = 0.34
            basis = "field_degree_class"

    return {
        "support": support,
        "basis": basis,
        "tokens": sorted(set(tokens))[:16],
    }


def _candidate_root(row: Dict[str, Any]) -> str:
    for k in (
        "source_root_hint_micro",
        "target_root_hint_micro",
        "root_hint_micro_not_identity",
        "root_hint_not_identity",
        "note_candidate",
    ):
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def _note_family_tokens(row: Dict[str, Any]) -> Set[str]:
    toks = set()
    toks.update(_split_tokens(row.get("source_micro_preview", "")))
    toks.update(_split_tokens(row.get("target_micro_preview", "")))
    toks.update(_split_tokens(row.get("note_family_preview", "")))
    return toks


def _field_pool_tokens(field_row: Dict[str, Any]) -> Set[str]:
    toks = set()
    toks.update(_split_tokens(field_row.get("micro_token_preview", "")))
    toks.update(_split_tokens(field_row.get("field_pool_preview", "")))
    return toks


def _harmonic_cloud(root_token: str, note_tokens: Set[str], field_tokens: Set[str]) -> Dict[str, Any]:
    expected = _expected_harmonics(root_token)

    if not expected:
        return {
            "root_parse_ok": False,
            "harmonic_gravity_score": 0.0,
            "harmonic_5_7_gravity": 0.0,
            "present_harmonics": "",
            "missing_harmonics": "",
            "harmonic_cloud_tokens": "",
            "harmonic_basis": "",
            "root_attraction_class": "ROOT_PARSE_FAILED",
            "note_owned_harmonic_mean": 0.0,
            "field_harmonic_mean": 0.0,
        }

    note_by_pitch, note_by_degree = _bucket_tokens(note_tokens)
    field_by_pitch, field_by_degree = _bucket_tokens(field_tokens)

    total_weight = sum(_HARMONIC_WEIGHTS.values())
    h57_total = _HARMONIC_WEIGHTS[5] + _HARMONIC_WEIGHTS[7]

    gravity = 0.0
    h57_gravity = 0.0
    present = []
    missing = []
    cloud_tokens: List[str] = []
    basis_parts = []
    note_owned_supports = []
    field_supports = []

    for h, expected_pitch in expected.items():
        match = _match_harmonic(
            expected_pitch=expected_pitch,
            note_by_pitch=note_by_pitch,
            note_by_degree=note_by_degree,
            field_by_pitch=field_by_pitch,
            field_by_degree=field_by_degree,
        )

        support = _safe_float(match.get("support"), 0.0)
        basis = str(match.get("basis", ""))

        if support > 0.0:
            present.append(str(h))
            cloud_tokens.extend(match.get("tokens", []))
            basis_parts.append(f"h{h}:{basis}:{support:.2f}")
            gravity += _HARMONIC_WEIGHTS.get(h, 0.0) * support

            if h in (5, 7):
                h57_gravity += _HARMONIC_WEIGHTS.get(h, 0.0) * support

            if basis.startswith("note_"):
                note_owned_supports.append(support)
            elif basis.startswith("field_"):
                field_supports.append(support)
        else:
            missing.append(str(h))

    harmonic_gravity_score = gravity / max(total_weight, 1e-9)
    harmonic_5_7_gravity = h57_gravity / max(h57_total, 1e-9)
    note_owned_mean = _mean(note_owned_supports)
    field_mean = _mean(field_supports)

    if harmonic_gravity_score >= 0.62 and harmonic_5_7_gravity >= 0.45:
        attraction = "STRONG_ROOT_HARMONIC_GRAVITY"
    elif harmonic_gravity_score >= 0.42 or harmonic_5_7_gravity >= 0.50:
        attraction = "ROOT_HARMONIC_GRAVITY"
    elif field_mean > note_owned_mean and field_mean >= 0.25:
        attraction = "FIELD_SUPPORTED_HARMONIC_CLOUD"
    elif harmonic_gravity_score > 0.16:
        attraction = "WEAK_HARMONIC_CLOUD"
    else:
        attraction = "NO_STABLE_HARMONIC_CLOUD"

    return {
        "root_parse_ok": True,
        "harmonic_gravity_score": harmonic_gravity_score,
        "harmonic_5_7_gravity": harmonic_5_7_gravity,
        "present_harmonics": " ".join(present),
        "missing_harmonics": " ".join(missing),
        "harmonic_cloud_tokens": _join_preview(cloud_tokens, 96),
        "harmonic_basis": " | ".join(basis_parts),
        "note_owned_harmonic_mean": note_owned_mean,
        "field_harmonic_mean": field_mean,
        "root_attraction_class": attraction,
    }


def _register_compensation(root_token: str, cloud: Dict[str, Any]) -> Tuple[float, str]:
    reg = _register_class(root_token)
    g = _safe_float(cloud.get("harmonic_gravity_score"), 0.0)
    h57 = _safe_float(cloud.get("harmonic_5_7_gravity"), 0.0)

    if reg == "LOW_REGISTER":
        return 0.24 * g + 0.22 * h57, "low_register_harmonic_cloud_compensation"

    if reg == "HIGH_REGISTER":
        return 0.14 * g + 0.30 * h57, "high_register_harmonic_cloud_compensation"

    if reg == "MID_REGISTER":
        return 0.16 * g + 0.20 * h57, "mid_register_harmonic_cloud_compensation"

    return 0.08 * g + 0.08 * h57, "unknown_register_harmonic_cloud_compensation"


def _root_scene_score(
    *,
    episode_confidence: float,
    ownership_stability: float,
    continuity_support: float,
    cloud: Dict[str, Any],
    register_compensation: float,
    masking_warning: str,
) -> float:
    g = _safe_float(cloud.get("harmonic_gravity_score"), 0.0)
    h57 = _safe_float(cloud.get("harmonic_5_7_gravity"), 0.0)
    note_owned = _safe_float(cloud.get("note_owned_harmonic_mean"), 0.0)
    penalty = 0.10 if masking_warning else 0.0

    return _clamp(
        episode_confidence * 0.17
        + ownership_stability * 0.16
        + continuity_support * 0.15
        + g * 0.22
        + h57 * 0.20
        + note_owned * 0.06
        + register_compensation
        - penalty
    )


def _stabilized_label(root_score: float, cloud: Dict[str, Any], masking_warning: str) -> str:
    if masking_warning and root_score < 0.56:
        return "ROOT_WITH_MASKING_RISK"

    attraction = str(cloud.get("root_attraction_class", ""))

    if root_score >= 0.66 and attraction == "STRONG_ROOT_HARMONIC_GRAVITY":
        return "STABLE_NOTE_ROOT"

    if root_score >= 0.52 and attraction in {
        "STRONG_ROOT_HARMONIC_GRAVITY",
        "ROOT_HARMONIC_GRAVITY",
        "FIELD_SUPPORTED_HARMONIC_CLOUD",
    }:
        return "SUPPORTED_NOTE_ROOT"

    if root_score >= 0.36 and attraction != "NO_STABLE_HARMONIC_CLOUD":
        return "WEAK_SUPPORTED_ROOT"

    return "UNSTABLE_ROOT_CANDIDATE"


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stabilize root-centered harmonic clouds before note+box scene building. "
            "This module does not use instrument passports. It strengthens root decisions "
            "by harmonic gravity, especially 5th and 7th harmonics."
        )
    )

    ap.add_argument("--ownership_episodes_csv", required=True)
    ap.add_argument("--entity_timeline_csv", required=True)
    ap.add_argument("--field_windows_csv", required=True)

    ap.add_argument("--out_harmonic_clouds_csv", required=True)
    ap.add_argument("--out_stabilized_roots_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_root_scene_score", type=float, default=0.16)

    args = ap.parse_args()

    episodes = _load_csv(Path(args.ownership_episodes_csv))
    timelines = _load_csv(Path(args.entity_timeline_csv))
    fields = _load_csv(Path(args.field_windows_csv))

    timeline_by_entity = {
        str(r.get("entity_id", "")).strip(): r
        for r in timelines
    }

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

    cloud_rows = []
    root_rows = []
    readable_rows = []

    label_counts = defaultdict(int)
    attraction_counts = defaultdict(int)
    compensation_counts = defaultdict(int)

    for ep in episodes:
        frame = _safe_int(ep.get("source_birth_frame"), 0)
        src = str(ep.get("source_entity", "")).strip()
        dst = str(ep.get("target_entity", "")).strip()

        root = _candidate_root(ep)
        if not root:
            continue

        fw = _field_for_frame(frame)
        note_tokens = _note_family_tokens(ep)
        field_tokens = _field_pool_tokens(fw)
        cloud = _harmonic_cloud(root, note_tokens, field_tokens)
        register_compensation, compensation_basis = _register_compensation(root, cloud)

        source_timeline = timeline_by_entity.get(src, {})
        target_timeline = timeline_by_entity.get(dst, {})

        ownership_stability = max(
            _safe_float(source_timeline.get("ownership_stability"), 0.0),
            _safe_float(target_timeline.get("ownership_stability"), 0.0),
        )

        episode_confidence = _safe_float(ep.get("episode_confidence"), 0.0)
        continuity_support = _safe_float(ep.get("continuity_support"), 0.0)
        masking_warning = str(ep.get("masking_warning", "")).strip()

        root_score = _root_scene_score(
            episode_confidence=episode_confidence,
            ownership_stability=ownership_stability,
            continuity_support=continuity_support,
            cloud=cloud,
            register_compensation=register_compensation,
            masking_warning=masking_warning,
        )

        if root_score < args.min_root_scene_score:
            continue

        label = _stabilized_label(root_score, cloud, masking_warning)

        label_counts[label] += 1
        attraction_counts[str(cloud.get("root_attraction_class", ""))] += 1
        compensation_counts[compensation_basis] += 1

        common = {
            "frame": frame,
            "source_entity": src,
            "target_entity": dst,
            "root_candidate": root,
            "stabilized_root_label": label,
            "root_scene_score": f"{root_score:.9f}",
            "root_attraction_class": str(cloud.get("root_attraction_class", "")),
            "harmonic_gravity_score": f"{_safe_float(cloud.get('harmonic_gravity_score'), 0.0):.9f}",
            "harmonic_5_7_gravity": f"{_safe_float(cloud.get('harmonic_5_7_gravity'), 0.0):.9f}",
            "present_harmonics": str(cloud.get("present_harmonics", "")),
            "missing_harmonics": str(cloud.get("missing_harmonics", "")),
            "harmonic_cloud_tokens": str(cloud.get("harmonic_cloud_tokens", "")),
            "harmonic_basis": str(cloud.get("harmonic_basis", "")),
            "note_owned_harmonic_mean": f"{_safe_float(cloud.get('note_owned_harmonic_mean'), 0.0):.9f}",
            "field_harmonic_mean": f"{_safe_float(cloud.get('field_harmonic_mean'), 0.0):.9f}",
            "register_class": _register_class(root),
            "register_compensation": f"{register_compensation:.9f}",
            "compensation_basis": compensation_basis,
            "episode_role": str(ep.get("episode_role", "")).strip(),
            "episode_confidence": f"{episode_confidence:.9f}",
            "ownership_stability": f"{ownership_stability:.9f}",
            "continuity_support": f"{continuity_support:.9f}",
            "source_micro_count": _safe_int(ep.get("source_micro_count"), 0),
            "target_micro_count": _safe_int(ep.get("target_micro_count"), 0),
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

        cloud_rows.append(common)
        root_rows.append({
            "frame": frame,
            "root_candidate": root,
            "stabilized_root_label": label,
            "root_scene_score": f"{root_score:.9f}",
            "root_attraction_class": str(cloud.get("root_attraction_class", "")),
            "harmonic_gravity_score": f"{_safe_float(cloud.get('harmonic_gravity_score'), 0.0):.9f}",
            "harmonic_5_7_gravity": f"{_safe_float(cloud.get('harmonic_5_7_gravity'), 0.0):.9f}",
            "present_harmonics": str(cloud.get("present_harmonics", "")),
            "missing_harmonics": str(cloud.get("missing_harmonics", "")),
            "register_class": _register_class(root),
            "register_compensation": f"{register_compensation:.9f}",
            "source_entity": src,
            "target_entity": dst,
        })

        readable_rows.append({
            "frame": frame,
            "root": root,
            "label": label,
            "score": f"{root_score:.3f}",
            "gravity": f"{_safe_float(cloud.get('harmonic_gravity_score'), 0.0):.3f}",
            "h57": f"{_safe_float(cloud.get('harmonic_5_7_gravity'), 0.0):.3f}",
            "harmonics": str(cloud.get("present_harmonics", "")),
            "register": _register_class(root),
            "basis": str(cloud.get("root_attraction_class", "")),
        })

    cloud_rows.sort(key=lambda r: (_safe_int(r.get("frame"), 0), -_safe_float(r.get("root_scene_score"), 0.0)))
    root_rows.sort(key=lambda r: (_safe_int(r.get("frame"), 0), -_safe_float(r.get("root_scene_score"), 0.0)))

    out_clouds = Path(args.out_harmonic_clouds_csv)
    out_roots = Path(args.out_stabilized_roots_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)
    out_clouds.parent.mkdir(parents=True, exist_ok=True)

    cloud_fields = [
        "frame", "source_entity", "target_entity", "root_candidate",
        "stabilized_root_label", "root_scene_score", "root_attraction_class",
        "harmonic_gravity_score", "harmonic_5_7_gravity", "present_harmonics",
        "missing_harmonics", "harmonic_cloud_tokens", "harmonic_basis",
        "note_owned_harmonic_mean", "field_harmonic_mean", "register_class",
        "register_compensation", "compensation_basis", "episode_role",
        "episode_confidence", "ownership_stability", "continuity_support",
        "source_micro_count", "target_micro_count", "source_micro_preview",
        "target_micro_preview", "field_state", "micro_field_texture",
        "continuity_texture", "field_micro_token_richness",
        "field_mean_continuity_support", "field_mean_causal_confidence",
        "masking_warning", "carrier_transition", "confidence_reason",
    ]

    with out_clouds.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cloud_fields)
        w.writeheader()
        w.writerows(cloud_rows)

    root_fields = [
        "frame", "root_candidate", "stabilized_root_label", "root_scene_score",
        "root_attraction_class", "harmonic_gravity_score", "harmonic_5_7_gravity",
        "present_harmonics", "missing_harmonics", "register_class",
        "register_compensation", "source_entity", "target_entity",
    ]

    with out_roots.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=root_fields)
        w.writeheader()
        w.writerows(root_rows)

    readable_fields = ["frame", "root", "label", "score", "gravity", "h57", "harmonics", "register", "basis"]
    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=readable_fields)
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "harmonic_chain_scene_stabilizer",
        "semantic_version": "harmonic_cloud_stabilizer_v1",
        "inputs": {
            "ownership_episodes_csv": args.ownership_episodes_csv,
            "entity_timeline_csv": args.entity_timeline_csv,
            "field_windows_csv": args.field_windows_csv,
        },
        "outputs": {
            "harmonic_clouds_csv": args.out_harmonic_clouds_csv,
            "stabilized_roots_csv": args.out_stabilized_roots_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {"min_root_scene_score": args.min_root_scene_score},
        "result": {
            "episodes_in": len(episodes),
            "harmonic_cloud_rows": len(cloud_rows),
            "stabilized_root_rows": len(root_rows),
            "label_counts": dict(label_counts),
            "attraction_counts": dict(attraction_counts),
            "compensation_counts": dict(compensation_counts),
        },
        "ontology_note": (
            "This module stabilizes root-centered harmonic clouds before note+box scene building. "
            "It does not use instrument passports. It emphasizes 5th and 7th harmonics and separates "
            "note-owned harmonic support from field-supported clouds."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "HARMONIC CHAIN SCENE STABILIZER",
        "=" * 72,
        f"ownership_episodes_csv : {args.ownership_episodes_csv}",
        f"entity_timeline_csv     : {args.entity_timeline_csv}",
        f"field_windows_csv       : {args.field_windows_csv}",
        "",
        f"episodes_in             : {len(episodes)}",
        f"harmonic_cloud_rows     : {len(cloud_rows)}",
        f"stabilized_root_rows    : {len(root_rows)}",
        "",
        "Stabilized root label counts:",
    ]
    for k in sorted(label_counts):
        txt.append(f"  {k}: {label_counts[k]}")

    txt.append("")
    txt.append("Root attraction counts:")
    for k in sorted(attraction_counts):
        txt.append(f"  {k}: {attraction_counts[k]}")

    txt.append("")
    txt.append("Register compensation counts:")
    for k in sorted(compensation_counts):
        txt.append(f"  {k}: {compensation_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  A root candidate should not become a note only because it exists.",
        "  It must form a harmonic cloud around itself.",
        "  The 5th and 7th harmonics are weighted strongly because they help",
        "  stabilize polyphonic scenes and compensate low/high register weakness.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("harmonic chain scene stabilizer complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
