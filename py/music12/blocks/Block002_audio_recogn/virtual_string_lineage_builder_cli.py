# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s.replace(",", "."))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s.replace(",", ".")))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _split_tokens(raw: Any) -> Set[str]:
    return {
        x.strip()
        for x in str(raw or "").replace(",", " ").replace("|", " ").split()
        if x.strip()
    }


def _join(tokens: Iterable[str], limit: int = 96) -> str:
    return " ".join(sorted(set(tokens))[:limit])


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


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


def _coarse_token(token: str) -> str:
    s = str(token or "").strip()
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if "." in s:
        return s + "'-"
    return s


def _normalize_note(token: Any) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


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
    2: 12, 3: 19, 4: 24, 5: 28, 6: 31, 7: 34,
    8: 36, 9: 38, 10: 40, 11: 42, 12: 43,
}

_HARMONIC_WEIGHTS = {
    2: 0.07, 3: 0.12, 4: 0.06, 5: 0.28, 6: 0.07, 7: 0.26,
    8: 0.05, 9: 0.03, 10: 0.03, 11: 0.02, 12: 0.01,
}


def _expected_harmonic_pitch(root_token: str) -> Dict[int, int]:
    root_idx = _pitch_index12(root_token)
    if root_idx is None:
        return {}
    return {h: root_idx + off for h, off in _HARMONIC_OFFSETS_12.items()}


def _root_from_row(row: Dict[str, Any]) -> str:
    for k in (
        "root_candidate",
        "note_candidate",
        "attractor_note_micro",
        "attractor_note",
        "source_root_hint_micro",
        "target_root_hint_micro",
    ):
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def _frame_from_row(row: Dict[str, Any]) -> int:
    for k in ("frame", "birth_frame", "source_birth_frame", "frame_index"):
        if str(row.get(k, "")).strip():
            return _safe_int(row.get(k), 0)
    return 0


def _tokens_from_row(row: Dict[str, Any]) -> Set[str]:
    toks = set()
    for k in (
        "source_micro_preview",
        "target_micro_preview",
        "harmonic_cloud_tokens",
        "note_family_preview",
        "field_pool_preview",
        "box_residual_preview",
        "excitation_core_tokens",
        "instrument_body_tokens",
        "secondary_field_tokens",
    ):
        toks.update(_split_tokens(row.get(k, "")))
    return toks


def _row_strength(row: Dict[str, Any]) -> float:
    vals = [
        _safe_float(row.get("root_scene_score"), 0.0),
        _safe_float(row.get("note_candidate_score"), 0.0),
        _safe_float(row.get("episode_confidence"), 0.0),
        _safe_float(row.get("attractor_confidence"), 0.0),
        _safe_float(row.get("harmonic_gravity_score"), 0.0),
        _safe_float(row.get("harmonic_5_7_gravity"), 0.0),
        _safe_float(row.get("ownership_stability"), 0.0),
        _safe_float(row.get("continuity_support"), 0.0),
    ]
    return max(vals)


def _source_target(row: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(row.get("source_entity", "")).strip(),
        str(row.get("target_entity", "")).strip(),
    )


def _token_match_support(expected_pitch: int, token: str) -> float:
    pi = _pitch_index12(token)
    if pi is not None:
        delta = abs(pi - expected_pitch)
        if delta == 0:
            return 1.00
        if delta == 1:
            return 0.82
        if delta == 2:
            return 0.58

    di = _degree_index(token)
    if di is not None and di == (expected_pitch % 12):
        return 0.42

    return 0.0


def _find_offspring_for_parent(
    *,
    parent_root: str,
    parent_frame: int,
    search_rows: List[Dict[str, Any]],
    max_delay_frames: int,
) -> Dict[int, Dict[str, Any]]:
    expected = _expected_harmonic_pitch(parent_root)
    if not expected:
        return {}

    best: Dict[int, Dict[str, Any]] = {}

    for row in search_rows:
        child_frame = _frame_from_row(row)
        delay = child_frame - parent_frame
        if delay < 0 or delay > max_delay_frames:
            continue

        toks = _tokens_from_row(row)
        if not toks:
            continue

        row_strength = _row_strength(row)
        src, dst = _source_target(row)

        for h, expected_pitch in expected.items():
            best_token = ""
            best_support = 0.0

            for t in toks:
                s = _token_match_support(expected_pitch, t)
                if s > best_support:
                    best_support = s
                    best_token = t

            if best_support <= 0.0:
                continue

            delay_support = 1.0 - min(delay / max(max_delay_frames, 1), 1.0) * 0.42

            score = (
                best_support * 0.52
                + delay_support * 0.20
                + row_strength * 0.20
                + _HARMONIC_WEIGHTS.get(h, 0.0) * 0.08
            )

            if h not in best or score > best[h]["offspring_score"]:
                best[h] = {
                    "h": h,
                    "offspring_token": best_token,
                    "offspring_frame": child_frame,
                    "offspring_delay": delay,
                    "offspring_score": score,
                    "offspring_support": best_support,
                    "offspring_row_strength": row_strength,
                    "offspring_source_entity": src,
                    "offspring_target_entity": dst,
                }

    return best


def _lineage_strength(
    *,
    parent_strength: float,
    offspring: Dict[int, Dict[str, Any]],
    parent_tokens: Set[str],
) -> Dict[str, Any]:
    total_w = sum(_HARMONIC_WEIGHTS.values())
    got = 0.0
    present = []
    missing = []
    offspring_tokens = set()

    for h, w in _HARMONIC_WEIGHTS.items():
        if h in offspring:
            present.append(str(h))
            got += w * _safe_float(offspring[h].get("offspring_support"), 0.0)
            tok = str(offspring[h].get("offspring_token", "")).strip()
            if tok:
                offspring_tokens.add(tok)
        else:
            missing.append(str(h))

    harmonic_parenthood = got / max(total_w, 1e-9)

    h57_w = _HARMONIC_WEIGHTS[5] + _HARMONIC_WEIGHTS[7]
    h57_got = 0.0
    if 5 in offspring:
        h57_got += _HARMONIC_WEIGHTS[5] * _safe_float(offspring[5].get("offspring_support"), 0.0)
    if 7 in offspring:
        h57_got += _HARMONIC_WEIGHTS[7] * _safe_float(offspring[7].get("offspring_support"), 0.0)

    h57_parenthood = h57_got / max(h57_w, 1e-9)
    odd_present = sum(1 for h in (3, 5, 7, 9, 11) if h in offspring)
    delayed = [h for h, r in offspring.items() if _safe_int(r.get("offspring_delay"), 0) > 0]
    token_continuity = _jaccard(parent_tokens, offspring_tokens)

    score = _clamp(
        parent_strength * 0.16
        + harmonic_parenthood * 0.34
        + h57_parenthood * 0.28
        + min(odd_present / 5.0, 1.0) * 0.10
        + min(len(delayed) / 4.0, 1.0) * 0.06
        + token_continuity * 0.06
    )

    if score >= 0.62 and h57_parenthood >= 0.45:
        label = "STRONG_VIRTUAL_STRING_LINEAGE"
    elif score >= 0.46:
        label = "SUPPORTED_VIRTUAL_STRING_LINEAGE"
    elif score >= 0.30:
        label = "WEAK_VIRTUAL_STRING_LINEAGE"
    else:
        label = "UNSTABLE_VIRTUAL_STRING_LINEAGE"

    return {
        "lineage_strength": score,
        "harmonic_parenthood_score": harmonic_parenthood,
        "harmonic_5_7_parenthood_score": h57_parenthood,
        "present_harmonics": " ".join(present),
        "missing_harmonics": " ".join(missing),
        "offspring_tokens": _join(offspring_tokens, 96),
        "offspring_count": len(offspring),
        "odd_harmonic_count": odd_present,
        "delayed_offspring_count": len(delayed),
        "token_continuity": token_continuity,
        "lineage_label": label,
    }


def _residual_box_tokens(parent_tokens: Set[str], offspring_tokens: Set[str], all_tokens: Set[str], root: str) -> Set[str]:
    root_coarse = _coarse_token(root)
    explained = set(parent_tokens) | set(offspring_tokens)
    residual = set()

    for t in all_tokens:
        if not t:
            continue
        if t in explained:
            continue
        if _coarse_token(t) == root_coarse:
            continue
        residual.add(t)

    return residual


def _parent_id(prefix: str, idx: int, root: str, frame: int) -> str:
    safe_root = (
        str(root or "")
        .replace("'", "")
        .replace(".", "_")
        .replace("-", "m")
        .replace("+", "p")
    )
    return f"{prefix}_{idx:06d}_{safe_root}_{frame}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build virtual-string parent→offspring harmonic lineages. "
            "A note is treated as a time-distributed chain of harmonic descendants "
            "born from a virtual string excitation, not as an instantaneous root label."
        )
    )

    ap.add_argument("--source_events_csv", required=True)
    ap.add_argument("--out_lineages_csv", required=True)
    ap.add_argument("--out_offspring_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--identity_prefix", default="VSTR")
    ap.add_argument("--max_delay_frames", type=int, default=180)
    ap.add_argument("--min_parent_strength", type=float, default=0.08)
    ap.add_argument("--min_lineage_strength", type=float, default=0.16)

    args = ap.parse_args()

    rows = _load_csv(Path(args.source_events_csv))
    rows.sort(key=lambda r: (_frame_from_row(r), _root_from_row(r)))

    lineages = []
    offspring_rows = []
    readable = []

    label_counts = defaultdict(int)
    register_counts = defaultdict(int)

    idx = 0

    for parent in rows:
        parent_frame = _frame_from_row(parent)
        parent_root = _root_from_row(parent)
        parent_strength = _row_strength(parent)

        if not parent_root or parent_strength < args.min_parent_strength:
            continue

        parent_tokens = _tokens_from_row(parent)

        future_rows = [
            r for r in rows
            if parent_frame <= _frame_from_row(r) <= parent_frame + args.max_delay_frames
        ]

        offspring = _find_offspring_for_parent(
            parent_root=parent_root,
            parent_frame=parent_frame,
            search_rows=future_rows,
            max_delay_frames=args.max_delay_frames,
        )

        metrics = _lineage_strength(
            parent_strength=parent_strength,
            offspring=offspring,
            parent_tokens=parent_tokens,
        )

        if _safe_float(metrics["lineage_strength"], 0.0) < args.min_lineage_strength:
            continue

        idx += 1
        lineage_id = _parent_id(args.identity_prefix, idx, parent_root, parent_frame)

        all_tokens = set(parent_tokens)
        offspring_tokens = set()
        for _h, off in offspring.items():
            tok = str(off.get("offspring_token", "")).strip()
            if tok:
                offspring_tokens.add(tok)
                all_tokens.add(tok)

        for r in future_rows:
            all_tokens.update(_tokens_from_row(r))

        residual_box = _residual_box_tokens(
            parent_tokens=parent_tokens,
            offspring_tokens=offspring_tokens,
            all_tokens=all_tokens,
            root=parent_root,
        )

        src, dst = _source_target(parent)

        register = _register_class(parent_root)
        register_counts[register] += 1
        label_counts[metrics["lineage_label"]] += 1

        lineages.append({
            "lineage_id": lineage_id,
            "parent_exciter_id": str(parent.get("identity_id", "")).strip(),
            "root_candidate_micro": parent_root,
            "root_candidate": _normalize_note(parent_root),

            "birth_frame": parent_frame,
            "end_frame": parent_frame + args.max_delay_frames,
            "duration_frames": args.max_delay_frames + 1,

            "source_entity": src,
            "target_entity": dst,

            "parent_strength": f"{parent_strength:.9f}",
            "lineage_strength": f"{metrics['lineage_strength']:.9f}",
            "lineage_label": metrics["lineage_label"],

            "harmonic_parenthood_score": f"{metrics['harmonic_parenthood_score']:.9f}",
            "harmonic_5_7_parenthood_score": f"{metrics['harmonic_5_7_parenthood_score']:.9f}",

            "present_harmonics": metrics["present_harmonics"],
            "missing_harmonics": metrics["missing_harmonics"],

            "offspring_count": metrics["offspring_count"],
            "odd_harmonic_count": metrics["odd_harmonic_count"],
            "delayed_offspring_count": metrics["delayed_offspring_count"],
            "token_continuity": f"{metrics['token_continuity']:.9f}",

            "parent_tokens": _join(parent_tokens, 120),
            "offspring_tokens": metrics["offspring_tokens"],

            "residual_box_count": len(residual_box),
            "residual_box_tokens": _join(residual_box, 160),

            "register_class": register,

            "note_identity_candidate": _normalize_note(parent_root),
            "box_residual_signature": _join(residual_box, 160),
        })

        readable.append({
            "lineage_id": lineage_id,
            "frame": parent_frame,
            "root": _normalize_note(parent_root),
            "label": metrics["lineage_label"],
            "strength": f"{metrics['lineage_strength']:.3f}",
            "h57": f"{metrics['harmonic_5_7_parenthood_score']:.3f}",
            "present": metrics["present_harmonics"],
            "offspring": metrics["offspring_tokens"],
            "residual_box_count": len(residual_box),
        })

        for h, off in sorted(offspring.items()):
            offspring_rows.append({
                "lineage_id": lineage_id,
                "root_candidate": _normalize_note(parent_root),
                "harmonic_number": h,
                "offspring_token": off.get("offspring_token", ""),
                "offspring_frame": off.get("offspring_frame", ""),
                "offspring_delay": off.get("offspring_delay", ""),
                "offspring_score": f"{_safe_float(off.get('offspring_score'), 0.0):.9f}",
                "offspring_support": f"{_safe_float(off.get('offspring_support'), 0.0):.9f}",
                "offspring_row_strength": f"{_safe_float(off.get('offspring_row_strength'), 0.0):.9f}",
                "offspring_source_entity": off.get("offspring_source_entity", ""),
                "offspring_target_entity": off.get("offspring_target_entity", ""),
            })

    lineages.sort(
        key=lambda r: (
            _safe_int(r.get("birth_frame"), 0),
            -_safe_float(r.get("lineage_strength"), 0.0),
        )
    )

    out_lineages = Path(args.out_lineages_csv)
    out_offspring = Path(args.out_offspring_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    lineage_fields = [
        "lineage_id",
        "parent_exciter_id",
        "root_candidate_micro",
        "root_candidate",

        "birth_frame",
        "end_frame",
        "duration_frames",

        "source_entity",
        "target_entity",

        "parent_strength",
        "lineage_strength",
        "lineage_label",

        "harmonic_parenthood_score",
        "harmonic_5_7_parenthood_score",

        "present_harmonics",
        "missing_harmonics",

        "offspring_count",
        "odd_harmonic_count",
        "delayed_offspring_count",
        "token_continuity",

        "parent_tokens",
        "offspring_tokens",

        "residual_box_count",
        "residual_box_tokens",

        "register_class",

        "note_identity_candidate",
        "box_residual_signature",
    ]

    _write_csv(out_lineages, lineages, lineage_fields)

    offspring_fields = [
        "lineage_id",
        "root_candidate",
        "harmonic_number",
        "offspring_token",
        "offspring_frame",
        "offspring_delay",
        "offspring_score",
        "offspring_support",
        "offspring_row_strength",
        "offspring_source_entity",
        "offspring_target_entity",
    ]

    _write_csv(out_offspring, offspring_rows, offspring_fields)

    readable_fields = [
        "lineage_id",
        "frame",
        "root",
        "label",
        "strength",
        "h57",
        "present",
        "offspring",
        "residual_box_count",
    ]

    _write_csv(out_readable, readable, readable_fields)

    meta = {
        "stage": "virtual_string_lineage_builder",
        "semantic_version": "virtual_string_lineage_v1",
        "inputs": {
            "source_events_csv": args.source_events_csv,
        },
        "outputs": {
            "lineages_csv": args.out_lineages_csv,
            "offspring_csv": args.out_offspring_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "identity_prefix": args.identity_prefix,
            "max_delay_frames": args.max_delay_frames,
            "min_parent_strength": args.min_parent_strength,
            "min_lineage_strength": args.min_lineage_strength,
        },
        "result": {
            "source_rows": len(rows),
            "lineages": len(lineages),
            "offspring_rows": len(offspring_rows),
            "lineage_label_counts": dict(label_counts),
            "register_counts": dict(register_counts),
        },
        "ontology_note": (
            "This layer follows parent→offspring logic. A note is not forced from an "
            "instantaneous root. It is a time-distributed lineage born from virtual string "
            "excitation, with 5th and 7th harmonics as strong parenthood witnesses. "
            "Residual box tokens are kept for later instrument passport matching."
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "VIRTUAL STRING LINEAGE BUILDER",
        "=" * 72,
        f"source_events_csv     : {args.source_events_csv}",
        "",
        f"source_rows           : {len(rows)}",
        f"lineages              : {len(lineages)}",
        f"offspring_rows        : {len(offspring_rows)}",
        "",
        "Lineage label counts:",
    ]

    for k in sorted(label_counts):
        txt.append(f"  {k}: {label_counts[k]}")

    txt.append("")
    txt.append("Register counts:")
    for k in sorted(register_counts):
        txt.append(f"  {k}: {register_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Virtual string action is the parent.",
        "  Harmonics are offspring distributed in time.",
        "  Note identity emerges from lineage strength, not from a single root hit.",
        "  5th and 7th harmonics are strong witnesses for parenthood.",
        "  Residual unexplained tokens are kept as box/instrument-body candidates.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("virtual string lineage builder complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
