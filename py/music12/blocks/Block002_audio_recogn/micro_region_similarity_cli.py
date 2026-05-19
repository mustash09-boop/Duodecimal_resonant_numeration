# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


ALPHABET12 = "123456789ABC"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_token(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _note_from_profile_filename(path: Path) -> str:
    left = path.name.split("__note_box_profile", 1)[0]
    raw = left.split("_")[-1]
    return _normalize_token(raw)


def _degree(token: str) -> str:
    try:
        return token.split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _anchor(token: str) -> str:
    if "'" not in token:
        return token
    return token.split("'", 1)[0] + "'-"


def _micro_side(token: str) -> str:
    if "'" not in token:
        return "-"
    tail = token.split("'", 1)[1]
    if tail.startswith("a"):
        return "a"
    if tail.startswith("i"):
        return "i"
    return "-"


def _micro_depth(token: str) -> int:
    if "'" not in token:
        return 0
    tail = token.split("'", 1)[1]
    digits = "".join(ch for ch in tail if ch in ALPHABET12)
    if not digits:
        return 0
    # first-level depth only for now
    return ALPHABET12.index(digits[-1]) + 1 if digits[-1] in ALPHABET12 else 0


def _micro_distance(a: str, b: str) -> float:
    if _anchor(a) != _anchor(b):
        return 3.0

    sa = _micro_side(a)
    sb = _micro_side(b)
    da = _micro_depth(a)
    db = _micro_depth(b)

    if sa == sb:
        return abs(da - db) / 12.0

    if sa == "-" or sb == "-":
        return (da + db) / 12.0

    return (da + db) / 6.0


def _radius(row: Dict[str, Any]) -> float:
    x = _safe_float(row.get("mean_x12"), 0.0)
    y = _safe_float(row.get("mean_y12"), 0.0)
    return math.sqrt(x * x + y * y)


def _load_profile_regions(folder: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    for p in sorted(folder.glob("*__note_box_profile.csv")):
        note = _note_from_profile_filename(p)
        rows = _load_csv(p)

        regions = []

        for r in rows:
            token = _normalize_token(r.get("token", ""))
            if not token:
                continue

            amp = _safe_float(r.get("mean_amp"), 0.0)
            presence = _safe_float(r.get("presence_ratio"), 0.0)
            rad = _radius(r)

            if amp >= 0.16 and presence >= 0.06:
                kind = "note"
            elif presence >= 0.16 and amp >= 0.035:
                kind = "box"
            elif presence >= 0.10 and amp < 0.05:
                kind = "echo"
            elif presence >= 0.08 and rad > 0.35:
                kind = "echo"
            else:
                kind = "weak"

            regions.append({
                "profile_note": note,
                "token": token,
                "anchor": _anchor(token),
                "degree": _degree(token),
                "kind": kind,
                "amp": amp,
                "presence": presence,
                "radius": rad,
            })

        out[note] = regions

    return out


def _members(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split() if x.strip()]


def _find_candidate_regions(root: str, profiles: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # same root first
    if root in profiles:
        return profiles[root]

    # then same degree across octave
    d = _degree(root)
    found = []
    for note, regions in profiles.items():
        if _degree(note) == d:
            found.extend(regions)

    return found


def _region_similarity(member: str, region: Dict[str, Any]) -> float:
    score = 0.0

    if _anchor(member) == region["anchor"]:
        score += 0.55
    elif _degree(member) == region["degree"]:
        score += 0.25

    md = _micro_distance(member, region["token"])
    score += max(0.0, 0.25 - md * 0.20)

    score += min(region["presence"], 1.0) * 0.12
    score += min(region["amp"], 1.0) * 0.08

    return min(score, 1.0)


def _classify_family(root: str, members: List[str], regions: List[Dict[str, Any]], threshold: float):
    note_score = 0.0
    box_score = 0.0
    echo_score = 0.0
    weak_score = 0.0

    note_hits = []
    box_hits = []
    echo_hits = []
    weak_hits = []
    unknown = []

    for m in members:
        best = None
        best_score = 0.0

        for reg in regions:
            s = _region_similarity(m, reg)
            if s > best_score:
                best_score = s
                best = reg

        if best is None or best_score < threshold:
            unknown.append(m)
            continue

        kind = best["kind"]

        if kind == "note":
            note_score += best_score
            note_hits.append(m)
        elif kind == "box":
            box_score += best_score
            box_hits.append(m)
        elif kind == "echo":
            echo_score += best_score
            echo_hits.append(m)
        else:
            weak_score += best_score
            weak_hits.append(m)

    return {
        "note_region_score": note_score,
        "box_region_score": box_score,
        "echo_region_score": echo_score,
        "weak_region_score": weak_score,
        "note_hits": sorted(set(note_hits)),
        "box_hits": sorted(set(box_hits)),
        "echo_hits": sorted(set(echo_hits)),
        "weak_hits": sorted(set(weak_hits)),
        "unknown": sorted(set(unknown)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Match micro harmonic families to Block004 resonance regions by similarity."
    )

    ap.add_argument("--micro_family_csv", required=True)
    ap.add_argument("--box_profile_folder", required=True)

    ap.add_argument("--out_region_csv", required=True)
    ap.add_argument("--out_frame_summary_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--similarity_threshold", type=float, default=0.42)

    args = ap.parse_args()

    family_csv = Path(args.micro_family_csv)
    profile_folder = Path(args.box_profile_folder)

    out_csv = Path(args.out_region_csv)
    out_frame = Path(args.out_frame_summary_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    families = _load_csv(family_csv)
    profiles = _load_profile_regions(profile_folder)

    out_rows = []

    total_note = 0
    total_box = 0
    total_echo = 0
    total_unknown = 0

    for r in families:
        root = str(r.get("family_root_note", "")).strip()
        members = _members(r.get("family_members", ""))

        regions = _find_candidate_regions(root, profiles)
        classified = _classify_family(
            root=root,
            members=members,
            regions=regions,
            threshold=args.similarity_threshold,
        )

        base = _safe_float(r.get("family_score"), 0.0)

        causal_score = base
        causal_score += classified["note_region_score"] * 0.12
        causal_score -= classified["box_region_score"] * 0.05
        causal_score -= classified["echo_region_score"] * 0.03
        causal_score += classified["weak_region_score"] * 0.01
        causal_score = max(causal_score, 0.0)

        total_note += len(classified["note_hits"])
        total_box += len(classified["box_hits"])
        total_echo += len(classified["echo_hits"])
        total_unknown += len(classified["unknown"])

        row = dict(r)
        row["causal_note_score"] = f"{causal_score:.9f}"
        row["note_region_score"] = f"{classified['note_region_score']:.9f}"
        row["box_region_score"] = f"{classified['box_region_score']:.9f}"
        row["echo_region_score"] = f"{classified['echo_region_score']:.9f}"
        row["weak_region_score"] = f"{classified['weak_region_score']:.9f}"

        row["note_hits"] = " ".join(classified["note_hits"])
        row["box_hits"] = " ".join(classified["box_hits"])
        row["echo_hits"] = " ".join(classified["echo_hits"])
        row["weak_hits"] = " ".join(classified["weak_hits"])
        row["unknown_hits"] = " ".join(classified["unknown"])

        row["note_hit_count"] = len(classified["note_hits"])
        row["box_hit_count"] = len(classified["box_hits"])
        row["echo_hit_count"] = len(classified["echo_hits"])
        row["unknown_hit_count"] = len(classified["unknown"])

        out_rows.append(row)

    out_rows.sort(
        key=lambda x: (
            _safe_int(x.get("frame_index"), 0),
            -_safe_float(x.get("causal_note_score"), 0.0),
        )
    )

    frame_map: Dict[int, List[str]] = {}
    for r in out_rows:
        frame = _safe_int(r.get("frame_index"), 0)
        frame_map.setdefault(frame, []).append(
            f"{r.get('family_root_note')}:{r.get('causal_note_score')}"
            f"[N{r.get('note_hit_count')}/B{r.get('box_hit_count')}/E{r.get('echo_hit_count')}/U{r.get('unknown_hit_count')}]"
        )

    frame_rows = [
        {
            "frame_index": frame,
            "top_region_families": " | ".join(items[:12]),
        }
        for frame, items in sorted(frame_map.items())
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(out_rows[0].keys()) if out_rows else []

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    with out_frame.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "top_region_families"])
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "stage": "micro_region_similarity",
        "inputs": {
            "micro_family_csv": str(family_csv),
            "box_profile_folder": str(profile_folder),
        },
        "outputs": {
            "region_csv": str(out_csv),
            "frame_summary_csv": str(out_frame),
            "meta_json": str(out_meta),
            "summary_txt": str(out_txt),
        },
        "parameters": {
            "similarity_threshold": args.similarity_threshold,
        },
        "result": {
            "profiles_loaded": len(profiles),
            "family_rows": len(out_rows),
            "note_region_hits": total_note,
            "box_region_hits": total_box,
            "echo_region_hits": total_echo,
            "unknown_hits": total_unknown,
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "MICRO REGION SIMILARITY",
        "=" * 72,
        f"micro_family_csv    : {family_csv}",
        f"box_profile_folder  : {profile_folder}",
        "",
        f"profiles_loaded     : {len(profiles)}",
        f"family_rows         : {len(out_rows)}",
        "",
        f"note_region_hits    : {total_note}",
        f"box_region_hits     : {total_box}",
        f"echo_region_hits    : {total_echo}",
        f"unknown_hits        : {total_unknown}",
        "",
        "Principle:",
        "  Resonance is treated as causality:",
        "  exciter -> response chains -> note / box / echo regions.",
        "  Matching is region-similarity, not exact token equality.",
        "",
    ]

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("micro region similarity complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()