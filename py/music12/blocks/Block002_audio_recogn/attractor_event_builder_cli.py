# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


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


def _join(tokens: Set[str], limit: int = 128) -> str:
    return " ".join(sorted(tokens)[:limit])


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _normalize_note(token: Any) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _coarse_note(token: Any) -> str:
    return _normalize_note(token)


def _event_id(prefix: str, idx: int, root: str, birth: int, end: int) -> str:
    safe_root = (
        str(root or "")
        .replace("'", "")
        .replace(".", "_")
        .replace("-", "m")
        .replace("+", "p")
    )
    return f"{prefix}_{idx:06d}_{safe_root}_{birth}_{end}"


def _scene_key(row: Dict[str, Any]) -> Tuple[int, str, str, str]:
    return (
        _safe_int(row.get("frame"), 0),
        str(row.get("source_entity", "")).strip(),
        str(row.get("target_entity", "")).strip(),
        str(row.get("note_candidate", "")).strip(),
    )


def _cloud_key(row: Dict[str, Any]) -> Tuple[int, str, str, str]:
    return (
        _safe_int(row.get("frame"), 0),
        str(row.get("source_entity", "")).strip(),
        str(row.get("target_entity", "")).strip(),
        str(row.get("root_candidate", "")).strip(),
    )


def _combined_confidence(
    *,
    root_scene_score: float,
    note_candidate_score: float,
    box_signature_score: float,
    harmonic_gravity_score: float,
    h57_score: float,
    ownership_stability: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            root_scene_score * 0.24
            + note_candidate_score * 0.18
            + harmonic_gravity_score * 0.22
            + h57_score * 0.18
            + ownership_stability * 0.12
            + box_signature_score * 0.06,
        ),
    )


def _row_tokens(cloud: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Set[str]]:
    excitation = set()
    excitation.update(_split_tokens(cloud.get("harmonic_cloud_tokens", "")))
    excitation.update(_split_tokens(cloud.get("source_micro_preview", "")))
    excitation.update(_split_tokens(cloud.get("target_micro_preview", "")))

    body = set()
    body.update(_split_tokens(scene.get("box_residual_preview", "")))

    secondary = set()
    secondary.update(_split_tokens(scene.get("field_pool_preview", "")))
    secondary.update(_split_tokens(scene.get("field_micro_token_preview", "")))

    body = body - excitation
    secondary = secondary - excitation

    return {"excitation": excitation, "body": body, "secondary": secondary}


def _proto_from_rows(cloud: Dict[str, Any], scene: Dict[str, Any], min_conf: float) -> Dict[str, Any] | None:
    frame = _safe_int(cloud.get("frame"), 0)
    src = str(cloud.get("source_entity", "")).strip()
    dst = str(cloud.get("target_entity", "")).strip()
    root_micro = str(cloud.get("root_candidate", "")).strip()
    root_coarse = _coarse_note(root_micro)

    if not root_micro:
        return None

    root_scene_score = _safe_float(cloud.get("root_scene_score"), 0.0)
    harmonic_gravity_score = _safe_float(cloud.get("harmonic_gravity_score"), 0.0)
    h57_score = _safe_float(cloud.get("harmonic_5_7_gravity"), 0.0)
    ownership_stability = _safe_float(cloud.get("ownership_stability"), 0.0)

    note_candidate_score = _safe_float(scene.get("note_candidate_score"), root_scene_score)
    box_signature_score = _safe_float(scene.get("box_signature_score"), 0.0)

    confidence = _combined_confidence(
        root_scene_score=root_scene_score,
        note_candidate_score=note_candidate_score,
        box_signature_score=box_signature_score,
        harmonic_gravity_score=harmonic_gravity_score,
        h57_score=h57_score,
        ownership_stability=ownership_stability,
    )

    if confidence < min_conf:
        return None

    toks = _row_tokens(cloud, scene)

    return {
        "frame": frame,
        "birth_frame": frame,
        "end_frame": frame,
        "source_entities": {src} if src else set(),
        "target_entities": {dst} if dst else set(),
        "root_micro_values": {root_micro},
        "root_coarse": root_coarse,
        "confidence_values": [confidence],
        "root_scene_scores": [root_scene_score],
        "note_candidate_scores": [note_candidate_score],
        "box_signature_scores": [box_signature_score],
        "harmonic_gravity_scores": [harmonic_gravity_score],
        "h57_scores": [h57_score],
        "ownership_stability_values": [ownership_stability],
        "stabilized_root_labels": {str(cloud.get("stabilized_root_label", "")).strip()},
        "root_attraction_classes": {str(cloud.get("root_attraction_class", "")).strip()},
        "scene_labels": {str(scene.get("scene_label", "")).strip()},
        "field_states": {str(cloud.get("field_state", scene.get("field_state", ""))).strip()},
        "present_harmonics": _split_tokens(cloud.get("present_harmonics", "")),
        "missing_harmonics": _split_tokens(cloud.get("missing_harmonics", "")),
        "register_classes": {str(cloud.get("register_class", "")).strip()},
        "excitation_core_tokens": set(toks["excitation"]),
        "instrument_body_tokens": set(toks["body"]),
        "secondary_field_tokens": set(toks["secondary"]),
        "harmonic_basis": [str(cloud.get("harmonic_basis", "")).strip()],
        "box_signature_basis": [str(scene.get("box_signature_basis", "")).strip()],
        "confidence_reason": [str(cloud.get("confidence_reason", scene.get("confidence_reason", ""))).strip()],
        "member_count": 1,
    }


def _cluster_similarity(cluster: Dict[str, Any], proto: Dict[str, Any], max_gap: int) -> float:
    if proto["root_coarse"] != cluster["root_coarse"]:
        return 0.0

    gap = proto["birth_frame"] - cluster["end_frame"]
    if gap < 0:
        gap = 0
    if gap > max_gap:
        return 0.0

    time_score = 1.0 - min(gap / max(max_gap, 1), 1.0)
    exc_sim = _jaccard(cluster["excitation_core_tokens"], proto["excitation_core_tokens"])
    body_sim = _jaccard(cluster["instrument_body_tokens"], proto["instrument_body_tokens"])
    sec_sim = _jaccard(cluster["secondary_field_tokens"], proto["secondary_field_tokens"])
    h57_score = min(max(cluster["h57_scores"] or [0.0]), max(proto["h57_scores"] or [0.0]))
    gravity_score = min(max(cluster["harmonic_gravity_scores"] or [0.0]), max(proto["harmonic_gravity_scores"] or [0.0]))

    return (
        time_score * 0.28
        + exc_sim * 0.24
        + body_sim * 0.14
        + sec_sim * 0.08
        + h57_score * 0.14
        + gravity_score * 0.12
    )


def _merge_cluster(cluster: Dict[str, Any], proto: Dict[str, Any]) -> None:
    cluster["birth_frame"] = min(cluster["birth_frame"], proto["birth_frame"])
    cluster["end_frame"] = max(cluster["end_frame"], proto["end_frame"])

    for k in (
        "source_entities", "target_entities", "root_micro_values", "stabilized_root_labels",
        "root_attraction_classes", "scene_labels", "field_states", "present_harmonics",
        "missing_harmonics", "register_classes", "excitation_core_tokens",
        "instrument_body_tokens", "secondary_field_tokens",
    ):
        cluster[k].update(proto[k])

    for k in (
        "confidence_values", "root_scene_scores", "note_candidate_scores", "box_signature_scores",
        "harmonic_gravity_scores", "h57_scores", "ownership_stability_values", "harmonic_basis",
        "box_signature_basis", "confidence_reason",
    ):
        cluster[k].extend(proto[k])

    cluster["member_count"] += proto["member_count"]


def _best_label(labels: Set[str], priority: List[str], default: str = "") -> str:
    for p in priority:
        if p in labels:
            return p
    vals = sorted([x for x in labels if x])
    return vals[0] if vals else default


def _status_from_cluster(cluster: Dict[str, Any], confidence: float) -> str:
    root_label = _best_label(cluster["stabilized_root_labels"], ["STABLE_NOTE_ROOT", "SUPPORTED_NOTE_ROOT", "WEAK_SUPPORTED_ROOT", "UNSTABLE_ROOT_CANDIDATE"])
    scene_label = _best_label(cluster["scene_labels"], ["NOTE_WITH_BOX_SIGNATURE", "NOTE_DOMINANT_SCENE", "HARMONICALLY_SUPPORTED_NOTE_SCENE", "NOTE_CANDIDATE_SCENE"])

    if root_label in ("STABLE_NOTE_ROOT", "SUPPORTED_NOTE_ROOT") and confidence >= 0.52:
        return "SUPPORTED_ATTRACTOR"
    if scene_label in ("NOTE_WITH_BOX_SIGNATURE", "NOTE_DOMINANT_SCENE", "HARMONICALLY_SUPPORTED_NOTE_SCENE") and confidence >= 0.36:
        return "NOTE_SCENE_ATTRACTOR"
    if root_label == "WEAK_SUPPORTED_ROOT" and confidence >= 0.28:
        return "WEAK_ATTRACTOR"
    if cluster["member_count"] >= 3 and confidence >= 0.22:
        return "CLUSTERED_CANDIDATE_ATTRACTOR"
    return "CANDIDATE_ATTRACTOR"


def _field_behavior_from_cluster(cluster: Dict[str, Any]) -> str:
    if "NOTE_WITH_BOX_SIGNATURE" in cluster["scene_labels"]:
        return "NOTE_WITH_BODY_FIELD"
    if "CARRIER_DOMINANT_FIELD" in cluster["field_states"]:
        return "CARRIER_BODY_FIELD"
    if "SOURCE_DOMINANT_FIELD" in cluster["field_states"]:
        return "SOURCE_PRESSURE_FIELD"
    if "ABSORPTION_DOMINANT_FIELD" in cluster["field_states"]:
        return "ABSORPTION_FIELD"
    if "BALANCED_RESONANCE_FIELD" in cluster["field_states"]:
        return "BALANCED_RESONANCE_FIELD"
    return "UNRESOLVED_FIELD"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build clustered attractor_events_csv for polyphonic_attractor_ecology_cli.")
    ap.add_argument("--harmonic_clouds_csv", required=True)
    ap.add_argument("--note_box_scene_csv", required=True)
    ap.add_argument("--out_attractor_events_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--identity_prefix", default="ATTR")
    ap.add_argument("--min_attractor_confidence", type=float, default=0.10)
    ap.add_argument("--max_cluster_gap_frames", type=int, default=90)
    ap.add_argument("--min_cluster_similarity", type=float, default=0.28)
    args = ap.parse_args()

    cloud_rows = _load_csv(Path(args.harmonic_clouds_csv))
    scene_rows = _load_csv(Path(args.note_box_scene_csv))
    scene_index = {_scene_key(r): r for r in scene_rows}

    protos = []
    for cloud in cloud_rows:
        scene = scene_index.get(_cloud_key(cloud), {})
        proto = _proto_from_rows(cloud, scene, args.min_attractor_confidence)
        if proto:
            protos.append(proto)

    protos.sort(key=lambda r: (r["birth_frame"], r["root_coarse"], -max(r["confidence_values"])))

    clusters = []
    for proto in protos:
        best_i = -1
        best_score = 0.0
        for i, cluster in enumerate(clusters):
            score = _cluster_similarity(cluster, proto, args.max_cluster_gap_frames)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0 and best_score >= args.min_cluster_similarity:
            _merge_cluster(clusters[best_i], proto)
        else:
            clusters.append(proto)

    out_rows = []
    readable_rows = []
    status_counts = defaultdict(int)
    behavior_counts = defaultdict(int)
    member_count_hist = defaultdict(int)

    for idx, cluster in enumerate(clusters, 1):
        vals = cluster["confidence_values"]
        confidence = max(vals) * 0.58 + (sum(vals) / max(len(vals), 1)) * 0.42
        root_scene_score = max(cluster["root_scene_scores"] or [0.0])
        note_candidate_score = max(cluster["note_candidate_scores"] or [0.0])
        box_signature_score = max(cluster["box_signature_scores"] or [0.0])
        harmonic_gravity_score = max(cluster["harmonic_gravity_scores"] or [0.0])
        h57_score = max(cluster["h57_scores"] or [0.0])
        ownership_stability = max(cluster["ownership_stability_values"] or [0.0])
        root_micro = sorted(cluster["root_micro_values"])[0] if cluster["root_micro_values"] else cluster["root_coarse"]
        root_coarse = cluster["root_coarse"]
        status = _status_from_cluster(cluster, confidence)
        behavior = _field_behavior_from_cluster(cluster)
        duration = max(1, cluster["end_frame"] - cluster["birth_frame"] + 1)
        identity_id = _event_id(args.identity_prefix, idx, root_coarse, cluster["birth_frame"], cluster["end_frame"])

        row = {
            "identity_id": identity_id,
            "attractor_note": root_coarse,
            "attractor_note_micro": root_micro,
            "attractor_confidence": f"{confidence:.9f}",
            "attractor_status": status,
            "birth_frame": cluster["birth_frame"],
            "end_frame": cluster["end_frame"],
            "duration_frames": duration,
            "field_behavior": behavior,
            "field_state": _best_label(cluster["field_states"], ["SOURCE_DOMINANT_FIELD", "CARRIER_DOMINANT_FIELD", "BALANCED_RESONANCE_FIELD", "ABSORPTION_DOMINANT_FIELD"]),
            "source_entity": " ".join(sorted(cluster["source_entities"])[:24]),
            "target_entity": " ".join(sorted(cluster["target_entities"])[:24]),
            "root_scene_score": f"{root_scene_score:.9f}",
            "note_candidate_score": f"{note_candidate_score:.9f}",
            "box_signature_score": f"{box_signature_score:.9f}",
            "harmonic_gravity_score": f"{harmonic_gravity_score:.9f}",
            "harmonic_5_7_gravity": f"{h57_score:.9f}",
            "ownership_stability": f"{ownership_stability:.9f}",
            "stabilized_root_label": _best_label(cluster["stabilized_root_labels"], ["STABLE_NOTE_ROOT", "SUPPORTED_NOTE_ROOT", "WEAK_SUPPORTED_ROOT", "UNSTABLE_ROOT_CANDIDATE"]),
            "root_attraction_class": _best_label(cluster["root_attraction_classes"], ["STRONG_ROOT_HARMONIC_GRAVITY", "ROOT_HARMONIC_GRAVITY", "FIELD_SUPPORTED_HARMONIC_CLOUD", "WEAK_HARMONIC_CLOUD", "NO_STABLE_HARMONIC_CLOUD"]),
            "scene_label": _best_label(cluster["scene_labels"], ["NOTE_WITH_BOX_SIGNATURE", "NOTE_DOMINANT_SCENE", "HARMONICALLY_SUPPORTED_NOTE_SCENE", "NOTE_CANDIDATE_SCENE", "RESIDUAL_BOX_CANDIDATE_SCENE", "AMBIGUOUS_RESONANCE_SCENE"]),
            "present_harmonics": " ".join(sorted(cluster["present_harmonics"])),
            "missing_harmonics": " ".join(sorted(cluster["missing_harmonics"])),
            "register_class": _best_label(cluster["register_classes"], ["LOW_REGISTER", "MID_REGISTER", "HIGH_REGISTER", "UNKNOWN_REGISTER"]),
            "excitation_core_tokens": _join(cluster["excitation_core_tokens"], 160),
            "instrument_body_tokens": _join(cluster["instrument_body_tokens"], 160),
            "secondary_field_tokens": _join(cluster["secondary_field_tokens"], 160),
            "harmonic_basis": " || ".join([x for x in cluster["harmonic_basis"] if x][:8]),
            "box_signature_basis": " || ".join([x for x in cluster["box_signature_basis"] if x][:8]),
            "confidence_reason": " || ".join([x for x in cluster["confidence_reason"] if x][:8]),
            "cluster_member_count": cluster["member_count"],
            "root_micro_values": " ".join(sorted(cluster["root_micro_values"])[:36]),
        }
        out_rows.append(row)
        status_counts[status] += 1
        behavior_counts[behavior] += 1
        member_count_hist[str(min(cluster["member_count"], 10))] += 1
        readable_rows.append({
            "identity_id": identity_id,
            "birth_frame": cluster["birth_frame"],
            "end_frame": cluster["end_frame"],
            "note": root_coarse,
            "micro": root_micro,
            "confidence": f"{confidence:.3f}",
            "status": status,
            "field_behavior": behavior,
            "members": cluster["member_count"],
            "h57": f"{h57_score:.3f}",
            "box": f"{box_signature_score:.3f}",
        })

    out_rows.sort(key=lambda r: (_safe_int(r.get("birth_frame"), 0), -_safe_float(r.get("attractor_confidence"), 0.0)))

    fields = [
        "identity_id", "attractor_note", "attractor_note_micro", "attractor_confidence", "attractor_status",
        "birth_frame", "end_frame", "duration_frames", "field_behavior", "field_state", "source_entity", "target_entity",
        "root_scene_score", "note_candidate_score", "box_signature_score", "harmonic_gravity_score", "harmonic_5_7_gravity", "ownership_stability",
        "stabilized_root_label", "root_attraction_class", "scene_label", "present_harmonics", "missing_harmonics", "register_class",
        "excitation_core_tokens", "instrument_body_tokens", "secondary_field_tokens", "harmonic_basis", "box_signature_basis", "confidence_reason",
        "cluster_member_count", "root_micro_values",
    ]
    _write_csv(Path(args.out_attractor_events_csv), out_rows, fields)

    readable_fields = ["identity_id", "birth_frame", "end_frame", "note", "micro", "confidence", "status", "field_behavior", "members", "h57", "box"]
    _write_csv(Path(args.out_readable_csv), readable_rows, readable_fields)

    meta = {
        "stage": "attractor_event_builder",
        "semantic_version": "attractor_event_builder_v2_clustered",
        "inputs": {"harmonic_clouds_csv": args.harmonic_clouds_csv, "note_box_scene_csv": args.note_box_scene_csv},
        "outputs": {
            "attractor_events_csv": args.out_attractor_events_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "identity_prefix": args.identity_prefix,
            "min_attractor_confidence": args.min_attractor_confidence,
            "max_cluster_gap_frames": args.max_cluster_gap_frames,
            "min_cluster_similarity": args.min_cluster_similarity,
        },
        "result": {
            "harmonic_cloud_rows": len(cloud_rows),
            "scene_rows": len(scene_rows),
            "proto_events": len(protos),
            "attractor_events": len(out_rows),
            "status_counts": dict(status_counts),
            "behavior_counts": dict(behavior_counts),
            "cluster_member_count_hist": dict(member_count_hist),
        },
        "ontology_note": "v2 clusters resonance continuity before naming attractor events, reducing graph-density bias.",
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = [
        "ATTRACTOR EVENT BUILDER",
        "=" * 72,
        f"harmonic_clouds_csv : {args.harmonic_clouds_csv}",
        f"note_box_scene_csv  : {args.note_box_scene_csv}",
        "",
        f"harmonic_cloud_rows : {len(cloud_rows)}",
        f"scene_rows          : {len(scene_rows)}",
        f"proto_events        : {len(protos)}",
        f"attractor_events    : {len(out_rows)}",
        "",
        "Attractor status counts:",
    ]
    for k in sorted(status_counts):
        txt.append(f"  {k}: {status_counts[k]}")
    txt.append("")
    txt.append("Field behavior counts:")
    for k in sorted(behavior_counts):
        txt.append(f"  {k}: {behavior_counts[k]}")
    txt.append("")
    txt.append("Cluster member count histogram:")
    for k in sorted(member_count_hist, key=lambda x: int(x)):
        txt.append(f"  {k}: {member_count_hist[k]}")
    txt.extend([
        "",
        "Principle:",
        "  Attractors are not ownership rows.",
        "  They are stabilized resonance continuities.",
        "  This builder clusters root/coarse note, time proximity, harmonic cloud similarity,",
        "  5/7 gravity and residual body similarity, and only then names attractor events.",
        "",
    ])
    Path(args.out_summary_txt).write_text("\n".join(txt), encoding="utf-8")
    print("attractor event builder complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
