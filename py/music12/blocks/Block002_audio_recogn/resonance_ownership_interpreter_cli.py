# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Safe helpers
# ============================================================

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


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _role_from_probs(row: Dict[str, Any]) -> str:
    probs = {
        "ownership": _safe_float(row.get("ownership_probability"), 0.0),
        "feeding": _safe_float(row.get("feeding_probability"), 0.0),
        "carrying": _safe_float(row.get("carrying_probability"), 0.0),
        "masking": _safe_float(row.get("masking_probability"), 0.0),
    }

    top = max(probs, key=probs.get)
    second = sorted(probs.values(), reverse=True)[1]
    gap = probs[top] - second

    if top == "ownership" and probs[top] >= 0.46 and gap >= 0.12:
        return "PRIMARY_OWNER_EPISODE"

    if top == "ownership" and probs[top] >= 0.28:
        return "MICRO_DOMINANCE_EPISODE"

    if top == "carrying" and probs[top] >= 0.26:
        return "CARRIER_TRANSFER_EPISODE"

    if top == "masking" and probs[top] >= 0.24:
        return "LOCAL_MASKING_EPISODE"

    if top == "feeding" and probs[top] >= 0.22:
        return "FEEDING_SUPPORT_EPISODE"

    return "SHARED_RESONANCE_EPISODE"


def _episode_narrative(role: str) -> str:
    return {
        "PRIMARY_OWNER_EPISODE": (
            "local resonance ownership is strongly concentrated around one source"
        ),
        "MICRO_DOMINANCE_EPISODE": (
            "micro-topology suggests local dominance without full ownership closure"
        ),
        "CARRIER_TRANSFER_EPISODE": (
            "resonance structure is being carried or transferred through the field"
        ),
        "LOCAL_MASKING_EPISODE": (
            "a local region shows masking or absorption pressure"
        ),
        "FEEDING_SUPPORT_EPISODE": (
            "one structure supports another through delayed or feeding resonance"
        ),
        "SHARED_RESONANCE_EPISODE": (
            "multiple structures coexist without decisive ownership"
        ),
    }.get(role, "ownership state is ambiguous")


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.46:
        return "MEDIUM"
    if score > 0.0:
        return "LOW"
    return "NONE"


def _confidence_reason(
    *,
    role: str,
    ownership_p: float,
    feeding_p: float,
    carrying_p: float,
    masking_p: float,
    continuity: float,
    causal: float,
    causal_conf: float,
    micro_richness: float,
    field_state: str,
    micro_field_texture: str,
    continuity_texture: str,
) -> str:
    parts = []

    if role == "PRIMARY_OWNER_EPISODE":
        parts.append("ownership_probability_dominant")
    elif role == "MICRO_DOMINANCE_EPISODE":
        parts.append("micro_dominance_without_full_closure")
    elif role == "CARRIER_TRANSFER_EPISODE":
        parts.append("carrying_probability_dominant")
    elif role == "LOCAL_MASKING_EPISODE":
        parts.append("masking_probability_dominant")
    elif role == "FEEDING_SUPPORT_EPISODE":
        parts.append("feeding_probability_dominant")
    else:
        parts.append("shared_probabilistic_field")

    if continuity >= 0.38:
        parts.append("strong_continuity")
    elif continuity >= 0.18:
        parts.append("moderate_continuity")
    elif continuity > 0.0:
        parts.append("weak_continuity")

    if causal_conf >= 0.55:
        parts.append("strong_causal_confidence")
    elif causal_conf >= 0.28:
        parts.append("moderate_causal_confidence")

    if causal >= 0.30:
        parts.append("causal_support_present")

    if micro_richness >= 0.70:
        parts.append("rich_micro_family")
    elif micro_richness >= 0.30:
        parts.append("active_micro_family")

    if field_state:
        parts.append(f"field={field_state}")

    if micro_field_texture:
        parts.append(f"micro_field={micro_field_texture}")

    if continuity_texture:
        parts.append(f"continuity_field={continuity_texture}")

    return "+".join(parts)


def _masking_warning(
    *,
    role: str,
    masking_p: float,
    field_state: str,
    continuity: float,
) -> str:
    if role == "LOCAL_MASKING_EPISODE":
        return "MASKING_ACTIVE"

    if masking_p >= 0.20 and field_state == "ABSORPTION_DOMINANT_FIELD":
        return "MASKING_FIELD_PRESSURE"

    if masking_p >= 0.18 and continuity < 0.18:
        return "MASKING_RISK_LOW_CONTINUITY"

    return ""


def _carrier_transition(
    *,
    role: str,
    carrying_p: float,
    flow_kind: str,
    field_state: str,
) -> str:
    if role == "CARRIER_TRANSFER_EPISODE":
        return "CARRIER_TRANSFER_ACTIVE"

    if carrying_p >= 0.22 and flow_kind in ("SUSTAINED_COUPLING", "DELAYED_FEEDING"):
        return "CARRIER_TRANSITION_PROBABLE"

    if carrying_p >= 0.20 and field_state == "CARRIER_DOMINANT_FIELD":
        return "CARRIER_FIELD_SUPPORT"

    return ""


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Interpret ownership resolution as resonance ecology episodes. "
            "This is a diagnostic/explanatory layer, not the final note resolver."
        )
    )

    ap.add_argument("--ownership_csv", required=True)
    ap.add_argument("--entity_roles_csv", required=True)
    ap.add_argument("--field_windows_csv", required=True)

    ap.add_argument("--out_episodes_csv", required=True)
    ap.add_argument("--out_entity_timeline_csv", required=True)
    ap.add_argument("--out_readable_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--out_summary_txt", required=True)

    ap.add_argument("--min_episode_confidence", type=float, default=0.16)

    args = ap.parse_args()

    ownership_rows = _load_csv(Path(args.ownership_csv))
    entity_role_rows = _load_csv(Path(args.entity_roles_csv))
    field_rows = _load_csv(Path(args.field_windows_csv))

    entity_roles = {
        str(r.get("entity_id", "")).strip(): str(r.get("ownership_role", "")).strip()
        for r in entity_role_rows
    }

    field_windows = []
    for r in field_rows:
        field_windows.append({
            "start": _safe_int(r.get("window_start_frame"), 0),
            "end": _safe_int(r.get("window_end_frame"), 0),
            "field_state": str(r.get("field_state", "")).strip(),
            "micro_field_texture": str(r.get("micro_field_texture", "")).strip(),
            "continuity_texture": str(r.get("continuity_texture", "")).strip(),
            "micro_token_richness": _safe_int(r.get("micro_token_richness"), 0),
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
            "mean_continuity_support": 0.0,
            "mean_causal_confidence": 0.0,
        }

    episodes = []
    timeline_by_entity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in ownership_rows:
        src = str(r.get("source_entity", "")).strip()
        dst = str(r.get("target_entity", "")).strip()

        if not src or not dst:
            continue

        source_birth_frame = _safe_int(r.get("source_birth_frame"), 0)
        if source_birth_frame <= 0:
            source_birth_frame = _safe_int(r.get("birth_frame", 0), 0)

        role = _role_from_probs(r)

        ownership_p = _safe_float(r.get("ownership_probability"), 0.0)
        feeding_p = _safe_float(r.get("feeding_probability"), 0.0)
        carrying_p = _safe_float(r.get("carrying_probability"), 0.0)
        masking_p = _safe_float(r.get("masking_probability"), 0.0)

        continuity = _safe_float(r.get("continuity_support"), 0.0)
        causal = _safe_float(r.get("causal_support"), 0.0)
        causal_conf = _safe_float(r.get("causal_confidence"), 0.0)

        source_micro_count = _safe_int(r.get("source_micro_count"), 0)
        target_micro_count = _safe_int(r.get("target_micro_count"), 0)

        micro_richness = min((source_micro_count + target_micro_count) / 120.0, 1.0)

        fw = _field_for_frame(source_birth_frame)

        episode_confidence = (
            max(ownership_p, feeding_p, carrying_p, masking_p) * 0.36
            + continuity * 0.23
            + causal * 0.13
            + causal_conf * 0.13
            + micro_richness * 0.10
            + fw["mean_continuity_support"] * 0.05
        )

        if episode_confidence < args.min_episode_confidence:
            continue

        field_state = fw["field_state"]
        micro_field_texture = fw["micro_field_texture"]
        continuity_texture = fw["continuity_texture"]

        source_entity_role = entity_roles.get(src, "")
        target_entity_role = entity_roles.get(dst, "")
        flow_kind = str(r.get("flow_kind", "")).strip()

        confidence_reason = _confidence_reason(
            role=role,
            ownership_p=ownership_p,
            feeding_p=feeding_p,
            carrying_p=carrying_p,
            masking_p=masking_p,
            continuity=continuity,
            causal=causal,
            causal_conf=causal_conf,
            micro_richness=micro_richness,
            field_state=field_state,
            micro_field_texture=micro_field_texture,
            continuity_texture=continuity_texture,
        )

        masking_warning = _masking_warning(
            role=role,
            masking_p=masking_p,
            field_state=field_state,
            continuity=continuity,
        )

        carrier_transition = _carrier_transition(
            role=role,
            carrying_p=carrying_p,
            flow_kind=flow_kind,
            field_state=field_state,
        )

        ownership_gap = ownership_p - max(feeding_p, carrying_p, masking_p)

        episode = {
            "source_entity": src,
            "target_entity": dst,

            "episode_role": role,
            "episode_confidence": f"{episode_confidence:.9f}",
            "episode_confidence_label": _confidence_label(episode_confidence),
            "confidence_reason": confidence_reason,

            "masking_warning": masking_warning,
            "carrier_transition": carrier_transition,

            "ownership_gap": f"{ownership_gap:.9f}",
            "micro_richness": f"{micro_richness:.9f}",

            "episode_narrative": _episode_narrative(role),

            "source_entity_ownership_role": source_entity_role,
            "target_entity_ownership_role": target_entity_role,

            "flow_kind": flow_kind,

            "ownership_probability": f"{ownership_p:.9f}",
            "feeding_probability": f"{feeding_p:.9f}",
            "carrying_probability": f"{carrying_p:.9f}",
            "masking_probability": f"{masking_p:.9f}",

            "continuity_support": f"{continuity:.9f}",
            "causal_support": f"{causal:.9f}",
            "causal_confidence": f"{causal_conf:.9f}",

            "micro_token_similarity": str(r.get("micro_token_similarity", "")).strip(),
            "coarse_token_similarity": str(r.get("coarse_token_similarity", "")).strip(),
            "micro_root_similarity": str(r.get("micro_root_similarity", "")).strip(),
            "topology_similarity": str(r.get("topology_similarity", "")).strip(),

            "source_root_hint_micro": str(r.get("source_root_hint_micro", "")).strip(),
            "target_root_hint_micro": str(r.get("target_root_hint_micro", "")).strip(),

            "source_micro_count": source_micro_count,
            "target_micro_count": target_micro_count,

            "source_micro_preview": str(r.get("source_micro_preview", "")).strip(),
            "target_micro_preview": str(r.get("target_micro_preview", "")).strip(),

            "field_state": field_state,
            "micro_field_texture": micro_field_texture,
            "continuity_texture": continuity_texture,
            "field_micro_token_richness": fw["micro_token_richness"],
            "field_mean_continuity_support": f"{fw['mean_continuity_support']:.9f}",
            "field_mean_causal_confidence": f"{fw['mean_causal_confidence']:.9f}",

            "source_birth_frame": source_birth_frame,
        }

        episodes.append(episode)
        timeline_by_entity[src].append(episode)
        timeline_by_entity[dst].append(episode)

    episodes.sort(
        key=lambda r: (
            _safe_int(r.get("source_birth_frame"), 0),
            r.get("source_entity", ""),
            -_safe_float(r.get("episode_confidence"), 0.0),
        )
    )

    entity_timeline_rows = []

    for eid, rows in sorted(timeline_by_entity.items(), key=lambda kv: _safe_int(kv[0], 0)):
        roles = defaultdict(int)
        confidences = []
        ownership_values = []
        micro_counts = []
        fields = defaultdict(int)
        frames = []

        masking_warning_count = 0
        carrier_transition_count = 0
        micro_dominance_episode_count = 0
        shared_cluster_duration = 0
        primary_owner_episode_count = 0

        for r in rows:
            role = r["episode_role"]
            roles[role] += 1
            confidences.append(_safe_float(r.get("episode_confidence"), 0.0))
            ownership_values.append(_safe_float(r.get("ownership_probability"), 0.0))
            micro_counts.append(_safe_int(r.get("source_micro_count"), 0))
            micro_counts.append(_safe_int(r.get("target_micro_count"), 0))
            fields[r.get("field_state", "")] += 1
            frames.append(_safe_int(r.get("source_birth_frame"), 0))

            if r.get("masking_warning"):
                masking_warning_count += 1

            if r.get("carrier_transition"):
                carrier_transition_count += 1

            if role == "MICRO_DOMINANCE_EPISODE":
                micro_dominance_episode_count += 1

            if role == "SHARED_RESONANCE_EPISODE":
                shared_cluster_duration += 1

            if role == "PRIMARY_OWNER_EPISODE":
                primary_owner_episode_count += 1

        dominant_episode_role = sorted(
            roles.items(),
            key=lambda x: (-x[1], x[0]),
        )[0][0] if roles else ""

        dominant_field_state = sorted(
            fields.items(),
            key=lambda x: (-x[1], x[0]),
        )[0][0] if fields else ""

        # Stability means ownership is sustained and not wildly fluctuating.
        # It is not final note confidence.
        if ownership_values:
            own_mean = _mean(ownership_values)
            own_span = max(ownership_values) - min(ownership_values)
            ownership_stability = max(0.0, min(1.0, own_mean * (1.0 - min(own_span, 1.0))))
        else:
            ownership_stability = 0.0

        if shared_cluster_duration > 0:
            shared_cluster_status = "SHARED_CLUSTER_PRESENT"
        else:
            shared_cluster_status = ""

        if masking_warning_count > 0:
            masking_status = "MASKING_WARNINGS_PRESENT"
        else:
            masking_status = ""

        if carrier_transition_count > 0:
            carrier_status = "CARRIER_TRANSITIONS_PRESENT"
        else:
            carrier_status = ""

        entity_timeline_rows.append({
            "entity_id": eid,
            "episode_count": len(rows),
            "dominant_episode_role": dominant_episode_role,
            "dominant_field_state": dominant_field_state,

            "ownership_stability": f"{ownership_stability:.9f}",
            "ownership_mean": f"{_mean(ownership_values):.9f}",
            "ownership_span": f"{(max(ownership_values) - min(ownership_values)) if ownership_values else 0.0:.9f}",

            "shared_cluster_duration": shared_cluster_duration,
            "shared_cluster_status": shared_cluster_status,

            "micro_dominance_episode_count": micro_dominance_episode_count,
            "primary_owner_episode_count": primary_owner_episode_count,

            "masking_warning_count": masking_warning_count,
            "masking_status": masking_status,

            "carrier_transition_count": carrier_transition_count,
            "carrier_status": carrier_status,

            "mean_episode_confidence": f"{_mean(confidences):.9f}",
            "max_episode_confidence": f"{max(confidences) if confidences else 0.0:.9f}",
            "mean_micro_count": f"{_mean([float(x) for x in micro_counts]):.9f}",
            "first_frame": min(frames) if frames else 0,
            "last_frame": max(frames) if frames else 0,

            "episode_role_counts_json": json.dumps(dict(roles), ensure_ascii=False, sort_keys=True),
            "field_state_counts_json": json.dumps(dict(fields), ensure_ascii=False, sort_keys=True),
        })

    readable_rows = []
    for r in episodes[:5000]:
        readable_rows.append({
            "frame": r["source_birth_frame"],
            "scene": (
                f"E{r['source_entity']}:{r['source_root_hint_micro']} "
                f"→ E{r['target_entity']}:{r['target_root_hint_micro']}"
            ),
            "episode_role": r["episode_role"],
            "confidence": r["episode_confidence_label"],
            "confidence_reason": r["confidence_reason"],
            "masking_warning": r["masking_warning"],
            "carrier_transition": r["carrier_transition"],
            "field": f"{r['field_state']} / {r['micro_field_texture']}",
            "micro": f"{r['source_micro_count']}→{r['target_micro_count']}",
            "narrative": r["episode_narrative"],
        })

    episode_counts = defaultdict(int)
    confidence_counts = defaultdict(int)
    warning_counts = defaultdict(int)

    for r in episodes:
        episode_counts[r["episode_role"]] += 1
        confidence_counts[r["episode_confidence_label"]] += 1

        if r.get("masking_warning"):
            warning_counts[r["masking_warning"]] += 1

        if r.get("carrier_transition"):
            warning_counts[r["carrier_transition"]] += 1

    out_episodes = Path(args.out_episodes_csv)
    out_timeline = Path(args.out_entity_timeline_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    out_episodes.parent.mkdir(parents=True, exist_ok=True)

    episode_fields = [
        "source_entity",
        "target_entity",
        "episode_role",
        "episode_confidence",
        "episode_confidence_label",
        "confidence_reason",
        "masking_warning",
        "carrier_transition",
        "ownership_gap",
        "micro_richness",
        "episode_narrative",

        "source_entity_ownership_role",
        "target_entity_ownership_role",
        "flow_kind",

        "ownership_probability",
        "feeding_probability",
        "carrying_probability",
        "masking_probability",

        "continuity_support",
        "causal_support",
        "causal_confidence",

        "micro_token_similarity",
        "coarse_token_similarity",
        "micro_root_similarity",
        "topology_similarity",

        "source_root_hint_micro",
        "target_root_hint_micro",

        "source_micro_count",
        "target_micro_count",

        "source_micro_preview",
        "target_micro_preview",

        "field_state",
        "micro_field_texture",
        "continuity_texture",
        "field_micro_token_richness",
        "field_mean_continuity_support",
        "field_mean_causal_confidence",

        "source_birth_frame",
    ]

    with out_episodes.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=episode_fields)
        w.writeheader()
        w.writerows(episodes)

    timeline_fields = [
        "entity_id",
        "episode_count",
        "dominant_episode_role",
        "dominant_field_state",

        "ownership_stability",
        "ownership_mean",
        "ownership_span",

        "shared_cluster_duration",
        "shared_cluster_status",

        "micro_dominance_episode_count",
        "primary_owner_episode_count",

        "masking_warning_count",
        "masking_status",

        "carrier_transition_count",
        "carrier_status",

        "mean_episode_confidence",
        "max_episode_confidence",
        "mean_micro_count",
        "first_frame",
        "last_frame",

        "episode_role_counts_json",
        "field_state_counts_json",
    ]

    with out_timeline.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timeline_fields)
        w.writeheader()
        w.writerows(entity_timeline_rows)

    with out_readable.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "scene",
                "episode_role",
                "confidence",
                "confidence_reason",
                "masking_warning",
                "carrier_transition",
                "field",
                "micro",
                "narrative",
            ],
        )
        w.writeheader()
        w.writerows(readable_rows)

    meta = {
        "stage": "resonance_ownership_interpreter",
        "semantic_version": "ownership_ecology_episode_interpreter_v2",
        "inputs": {
            "ownership_csv": args.ownership_csv,
            "entity_roles_csv": args.entity_roles_csv,
            "field_windows_csv": args.field_windows_csv,
        },
        "outputs": {
            "episodes_csv": args.out_episodes_csv,
            "entity_timeline_csv": args.out_entity_timeline_csv,
            "readable_csv": args.out_readable_csv,
            "meta_json": args.out_meta_json,
            "summary_txt": args.out_summary_txt,
        },
        "parameters": {
            "min_episode_confidence": args.min_episode_confidence,
        },
        "result": {
            "ownership_rows": len(ownership_rows),
            "episodes": len(episodes),
            "entity_timelines": len(entity_timeline_rows),
            "episode_counts": dict(episode_counts),
            "confidence_counts": dict(confidence_counts),
            "warning_counts": dict(warning_counts),
        },
        "ontology_note": (
            "This layer interprets ownership resolution as resonance ecology episodes. "
            "It does not decide final notes; it explains micro-dominance, shared clusters, "
            "carrying, masking and feeding as context for later note identity resolution. "
            "Version 2 adds ownership stability, confidence reasons, masking warnings, "
            "carrier transitions and shared-cluster diagnostics."
        ),
    }

    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = [
        "RESONANCE OWNERSHIP INTERPRETER",
        "=" * 72,
        f"ownership_csv     : {args.ownership_csv}",
        f"entity_roles_csv  : {args.entity_roles_csv}",
        f"field_windows_csv : {args.field_windows_csv}",
        "",
        f"ownership_rows    : {len(ownership_rows)}",
        f"episodes          : {len(episodes)}",
        f"entity_timelines  : {len(entity_timeline_rows)}",
        "",
        "Episode counts:",
    ]

    for k in sorted(episode_counts):
        txt.append(f"  {k}: {episode_counts[k]}")

    txt.append("")
    txt.append("Confidence counts:")
    for k in sorted(confidence_counts):
        txt.append(f"  {k}: {confidence_counts[k]}")

    txt.append("")
    txt.append("Warning / transition counts:")
    for k in sorted(warning_counts):
        txt.append(f"  {k}: {warning_counts[k]}")

    txt.extend([
        "",
        "Principle:",
        "  Ownership interpretation is not final note recognition.",
        "  It explains how resonance structures coexist, dominate, feed, mask,",
        "  and migrate through a micro-aware acoustic field.",
        "  The v2 interpreter also exposes ownership stability, shared cluster",
        "  duration, masking warnings, carrier transitions and confidence reasons.",
        "",
    ])

    out_txt.write_text("\n".join(txt), encoding="utf-8")

    print("resonance ownership interpreter complete")
    print(json.dumps(meta["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
