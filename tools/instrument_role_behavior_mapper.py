from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


INSTRUMENTS = {"piano", "violin", "cello", "organ"}


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _index_by_id(path: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv(path)
    return {_safe_int(r.get("merged_event_id")): r for r in rows}


def _support_list(row: dict[str, str]) -> list[str]:
    raw = str(row.get("support_instruments", "")).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split() if x.strip()]


def _join(items: list[str]) -> str:
    items = [x for x in items if x]
    return " ".join(items)


def _assign_roles(
    layered: dict[str, str],
    lifecycle: dict[str, str] | None,
    acoustic: dict[str, str] | None,
    shared: dict[str, str] | None,
) -> dict[str, str]:
    dominant = str(layered.get("dominant_instrument", "")).strip()
    dom_state = str(layered.get("dominant_state_layered", "")).strip()
    supports = _support_list(layered)
    candidate_note = str(layered.get("candidate_note", "")).strip()

    acoustic_class = str(acoustic.get("acoustic_cause_class", "")).strip() if acoustic else ""
    fragment_class = str(acoustic.get("residual_fragmentation_class", "")).strip() if acoustic else ""
    body_ratio = _safe_float(acoustic.get("body_ratio")) if acoustic else 0.0
    trace_ratio = _safe_float(acoustic.get("trace_ratio")) if acoustic else 0.0
    internal_wave = str(acoustic.get("re_as_internal_wave", "")).strip() == "YES" if acoustic else False

    birth_count = _safe_int(lifecycle.get("birth_count")) if lifecycle else 0
    birth_like_count = _safe_int(lifecycle.get("birth_like_count")) if lifecycle else 0
    duration_frames = _safe_int(lifecycle.get("duration_frames")) if lifecycle else 0
    refined_kind = str(lifecycle.get("refined_lifecycle_kind", "")).strip() if lifecycle else ""

    shared_mode = str(shared.get("shared_mode", "")).strip() if shared else ""
    ownership_mode = str(shared.get("ownership_mode", "")).strip() if shared else ""

    attack_owner = ""
    sustain_owner = ""
    body_owner = ""
    field_owner = ""
    support_owners: list[str] = []
    role_pattern = ""
    confidence = "LOW"

    if acoustic_class == "LIKELY_HALL_OR_FIELD_TRACE":
        field_owner = "hall_or_field"
        support_owners = [x for x in supports if x in INSTRUMENTS]
        role_pattern = "FIELD_TRACE_EVENT"
        confidence = "MEDIUM"
    elif acoustic_class == "LIKELY_INSTRUMENT_BODY_RETURN":
        body_owner = dominant if dominant in INSTRUMENTS else "unknown_body"
        if dominant in INSTRUMENTS and dom_state in {"CLEAR_DOMINANT", "OWN_WINDOW_DOMINANT"}:
            sustain_owner = dominant
        support_owners = [x for x in supports if x in INSTRUMENTS]
        role_pattern = "BODY_RETURN_EVENT"
        confidence = "HIGH" if dominant in INSTRUMENTS else "LOW"
    elif acoustic_class == "LIKELY_INTERNAL_WAVE" or internal_wave or fragment_class == "INTERNAL_WAVE_HEAVY":
        sustain_owner = dominant if dominant in INSTRUMENTS else ""
        if body_ratio >= 0.35 and dominant in INSTRUMENTS:
            body_owner = dominant
        support_owners = [x for x in supports if x in INSTRUMENTS]
        role_pattern = "INTERNAL_WAVE_EVENT"
        confidence = "MEDIUM"
    elif ownership_mode == "PIANO_ATTACK_CELLO_SUSTAIN":
        attack_owner = "piano"
        sustain_owner = "cello"
        if "organ" in supports:
            support_owners.append("organ")
        role_pattern = "SHARED_ATTACK_SUSTAIN_EVENT"
        confidence = "HIGH"
    elif ownership_mode == "SHARED_SIMILAR_DURATION":
        attack_owner = "piano+cello"
        sustain_owner = "piano+cello"
        support_owners = [x for x in supports if x in INSTRUMENTS and x not in {"piano", "cello"}]
        role_pattern = "SHARED_CO_OWNED_EVENT"
        confidence = "MEDIUM"
    elif dominant in INSTRUMENTS:
        if birth_count > 0 or birth_like_count > 0:
            attack_owner = dominant
        if dominant in {"cello", "violin", "organ"}:
            sustain_owner = dominant
        elif dominant == "piano":
            if duration_frames >= 18 or dom_state in {"OWN_WINDOW_DOMINANT", "CLEAR_DOMINANT"}:
                sustain_owner = dominant

        if body_ratio >= 0.42:
            body_owner = dominant
        if trace_ratio >= 0.20 and not field_owner:
            field_owner = "mixed_field"

        if shared_mode in {"PIANO_ONLY_EXACT", "PIANO_ONLY_PITCHCLASS"} and dominant == "piano" and "cello" in supports:
            support_owners = [x for x in supports if x in INSTRUMENTS and x != "cello"]
            role_pattern = "PIANO_PRIMARY_WITH_FALSE_CELLO_SUPPORT"
            confidence = "HIGH"
        elif shared_mode in {"CELLO_ONLY_EXACT", "CELLO_ONLY_PITCHCLASS"} and dominant == "cello" and "piano" in supports:
            support_owners = [x for x in supports if x in INSTRUMENTS and x != "piano"]
            role_pattern = "CELLO_PRIMARY_WITH_FALSE_PIANO_SUPPORT"
            confidence = "HIGH"
        else:
            support_owners = [x for x in supports if x in INSTRUMENTS]
            if support_owners:
                role_pattern = "PRIMARY_WITH_SUPPORT_EVENT"
                confidence = "MEDIUM" if dom_state in {"MIXED_DOMINANT", "LEANING_DOMINANT"} else "HIGH"
            else:
                role_pattern = "PRIMARY_SINGLE_OWNER_EVENT"
                confidence = "HIGH" if dom_state in {"OWN_WINDOW_DOMINANT", "CLEAR_DOMINANT"} else "MEDIUM"
    else:
        field_owner = "unresolved_field"
        support_owners = [x for x in supports if x in INSTRUMENTS]
        role_pattern = "UNRESOLVED_FIELD_EVENT"
        confidence = "LOW"

    # Tighten sustain for short piano attacks.
    if attack_owner == "piano" and sustain_owner == "piano" and duration_frames <= 8 and body_owner == "":
        sustain_owner = ""
        if role_pattern == "PRIMARY_SINGLE_OWNER_EVENT":
            role_pattern = "PIANO_ATTACK_EVENT"

    # If a field trace dominates and no real attack is present, do not invent one.
    if field_owner in {"hall_or_field", "unresolved_field"} and birth_count == 0 and birth_like_count == 0:
        attack_owner = ""
        if sustain_owner and sustain_owner not in {"organ", "cello", "violin", "piano"}:
            sustain_owner = ""

    return {
        "candidate_note": candidate_note,
        "attack_owner": attack_owner,
        "sustain_owner": sustain_owner,
        "body_owner": body_owner,
        "field_owner": field_owner,
        "support_owners": _join(support_owners),
        "role_pattern": role_pattern,
        "role_confidence": confidence,
        "acoustic_cause_class": acoustic_class,
        "residual_fragmentation_class": fragment_class,
        "refined_lifecycle_kind": refined_kind,
        "shared_mode": shared_mode,
        "ownership_mode": ownership_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Map ensemble events to phased instrument behavior roles.")
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--lifecycle_csv", required=True)
    ap.add_argument("--acoustic_audit_csv", required=True)
    ap.add_argument("--shared_guard_csv")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    layered_rows = _read_csv(Path(args.layered_csv))
    lifecycle_index = _index_by_id(Path(args.lifecycle_csv))
    acoustic_index = _index_by_id(Path(args.acoustic_audit_csv))
    shared_index = _index_by_id(Path(args.shared_guard_csv)) if args.shared_guard_csv else {}

    out_rows: list[dict[str, str]] = []
    pattern_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    sustain_counts: Counter[str] = Counter()
    body_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    support_combo_counts: Counter[str] = Counter()

    for row in layered_rows:
        event_id = _safe_int(row.get("merged_event_id"))
        mapped = _assign_roles(
            layered=row,
            lifecycle=lifecycle_index.get(event_id),
            acoustic=acoustic_index.get(event_id),
            shared=shared_index.get(event_id),
        )
        out = {
            "merged_event_id": str(event_id),
            "birth_frame": str(row.get("birth_frame", "")),
            "dominant_instrument": str(row.get("dominant_instrument", "")).strip(),
            "dominant_state_layered": str(row.get("dominant_state_layered", "")).strip(),
            "support_combo_key": str(row.get("support_combo_key", "")).strip(),
            **mapped,
        }
        out_rows.append(out)

        pattern_counts[out["role_pattern"]] += 1
        attack_counts[out["attack_owner"] or "<NONE>"] += 1
        sustain_counts[out["sustain_owner"] or "<NONE>"] += 1
        body_counts[out["body_owner"] or "<NONE>"] += 1
        field_counts[out["field_owner"] or "<NONE>"] += 1
        support_combo_counts[out["support_owners"] or "<NONE>"] += 1

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as fh:
        fields = list(out_rows[0].keys()) if out_rows else [
            "merged_event_id", "candidate_note", "attack_owner", "sustain_owner",
            "body_owner", "field_owner", "support_owners", "role_pattern",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        "INSTRUMENT ROLE BEHAVIOR MAPPER",
        "=" * 72,
        f"input_events: {len(out_rows)}",
        "",
        "role_pattern_counts:",
    ]
    for key, value in pattern_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("attack_owner_counts:")
    for key, value in attack_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("sustain_owner_counts:")
    for key, value in sustain_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("body_owner_counts:")
    for key, value in body_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("field_owner_counts:")
    for key, value in field_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("support_owner_combos:")
    for key, value in support_combo_counts.most_common(12):
        lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "input_events": len(out_rows),
        "role_pattern_counts": dict(pattern_counts),
        "attack_owner_counts": dict(attack_counts),
        "sustain_owner_counts": dict(sustain_counts),
        "body_owner_counts": dict(body_counts),
        "field_owner_counts": dict(field_counts),
        "support_owner_combos": dict(support_combo_counts),
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
