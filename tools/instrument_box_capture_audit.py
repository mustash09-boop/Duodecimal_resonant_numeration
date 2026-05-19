from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_list(value: Any) -> list[Any]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return raw
    except Exception:
        return []
    return []


def _normalize_note(note: str) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if "'" in s:
        return s.split("'", 1)[0] + "'-"
    if s.endswith("-"):
        return s[:-1] + "'-"
    return s + "'-"


def _pitch_class(note: str) -> str:
    try:
        return _normalize_note(note).split(".", 1)[1].split("'", 1)[0]
    except Exception:
        return ""


def _register_band(note: str) -> str:
    octave = str(_normalize_note(note).split(".", 1)[0])
    if octave in {"6", "7"}:
        return "low_zone"
    if octave in {"8", "9"}:
        return "mid_zone"
    if octave in {"A", "B", "C", "D", "E", "F"}:
        return "high_zone"
    return "other_zone"


def _group_truth(midi_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in midi_rows:
        gid = str(row.get("onset_group", "")).strip()
        note = _normalize_note(row.get("expected_note_token", row.get("note_token", "")))
        if gid and note:
            out[gid].append(note)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit whether residual pitch-class capture looks like an instrument box effect: "
            "persistent background classes present across many onset-groups even when the truth class is different."
        )
    )
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--ownership-window-candidates-csv", required=True)
    ap.add_argument("--dsp-group-audit-csv", required=True)
    ap.add_argument("--residual-substitution-audit-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--ownership-topk", type=int, default=8)
    ap.add_argument("--dsp-topk", type=int, default=6)
    args = ap.parse_args()

    midi_rows = _load_csv(Path(args.midi_events_csv))
    ownership_rows = _load_csv(Path(args.ownership_window_candidates_csv))
    dsp_rows = _load_csv(Path(args.dsp_group_audit_csv))
    residual_rows = _load_csv(Path(args.residual_substitution_audit_csv))

    truth_by_group = _group_truth(midi_rows)
    all_groups = sorted(truth_by_group.keys(), key=lambda x: int(x))

    own_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ownership_rows:
        own_by_group[str(row.get("onset_group_id", "")).strip()].append(row)
    for gid in own_by_group:
        own_by_group[gid].sort(key=lambda r: _safe_int(r.get("candidate_rank"), 0))

    dsp_by_group = {
        str(row.get("onset_group", "")).strip(): row
        for row in dsp_rows
    }

    truth_group_classes: dict[str, set[str]] = {}
    zone_by_group: dict[str, str] = {}
    own_presence: dict[str, set[str]] = {}
    dsp_presence: dict[str, set[str]] = {}
    joint_presence: dict[str, set[str]] = {}
    own_background: dict[str, set[str]] = {}
    dsp_background: dict[str, set[str]] = {}
    joint_background: dict[str, set[str]] = {}

    for gid in all_groups:
        truth_notes = truth_by_group.get(gid, [])
        truth_pcs = {_pitch_class(x) for x in truth_notes if _pitch_class(x)}
        truth_group_classes[gid] = truth_pcs
        zone_by_group[gid] = _register_band(truth_notes[0]) if truth_notes else "other_zone"

        own_notes = [
            _normalize_note(r.get("note_token", ""))
            for r in own_by_group.get(gid, [])[: int(args.ownership_topk)]
            if _normalize_note(r.get("note_token", ""))
        ]
        own_pcs = {_pitch_class(x) for x in own_notes if _pitch_class(x)}

        dsp_chain_rows = _json_list(dsp_by_group.get(gid, {}).get("dsp_chain_rows_json", "[]"))
        dsp_notes = [
            _normalize_note(r.get("note_token", ""))
            for r in dsp_chain_rows[: int(args.dsp_topk)]
            if isinstance(r, dict) and _normalize_note(r.get("note_token", ""))
        ]
        dsp_pcs = {_pitch_class(x) for x in dsp_notes if _pitch_class(x)}

        own_presence[gid] = own_pcs
        dsp_presence[gid] = dsp_pcs
        joint_presence[gid] = own_pcs & dsp_pcs

        own_background[gid] = {pc for pc in own_pcs if pc not in truth_pcs}
        dsp_background[gid] = {pc for pc in dsp_pcs if pc not in truth_pcs}
        joint_background[gid] = {pc for pc in joint_presence[gid] if pc not in truth_pcs}

    capture_counter: Counter[str] = Counter()
    capture_zone_counter: dict[str, Counter[str]] = defaultdict(Counter)
    capture_truthpair_counter: Counter[str] = Counter()
    for row in residual_rows:
        captured = str(row.get("captured_pitch_class", "")).strip()
        truth_pc = str(row.get("truth_pitch_class", "")).strip()
        zone = str(row.get("register_band", "")).strip()
        if captured:
            capture_counter[captured] += 1
            capture_zone_counter[zone][captured] += 1
            if truth_pc:
                capture_truthpair_counter[f"{truth_pc}->{captured}"] += 1

    own_presence_counter: Counter[str] = Counter()
    dsp_presence_counter: Counter[str] = Counter()
    joint_presence_counter: Counter[str] = Counter()
    own_background_counter: Counter[str] = Counter()
    dsp_background_counter: Counter[str] = Counter()
    joint_background_counter: Counter[str] = Counter()
    own_background_zone_counter: dict[str, Counter[str]] = defaultdict(Counter)
    joint_background_zone_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for gid in all_groups:
        zone = zone_by_group.get(gid, "other_zone")
        for pc in own_presence[gid]:
            own_presence_counter[pc] += 1
        for pc in dsp_presence[gid]:
            dsp_presence_counter[pc] += 1
        for pc in joint_presence[gid]:
            joint_presence_counter[pc] += 1
        for pc in own_background[gid]:
            own_background_counter[pc] += 1
            own_background_zone_counter[zone][pc] += 1
        for pc in dsp_background[gid]:
            dsp_background_counter[pc] += 1
        for pc in joint_background[gid]:
            joint_background_counter[pc] += 1
            joint_background_zone_counter[zone][pc] += 1

    all_classes = sorted(
        set(capture_counter)
        | set(own_presence_counter)
        | set(dsp_presence_counter)
        | set(joint_background_counter)
    )

    audit_rows: list[dict[str, Any]] = []
    total_groups = max(len(all_groups), 1)
    zone_totals = Counter(zone_by_group.values())
    for pc in all_classes:
        own_cov = own_presence_counter[pc] / total_groups
        dsp_cov = dsp_presence_counter[pc] / total_groups
        joint_bg_cov = joint_background_counter[pc] / total_groups
        capture = capture_counter[pc]
        own_bg = own_background_counter[pc]
        joint_bg = joint_background_counter[pc]
        box_bias = joint_bg / max(capture, 1)
        audit_rows.append(
            {
                "pitch_class": pc,
                "capture_count": capture,
                "own_presence_groups": own_presence_counter[pc],
                "dsp_presence_groups": dsp_presence_counter[pc],
                "joint_presence_groups": joint_presence_counter[pc],
                "own_background_groups": own_bg,
                "dsp_background_groups": dsp_background_counter[pc],
                "joint_background_groups": joint_bg,
                "own_presence_ratio": f"{own_cov:.6f}",
                "dsp_presence_ratio": f"{dsp_cov:.6f}",
                "joint_background_ratio": f"{joint_bg_cov:.6f}",
                "background_to_capture_ratio": f"{box_bias:.6f}",
                "mid_joint_background": joint_background_zone_counter["mid_zone"][pc],
                "high_joint_background": joint_background_zone_counter["high_zone"][pc],
                "low_joint_background": joint_background_zone_counter["low_zone"][pc],
                "mid_capture": capture_zone_counter["mid_zone"][pc],
                "high_capture": capture_zone_counter["high_zone"][pc],
                "low_capture": capture_zone_counter["low_zone"][pc],
            }
        )

    audit_rows.sort(
        key=lambda r: (
            -int(r["capture_count"]),
            -int(r["joint_background_groups"]),
            str(r["pitch_class"]),
        )
    )

    out_audit = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_audit.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_audit.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "INSTRUMENT BOX CAPTURE AUDIT",
        "=" * 72,
        f"onset_groups                   : {len(all_groups)}",
        f"ownership_topk                 : {args.ownership_topk}",
        f"dsp_topk                       : {args.dsp_topk}",
        "",
        "TOP CAPTURE CLASSES VS BACKGROUND PERSISTENCE",
        "-" * 72,
    ]
    for row in audit_rows[:12]:
        lines.append(
            f"{row['pitch_class']:>2s}  capture={int(row['capture_count']):3d}  "
            f"joint_background={int(row['joint_background_groups']):3d}  "
            f"ratio={float(row['background_to_capture_ratio']):.3f}  "
            f"own_cov={float(row['own_presence_ratio']):.3f}  "
            f"dsp_cov={float(row['dsp_presence_ratio']):.3f}"
        )

    lines.extend(["", "JOINT BACKGROUND BY REGISTER BAND", "-" * 72])
    for zone in ["low_zone", "mid_zone", "high_zone"]:
        lines.append(f"{zone} (groups={zone_totals.get(zone, 0)})")
        for pc, count in joint_background_zone_counter[zone].most_common(10):
            lines.append(f"  {pc:>2s}: {count}")
        lines.append("")

    lines.extend(["TOP TRUTH->CAPTURE PAIRS", "-" * 72])
    for key, count in capture_truthpair_counter.most_common(16):
        lines.append(f"{key:>8s}: {count}")

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "instrument_box_capture_audit",
        "inputs": {
            "midi_events_csv": args.midi_events_csv,
            "ownership_window_candidates_csv": args.ownership_window_candidates_csv,
            "dsp_group_audit_csv": args.dsp_group_audit_csv,
            "residual_substitution_audit_csv": args.residual_substitution_audit_csv,
        },
        "result": {
            "capture_counter": dict(capture_counter),
            "own_presence_counter": dict(own_presence_counter),
            "dsp_presence_counter": dict(dsp_presence_counter),
            "joint_background_counter": dict(joint_background_counter),
            "capture_truthpair_counter": dict(capture_truthpair_counter),
            "joint_background_zone_counter": {
                zone: dict(counter) for zone, counter in joint_background_zone_counter.items()
            },
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
