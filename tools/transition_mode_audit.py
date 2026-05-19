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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _build_families_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("frame_index"), 0)].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: _safe_int(r.get("family_rank"), 999999))
    return out


def _build_proto_by_start_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_safe_int(row.get("start_frame"), 0)].append(row)
    return out


def _previous_tail_notes(
    *,
    proto_by_start_frame: dict[int, list[dict[str, Any]]],
    current_proto_id: str,
    start_frame: int,
    note_token: str,
    rhythm_window_frames: int,
    min_confidence: float,
) -> list[str]:
    proto_pc = _pitch_class(note_token)
    notes: list[str] = []
    for frame in range(start_frame - rhythm_window_frames, start_frame):
        for row in proto_by_start_frame.get(frame, []):
            proto_id = str(row.get("proto_exciter_id", "")).strip()
            if proto_id == current_proto_id:
                continue
            other_note = str(row.get("rescue_group_dominant_note", "")).strip() or str(row.get("coarse_note", "")).strip()
            other_note = _normalize_note(other_note)
            if not other_note or _pitch_class(other_note) == proto_pc:
                continue
            if _safe_float(row.get("exciter_confidence"), 0.0) < min_confidence:
                continue
            if _safe_int(row.get("end_frame"), 0) < start_frame - rhythm_window_frames:
                continue
            notes.append(other_note)
    return sorted(set(notes))


def _score_balance(current_score: float, tail_score: float) -> float:
    denom = max(current_score + tail_score, 1e-9)
    return (current_score - tail_score) / denom


def _transition_mode(
    *,
    current_support_frames: int,
    tail_support_frames: int,
    overlap_frames: int,
    inspect_frames: int,
    mean_current_score: float,
    mean_tail_score: float,
) -> str:
    if tail_support_frames == 0:
        return "CLEAN_NEW_BIRTH"

    balance = _score_balance(mean_current_score, mean_tail_score)
    current_full = current_support_frames >= inspect_frames
    tail_full = tail_support_frames >= inspect_frames

    if overlap_frames >= 1 and current_support_frames >= 1:
        if balance >= 0.20:
            return "SHARED_CURRENT_LED_TRANSITION"
        if balance <= -0.20:
            return "SHARED_TAIL_LED_TRANSITION"
        return "SHARED_BALANCED_TRANSITION"

    if current_support_frames >= 1 and tail_support_frames >= 1:
        if current_full and balance >= 0.20:
            return "CURRENT_DOMINANT_WITH_TAIL_NEARBY"
        if tail_full and balance <= -0.20:
            return "TAIL_DOMINANT_CONFLICT"
        if balance >= 0.20:
            return "INTERLEAVED_CURRENT_LED"
        if balance <= -0.20:
            return "INTERLEAVED_TAIL_LED"
        return "INTERLEAVED_BALANCED"

    if current_support_frames == 0 and tail_support_frames >= 1:
        return "TAIL_DOMINANT_CONFLICT"
    if current_support_frames == 0 and tail_support_frames == 0:
        return "WEAK_UNRESOLVED_TRANSITION"
    if current_full:
        return "CURRENT_DOMINANT_WITH_TAIL_NEARBY"
    return "MIXED_UNRESOLVED_TRANSITION"


def _mode_explanation(mode: str) -> str:
    explanations = {
        "CLEAN_NEW_BIRTH": "Новая нота рождается без заметной активной опоры хвоста чужой ноты в ближайших кадрах.",
        "CURRENT_DOMINANT_WITH_TAIL_NEARBY": "Новая нота уже сильнее, но хвост предыдущей ноты всё ещё живёт рядом как инерция сцены.",
        "SHARED_CURRENT_LED_TRANSITION": "Есть реальное совместное присутствие, но новая нота уже ведёт переход.",
        "SHARED_BALANCED_TRANSITION": "Есть общая переходная зона, где новая нота и хвост предыдущей делят опору почти на равных.",
        "SHARED_TAIL_LED_TRANSITION": "Есть общая зона, но старый хвост пока сильнее и может затягивать новую ноту в ошибочную область.",
        "INTERLEAVED_CURRENT_LED": "Поддержка новой и старой ноты чередуется по кадрам, но новая в целом сильнее.",
        "INTERLEAVED_BALANCED": "Поддержка новой и старой ноты чередуется без явного победителя.",
        "INTERLEAVED_TAIL_LED": "Поддержка чередуется, но хвост старой ноты пока сильнее новой.",
        "TAIL_DOMINANT_CONFLICT": "Хвост предыдущей ноты доминирует и рождение новой ноты пока не удерживает свою опору.",
        "WEAK_UNRESOLVED_TRANSITION": "В окне начала нет достаточной опоры ни у новой ноты, ни у хвоста.",
        "MIXED_UNRESOLVED_TRANSITION": "Ситуация смешанная и пока не раскладывается в устойчивый тип перехода.",
    }
    return explanations.get(mode, "")


def _mode_sort_key(mode: str) -> tuple[int, str]:
    order = {
        "CLEAN_NEW_BIRTH": 10,
        "CURRENT_DOMINANT_WITH_TAIL_NEARBY": 20,
        "SHARED_CURRENT_LED_TRANSITION": 30,
        "SHARED_BALANCED_TRANSITION": 40,
        "SHARED_TAIL_LED_TRANSITION": 50,
        "INTERLEAVED_CURRENT_LED": 60,
        "INTERLEAVED_BALANCED": 70,
        "INTERLEAVED_TAIL_LED": 80,
        "TAIL_DOMINANT_CONFLICT": 90,
        "WEAK_UNRESOLVED_TRANSITION": 100,
        "MIXED_UNRESOLVED_TRANSITION": 110,
    }
    return (order.get(mode, 999), mode)


def _example_lines(rows: list[dict[str, Any]], limit: int = 3) -> list[str]:
    out: list[str] = []
    for row in rows[:limit]:
        proto = str(row.get("proto_exciter_id", "")).strip()
        note = str(row.get("coarse_note", "")).strip()
        frame = str(row.get("chain_start_frame", "")).strip()
        chain_status = str(row.get("chain_status", "")).strip()
        midi_status = str(row.get("midi_status", "")).strip()
        prev = str(row.get("previous_tail_notes_json", "")).strip()
        out.append(
            f"- proto {proto}, нота {note}, кадр {frame}, "
            f"статус цепи {chain_status or '(empty)'}, MIDI {midi_status or '(empty)'}, "
            f"хвосты {prev or '[]'}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit how transitions from previous note tails to new note births actually occur in the stream."
    )
    ap.add_argument("--primary-note-chains-csv", required=True)
    ap.add_argument("--chain-identity-audit-csv", required=True)
    ap.add_argument("--midi-identity-audit-csv", required=True)
    ap.add_argument("--proto-exciters-csv", required=True)
    ap.add_argument("--branch-analysis-csv", required=True)
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--out-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--out-map-md", default="")
    ap.add_argument("--inspect-frames", type=int, default=2)
    ap.add_argument("--top-family-rank", type=int, default=8)
    ap.add_argument("--rhythm-window-frames", type=int, default=2)
    ap.add_argument("--min-competing-confidence", type=float, default=0.58)
    args = ap.parse_args()

    chains = _load_csv(Path(args.primary_note_chains_csv))
    chain_audit = _load_csv(Path(args.chain_identity_audit_csv))
    midi_audit = _load_csv(Path(args.midi_identity_audit_csv))
    proto_rows = _load_csv(Path(args.proto_exciters_csv))
    branch_rows = _load_csv(Path(args.branch_analysis_csv))
    family_rows = _load_csv(Path(args.micro_families_csv))

    chain_audit_by_proto = {str(r.get("proto_exciter_id", "")).strip(): r for r in chain_audit}
    midi_status_by_proto = {
        str(r.get("best_chain_proto_exciter_id", "")).strip(): r
        for r in midi_audit
        if str(r.get("best_chain_proto_exciter_id", "")).strip()
    }
    branch_by_proto = {str(r.get("proto_exciter_id", "")).strip(): r for r in branch_rows}
    proto_by_start_frame = _build_proto_by_start_frame(proto_rows)
    families_by_frame = _build_families_by_frame(family_rows)

    audit_rows: list[dict[str, Any]] = []
    transition_counter: Counter[str] = Counter()
    transition_by_chain_status: dict[str, Counter[str]] = defaultdict(Counter)
    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chain in chains:
        proto_id = str(chain.get("proto_exciter_id", "")).strip()
        current_note = _normalize_note(chain.get("coarse_note", ""))
        start_frame = _safe_int(chain.get("chain_start_frame"), 0)
        previous_notes = _previous_tail_notes(
            proto_by_start_frame=proto_by_start_frame,
            current_proto_id=proto_id,
            start_frame=start_frame,
            note_token=current_note,
            rhythm_window_frames=int(args.rhythm_window_frames),
            min_confidence=float(args.min_competing_confidence),
        )

        current_support_frames = 0
        tail_support_frames = 0
        overlap_frames = 0
        current_scores: list[float] = []
        tail_scores: list[float] = []

        for frame in range(start_frame, start_frame + int(args.inspect_frames)):
            frame_rows = [
                fr for fr in families_by_frame.get(frame, [])
                if _safe_int(fr.get("family_rank"), 999999) <= int(args.top_family_rank)
            ]
            current_here = False
            tail_here = False
            for fr in frame_rows:
                note = _normalize_note(fr.get("family_root_note_micro", ""))
                score = _safe_float(fr.get("family_score"), 0.0)
                if note == current_note:
                    current_here = True
                    current_scores.append(score)
                if note in previous_notes:
                    tail_here = True
                    tail_scores.append(score)
            if current_here:
                current_support_frames += 1
            if tail_here:
                tail_support_frames += 1
            if current_here and tail_here:
                overlap_frames += 1

        mean_current_score = sum(current_scores) / max(len(current_scores), 1) if current_scores else 0.0
        mean_tail_score = sum(tail_scores) / max(len(tail_scores), 1) if tail_scores else 0.0
        balance = _score_balance(mean_current_score, mean_tail_score) if tail_support_frames > 0 else 1.0

        transition = _transition_mode(
            current_support_frames=current_support_frames,
            tail_support_frames=tail_support_frames,
            overlap_frames=overlap_frames,
            inspect_frames=int(args.inspect_frames),
            mean_current_score=mean_current_score,
            mean_tail_score=mean_tail_score,
        )
        transition_counter[transition] += 1

        chain_status = str(chain_audit_by_proto.get(proto_id, {}).get("status", "")).strip()
        midi_status = str(midi_status_by_proto.get(proto_id, {}).get("status", "")).strip()
        transition_by_chain_status[chain_status][transition] += 1
        branch_row = branch_by_proto.get(proto_id, {})

        row = {
            "proto_exciter_id": proto_id,
            "coarse_note": current_note,
            "chain_start_frame": start_frame,
            "branch_label": branch_row.get("branch_label", ""),
            "route_label": branch_row.get("route_label", ""),
            "chain_status": chain_status,
            "midi_status": midi_status,
            "previous_tail_notes_json": json.dumps(previous_notes, ensure_ascii=False),
            "current_support_frames": current_support_frames,
            "tail_support_frames": tail_support_frames,
            "overlap_frames": overlap_frames,
            "mean_current_score": f"{mean_current_score:.9f}" if current_scores else "",
            "mean_tail_score": f"{mean_tail_score:.9f}" if tail_scores else "",
            "support_balance": f"{balance:.9f}",
            "transition_mode": transition,
        }
        audit_rows.append(row)
        rows_by_mode[transition].append(row)

    out_csv = Path(args.out_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "TRANSITION MODE AUDIT",
        "=" * 72,
        f"chain_count : {len(audit_rows)}",
        "",
        "TRANSITION MODE COUNTS",
        "-" * 72,
    ]
    for key in sorted(transition_counter, key=_mode_sort_key):
        lines.append(f"{key:28s}: {transition_counter[key]}")
    lines.extend(["", "BY CHAIN STATUS", "-" * 72])
    for chain_status in sorted(transition_by_chain_status):
        lines.append(chain_status or "(empty)")
        for key in sorted(transition_by_chain_status[chain_status], key=_mode_sort_key):
            lines.append(f"  {key:26s}: {transition_by_chain_status[chain_status][key]}")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.out_map_md:
        out_map = Path(args.out_map_md)
        md_lines = [
            "# Карта переходов",
            "",
            f"- Всего цепей: `{len(audit_rows)}`",
            f"- Окно просмотра начала: `{int(args.inspect_frames)}` кадра",
            f"- Ритмическое окно поиска хвоста: `{int(args.rhythm_window_frames)}` кадра",
            "",
            "## Режимы",
            "",
        ]
        for mode in sorted(transition_counter, key=_mode_sort_key):
            md_lines.append(f"### {mode}")
            explanation = _mode_explanation(mode)
            if explanation:
                md_lines.append(explanation)
                md_lines.append("")
            md_lines.append(f"- Количество: `{transition_counter[mode]}`")
            exact_count = sum(1 for row in rows_by_mode[mode] if row.get('chain_status') == 'CHAIN_EXACT_IDENTITY')
            foreign_count = sum(1 for row in rows_by_mode[mode] if row.get('chain_status') == 'CHAIN_FOREIGN_CAPTURE')
            drift_count = sum(1 for row in rows_by_mode[mode] if row.get('chain_status') == 'CHAIN_STEP_DRIFT')
            md_lines.append(f"- Точных цепей: `{exact_count}`")
            md_lines.append(f"- Захватов чужой нотой: `{foreign_count}`")
            md_lines.append(f"- Съездов на соседнюю ступень: `{drift_count}`")
            examples = _example_lines(rows_by_mode[mode], limit=3)
            if examples:
                md_lines.append("- Примеры:")
                md_lines.extend(examples)
            md_lines.append("")
        out_map.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    meta = {
        "inputs": {
            "primary_note_chains_csv": args.primary_note_chains_csv,
            "chain_identity_audit_csv": args.chain_identity_audit_csv,
            "midi_identity_audit_csv": args.midi_identity_audit_csv,
            "proto_exciters_csv": args.proto_exciters_csv,
            "branch_analysis_csv": args.branch_analysis_csv,
            "micro_families_csv": args.micro_families_csv,
        },
        "parameters": {
            "inspect_frames": int(args.inspect_frames),
            "top_family_rank": int(args.top_family_rank),
            "rhythm_window_frames": int(args.rhythm_window_frames),
            "min_competing_confidence": float(args.min_competing_confidence),
        },
        "result": {
            "chain_count": len(audit_rows),
            "transition_mode_counts": dict(transition_counter),
            "by_chain_status": {k: dict(v) for k, v in transition_by_chain_status.items()},
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
