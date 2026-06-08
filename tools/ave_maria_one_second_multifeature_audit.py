from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
AVE_ROOT = PROJECT_ROOT / "Block001_data" / "Ave_Maria"
REPORTS_ROOT = AVE_ROOT / "10_reports_Ave_Maria"
MIDI_EVENTS_CSV = AVE_ROOT / "00_sources" / "midi" / "ave_maria_gounod_midi_events_with_parts_v1.csv"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_ROOT / name)


def list_or_empty(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x for x in text.split() if x]


def match_reference_parts(
    ref_df: pd.DataFrame,
    candidate_note: str,
    birth_frame: int,
    frame_tolerance: int = 12,
) -> tuple[str, int]:
    matched = ref_df[
        (ref_df["note12"] == candidate_note)
        & (ref_df["start_frame60"] >= birth_frame - frame_tolerance)
        & (ref_df["start_frame60"] <= birth_frame + frame_tolerance)
    ].copy()
    if len(matched) == 0:
        return "", 0
    parts = sorted(set(matched["track_name"].astype(str).tolist()))
    return " | ".join(parts), int(len(matched))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit one second of Ave Maria using the current multi-feature instrument model.")
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--duration-sec", type=float, default=1.0)
    ap.add_argument("--out-prefix", required=True, help="Output prefix without extension")
    args = ap.parse_args()

    start_frame = int(round(args.start_sec * 60.0))
    end_frame = int(round((args.start_sec + args.duration_sec) * 60.0))

    layered = load_csv("ave_maria_multi_instrument_layered_assignment_v1.csv")
    role_map = load_csv("ave_maria_instrument_role_behavior_map_v1.csv")
    hall_body = load_csv("ave_maria_legacy_hall_body_reexcitation_audit_v1.csv")
    midi_ref = pd.read_csv(MIDI_EVENTS_CSV)

    layered_sub = layered[(layered["birth_frame"] >= start_frame) & (layered["birth_frame"] < end_frame)].copy()
    role_sub = role_map[(role_map["birth_frame"] >= start_frame) & (role_map["birth_frame"] < end_frame)].copy()
    hall_sub = hall_body[(hall_body["birth_frame"] >= start_frame) & (hall_body["birth_frame"] < end_frame)].copy()
    ref_sub = midi_ref[(midi_ref["start_frame60"] < end_frame) & (midi_ref["end_frame60"] >= start_frame)].copy()

    merged = layered_sub.merge(
        role_sub[
            [
                "merged_event_id",
                "attack_owner",
                "sustain_owner",
                "body_owner",
                "field_owner",
                "support_owners",
                "role_pattern",
                "role_confidence",
                "acoustic_cause_class",
                "residual_fragmentation_class",
                "refined_lifecycle_kind",
            ]
        ],
        on="merged_event_id",
        how="left",
        suffixes=("", "_role"),
    )

    merged = merged.merge(
        hall_sub[
            [
                "merged_event_id",
                "end_frame",
                "duration_frames",
                "frame_count",
                "mean_score",
                "max_score",
                "birth_score",
                "final_score",
                "same_note_ratio",
                "body_ratio",
                "trace_ratio",
                "relative_energy_span",
                "birth_count",
                "re_excitation_count",
                "active_body_count",
                "sustain_body_count",
                "response_trace_count",
                "decay_trace_count",
            ]
        ],
        on="merged_event_id",
        how="left",
    )

    ref_part_rows = []
    for _, row in merged.iterrows():
        parts, count = match_reference_parts(ref_sub, str(row["candidate_note"]), int(row["birth_frame"]))
        ref_part_rows.append(
            {
                "merged_event_id": int(row["merged_event_id"]),
                "matched_reference_parts": parts,
                "matched_reference_count": count,
                "has_exact_reference_match": "YES" if count > 0 else "NO",
            }
        )
    ref_match_df = pd.DataFrame(ref_part_rows)
    if len(ref_match_df):
        merged = merged.merge(ref_match_df, on="merged_event_id", how="left")

    merged["support_instruments"] = merged["support_instruments"].fillna("")
    merged["support_owners"] = merged["support_owners"].fillna("")
    merged["attack_owner"] = merged["attack_owner"].fillna("")
    merged["sustain_owner"] = merged["sustain_owner"].fillna("")
    merged["body_owner"] = merged["body_owner"].fillna("")
    merged["field_owner"] = merged["field_owner"].fillna("")

    out_csv = Path(f"{args.out_prefix}.csv")
    out_txt = Path(f"{args.out_prefix}.txt")
    out_json = Path(f"{args.out_prefix}.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    merged = merged.sort_values(["birth_frame", "merged_event_id"]).reset_index(drop=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ref_track_counts = ref_sub.groupby("track_name").size().sort_values(ascending=False)
    model_dom_counts = merged.groupby("dominant_instrument").size().sort_values(ascending=False) if len(merged) else pd.Series(dtype=int)
    role_counts = merged.groupby("role_pattern").size().sort_values(ascending=False) if len(merged) else pd.Series(dtype=int)

    own_window = int((merged["winner_window_alignment"] == "WINNER_IN_OWN_TARGET_WINDOW").sum()) if len(merged) else 0
    outside_window = int((merged["winner_window_alignment"] == "WINNER_OUTSIDE_TARGET_WINDOW").sum()) if len(merged) else 0
    exact_ref = int((merged["has_exact_reference_match"] == "YES").sum()) if len(merged) else 0
    piano_attack = int((merged["role_pattern"] == "PIANO_ATTACK_EVENT").sum()) if len(merged) else 0
    ambiguous = int((merged["winner_state"] == "AMBIGUOUS").sum()) if len(merged) else 0
    mixed_dom = int((merged["dominant_state_layered"] == "MIXED_DOMINANT").sum()) if len(merged) else 0
    organ_dom = int((merged["dominant_instrument"] == "organ").sum()) if len(merged) else 0

    verdict_lines = []
    if own_window >= 2 and piano_attack >= 1:
        verdict_lines.append("Пиано-атаки в окне частично отделяются: есть собственные on-target события рояля.")
    else:
        verdict_lines.append("Чистого отделения пиано-атаки в окне пока мало.")

    if organ_dom > 0 and "Organ" not in " ".join(ref_track_counts.index.astype(str).tolist()):
        verdict_lines.append("Модель всё ещё порождает ложный organ-like слой там, где в MIDI-референсе орган не активен.")

    if ambiguous + mixed_dom >= 2:
        verdict_lines.append("Большая часть сложных событий остаётся в зоне shared ownership, а не single-owner separation.")

    if exact_ref <= max(1, len(merged) // 3):
        verdict_lines.append("Точное совпадение по note+time с MIDI у модельных событий низкое; модель пока лучше различает роли, чем ноты партий один-к-одному.")

    if not verdict_lines:
        verdict_lines.append("В этом окне текущая модель уже даёт приемлемое одиночное разделение.")

    lines: list[str] = []
    lines.append("AVE MARIA ONE-SECOND MULTI-FEATURE AUDIT")
    lines.append("=" * 72)
    lines.append(f"window_sec: {args.start_sec:.3f} -> {args.start_sec + args.duration_sec:.3f}")
    lines.append(f"window_frames60: {start_frame} -> {end_frame}")
    lines.append("")
    lines.append(f"reference_midi_events_in_window: {len(ref_sub)}")
    lines.append("reference_track_counts:")
    for name, count in ref_track_counts.items():
        lines.append(f"  {name}: {int(count)}")
    lines.append("")
    lines.append(f"model_events_born_in_window: {len(merged)}")
    lines.append(f"model_exact_reference_matches: {exact_ref}")
    lines.append(f"model_winner_in_own_target_window: {own_window}")
    lines.append(f"model_winner_outside_target_window: {outside_window}")
    lines.append(f"model_ambiguous_winners: {ambiguous}")
    lines.append(f"model_mixed_dominant_events: {mixed_dom}")
    lines.append(f"model_piano_attack_events: {piano_attack}")
    lines.append("")
    lines.append("model_dominant_instrument_counts:")
    for name, count in model_dom_counts.items():
        lines.append(f"  {name}: {int(count)}")
    lines.append("")
    lines.append("model_role_pattern_counts:")
    for name, count in role_counts.items():
        lines.append(f"  {name}: {int(count)}")
    lines.append("")
    lines.append("verdict:")
    for line in verdict_lines:
        lines.append(f"  - {line}")
    lines.append("")
    lines.append("event_rows_csv:")
    lines.append(f"  {out_csv}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "window_sec": [args.start_sec, args.start_sec + args.duration_sec],
                "window_frames60": [start_frame, end_frame],
                "reference_midi_events_in_window": int(len(ref_sub)),
                "reference_track_counts": {str(k): int(v) for k, v in ref_track_counts.items()},
                "model_events_born_in_window": int(len(merged)),
                "model_exact_reference_matches": exact_ref,
                "model_winner_in_own_target_window": own_window,
                "model_winner_outside_target_window": outside_window,
                "model_ambiguous_winners": ambiguous,
                "model_mixed_dominant_events": mixed_dom,
                "model_piano_attack_events": piano_attack,
                "model_dominant_instrument_counts": {str(k): int(v) for k, v in model_dom_counts.items()},
                "model_role_pattern_counts": {str(k): int(v) for k, v in role_counts.items()},
                "verdict": verdict_lines,
                "rows_csv": str(out_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"WROTE {out_csv}")
    print(f"WROTE {out_txt}")
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()

