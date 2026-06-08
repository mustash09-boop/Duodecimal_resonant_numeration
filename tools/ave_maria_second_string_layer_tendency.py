from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from music12.core.notation12 import bij12_to_int, parse_token, step_index0


REPORTS = Path(
    r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\10_reports_Ave_Maria"
)
MIDI_CSV = Path(
    r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\00_sources\midi\ave_maria_gounod_midi_events_with_parts_v1.csv"
)
ANCHOR_A4 = "9.A'-"
ANCHOR_A4_HZ = 440.0


def token_to_index(token: str) -> int:
    tok = parse_token(token)
    return (bij12_to_int(tok.oct) - 1) * 12 + step_index0(tok.step)


def token_to_hz(token: str) -> float:
    idx = token_to_index(token)
    a4_idx = token_to_index(ANCHOR_A4)
    return ANCHOR_A4_HZ * (2.0 ** ((idx - a4_idx) / 12.0))


def load_anchor_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("anchor_note_token", "")).strip()
            if not token:
                continue
            hz = token_to_hz(token)
            post_count = int(float(row.get("post_frame_count", 0) or 0))
            pre_count = int(float(row.get("pre_frame_count", 0) or 0))
            mean_strength = float(row.get("mean_post_field_strength", 0.0) or 0.0)
            weight = post_count * mean_strength
            persistent_new = int(float(row.get("persistent_new_anchor", 0) or 0))
            novel_after_split = int(float(row.get("novel_after_split", 0) or 0))
            rows.append(
                {
                    "anchor_note_token": token,
                    "pre_frame_count": pre_count,
                    "post_frame_count": post_count,
                    "persistent_new_anchor": persistent_new,
                    "novel_after_split": novel_after_split,
                    "mean_post_field_strength": mean_strength,
                    "approx_hz": hz,
                    "weight": weight,
                }
            )
    return rows


def load_midi_rows(path: Path, start_sec: float, end_sec: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row.get("start_sec", 0.0) or 0.0)
            if not (start_sec <= t <= end_sec):
                continue
            token = str(row.get("note_token", "")).strip()
            if not token:
                continue
            rows.append(
                {
                    "start_sec": t,
                    "track_name": str(row.get("track_name", "")).strip(),
                    "note_token": token,
                    "freq_hz": float(row.get("freq_hz", 0.0) or 0.0),
                    "duration_sec": float(row.get("duration_sec", 0.0) or 0.0),
                }
            )
    return rows


def classify_anchor_role(hz: float) -> str:
    if hz < 196.0:
        return "LOWER_SUPPORT_BAND"
    if hz < 220.0:
        return "BRIDGE_BAND"
    if hz < 330.0:
        return "UPPER_MAIN_BAND"
    return "HIGH_EXTENSION_BAND"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit whether the newly detected non-piano sustained layer is currently tracked as direct string fundamental or only as lower/shared support field."
    )
    ap.add_argument("--anchor-stats-csv", default=str(REPORTS / "ave_maria_11p95s_12p25s_new_sustained_string_layer_detector_v1__anchor_stats.csv"))
    ap.add_argument("--midi-csv", default=str(MIDI_CSV))
    ap.add_argument("--start-sec", type=float, default=11.95)
    ap.add_argument("--end-sec", type=float, default=12.25)
    ap.add_argument("--out-csv", default=str(REPORTS / "ave_maria_11p95s_12p25s_second_string_layer_tendency_v1.csv"))
    ap.add_argument("--out-summary-txt", default=str(REPORTS / "ave_maria_11p95s_12p25s_second_string_layer_tendency_v1.txt"))
    ap.add_argument("--out-meta-json", default=str(REPORTS / "ave_maria_11p95s_12p25s_second_string_layer_tendency_v1.json"))
    args = ap.parse_args()

    anchor_rows = load_anchor_rows(Path(args.anchor_stats_csv))
    midi_rows = load_midi_rows(Path(args.midi_csv), args.start_sec, args.end_sec)

    piano_tokens = {r["note_token"] for r in midi_rows if r["track_name"] == "Piano-Treble"}
    string_tokens = {r["note_token"] for r in midi_rows if r["track_name"] in {"Cello", "Violin"}}
    string_rows = [r for r in midi_rows if r["track_name"] in {"Cello", "Violin"}]

    selected = [
        r
        for r in anchor_rows
        if int(r["post_frame_count"]) >= 4
        and (int(r["persistent_new_anchor"]) == 1 or int(r["novel_after_split"]) == 1)
    ]

    out_rows: list[dict[str, object]] = []
    weight_total = sum(float(r["weight"]) for r in selected)
    weighted_hz_centroid = (
        sum(float(r["approx_hz"]) * float(r["weight"]) for r in selected) / weight_total
        if weight_total > 0.0
        else 0.0
    )

    band_weights = Counter()
    direct_piano_match_weight = 0.0
    direct_string_match_weight = 0.0
    direct_neither_match_weight = 0.0

    for row in selected:
        token = str(row["anchor_note_token"])
        hz = float(row["approx_hz"])
        role = classify_anchor_role(hz)
        band_weights[role] += float(row["weight"])

        if token in string_tokens:
            direct_kind = "DIRECT_STRING_MATCH"
            direct_string_match_weight += float(row["weight"])
        elif token in piano_tokens:
            direct_kind = "DIRECT_PIANO_MATCH"
            direct_piano_match_weight += float(row["weight"])
        else:
            direct_kind = "NO_DIRECT_WINDOW_MATCH"
            direct_neither_match_weight += float(row["weight"])

        out_rows.append(
            {
                "anchor_note_token": token,
                "pre_frame_count": int(row["pre_frame_count"]),
                "post_frame_count": int(row["post_frame_count"]),
                "persistent_new_anchor": int(row["persistent_new_anchor"]),
                "novel_after_split": int(row["novel_after_split"]),
                "mean_post_field_strength": f"{float(row['mean_post_field_strength']):.9f}",
                "approx_hz": f"{hz:.6f}",
                "weight": f"{float(row['weight']):.9f}",
                "register_role": role,
                "window_direct_match": direct_kind,
            }
        )

    out_rows.sort(key=lambda r: float(r["weight"]), reverse=True)

    upper_main_share = band_weights["UPPER_MAIN_BAND"] / weight_total if weight_total else 0.0
    lower_support_share = band_weights["LOWER_SUPPORT_BAND"] / weight_total if weight_total else 0.0
    bridge_share = band_weights["BRIDGE_BAND"] / weight_total if weight_total else 0.0
    high_extension_share = band_weights["HIGH_EXTENSION_BAND"] / weight_total if weight_total else 0.0
    piano_match_share = direct_piano_match_weight / weight_total if weight_total else 0.0
    string_match_share = direct_string_match_weight / weight_total if weight_total else 0.0
    neither_match_share = direct_neither_match_weight / weight_total if weight_total else 0.0

    if string_match_share > 0.25:
        verdict = "DIRECT_STRING_FUNDAMENTAL_TRACKING_PRESENT"
    elif upper_main_share >= 0.55 and lower_support_share >= 0.20 and string_match_share < 0.10:
        verdict = "UPPER_STRING_LIKE_ENTRANT_BUT_CURRENT_TRACKING_IS_MOSTLY_SHARED_OR_LOWER_SUPPORT"
    elif lower_support_share >= 0.55:
        verdict = "LOWER_SUPPORT_DOMINANT_STRING_LAYER_TRACKING"
    else:
        verdict = "STRING_LAYER_TRACKING_AMBIGUOUS"

    summary_lines = [
        "AVE MARIA SECOND STRING LAYER TENDENCY AUDIT",
        "=" * 72,
        f"window_sec: {args.start_sec:.3f} -> {args.end_sec:.3f}",
        f"selected_anchor_count: {len(selected)}",
        f"weighted_hz_centroid: {weighted_hz_centroid:.6f}",
        "",
        "weighted_register_shares:",
        f"  lower_support_band : {lower_support_share:.6f}",
        f"  bridge_band        : {bridge_share:.6f}",
        f"  upper_main_band    : {upper_main_share:.6f}",
        f"  high_extension     : {high_extension_share:.6f}",
        "",
        "window_direct_match_shares:",
        f"  direct_piano_match : {piano_match_share:.6f}",
        f"  direct_string_match: {string_match_share:.6f}",
        f"  no_direct_match    : {neither_match_share:.6f}",
        "",
        "window_reference_midi:",
    ]
    for row in string_rows:
        summary_lines.append(
            f"  {row['track_name']}: {row['note_token']}  hz={float(row['freq_hz']):.6f}  dur={float(row['duration_sec']):.6f}"
        )
    summary_lines.extend(
        [
            "",
            "top_selected_anchors:",
        ]
    )
    for row in out_rows[:8]:
        summary_lines.append(
            f"  {row['anchor_note_token']}: hz={row['approx_hz']}  role={row['register_role']}  match={row['window_direct_match']}  weight={row['weight']}"
        )
    summary_lines.extend(
        [
            "",
            f"verdict: {verdict}",
            "verdict_notes:",
            "  - This audit checks whether the post-split sustained layer is tracked as direct string fundamental or as lower/shared support field.",
            "  - Direct string note tracking is considered strong only when persistent post-split anchors coincide with string note tokens in the same window.",
            "  - If persistent anchors mostly match piano tokens or lower support bands, the string entrant is present but not yet isolated as direct fundamental.",
        ]
    )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_note_token",
                "pre_frame_count",
                "post_frame_count",
                "persistent_new_anchor",
                "novel_after_split",
                "mean_post_field_strength",
                "approx_hz",
                "weight",
                "register_role",
                "window_direct_match",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "ave_maria_second_string_layer_tendency",
        "inputs": {
            "anchor_stats_csv": args.anchor_stats_csv,
            "midi_csv": args.midi_csv,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
        },
        "result": {
            "selected_anchor_count": len(selected),
            "weighted_hz_centroid": weighted_hz_centroid,
            "upper_main_share": upper_main_share,
            "lower_support_share": lower_support_share,
            "direct_piano_match_share": piano_match_share,
            "direct_string_match_share": string_match_share,
            "no_direct_match_share": neither_match_share,
            "verdict": verdict,
        },
    }
    Path(args.out_meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
