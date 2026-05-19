from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _normalize_note(token: str) -> str:
    return token.replace("'", "").replace('"', "").strip()


def _pitchclass(token: str) -> str:
    token = _normalize_note(token)
    if "." not in token:
        return token
    return token.split(".", 1)[1]


def _group_midi_piano(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for row in rows:
        if not str(row.get("track_name", "")).startswith("Piano"):
            continue
        gid = str(row.get("onset_group", "")).strip()
        if gid not in groups:
            groups[gid] = {
                "midi_onset_group": gid,
                "start_frame60": _safe_int(row.get("start_frame60")),
                "note_set": set(),
                "pitchclass_set": set(),
                "event_count": 0,
            }
        g = groups[gid]
        note = _normalize_note(str(row.get("note12", "")).strip())
        g["note_set"].add(note)
        g["pitchclass_set"].add(_pitchclass(note))
        g["event_count"] = int(g["event_count"]) + 1
        g["start_frame60"] = min(int(g["start_frame60"]), _safe_int(row.get("start_frame60")))
    ordered = sorted(groups.values(), key=lambda x: (int(x["start_frame60"]), int(x["midi_onset_group"])))
    return ordered


def _group_real_piano(layered_rows: list[dict[str, str]], max_gap_frames: int) -> list[dict[str, object]]:
    filtered = []
    for row in layered_rows:
        if str(row.get("dominant_instrument", "")).strip() != "piano":
            continue
        if str(row.get("dominant_state_layered", "")).strip() == "NO_STRUCTURAL_OWNER":
            continue
        if str(row.get("piano_window", "")).strip() not in {"TARGET_ONLY_WINDOW", "MIXED_WINDOW"}:
            continue
        filtered.append(row)
    filtered.sort(key=lambda r: _safe_int(r.get("birth_frame")))

    groups: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for row in filtered:
        birth = _safe_int(row.get("birth_frame"))
        note = _normalize_note(str(row.get("candidate_note", "")).strip())
        if current is None or birth > int(current["end_frame"]) + max_gap_frames:
            current = {
                "real_group_id": len(groups) + 1,
                "anchor_frame": birth,
                "start_frame": birth,
                "end_frame": birth,
                "note_set": {note},
                "pitchclass_set": {_pitchclass(note)},
                "event_ids": [str(row.get("merged_event_id", "")).strip()],
                "state_counts": Counter([str(row.get("dominant_state_layered", "")).strip()]),
                "support_counts": Counter([str(row.get("support_combo_key", "")).strip() or "<NONE>"]),
            }
            groups.append(current)
        else:
            current["end_frame"] = birth
            current["note_set"].add(note)
            current["pitchclass_set"].add(_pitchclass(note))
            current["event_ids"].append(str(row.get("merged_event_id", "")).strip())
            current["state_counts"].update([str(row.get("dominant_state_layered", "")).strip()])
            current["support_counts"].update([str(row.get("support_combo_key", "")).strip() or "<NONE>"])
    return groups


def _group_score(midi_g: dict[str, object], real_g: dict[str, object]) -> tuple[float, int, int]:
    midi_notes = set(midi_g["note_set"])
    real_notes = set(real_g["note_set"])
    midi_pc = set(midi_g["pitchclass_set"])
    real_pc = set(real_g["pitchclass_set"])
    inter_note = len(midi_notes & real_notes)
    inter_pc = len(midi_pc & real_pc)
    union_note = len(midi_notes | real_notes)
    union_pc = len(midi_pc | real_pc)
    if inter_note > 0:
        score = inter_note * 4.0 - max(0, union_note - inter_note) * 0.35
    elif inter_pc > 0:
        score = inter_pc * 1.5 - max(0, union_pc - inter_pc) * 0.25
    else:
        score = -1.75
    return score, inter_note, inter_pc


def _align_groups(
    midi_groups: list[dict[str, object]],
    real_groups: list[dict[str, object]],
    gap_penalty: float,
) -> list[dict[str, object]]:
    m = len(midi_groups)
    n = len(real_groups)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        bt[i][0] = "up"
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        bt[0][j] = "left"

    score_cache: dict[tuple[int, int], tuple[float, int, int]] = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_cache[(i, j)] = _group_score(midi_groups[i - 1], real_groups[j - 1])
            match_score = dp[i - 1][j - 1] + score_cache[(i, j)][0]
            up_score = dp[i - 1][j] + gap_penalty
            left_score = dp[i][j - 1] + gap_penalty
            best = max(match_score, up_score, left_score)
            dp[i][j] = best
            if best == match_score:
                bt[i][j] = "diag"
            elif best == up_score:
                bt[i][j] = "up"
            else:
                bt[i][j] = "left"

    pairs: list[dict[str, object]] = []
    i, j = m, n
    while i > 0 or j > 0:
        step = bt[i][j]
        if step == "diag":
            score, inter_note, inter_pc = score_cache[(i, j)]
            if inter_note > 0 or (score > 0.55 and inter_pc > 0):
                mg = midi_groups[i - 1]
                rg = real_groups[j - 1]
                pairs.append(
                    {
                        "midi_onset_group": mg["midi_onset_group"],
                        "midi_start_frame60": int(mg["start_frame60"]),
                        "midi_note_set": " ".join(sorted(mg["note_set"])),
                        "real_group_id": int(rg["real_group_id"]),
                        "real_anchor_frame": int(rg["anchor_frame"]),
                        "real_note_set": " ".join(sorted(rg["note_set"])),
                        "match_score": score,
                        "exact_note_overlap": inter_note,
                        "pitchclass_overlap": inter_pc,
                    }
                )
            i -= 1
            j -= 1
        elif step == "up":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Align Ave Maria piano MIDI onset groups to real piano event groups.")
    ap.add_argument("--midi_parts_csv", required=True)
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--out_alignment_csv", required=True)
    ap.add_argument("--out_anchor_csv", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--real_group_gap_frames", type=int, default=3)
    ap.add_argument("--gap_penalty", type=float, default=-0.75)
    args = ap.parse_args()

    midi_rows = _read_csv(Path(args.midi_parts_csv))
    layered_rows = _read_csv(Path(args.layered_csv))
    midi_groups = _group_midi_piano(midi_rows)
    real_groups = _group_real_piano(layered_rows, max_gap_frames=args.real_group_gap_frames)
    alignment = _align_groups(midi_groups, real_groups, gap_penalty=args.gap_penalty)

    anchors: list[dict[str, str]] = []
    prev_midi = -10**9
    prev_real = -10**9
    exact_count = 0
    for pair in alignment:
        midi_frame = int(pair["midi_start_frame60"])
        real_frame = int(pair["real_anchor_frame"])
        if midi_frame <= prev_midi or real_frame <= prev_real:
            continue
        prev_midi = midi_frame
        prev_real = real_frame
        if int(pair["exact_note_overlap"]) > 0:
            exact_count += 1
        anchors.append(
            {
                "anchor_index": str(len(anchors) + 1),
                "midi_onset_group": str(pair["midi_onset_group"]),
                "midi_start_frame60": str(midi_frame),
                "real_group_id": str(pair["real_group_id"]),
                "real_anchor_frame": str(real_frame),
                "match_score": f"{float(pair['match_score']):.6f}",
                "exact_note_overlap": str(pair["exact_note_overlap"]),
                "pitchclass_overlap": str(pair["pitchclass_overlap"]),
                "midi_note_set": str(pair["midi_note_set"]),
                "real_note_set": str(pair["real_note_set"]),
            }
        )

    with Path(args.out_alignment_csv).open("w", encoding="utf-8", newline="") as fh:
        fields = list(alignment[0].keys()) if alignment else [
            "midi_onset_group", "midi_start_frame60", "midi_note_set",
            "real_group_id", "real_anchor_frame", "real_note_set",
            "match_score", "exact_note_overlap", "pitchclass_overlap",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(alignment)

    with Path(args.out_anchor_csv).open("w", encoding="utf-8", newline="") as fh:
        fields = list(anchors[0].keys()) if anchors else [
            "anchor_index", "midi_onset_group", "midi_start_frame60",
            "real_group_id", "real_anchor_frame", "match_score",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(anchors)

    summary_lines = [
        "PIANO MIDI TO REAL TIME ALIGNMENT",
        "=" * 72,
        f"midi_piano_onset_groups: {len(midi_groups)}",
        f"real_piano_groups: {len(real_groups)}",
        f"alignment_pairs: {len(alignment)}",
        f"usable_monotonic_anchors: {len(anchors)}",
        f"exact_note_anchor_count: {exact_count}",
    ]
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "midi_piano_onset_groups": len(midi_groups),
                "real_piano_groups": len(real_groups),
                "alignment_pairs": len(alignment),
                "usable_monotonic_anchors": len(anchors),
                "exact_note_anchor_count": exact_count,
                "real_group_gap_frames": args.real_group_gap_frames,
                "gap_penalty": args.gap_penalty,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
