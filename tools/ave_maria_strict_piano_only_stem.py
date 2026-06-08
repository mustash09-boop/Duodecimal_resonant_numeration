# -*- coding: ascii -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


FRAME_RATE = 60.0
ACTIVE_RATIO_THRESHOLD = 0.0195
STRONG_RATIO_THRESHOLD = 0.25


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _index_rows(path: Path, key: str) -> dict[int, dict[str, str]]:
    return {_safe_int(r.get(key)): r for r in _read_csv(path)}


def _split_set(raw: str) -> set[str]:
    raw = str(raw or "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split() if x.strip()}


def _weight_event(
    role_row: dict[str, str],
    layered_row: dict[str, str],
    guard_row: dict[str, str] | None,
) -> tuple[float, str]:
    attack = str(role_row.get("attack_owner", "")).strip()
    sustain = str(role_row.get("sustain_owner", "")).strip()
    body = str(role_row.get("body_owner", "")).strip()
    field = str(role_row.get("field_owner", "")).strip()
    support_role = _split_set(role_row.get("support_owners", ""))
    dominant = str(layered_row.get("dominant_instrument", "")).strip()
    support_layered = _split_set(layered_row.get("support_instruments", ""))
    role_pattern = str(role_row.get("role_pattern", "")).strip()
    window_alignment = str(layered_row.get("winner_window_alignment", "")).strip()
    dom_state = str(layered_row.get("dominant_state_layered", "")).strip()

    # Hard rejects first.
    if dominant in {"cello", "violin", "organ", "UNRESOLVED_FIELD"}:
        return 0.0, f"reject_dominant_{dominant or 'empty'}"
    if "cello" in support_layered or "violin" in support_layered:
        return 0.0, "reject_string_support_layered"
    if "cello" in support_role or "violin" in support_role:
        return 0.0, "reject_string_support_role"
    if sustain in {"cello", "violin"} or body in {"cello", "violin"}:
        return 0.0, "reject_string_owner"
    if field and field not in {"", "unknown_field", "unresolved_field"}:
        return 0.0, "reject_foreign_field"
    if guard_row is not None:
        shared_mode = str(guard_row.get("shared_mode", "")).strip()
        ownership_mode = str(guard_row.get("ownership_mode", "")).strip()
        if shared_mode in {
            "NO_DIRECT_PART_MATCH",
            "CELLO_ONLY_EXACT",
            "CELLO_ONLY_PITCHCLASS",
            "CELLO_EXACT_PIANO_PITCHCLASS",
        }:
            return 0.0, "reject_guard_shared_mode"
        if ownership_mode == "CELLO_DOMINANT_WITH_PIANO_SUPPORT":
            return 0.0, "reject_guard_cello_owner"

    # Positive piano cases.
    if role_pattern == "PIANO_ATTACK_EVENT" or attack == "piano":
        if dominant == "piano":
            return 1.0, "piano_attack_core"
        return 0.90, "piano_attack_non_dominant"

    if sustain == "piano":
        if dominant == "piano" and window_alignment == "WINNER_IN_OWN_TARGET_WINDOW":
            return 0.72, "piano_sustain_own_window"
        if dominant == "piano" and dom_state in {"CLEAR_DOMINANT", "OWN_WINDOW_DOMINANT"}:
            return 0.64, "piano_sustain_dominant"
        return 0.0, "reject_piano_sustain_weak"

    if dominant == "piano" and window_alignment == "WINNER_IN_OWN_TARGET_WINDOW":
        if role_pattern in {"PRIMARY_SINGLE_OWNER_EVENT", "PRIMARY_WITH_SUPPORT_EVENT"}:
            return 0.58, "piano_primary_window"
        return 0.44, "piano_window_trace"

    return 0.0, "reject_other"


def _build_frame_mask(
    role_index: dict[int, dict[str, str]],
    layered_index: dict[int, dict[str, str]],
    event_index: dict[int, dict[str, str]],
    guard_index: dict[int, dict[str, str]],
    tail_frames: int,
) -> tuple[np.ndarray, Counter[str], int]:
    max_end = 0
    for ev in event_index.values():
        max_end = max(max_end, _safe_int(ev.get("end_frame")))
    frame_sum = np.zeros(max_end + tail_frames + 4, dtype=np.float32)
    frame_peak = np.zeros_like(frame_sum)
    reason_counts: Counter[str] = Counter()
    used_events = 0

    for event_id, role_row in role_index.items():
        layered_row = layered_index.get(event_id)
        event_row = event_index.get(event_id)
        if layered_row is None or event_row is None:
            continue
        weight, reason = _weight_event(role_row, layered_row, guard_index.get(event_id))
        reason_counts[reason] += 1
        if weight <= 0.0:
            continue
        used_events += 1
        start = _safe_int(event_row.get("birth_frame"))
        end = _safe_int(event_row.get("end_frame"))
        attack_span = min(end, start + 4)
        for frame in range(start, end + 1):
            if frame <= attack_span:
                pos = (frame - start) / max(1, attack_span - start + 1)
                gain = weight * (0.90 + 0.12 * pos)
            else:
                gain = weight * 0.92
            frame_sum[frame] += float(gain)
            frame_peak[frame] = max(frame_peak[frame], float(gain))

        if str(role_row.get("sustain_owner", "")).strip() == "piano":
            for frame in range(end + 1, min(len(frame_sum), end + tail_frames + 1)):
                remain = 1.0 - ((frame - end) / max(1, tail_frames))
                gain = weight * 0.28 * max(0.0, remain)
                frame_sum[frame] += float(gain)
                frame_peak[frame] = max(frame_peak[frame], float(gain))

    if np.max(frame_sum) > 0:
        nonzero = frame_sum[frame_sum > 0]
        norm = frame_sum / max(1.0, float(np.quantile(nonzero, 0.97)))
    else:
        norm = frame_sum
    frame_mask = np.clip(np.maximum(frame_peak, norm), 0.0, 1.0)
    return frame_mask, reason_counts, used_events


def _frame_to_audio_mask(frame_mask: np.ndarray, sample_rate: int, sample_count: int, smoothing_ms: float) -> np.ndarray:
    frame_times = np.arange(len(frame_mask), dtype=np.float64) / FRAME_RATE
    sample_times = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    audio_mask = np.interp(sample_times, frame_times, frame_mask.astype(np.float64), left=0.0, right=0.0).astype(np.float32)
    window = max(1, int(round(sample_rate * smoothing_ms / 1000.0)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        audio_mask = np.convolve(audio_mask, kernel, mode="same")
    return np.clip(audio_mask, 0.0, 1.0)


def _write_mask_csv(path: Path, frame_mask: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_index", "time_sec", "mask"])
        writer.writeheader()
        for idx, value in enumerate(frame_mask):
            writer.writerow(
                {
                    "frame_index": idx,
                    "time_sec": f"{idx / FRAME_RATE:.6f}",
                    "mask": f"{float(value):.9f}",
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a strict piano-only Ave Maria stem by rejecting all string-attributed events.")
    ap.add_argument("--role_map_csv", required=True)
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--shared_guard_csv", required=True)
    ap.add_argument("--audio_wav", required=True)
    ap.add_argument("--tail_frames", type=int, default=6)
    ap.add_argument("--smoothing_ms", type=float, default=18.0)
    ap.add_argument("--out_mask_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    role_index = _index_rows(Path(args.role_map_csv), "merged_event_id")
    layered_index = _index_rows(Path(args.layered_csv), "merged_event_id")
    event_index = _index_rows(Path(args.events_csv), "merged_event_id")
    guard_index = _index_rows(Path(args.shared_guard_csv), "merged_event_id")

    frame_mask, reason_counts, used_events = _build_frame_mask(
        role_index=role_index,
        layered_index=layered_index,
        event_index=event_index,
        guard_index=guard_index,
        tail_frames=int(args.tail_frames),
    )

    audio, sr = sf.read(str(Path(args.audio_wav)))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    audio_mask = _frame_to_audio_mask(frame_mask, int(sr), len(audio), float(args.smoothing_ms))
    rendered = audio * audio_mask
    sf.write(str(Path(args.out_wav)), rendered, int(sr), subtype="PCM_16")
    _write_mask_csv(Path(args.out_mask_csv), frame_mask)

    active_frames = int(np.count_nonzero(frame_mask > ACTIVE_RATIO_THRESHOLD))
    strong_frames = int(np.count_nonzero(frame_mask > STRONG_RATIO_THRESHOLD))
    summary_lines = [
        "AVE MARIA STRICT PIANO ONLY STEM",
        "=" * 72,
        f"used_events: {used_events}",
        f"active_frames_gt_0_02: {active_frames}",
        f"strong_frames_gt_0_25: {strong_frames}",
        f"active_frame_ratio: {active_frames / max(1, len(frame_mask)):.6f}",
        f"strong_frame_ratio: {strong_frames / max(1, len(frame_mask)):.6f}",
        f"audio_active_ratio_gt_0_0195: {float(np.count_nonzero(audio_mask > ACTIVE_RATIO_THRESHOLD)) / max(1, len(audio_mask)):.6f}",
        f"audio_strong_ratio_gt_0_25: {float(np.count_nonzero(audio_mask > STRONG_RATIO_THRESHOLD)) / max(1, len(audio_mask)):.6f}",
        "",
        "reason_counts:",
    ]
    for key, value in reason_counts.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "used_events": used_events,
                "active_frames_gt_0_02": active_frames,
                "strong_frames_gt_0_25": strong_frames,
                "reason_counts": dict(reason_counts),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
