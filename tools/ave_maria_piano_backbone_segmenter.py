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
ACTIVE_THRESHOLD = 0.0195


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _index_rows(path: Path, key: str) -> dict[int, dict[str, str]]:
    return {_safe_int(row.get(key)): row for row in _read_csv(path)}


def _split_set(raw: str | None) -> set[str]:
    raw = str(raw or "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split() if part.strip()}


def _event_profile(
    role_row: dict[str, str],
    layered_row: dict[str, str],
    guard_row: dict[str, str] | None,
) -> tuple[float, float, float, str]:
    attack = str(role_row.get("attack_owner", "")).strip()
    sustain = str(role_row.get("sustain_owner", "")).strip()
    body = str(role_row.get("body_owner", "")).strip()
    dominant = str(layered_row.get("dominant_instrument", "")).strip()
    role_pattern = str(role_row.get("role_pattern", "")).strip()
    support_role = _split_set(role_row.get("support_owners"))
    support_layered = _split_set(layered_row.get("support_instruments"))
    attack_first_owner = str(layered_row.get("attack_first_owner", "")).strip()
    late_owner = str(layered_row.get("late_owner_after_attack", "")).strip()
    window_alignment = str(layered_row.get("winner_window_alignment", "")).strip()
    dominant_state = str(layered_row.get("dominant_state_layered", "")).strip()
    shared_mode = str((guard_row or {}).get("shared_mode", "")).strip()
    ownership_mode = str((guard_row or {}).get("ownership_mode", "")).strip()

    if dominant in {"cello", "violin"}:
        return 0.0, 0.0, 0.0, f"reject_dominant_{dominant}"
    if sustain in {"cello", "violin"} or body in {"cello", "violin"}:
        return 0.0, 0.0, 0.0, "reject_string_owner"
    if "cello" in support_role and attack_first_owner != "piano" and late_owner != "piano":
        return 0.0, 0.0, 0.0, "reject_role_string_support"
    if "violin" in support_role and attack_first_owner != "piano" and late_owner != "piano":
        return 0.0, 0.0, 0.0, "reject_role_string_support"
    if ownership_mode == "CELLO_DOMINANT_WITH_PIANO_SUPPORT":
        return 0.0, 0.0, 0.0, "reject_guard_cello_ownership"
    if shared_mode in {"CELLO_ONLY_EXACT", "CELLO_ONLY_PITCHCLASS"}:
        return 0.0, 0.0, 0.0, "reject_guard_cello_only"

    # Strongest case: explicit piano attack or early piano owner.
    if role_pattern == "PIANO_ATTACK_EVENT" or attack == "piano" or attack_first_owner == "piano":
        attack_gain = 1.00
        body_gain = 0.72 if sustain == "piano" or late_owner == "piano" else 0.56
        tail_gain = 0.46
        if dominant in {"organ", "UNRESOLVED_FIELD"}:
            return attack_gain, 0.54, 0.40, "piano_attack_rescued_from_late_mix"
        return attack_gain, body_gain, tail_gain, "piano_attack_backbone"

    # Sustained piano that should preserve continuity.
    if sustain == "piano" or late_owner == "piano":
        if dominant == "piano":
            if window_alignment == "WINNER_IN_OWN_TARGET_WINDOW":
                return 0.68, 0.74, 0.48, "piano_sustain_own_window"
            if dominant_state in {"OWN_WINDOW_DOMINANT", "CLEAR_DOMINANT", "LEANING_DOMINANT"}:
                return 0.60, 0.68, 0.42, "piano_sustain_dominant"
        if dominant in {"organ", "UNRESOLVED_FIELD"} and "piano" in support_layered:
            return 0.52, 0.60, 0.38, "piano_sustain_rescued_from_late_mix"
        return 0.44, 0.50, 0.34, "weak_piano_sustain"

    # Piano body continuation may still be needed for continuity, but weaker.
    if body == "piano" or dominant == "piano":
        return 0.36, 0.44, 0.30, "piano_body_continuation"

    return 0.0, 0.0, 0.0, "reject_other"


def _paint_event(mask: np.ndarray, start: int, end: int, attack_gain: float, body_gain: float, tail_gain: float, tail_frames: int) -> None:
    if end < start:
        return
    dur = max(1, end - start + 1)
    attack_len = min(dur, 5)
    for frame in range(start, end + 1):
        pos = frame - start
        if pos < attack_len:
            local = attack_gain - (attack_gain - body_gain) * (pos / max(1, attack_len - 1))
        else:
            decay_pos = (pos - attack_len) / max(1, dur - attack_len)
            local = body_gain * (0.96 - 0.12 * decay_pos)
        mask[frame] = max(mask[frame], float(local))
    for frame in range(end + 1, min(len(mask), end + tail_frames + 1)):
        remain = 1.0 - ((frame - end) / max(1, tail_frames))
        local = tail_gain * max(0.0, remain)
        mask[frame] = max(mask[frame], float(local))


def _bridge_short_gaps(mask: np.ndarray, bridge_frames: int) -> np.ndarray:
    out = mask.copy()
    active = out > ACTIVE_THRESHOLD
    idx = 0
    while idx < len(out):
        if active[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < len(out) and not active[idx]:
            idx += 1
        gap_end = idx - 1
        gap_len = gap_end - gap_start + 1
        if gap_len > bridge_frames:
            continue
        left = out[gap_start - 1] if gap_start > 0 else 0.0
        right = out[idx] if idx < len(out) else 0.0
        if left <= 0.0 or right <= 0.0:
            continue
        fill_peak = min(left, right) * 0.72
        for pos, frame in enumerate(range(gap_start, gap_end + 1), start=1):
            mix = pos / max(1, gap_len + 1)
            interp = left * (1.0 - mix) + right * mix
            out[frame] = max(out[frame], float(max(fill_peak, interp * 0.82)))
    return np.clip(out, 0.0, 1.0)


def _build_backbone_mask(
    role_index: dict[int, dict[str, str]],
    layered_index: dict[int, dict[str, str]],
    event_index: dict[int, dict[str, str]],
    guard_index: dict[int, dict[str, str]],
    tail_frames: int,
    bridge_frames: int,
) -> tuple[np.ndarray, Counter[str], int]:
    max_end = 0
    for event_row in event_index.values():
        max_end = max(max_end, _safe_int(event_row.get("end_frame")))
    mask = np.zeros(max_end + tail_frames + bridge_frames + 8, dtype=np.float32)
    reasons: Counter[str] = Counter()
    used_events = 0

    for event_id, event_row in event_index.items():
        role_row = role_index.get(event_id)
        layered_row = layered_index.get(event_id)
        if role_row is None or layered_row is None:
            continue
        attack_gain, body_gain, tail_gain, reason = _event_profile(role_row, layered_row, guard_index.get(event_id))
        reasons[reason] += 1
        if max(attack_gain, body_gain, tail_gain) <= 0.0:
            continue
        used_events += 1
        start = _safe_int(event_row.get("birth_frame"))
        end = _safe_int(event_row.get("end_frame"))
        _paint_event(mask, start, end, attack_gain, body_gain, tail_gain, tail_frames)

    mask = _bridge_short_gaps(mask, bridge_frames)
    return mask, reasons, used_events


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
    ap = argparse.ArgumentParser(description="Build a continuous piano backbone mask for Ave Maria before late shared-event collapse.")
    ap.add_argument("--role_map_csv", required=True)
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--shared_guard_csv", required=True)
    ap.add_argument("--audio_wav", required=True)
    ap.add_argument("--tail_frames", type=int, default=12)
    ap.add_argument("--bridge_frames", type=int, default=10)
    ap.add_argument("--smoothing_ms", type=float, default=24.0)
    ap.add_argument("--out_mask_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    role_index = _index_rows(Path(args.role_map_csv), "merged_event_id")
    layered_index = _index_rows(Path(args.layered_csv), "merged_event_id")
    event_index = _index_rows(Path(args.events_csv), "merged_event_id")
    guard_index = _index_rows(Path(args.shared_guard_csv), "merged_event_id")

    frame_mask, reasons, used_events = _build_backbone_mask(
        role_index=role_index,
        layered_index=layered_index,
        event_index=event_index,
        guard_index=guard_index,
        tail_frames=int(args.tail_frames),
        bridge_frames=int(args.bridge_frames),
    )

    audio, sample_rate = sf.read(str(Path(args.audio_wav)))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    audio_mask = _frame_to_audio_mask(frame_mask, int(sample_rate), len(audio), float(args.smoothing_ms))
    rendered = audio * audio_mask

    sf.write(str(Path(args.out_wav)), rendered, int(sample_rate), subtype="PCM_16")
    _write_mask_csv(Path(args.out_mask_csv), frame_mask)

    active_frames = int(np.count_nonzero(frame_mask > ACTIVE_THRESHOLD))
    audio_active = int(np.count_nonzero(audio_mask > ACTIVE_THRESHOLD))
    summary_lines = [
        "AVE MARIA PIANO BACKBONE SEGMENTER",
        "=" * 72,
        f"used_events: {used_events}",
        f"frame_count: {len(frame_mask)}",
        f"active_frames_gt_0_0195: {active_frames}",
        f"active_frame_ratio: {active_frames / max(1, len(frame_mask)):.6f}",
        f"audio_active_ratio_gt_0_0195: {audio_active / max(1, len(audio_mask)):.6f}",
        f"mean_frame_mask: {float(np.mean(frame_mask)):.6f}",
        f"max_frame_mask: {float(np.max(frame_mask)):.6f}",
        "",
        "reason_counts:",
    ]
    for key, value in reasons.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "used_events": used_events,
                "active_frames_gt_0_0195": active_frames,
                "audio_active_gt_0_0195": audio_active,
                "reason_counts": dict(reasons),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
