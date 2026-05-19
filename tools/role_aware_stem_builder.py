from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


FRAME_RATE = 60.0


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _index_events(path: Path) -> dict[int, dict[str, str]]:
    return {_safe_int(r.get("merged_event_id")): r for r in _read_csv(path)}


def _split_support(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split() if x.strip()]


def _event_weight(
    row: dict[str, str],
    target: str,
    mode: str,
) -> tuple[float, str]:
    attack = str(row.get("attack_owner", "")).strip()
    sustain = str(row.get("sustain_owner", "")).strip()
    body = str(row.get("body_owner", "")).strip()
    field = str(row.get("field_owner", "")).strip()
    support = _split_support(row.get("support_owners", ""))
    pattern = str(row.get("role_pattern", "")).strip()
    confidence = str(row.get("role_confidence", "")).strip()

    # Musical core only: ignore pure body/field unless backed by attack/sustain.
    if attack == target and sustain == target:
        return 1.0, "attack+sustain"
    if attack == target:
        if mode == "attack_only":
            return 1.0, "attack_only"
        return 0.88 if confidence == "HIGH" else 0.76, "attack_owner"
    if sustain == target:
        if pattern == "INTERNAL_WAVE_EVENT":
            return (0.48 if mode == "strict" else 0.58), "sustain_internal_wave"
        if body == target and mode == "strict":
            return 0.52, "sustain_with_body"
        return 0.74 if confidence == "HIGH" else 0.62, "sustain_owner"

    if target in support:
        if mode == "strict":
            return 0.18, "support_owner"
        return 0.26, "support_owner"

    # Shared co-owned event can still carry some useful musical energy.
    if attack == "piano+cello" and target in {"piano", "cello"}:
        return (0.34 if mode == "strict" else 0.42), "shared_attack"
    if sustain == "piano+cello" and target in {"piano", "cello"}:
        return (0.30 if mode == "strict" else 0.38), "shared_sustain"

    # Pure body and pure field are intentionally excluded from musical core.
    if body == target and not attack and not sustain:
        return 0.0, "reject_body_only"
    if field and not attack and not sustain:
        return 0.0, "reject_field_only"
    return 0.0, "reject_other"


def _build_frame_mask(
    role_rows: list[dict[str, str]],
    event_index: dict[int, dict[str, str]],
    target: str,
    mode: str,
    tail_frames: int,
) -> tuple[np.ndarray, Counter[str], int]:
    max_end = 0
    for ev in event_index.values():
        max_end = max(max_end, _safe_int(ev.get("end_frame")))
    frame_sum = np.zeros(max_end + tail_frames + 4, dtype=np.float32)
    frame_peak = np.zeros_like(frame_sum)
    reason_counts: Counter[str] = Counter()
    used_events = 0

    for row in role_rows:
        event_id = _safe_int(row.get("merged_event_id"))
        ev = event_index.get(event_id)
        if not ev:
            continue
        weight, reason = _event_weight(row, target=target, mode=mode)
        reason_counts[reason] += 1
        if weight <= 0.0:
            continue
        used_events += 1
        start = _safe_int(ev.get("birth_frame"))
        end = _safe_int(ev.get("end_frame"))
        attack_owner = str(row.get("attack_owner", "")).strip()
        sustain_owner = str(row.get("sustain_owner", "")).strip()
        body_owner = str(row.get("body_owner", "")).strip()

        attack_boost_frames = 4 if attack_owner == target else 1
        for frame in range(start, end + 1):
            if frame <= start + attack_boost_frames:
                pos = (frame - start) / max(1, attack_boost_frames + 1)
                gain = weight * (0.92 + 0.10 * pos)
            else:
                gain = weight
                if sustain_owner == target:
                    gain *= 0.94
                elif body_owner == target:
                    gain *= 0.72
            frame_sum[frame] += float(gain)
            frame_peak[frame] = max(frame_peak[frame], float(gain))

        if sustain_owner == target:
            for frame in range(end + 1, min(len(frame_sum), end + tail_frames + 1)):
                remain = 1.0 - ((frame - end) / max(1, tail_frames))
                gain = weight * 0.34 * max(0.0, remain)
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
    ap = argparse.ArgumentParser(description="Render role-aware musical-core stem from instrument role map.")
    ap.add_argument("--role_map_csv", required=True)
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--audio_wav", required=True)
    ap.add_argument("--target_instrument", required=True, choices=["piano", "organ", "cello", "violin"])
    ap.add_argument("--mode", choices=["strict", "balanced", "attack_only"], default="strict")
    ap.add_argument("--tail_frames", type=int, default=10)
    ap.add_argument("--smoothing_ms", type=float, default=20.0)
    ap.add_argument("--out_mask_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    role_rows = _read_csv(Path(args.role_map_csv))
    event_index = _index_events(Path(args.events_csv))
    frame_mask, reason_counts, used_events = _build_frame_mask(
        role_rows=role_rows,
        event_index=event_index,
        target=args.target_instrument,
        mode=args.mode,
        tail_frames=args.tail_frames,
    )

    audio, sr = sf.read(str(Path(args.audio_wav)))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    audio_mask = _frame_to_audio_mask(frame_mask, int(sr), len(audio), args.smoothing_ms)
    rendered = audio * audio_mask
    sf.write(str(Path(args.out_wav)), rendered, int(sr), subtype="PCM_16")
    _write_mask_csv(Path(args.out_mask_csv), frame_mask)

    active_frames = int(np.count_nonzero(frame_mask > 0.02))
    strong_frames = int(np.count_nonzero(frame_mask > 0.25))
    summary_lines = [
        "ROLE AWARE STEM",
        "=" * 72,
        f"target_instrument: {args.target_instrument}",
        f"mode: {args.mode}",
        f"used_events: {used_events}",
        f"active_frames_gt_0_02: {active_frames}",
        f"strong_frames_gt_0_25: {strong_frames}",
        f"active_frame_ratio: {active_frames / max(1, len(frame_mask)):.6f}",
        f"strong_frame_ratio: {strong_frames / max(1, len(frame_mask)):.6f}",
        f"audio_active_ratio_gt_0_02: {float(np.count_nonzero(audio_mask > 0.02)) / max(1, len(audio_mask)):.6f}",
        f"audio_strong_ratio_gt_0_25: {float(np.count_nonzero(audio_mask > 0.25)) / max(1, len(audio_mask)):.6f}",
        "",
        "reason_counts:",
    ]
    for key, value in reason_counts.most_common():
        summary_lines.append(f"  {key}: {value}")
    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "target_instrument": args.target_instrument,
                "mode": args.mode,
                "used_events": used_events,
                "active_frames_gt_0_02": active_frames,
                "strong_frames_gt_0_25": strong_frames,
                "reason_counts": dict(reason_counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
