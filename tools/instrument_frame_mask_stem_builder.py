from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


FRAME_RATE = 60.0


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


def _read_shared_guard(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv(path)
    return {_safe_int(r.get("merged_event_id")): r for r in rows}


def _index_events(path: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv(path)
    return {_safe_int(r.get("merged_event_id")): r for r in rows}


def _score_vector(row: dict[str, str]) -> dict[str, float]:
    return {
        "piano": _safe_float(row.get("piano_score")),
        "violin": _safe_float(row.get("violin_score")),
        "cello": _safe_float(row.get("cello_score")),
        "organ": _safe_float(row.get("organ_score")),
    }


def _support_set(row: dict[str, str]) -> set[str]:
    raw = str(row.get("support_instruments", "")).strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split() if x.strip()}


def _target_weight(
    row: dict[str, str],
    target_instrument: str,
    mode: str,
    shared_guard_row: dict[str, str] | None = None,
) -> tuple[float, str]:
    scores = _score_vector(row)
    total = sum(scores.values())
    target_score = scores.get(target_instrument, 0.0)
    dominant = str(row.get("dominant_instrument", "")).strip()
    dom_state = str(row.get("dominant_state_layered", "")).strip()
    supports = _support_set(row)
    target_window = str(row.get(f"{target_instrument}_window", "")).strip()

    if target_instrument == "piano" and "cello" in supports and shared_guard_row is not None:
        shared_mode = str(shared_guard_row.get("shared_mode", "")).strip()
        ownership_mode = str(shared_guard_row.get("ownership_mode", "")).strip()
        if shared_mode in {
            "NO_DIRECT_PART_MATCH",
            "CELLO_ONLY_EXACT",
            "CELLO_ONLY_PITCHCLASS",
            "CELLO_EXACT_PIANO_PITCHCLASS",
        }:
            if mode in {"core", "strict"}:
                return 0.0, "guard_reject_cello_capture"
        if ownership_mode == "CELLO_DOMINANT_WITH_PIANO_SUPPORT" and mode == "core":
            return 0.0, "guard_reject_cello_owner"

    if total <= 0.0 or target_score <= 0.0:
        return 0.0, "no_trace"

    share = target_score / total
    reason = "weak_background"

    if dominant == target_instrument:
        if dom_state == "OWN_WINDOW_DOMINANT":
            return min(1.0, 0.92 + share * 0.12), "own_window_dominant"
        if dom_state == "CLEAR_DOMINANT":
            return min(1.0, 0.84 + share * 0.14), "clear_dominant"
        if dom_state == "LEANING_DOMINANT":
            return min(1.0, 0.70 + share * 0.16), "leaning_dominant"
        if dom_state == "MIXED_DOMINANT":
            return min(1.0, 0.58 + share * 0.18), "mixed_dominant"
        return min(1.0, 0.48 + share * 0.14), "dominant_other_state"

    if target_instrument in supports:
        if mode == "core":
            if target_window == "TARGET_ONLY_WINDOW" and dominant in {"organ", "UNRESOLVED_FIELD"}:
                return min(0.56, 0.28 + share * 0.14), "core_support_target_window"
            return 0.0, "core_reject_support"
        if mode == "strict":
            base = 0.22
        elif mode == "balanced":
            base = 0.34
        else:
            base = 0.46
        if target_window == "TARGET_ONLY_WINDOW":
            base += 0.10
            reason = "support_target_window"
        elif target_window == "MIXED_WINDOW":
            base += 0.04
            reason = "support_mixed_window"
        else:
            reason = "support_other_window"
        return min(0.78, base + share * 0.20), reason

    if target_window == "TARGET_ONLY_WINDOW":
        if mode == "core":
            if dominant in {"UNRESOLVED_FIELD", "organ"}:
                return min(0.42, 0.14 + share * 0.12), "core_window_only_trace"
            return 0.0, "core_reject_window_only"
        if mode == "strict":
            base = 0.18
        elif mode == "balanced":
            base = 0.28
        else:
            base = 0.38
        return min(0.62, base + share * 0.16), "window_only_trace"

    if target_window == "MIXED_WINDOW" and mode == "broad":
        return min(0.48, 0.20 + share * 0.16), "mixed_window_trace"

    return 0.0, reason


def _build_frame_mask(
    layered_rows: list[dict[str, str]],
    event_index: dict[int, dict[str, str]],
    target_instrument: str,
    mode: str,
    boost_attack_frames: int,
    decay_tail_frames: int,
    shared_guard: dict[int, dict[str, str]],
) -> tuple[np.ndarray, Counter[str], int]:
    max_end = 0
    for ev in event_index.values():
        max_end = max(max_end, _safe_int(ev.get("end_frame")))
    frame_sum = np.zeros(max_end + decay_tail_frames + 4, dtype=np.float32)
    frame_peak = np.zeros_like(frame_sum)
    reason_counts: Counter[str] = Counter()
    used_events = 0

    for row in layered_rows:
        event_id = _safe_int(row.get("merged_event_id"))
        ev = event_index.get(event_id)
        if not ev:
            continue
        weight, reason = _target_weight(
            row=row,
            target_instrument=target_instrument,
            mode=mode,
            shared_guard_row=shared_guard.get(event_id),
        )
        if weight <= 0.0:
            continue
        used_events += 1
        reason_counts[reason] += 1
        start = _safe_int(ev.get("birth_frame"))
        end = _safe_int(ev.get("end_frame"))
        length = max(1, end - start + 1)
        attack_end = min(end, start + boost_attack_frames)
        tail_end = end + decay_tail_frames

        for frame in range(start, end + 1):
            if frame <= attack_end:
                pos = (frame - start) / max(1, (attack_end - start + 1))
                gain = weight * (0.92 + 0.12 * pos)
            else:
                pos = (frame - start) / max(1, length)
                gain = weight * (1.0 - 0.14 * pos)
            frame_sum[frame] += float(gain)
            frame_peak[frame] = max(frame_peak[frame], float(gain))

        for frame in range(end + 1, min(len(frame_sum), tail_end + 1)):
            remain = 1.0 - ((frame - end) / max(1, decay_tail_frames))
            gain = weight * 0.42 * max(0.0, remain)
            frame_sum[frame] += float(gain)
            frame_peak[frame] = max(frame_peak[frame], float(gain))

    if np.max(frame_sum) > 0:
        norm_sum = frame_sum / max(1.0, float(np.quantile(frame_sum[frame_sum > 0], 0.97)))
    else:
        norm_sum = frame_sum
    frame_mask = np.clip(np.maximum(frame_peak, norm_sum), 0.0, 1.0)
    return frame_mask, reason_counts, used_events


def _frame_mask_to_audio_mask(
    frame_mask: np.ndarray,
    sample_rate: int,
    sample_count: int,
    smoothing_ms: float,
) -> np.ndarray:
    frame_times = np.arange(len(frame_mask), dtype=np.float64) / FRAME_RATE
    sample_times = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    audio_mask = np.interp(sample_times, frame_times, frame_mask.astype(np.float64), left=0.0, right=0.0).astype(np.float32)

    window = max(1, int(round(sample_rate * smoothing_ms / 1000.0)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        audio_mask = np.convolve(audio_mask, kernel, mode="same")
    return np.clip(audio_mask, 0.0, 1.0)


def _write_frame_csv(path: Path, frame_mask: np.ndarray) -> None:
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
    ap = argparse.ArgumentParser(description="Build a frame-level stem mask for one instrument.")
    ap.add_argument("--layered_csv", required=True)
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--audio_wav", required=True)
    ap.add_argument("--target_instrument", required=True, choices=["piano", "violin", "cello", "organ"])
    ap.add_argument("--mode", choices=["core", "strict", "balanced", "broad"], default="balanced")
    ap.add_argument("--boost_attack_frames", type=int, default=3)
    ap.add_argument("--decay_tail_frames", type=int, default=8)
    ap.add_argument("--smoothing_ms", type=float, default=22.0)
    ap.add_argument("--shared_guard_csv")
    ap.add_argument("--out_frame_mask_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_summary_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    layered_rows = _read_csv(Path(args.layered_csv))
    event_index = _index_events(Path(args.events_csv))
    shared_guard = _read_shared_guard(Path(args.shared_guard_csv)) if args.shared_guard_csv else {}
    frame_mask, reason_counts, used_events = _build_frame_mask(
        layered_rows=layered_rows,
        event_index=event_index,
        target_instrument=args.target_instrument,
        mode=args.mode,
        boost_attack_frames=args.boost_attack_frames,
        decay_tail_frames=args.decay_tail_frames,
        shared_guard=shared_guard,
    )

    audio, sr = sf.read(str(Path(args.audio_wav)))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    audio_mask = _frame_mask_to_audio_mask(
        frame_mask=frame_mask,
        sample_rate=int(sr),
        sample_count=len(audio),
        smoothing_ms=args.smoothing_ms,
    )
    rendered = audio * audio_mask
    sf.write(str(Path(args.out_wav)), rendered, int(sr), subtype="PCM_16")
    _write_frame_csv(Path(args.out_frame_mask_csv), frame_mask)

    active_frames = int(np.count_nonzero(frame_mask > 0.02))
    strong_frames = int(np.count_nonzero(frame_mask > 0.25))
    summary_lines = [
        "INSTRUMENT FRAME MASK STEM",
        "=" * 72,
        f"target_instrument: {args.target_instrument}",
        f"mode: {args.mode}",
        f"layered_events: {len(layered_rows)}",
        f"used_events: {used_events}",
        f"frame_mask_count: {len(frame_mask)}",
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
                "shared_guard_csv": args.shared_guard_csv or "",
                "frame_mask_count": len(frame_mask),
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
