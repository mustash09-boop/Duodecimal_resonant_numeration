from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = PROJECT_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from music12.blocks.Block002_audio_recogn.dense_harmonic_chain_builder_cli import (
    hz_to_token_with_micro,
)


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


def _json_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
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


def _freq_to_note_token(freq: float) -> str:
    try:
        token = hz_to_token_with_micro(
            float(freq),
            anchor_token="9.A-",
            anchor_hz=440.0,
            micro_depth=0,
            exact_mark=True,
        )
    except Exception:
        return ""
    return _normalize_note(token)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float64), int(sr)


def _segment(audio: np.ndarray, sr: int, start_sec: float, duration_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * sr)))
    end = min(len(audio), int(round((start_sec + duration_sec) * sr)))
    if end <= start:
        return np.zeros(0, dtype=np.float64)
    return audio[start:end]


def _refine_peak_freq(fft_vals: np.ndarray, peak_idx: int, sr: int, n_fft: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(fft_vals) - 1:
        return peak_idx * sr / n_fft
    alpha = fft_vals[peak_idx - 1]
    beta = fft_vals[peak_idx]
    gamma = fft_vals[peak_idx + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-12:
        delta = 0.0
    else:
        delta = 0.5 * (alpha - gamma) / denom
    return (peak_idx + delta) * sr / n_fft


def _peak_list(segment: np.ndarray, sr: int) -> list[tuple[float, float]]:
    if len(segment) < 256:
        return []
    n = int(2 ** math.ceil(math.log2(len(segment))))
    win = np.hamming(len(segment))
    y = np.zeros(n, dtype=np.float64)
    y[: len(segment)] = segment * win
    fft_vals = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    peaks, _ = find_peaks(fft_vals)
    out: list[tuple[float, float]] = []
    if len(peaks) == 0:
        return out
    max_amp = float(np.max(fft_vals[peaks]))
    for idx in peaks:
        amp = float(fft_vals[idx])
        freq = float(freqs[idx])
        if freq < 27.0 or freq > 5000.0:
            continue
        if amp < max_amp * 0.04:
            continue
        refined = _refine_peak_freq(fft_vals, int(idx), sr, n)
        out.append((amp, refined))
    out.sort(reverse=True)
    if out and out[0][1] > 790.0:
        max_amp = out[0][0]
        out = [x for x in out if x[0] >= max_amp * 0.06]
    return out[:220]


def _bounds(freq: float) -> tuple[float, float]:
    if freq < 100:
        return 4.0, 6.0
    if freq < 130:
        return 7.0, 7.0
    if freq < 200:
        return 8.0, 9.0
    tol = 0.06 * freq
    return tol, tol


def _candidate_f0s(peaks: list[tuple[float, float]]) -> list[float]:
    if not peaks:
        return []
    max_amp = peaks[0][0]
    out: list[float] = []
    min_f0 = 26.5
    for amp, freq in peaks[:40]:
        if amp < max_amp * 0.18:
            break
        if freq < 60.0:
            min_f0 = 200.0
            continue
        if freq > 1860.0:
            out.append(freq)
        else:
            for j in range(1, 15):
                f0 = freq / j
                if f0 > min_f0:
                    out.append(f0)
    uniq = []
    for f0 in sorted(out):
        if not uniq or abs(uniq[-1] - f0) > 2.0:
            uniq.append(f0)
    return uniq[:80]


def _build_chain(peaks: list[tuple[float, float]], f0: float) -> tuple[float, list[tuple[int, float, float]], str]:
    if not peaks or f0 < 27.0:
        return 0.0, [], ""
    peak_max = peaks[0][0]
    weights = {
        1: 0.34,
        2: 0.18,
        3: 0.22,
        4: 0.07,
        5: 0.26,
        6: 0.06,
        7: 0.24,
        8: 0.04,
        9: 0.03,
        10: 0.03,
    }
    chain: list[tuple[int, float, float]] = []
    score = 0.0
    used_notes: Counter[str] = Counter()
    for h in range(1, 11):
        expected = f0 * h
        if expected > 5000.0:
            break
        d1, d2 = _bounds(expected)
        candidates = [(amp, freq) for amp, freq in peaks if expected - d1 <= freq <= expected + d2]
        if not candidates:
            continue
        best_amp, best_freq = min(candidates, key=lambda x: (abs(x[1] - expected), -x[0]))
        score += weights.get(h, 0.02) * (best_amp / max(peak_max, 1e-9))
        chain.append((h, best_freq, best_amp))
        used_notes[_freq_to_note_token(best_freq)] += 1
    root_note = _freq_to_note_token(f0)
    if len(chain) < 2:
        return 0.0, chain, root_note
    chain_h = {h for h, _f, _a in chain}
    if 5 in chain_h:
        score += 0.10
    if 7 in chain_h:
        score += 0.08
    if 1 not in chain_h and 2 in chain_h and 3 in chain_h:
        score -= 0.06
    if used_notes[root_note] > 1:
        score += 0.04
    return score, chain, root_note


def _select_notes(peaks: list[tuple[float, float]], max_notes: int) -> list[dict[str, Any]]:
    chains = []
    for f0 in _candidate_f0s(peaks):
        score, chain, note = _build_chain(peaks, f0)
        if score <= 0.0 or not note:
            continue
        chains.append(
            {
                "note_token": note,
                "score": score,
                "harmonic_count": len(chain),
                "chain_json": json.dumps(
                    [{"h": h, "freq": round(freq, 3), "amp": round(amp, 6)} for h, freq, amp in chain],
                    ensure_ascii=False,
                ),
            }
        )
    best_by_note: dict[str, dict[str, Any]] = {}
    for row in chains:
        note = row["note_token"]
        prev = best_by_note.get(note)
        if prev is None or row["score"] > prev["score"]:
            best_by_note[note] = row
    ranked = sorted(
        best_by_note.values(),
        key=lambda r: (-float(r["score"]), -int(r["harmonic_count"]), r["note_token"]),
    )
    return ranked[:max_notes]


def _group_midi_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("onset_group", "")).strip()].append(row)
    out = []
    for gid, items in sorted(groups.items(), key=lambda kv: min(_safe_float(x.get("start_sec", 0.0)) for x in kv[1])):
        items.sort(key=lambda r: _safe_float(r.get("start_sec", 0.0)))
        notes = [_normalize_note(r.get("expected_note_token", r.get("note_token", ""))) for r in items]
        out.append(
            {
                "onset_group": gid,
                "start_sec": min(_safe_float(r.get("start_sec", 0.0)) for r in items),
                "end_sec": max(_safe_float(r.get("end_sec", 0.0)) for r in items),
                "start_frame60": min(_safe_int(r.get("start_frame60"), 0) for r in items),
                "notes": notes,
                "polyphony": len(notes),
            }
        )
    return out


def _nearest_ownership_group(start_frame60: int, rows: list[dict[str, Any]], max_delta: int = 5) -> dict[str, Any] | None:
    nearby = [r for r in rows if abs(_safe_int(r.get("anchor_frame"), 0) - start_frame60) <= max_delta]
    nearby.sort(key=lambda r: abs(_safe_int(r.get("anchor_frame"), 0) - start_frame60))
    return nearby[0] if nearby else None


def _ownership_candidates(group_row: dict[str, Any] | None, topk: int) -> list[str]:
    if not group_row:
        return []
    return [_normalize_note(x) for x in _json_list(group_row.get("candidate_notes_json", ""))[:topk]]


def _combine_candidates(
    ownership: list[str],
    dsp_rows: list[dict[str, Any]],
    own_top_boost: float = 1.0,
    dsp_scale: float = 1.0,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for idx, note in enumerate(ownership, start=1):
        scores[note] += own_top_boost * (1.2 / idx)
    for idx, row in enumerate(dsp_rows, start=1):
        note = _normalize_note(row.get("note_token", ""))
        scores[note] += dsp_scale * (_safe_float(row.get("score"), 0.0) / idx)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [note for note, _score in ranked]


def _event_level_counts(midi_groups: list[dict[str, Any]], predicted_by_group: dict[str, list[str]], topk: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for group in midi_groups:
        pred = predicted_by_group.get(group["onset_group"], [])[:topk]
        pred_pc = {_pitch_class(x) for x in pred}
        for note in group["notes"]:
            if note in pred:
                counter["EXACT"] += 1
            elif _pitch_class(note) in pred_pc:
                counter["PITCHCLASS"] += 1
            elif pred:
                counter["WRONG"] += 1
            else:
                counter["EMPTY"] += 1
    return counter


def _group_level_counts(midi_groups: list[dict[str, Any]], predicted_by_group: dict[str, list[str]], topk: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for group in midi_groups:
        truth = [_normalize_note(x) for x in group["notes"]]
        pred = predicted_by_group.get(group["onset_group"], [])[:topk]
        truth_set = set(truth)
        pred_set = set(pred)
        truth_pc = {_pitch_class(x) for x in truth}
        pred_pc = {_pitch_class(x) for x in pred}
        if truth_set == pred_set and truth_set:
            counter["EXACT_GROUP"] += 1
        elif truth_pc == pred_pc and truth_pc:
            counter["PITCHCLASS_GROUP"] += 1
        elif pred:
            counter["WRONG_GROUP"] += 1
        else:
            counter["EMPTY_GROUP"] += 1
    return counter


def main() -> None:
    ap = argparse.ArgumentParser(
        description="External DSP harmonic-chain checker for Bach MIDI-audio, with pure DSP and mixed mode against the MIDI reference."
    )
    ap.add_argument("--audio-wav", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--ownership-window-groups-csv", required=True)
    ap.add_argument("--out-group-audit-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--window-min-sec", type=float, default=0.10)
    ap.add_argument("--window-max-sec", type=float, default=0.22)
    ap.add_argument("--ownership-topk", type=int, default=5)
    ap.add_argument("--dsp-max-notes", type=int, default=6)
    args = ap.parse_args()

    audio, sr = _read_audio(Path(args.audio_wav))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    ownership_rows = _load_csv(Path(args.ownership_window_groups_csv))
    midi_groups = _group_midi_events(midi_rows)

    dsp_predicted: dict[str, list[str]] = {}
    own_predicted: dict[str, list[str]] = {}
    mixed_predicted: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []

    for idx, group in enumerate(midi_groups):
        start_sec = float(group["start_sec"])
        if idx + 1 < len(midi_groups):
            next_start = float(midi_groups[idx + 1]["start_sec"])
            gap = max(0.0, next_start - start_sec)
        else:
            gap = float(group["end_sec"]) - start_sec
        duration = min(max(gap * 0.92 if gap > 0 else 0.125, args.window_min_sec), args.window_max_sec)
        seg = _segment(audio, sr, start_sec, duration)
        peaks = _peak_list(seg, sr)
        dsp_rows = _select_notes(peaks, int(args.dsp_max_notes))
        dsp_notes = [_normalize_note(r["note_token"]) for r in dsp_rows]
        own_group = _nearest_ownership_group(int(group["start_frame60"]), ownership_rows, max_delta=5)
        own_notes = _ownership_candidates(own_group, int(args.ownership_topk))
        mixed_notes = _combine_candidates(own_notes, dsp_rows)

        gid = str(group["onset_group"])
        dsp_predicted[gid] = dsp_notes
        own_predicted[gid] = own_notes
        mixed_predicted[gid] = mixed_notes

        audit_rows.append(
            {
                "onset_group": gid,
                "start_sec": f"{start_sec:.6f}",
                "duration_sec": f"{duration:.6f}",
                "polyphony": group["polyphony"],
                "truth_notes_json": json.dumps(group["notes"], ensure_ascii=False),
                "ownership_notes_json": json.dumps(own_notes, ensure_ascii=False),
                "dsp_notes_json": json.dumps(dsp_notes, ensure_ascii=False),
                "mixed_notes_json": json.dumps(mixed_notes[:8], ensure_ascii=False),
                "dsp_chain_rows_json": json.dumps(dsp_rows, ensure_ascii=False),
                "peak_count": len(peaks),
                "ownership_group_id": str(own_group.get("onset_group_id", "")).strip() if own_group else "",
                "ownership_anchor_frame": _safe_int(own_group.get("anchor_frame"), 0) if own_group else "",
            }
        )

    summaries = {}
    for mode_name, predicted in (
        ("dsp_only", dsp_predicted),
        ("ownership_only", own_predicted),
        ("mixed_mode", mixed_predicted),
    ):
        summaries[mode_name] = {
            "event_top1": dict(_event_level_counts(midi_groups, predicted, 1)),
            "event_top3": dict(_event_level_counts(midi_groups, predicted, 3)),
            "event_top5": dict(_event_level_counts(midi_groups, predicted, 5)),
            "group_top3": dict(_group_level_counts(midi_groups, predicted, 3)),
            "group_top5": dict(_group_level_counts(midi_groups, predicted, 5)),
        }

    out_csv = Path(args.out_group_audit_csv)
    out_summary = Path(args.out_summary_txt)
    out_meta = Path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if audit_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    lines = [
        "DSP HARMONIC CHAIN CHECKER",
        "=" * 72,
        f"midi_onset_groups             : {len(midi_groups)}",
        f"midi_note_events              : {len(midi_rows)}",
        f"window_min_sec                : {args.window_min_sec}",
        f"window_max_sec                : {args.window_max_sec}",
        "",
    ]
    for mode_name in ("dsp_only", "ownership_only", "mixed_mode"):
        lines.extend([mode_name.upper(), "-" * 72])
        mode = summaries[mode_name]
        for block_name in ("event_top1", "event_top3", "event_top5", "group_top3", "group_top5"):
            lines.append(block_name)
            counter = mode[block_name]
            for key in sorted(counter):
                lines.append(f"  {key:28s}: {counter[key]}")
        lines.append("")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": "dsp_harmonic_chain_checker",
        "inputs": {
            "audio_wav": args.audio_wav,
            "midi_events_csv": args.midi_events_csv,
            "ownership_window_groups_csv": args.ownership_window_groups_csv,
        },
        "parameters": {
            "window_min_sec": args.window_min_sec,
            "window_max_sec": args.window_max_sec,
            "ownership_topk": args.ownership_topk,
            "dsp_max_notes": args.dsp_max_notes,
        },
        "result": summaries,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
