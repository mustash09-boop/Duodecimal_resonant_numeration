from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from music12.core.spiral12_geometry import parse_token_to_spiral


@dataclass(frozen=True)
class VoiceEvent:
    voice_id: int
    note_index: int
    note_token: str

    time_start: float
    time_end: float
    duration: float

    representative_rc_hz_mean: float
    representative_rc_energy_mean: float
    stabilization_score_mean: float

    theoretical_chain_verdict_mode: str
    stabilization_role_mode: str
    best_theoretical_chain_string_mode: str

    spiral_arc: float


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return int(s)
    except Exception:
        return default


def _load_voice_events_csv(path: Path) -> list[VoiceEvent]:
    events: list[VoiceEvent] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "voice_id",
            "note_index",
            "note_token",
            "time_start",
            "time_end",
            "duration",
            "representative_rc_hz_mean",
            "representative_rc_energy_mean",
            "stabilization_score_mean",
            "theoretical_chain_verdict_mode",
            "stabilization_role_mode",
            "best_theoretical_chain_string_mode",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            note_token = _safe_str(row["note_token"])
            spiral = parse_token_to_spiral(note_token)
            if spiral is None:
                continue

            events.append(
                VoiceEvent(
                    voice_id=_safe_int(row["voice_id"], 0),
                    note_index=_safe_int(row["note_index"], 0),
                    note_token=note_token,
                    time_start=_safe_float(row["time_start"], 0.0),
                    time_end=_safe_float(row["time_end"], 0.0),
                    duration=_safe_float(row["duration"], 0.0),
                    representative_rc_hz_mean=_safe_float(row["representative_rc_hz_mean"], 0.0),
                    representative_rc_energy_mean=_safe_float(row["representative_rc_energy_mean"], 0.0),
                    stabilization_score_mean=_safe_float(row["stabilization_score_mean"], 0.0),
                    theoretical_chain_verdict_mode=_safe_str(row["theoretical_chain_verdict_mode"]),
                    stabilization_role_mode=_safe_str(row["stabilization_role_mode"]),
                    best_theoretical_chain_string_mode=_safe_str(row["best_theoretical_chain_string_mode"]),
                    spiral_arc=float(spiral.absolute_arc),
                )
            )

    events.sort(key=lambda e: (e.time_start, e.voice_id, e.note_index))
    return events


def _simultaneous_voice_count_sweepline(
    events: list[VoiceEvent],
) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

    points: list[tuple[float, int]] = []
    for ev in events:
        points.append((ev.time_start, +1))
        points.append((ev.time_end, -1))

    points.sort(key=lambda x: (x[0], -x[1]))

    xs: list[float] = []
    ys: list[int] = []

    active = 0
    for t, delta in points:
        active += delta
        xs.append(float(t))
        ys.append(max(0, active))

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def _voice_summary(events: list[VoiceEvent]) -> dict[str, Any]:
    by_voice: Dict[int, list[VoiceEvent]] = {}

    for ev in events:
        by_voice.setdefault(ev.voice_id, []).append(ev)

    voices = []
    for voice_id, items in sorted(by_voice.items()):
        items = sorted(items, key=lambda ev: (ev.time_start, ev.time_end))
        note_counts: Dict[str, int] = {}
        chain_counts: Dict[str, int] = {}

        for ev in items:
            note_counts[ev.note_token] = note_counts.get(ev.note_token, 0) + 1
            chain = _safe_str(ev.best_theoretical_chain_string_mode)
            if chain:
                chain_counts[chain] = chain_counts.get(chain, 0) + 1

        dominant_note = max(note_counts.items(), key=lambda x: x[1])[0] if note_counts else ""
        dominant_chain = max(chain_counts.items(), key=lambda x: x[1])[0] if chain_counts else ""

        voices.append(
            {
                "voice_id": voice_id,
                "event_count": len(items),
                "time_start": items[0].time_start,
                "time_end": items[-1].time_end,
                "duration": items[-1].time_end - items[0].time_start,
                "dominant_note": dominant_note,
                "dominant_chain": dominant_chain,
                "mean_spiral_arc": float(sum(ev.spiral_arc for ev in items) / len(items)),
            }
        )

    return {
        "voice_count": len(by_voice),
        "voices": voices,
    }


def _line_width_from_stability(stability: float) -> float:
    s = max(0.0, min(1.0, float(stability)))
    return 1.0 + 4.0 * s


def _plot_voice_arcs(
    events: list[VoiceEvent],
    out_png: Path,
    title: str,
) -> None:
    plt.figure(figsize=(16, 8))

    voice_ids = sorted({ev.voice_id for ev in events})
    voice_order = {voice_id: i for i, voice_id in enumerate(voice_ids)}

    for ev in events:
        lw = _line_width_from_stability(ev.stabilization_score_mean)

        # горизонтальный отрезок по времени на уровне спиральной дуги
        plt.plot(
            [ev.time_start, ev.time_end],
            [ev.spiral_arc, ev.spiral_arc],
            linewidth=lw,
            alpha=0.85,
        )

        # центральная точка события
        t_mid = (ev.time_start + ev.time_end) / 2.0
        plt.scatter([t_mid], [ev.spiral_arc], s=10 + 20 * max(0.0, ev.stabilization_score_mean))

        # лёгкая подпись note token
        plt.text(
            t_mid,
            ev.spiral_arc,
            ev.note_token,
            fontsize=7,
            ha="center",
            va="bottom",
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Spiral arc")
    plt.title(title.strip() or "Voices in spiral arc space")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_polyphony(
    events: list[VoiceEvent],
    out_png: Path,
    title: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = _simultaneous_voice_count_sweepline(events)

    plt.figure(figsize=(16, 5))
    if len(xs) > 0:
        plt.step(xs, ys, where="post")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Simultaneous voices")
    plt.title(title.strip() or "Simultaneous voice count")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()

    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot chain-based voice events in time × spiral_arc space"
    )
    parser.add_argument("--voice_events_csv", required=True, help="Input voice_events.csv")
    parser.add_argument("--out_voice_arc_png", required=True, help="Output PNG for spiral-arc voice plot")
    parser.add_argument("--out_polyphony_png", required=True, help="Output PNG for simultaneous voice count")
    parser.add_argument("--title", default="", help="Optional title prefix")
    args = parser.parse_args()

    voice_events_csv = Path(args.voice_events_csv).resolve()
    out_voice_arc_png = Path(args.out_voice_arc_png).resolve()
    out_polyphony_png = Path(args.out_polyphony_png).resolve()

    events = _load_voice_events_csv(voice_events_csv)
    if not events:
        raise ValueError("voice_events.csv is empty")

    _plot_voice_arcs(
        events,
        out_png=out_voice_arc_png,
        title=args.title.strip() or "Voice arcs in spiral space",
    )

    xs, ys = _plot_polyphony(
        events,
        out_png=out_polyphony_png,
        title=args.title.strip() or "Simultaneous voice count",
    )

    summary = {
        "event_count": len(events),
        "voice_count": len({ev.voice_id for ev in events}),
        "time_start": min(ev.time_start for ev in events),
        "time_end": max(ev.time_end for ev in events),
        "polyphony_max": int(max(ys)) if len(ys) > 0 else 0,
        "voice_summary": _voice_summary(events),
        "outputs": {
            "voice_arc_png": str(out_voice_arc_png),
            "polyphony_png": str(out_polyphony_png),
        },
        "semantic_note": (
            "Voice visualization in time × spiral_arc space. "
            "No horizontal voice-id bars are used as the main representation."
        ),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"voice arc plot saved: {out_voice_arc_png}")
    print(f"polyphony plot saved: {out_polyphony_png}")


if __name__ == "__main__":
    main()