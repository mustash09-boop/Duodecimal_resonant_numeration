from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _split_tokens(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split() if x.strip()]


def _note_root(token: str) -> str:
    t = str(token or "").strip()
    apos = t.find("'")
    if apos >= 0:
        return t[: apos + 1] + "-"
    return t


def _frame_target_window_map(
    midi_rows: List[Dict[str, Any]],
    target_prefixes: List[str],
    window: int,
) -> Dict[int, str]:
    out: Dict[int, Dict[str, int]] = defaultdict(lambda: {"target": 0, "other": 0})
    prefixes = [p.strip() for p in target_prefixes if p.strip()]
    for r in midi_rows:
        frame = _safe_int(r.get("start_frame60"), -999999)
        if frame < 0:
            continue
        track_name = str(r.get("track_name", "")).strip()
        is_target = any(track_name.startswith(pref) for pref in prefixes)
        for f in range(frame - window, frame + window + 1):
            if is_target:
                out[f]["target"] += 1
            else:
                out[f]["other"] += 1
    labels: Dict[int, str] = {}
    for frame, v in out.items():
        if v["target"] > 0 and v["other"] == 0:
            labels[frame] = "TARGET_ONLY_WINDOW"
        elif v["target"] > 0 and v["other"] > 0:
            labels[frame] = "MIXED_WINDOW"
        elif v["target"] == 0 and v["other"] > 0:
            labels[frame] = "OTHER_ONLY_WINDOW"
        else:
            labels[frame] = "EMPTY_WINDOW"
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a temporary organ pseudo-passport from target MIDI windows and micro families.")
    ap.add_argument("--micro-families-csv", required=True)
    ap.add_argument("--midi-events-csv", required=True)
    ap.add_argument("--target-track-prefixes", required=True)
    ap.add_argument("--out-passport-csv", required=True)
    ap.add_argument("--out-summary-txt", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--window-frames", type=int, default=3)
    ap.add_argument("--top-body", type=int, default=120)
    ap.add_argument("--top-secondary", type=int, default=64)
    args = ap.parse_args()

    families = _load_csv(Path(args.micro_families_csv))
    midi_rows = _load_csv(Path(args.midi_events_csv))
    target_prefixes = [x.strip() for x in str(args.target_track_prefixes).split(",") if x.strip()]
    frame_labels = _frame_target_window_map(midi_rows, target_prefixes, args.window_frames)

    token_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "frame_hits_target_only": 0.0,
        "frame_hits_mixed": 0.0,
        "score_target_only": 0.0,
        "score_mixed": 0.0,
        "root_hits_target_only": 0.0,
        "root_hits_mixed": 0.0,
    })

    for r in families:
        frame = _safe_int(r.get("frame_index"), -1)
        if frame < 0:
            continue
        label = frame_labels.get(frame, "EMPTY_WINDOW")
        if label not in {"TARGET_ONLY_WINDOW", "MIXED_WINDOW"}:
            continue
        score = _safe_float(r.get("family_score"), 0.0)
        root_token = str(r.get("family_root_note_micro", "")).strip()
        members = _split_tokens(r.get("root_micro_members", ""))
        seen: set[str] = set()
        for tok in members:
            st = token_stats[tok]
            if label == "TARGET_ONLY_WINDOW":
                st["score_target_only"] += score
                if tok not in seen:
                    st["frame_hits_target_only"] += 1
            else:
                st["score_mixed"] += score
                if tok not in seen:
                    st["frame_hits_mixed"] += 1
            if _note_root(tok) == root_token:
                if label == "TARGET_ONLY_WINDOW":
                    st["root_hits_target_only"] += 1
                else:
                    st["root_hits_mixed"] += 1
            seen.add(tok)

    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for tok, st in token_stats.items():
        body_score = st["score_target_only"] * 1.2 + st["score_mixed"] * 0.4 + st["root_hits_target_only"] * 3.0 + st["frame_hits_target_only"] * 1.5
        ranked.append((tok, body_score, st))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    body_tokens = [tok for tok, _, _ in ranked[: args.top_body]]

    secondary_ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for tok, _, st in ranked:
        sec_score = st["frame_hits_target_only"] * 0.4 + st["frame_hits_mixed"] * 1.0 + st["score_mixed"] * 0.3
        if sec_score <= 0:
            continue
        secondary_ranked.append((tok, sec_score, st))
    secondary_ranked.sort(key=lambda x: (-x[1], x[0]))
    secondary_tokens = [tok for tok, _, _ in secondary_ranked if tok not in body_tokens][: args.top_secondary]

    out_row = {
        "passport_name": "ave_maria_organ_pseudopassport",
        "target_track_prefixes": ",".join(target_prefixes),
        "body_token_count": str(len(body_tokens)),
        "secondary_token_count": str(len(secondary_tokens)),
        "body_tokens": " ".join(body_tokens),
        "secondary_tokens": " ".join(secondary_tokens),
    }
    _write_csv(Path(args.out_passport_csv), [out_row], out_row.keys())

    top_preview = ranked[:20]
    lines = [
        "ORGAN PSEUDOPASSPORT FROM TARGET WINDOWS",
        "=" * 72,
        f"micro_families_csv   : {args.micro_families_csv}",
        f"midi_events_csv      : {args.midi_events_csv}",
        f"target_track_prefixes: {','.join(target_prefixes)}",
        f"body_token_count     : {len(body_tokens)}",
        f"secondary_token_count: {len(secondary_tokens)}",
        "",
        "top_body_tokens_preview:",
    ]
    for tok, score, st in top_preview:
        lines.append(
            f"  {tok}: score={score:.3f}, target_only_frames={st['frame_hits_target_only']:.0f}, mixed_frames={st['frame_hits_mixed']:.0f}, root_target_only={st['root_hits_target_only']:.0f}"
        )

    Path(args.out_summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "inputs": {
                    "micro_families_csv": args.micro_families_csv,
                    "midi_events_csv": args.midi_events_csv,
                    "target_track_prefixes": target_prefixes,
                },
                "result": {
                    "body_token_count": len(body_tokens),
                    "secondary_token_count": len(secondary_tokens),
                    "body_tokens_preview": body_tokens[:20],
                    "secondary_tokens_preview": secondary_tokens[:20],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
