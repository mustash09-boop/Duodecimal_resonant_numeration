from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPORTS = Path(
    r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\10_reports_Ave_Maria"
)
MIDI_CSV = Path(
    r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\00_sources\midi\ave_maria_gounod_midi_events_with_parts_v1.csv"
)


def load_midi_window(path: Path, start_sec: float, end_sec: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row.get("start_sec", 0.0) or 0.0)
            if not (start_sec <= t <= end_sec):
                continue
            out.append(
                {
                    "track_name": str(row.get("track_name", "")).strip(),
                    "note_token": str(row.get("note_token", "")).strip(),
                    "freq_hz": float(row.get("freq_hz", 0.0) or 0.0),
                    "duration_sec": float(row.get("duration_sec", 0.0) or 0.0),
                    "start_sec": t,
                }
            )
    return out


def load_probe_coords(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "probe_index": int(row["probe_index"]),
                    "note_token": str(row["note_token"]).strip(),
                    "frequency_hz": float(row["frequency_hz"]),
                }
            )
    return out


def nearest_probes(coords: list[dict[str, object]], target_hz: float, count: int = 7) -> list[dict[str, object]]:
    ranked = sorted(coords, key=lambda r: abs(float(r["frequency_hz"]) - target_hz))
    return ranked[:count]


def load_probe_matrix_rows(path: Path, wanted_probe_ids: set[int]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe_id = int(row["probe_index"])
            if probe_id not in wanted_probe_ids:
                continue
            out[probe_id] = {k: float(v) for k, v in row.items() if k != "probe_index"}
    return out


def load_framewise_summary(path: Path, start_frame: int, end_frame: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["frame_index"]))
            if start_frame <= frame <= end_frame:
                out.append(
                    {
                        "frame_index": frame,
                        "time_sec": float(row.get("time_sec", 0.0) or 0.0),
                        "selected_notes_report": str(row.get("selected_notes_report", "")).strip(),
                    }
                )
    return out


def load_proxy_rows(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "anchor_note_token": str(row["anchor_note_token"]).strip(),
                    "register_role": str(row["register_role"]).strip(),
                    "window_direct_match": str(row["window_direct_match"]).strip(),
                    "weight": float(row["weight"]),
                    "approx_hz": float(row["approx_hz"]),
                }
            )
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect whether a new sustained entrant appears first as direct string fundamental or only as shared-band proxy."
    )
    ap.add_argument("--midi-csv", default=str(MIDI_CSV))
    ap.add_argument("--probe-coords-csv", default=str(REPORTS / "ave_maria_probe_coords_micro_full.csv"))
    ap.add_argument("--probe-matrix-csv", default=str(REPORTS / "ave_maria_probe_matrix_micro_full.csv"))
    ap.add_argument("--framewise-summary-csv", default=str(REPORTS / "ave_maria_framewise_candidates_micro_v1.csv"))
    ap.add_argument("--proxy-csv", default=str(REPORTS / "ave_maria_11p95s_12p25s_second_string_layer_tendency_v1.csv"))
    ap.add_argument("--start-sec", type=float, default=11.95)
    ap.add_argument("--end-sec", type=float, default=12.25)
    ap.add_argument("--start-frame", type=int, default=717)
    ap.add_argument("--split-frame", type=int, default=720)
    ap.add_argument("--end-frame", type=int, default=735)
    ap.add_argument("--out-csv", default=str(REPORTS / "ave_maria_11p95s_12p25s_direct_sustained_entrant_fundamental_v1.csv"))
    ap.add_argument("--out-summary-txt", default=str(REPORTS / "ave_maria_11p95s_12p25s_direct_sustained_entrant_fundamental_v1.txt"))
    ap.add_argument("--out-meta-json", default=str(REPORTS / "ave_maria_11p95s_12p25s_direct_sustained_entrant_fundamental_v1.json"))
    args = ap.parse_args()

    midi_rows = load_midi_window(Path(args.midi_csv), args.start_sec, args.end_sec)
    coords = load_probe_coords(Path(args.probe_coords_csv))
    framewise = load_framewise_summary(Path(args.framewise_summary_csv), args.start_frame, args.end_frame)
    proxies = load_proxy_rows(Path(args.proxy_csv))

    string_targets = [r for r in midi_rows if r["track_name"] in {"Cello", "Violin"}]
    piano_controls = [r for r in midi_rows if r["track_name"] == "Piano-Treble"]

    target_specs: list[dict[str, object]] = []
    for row in string_targets + piano_controls[:2]:
        nearest = nearest_probes(coords, float(row["freq_hz"]), count=7)
        target_specs.append(
            {
                "label": f"{row['track_name']}::{row['note_token']}",
                "track_name": row["track_name"],
                "note_token": row["note_token"],
                "freq_hz": float(row["freq_hz"]),
                "duration_sec": float(row["duration_sec"]),
                "nearest": nearest,
            }
        )

    wanted_probe_ids = {int(p["probe_index"]) for spec in target_specs for p in spec["nearest"]}
    matrix_rows = load_probe_matrix_rows(Path(args.probe_matrix_csv), wanted_probe_ids)

    pre_frames = [f for f in range(args.start_frame, args.split_frame)]
    post_frames = [f for f in range(args.split_frame, args.end_frame + 1)]
    min_signal_floor = 0.0195

    out_rows: list[dict[str, object]] = []
    direct_string_supported = False
    best_proxy_frame = None
    best_proxy_token = ""
    best_proxy_weight = -1.0

    for spec in target_specs:
        nearest = spec["nearest"]
        exact = nearest[0]
        exact_values = matrix_rows.get(int(exact["probe_index"]), {})
        exact_pre = [float(exact_values.get(f"frame_{f}", 0.0)) for f in pre_frames]
        exact_post = [float(exact_values.get(f"frame_{f}", 0.0)) for f in post_frames]
        exact_pre_mean = mean(exact_pre)
        exact_post_mean = mean(exact_post)
        exact_ratio = exact_post_mean / max(exact_pre_mean, 1e-9)
        exact_post_max = max(exact_post) if exact_post else 0.0
        exact_threshold = max(exact_pre_mean * 1.20, min_signal_floor)
        exact_hit_frames = [f for f, v in zip(post_frames, exact_post) if v >= exact_threshold]
        exact_persistent = len(exact_hit_frames)
        exact_first_hit_frame = exact_hit_frames[0] if exact_hit_frames else None
        exact_best_run_start = None
        exact_best_run_end = None
        exact_best_run_len = 0
        if exact_hit_frames:
            run_start = exact_hit_frames[0]
            prev_frame = exact_hit_frames[0]
            for frame in exact_hit_frames[1:]:
                if frame == prev_frame + 1:
                    prev_frame = frame
                    continue
                run_len = prev_frame - run_start + 1
                if run_len > exact_best_run_len:
                    exact_best_run_start = run_start
                    exact_best_run_end = prev_frame
                    exact_best_run_len = run_len
                run_start = frame
                prev_frame = frame
            run_len = prev_frame - run_start + 1
            if run_len > exact_best_run_len:
                exact_best_run_start = run_start
                exact_best_run_end = prev_frame
                exact_best_run_len = run_len

        neighborhood_probe_ids = [int(p["probe_index"]) for p in nearest[:4]]
        neigh_pre_vals: list[float] = []
        neigh_post_vals: list[float] = []
        for pid in neighborhood_probe_ids:
            row = matrix_rows.get(pid, {})
            neigh_pre_vals.extend(float(row.get(f"frame_{f}", 0.0)) for f in pre_frames)
            neigh_post_vals.extend(float(row.get(f"frame_{f}", 0.0)) for f in post_frames)
        neigh_pre_mean = mean(neigh_pre_vals)
        neigh_post_mean = mean(neigh_post_vals)
        neigh_ratio = neigh_post_mean / max(neigh_pre_mean, 1e-9)

        if spec["track_name"] in {"Cello", "Violin"} and exact_ratio >= 1.15 and exact_persistent >= 4:
            direct_string_supported = True

        out_rows.append(
            {
                "label": spec["label"],
                "track_name": spec["track_name"],
                "reference_note_token": spec["note_token"],
                "target_hz": f"{float(spec['freq_hz']):.6f}",
                "exact_probe_index": int(exact["probe_index"]),
                "exact_probe_band_token": str(exact["note_token"]),
                "exact_probe_hz": f"{float(exact['frequency_hz']):.6f}",
                "exact_pre_mean": f"{exact_pre_mean:.9f}",
                "exact_post_mean": f"{exact_post_mean:.9f}",
                "exact_post_max": f"{exact_post_max:.9f}",
                "exact_post_pre_ratio": f"{exact_ratio:.9f}",
                "exact_threshold": f"{exact_threshold:.9f}",
                "exact_persistent_post_frames": exact_persistent,
                "exact_first_hit_frame": exact_first_hit_frame if exact_first_hit_frame is not None else "",
                "exact_best_run_start_frame": exact_best_run_start if exact_best_run_start is not None else "",
                "exact_best_run_end_frame": exact_best_run_end if exact_best_run_end is not None else "",
                "exact_best_run_len": exact_best_run_len,
                "neighborhood_pre_mean": f"{neigh_pre_mean:.9f}",
                "neighborhood_post_mean": f"{neigh_post_mean:.9f}",
                "neighborhood_post_pre_ratio": f"{neigh_ratio:.9f}",
            }
        )

    proxy_candidates = [
        r
        for r in proxies
        if r["window_direct_match"] == "NO_DIRECT_WINDOW_MATCH" and r["register_role"] == "UPPER_MAIN_BAND"
    ]
    for proxy in proxy_candidates:
        token = str(proxy["anchor_note_token"])
        first_frame = None
        for row in framewise:
            if token in str(row["selected_notes_report"]):
                first_frame = int(row["frame_index"])
                break
        if first_frame is not None and (best_proxy_frame is None or first_frame < best_proxy_frame or (first_frame == best_proxy_frame and float(proxy["weight"]) > best_proxy_weight)):
            best_proxy_frame = first_frame
            best_proxy_token = token
            best_proxy_weight = float(proxy["weight"])

    if direct_string_supported:
        verdict = "DIRECT_STRING_FUNDAMENTAL_ENTRY_SUPPORTED"
    elif best_proxy_frame is not None:
        verdict = "NO_DIRECT_STRING_FUNDAMENTAL_ENTRY__EARLIEST_MEASURABLE_ENTRY_IS_SHARED_UPPER_BAND_PROXY"
    else:
        verdict = "NO_DIRECT_STRING_FUNDAMENTAL_ENTRY_DETECTED"

    summary_lines = [
        "AVE MARIA DIRECT SUSTAINED ENTRANT FUNDAMENTAL DETECTOR",
        "=" * 72,
        f"window_sec: {args.start_sec:.3f} -> {args.end_sec:.3f}",
        f"window_frames60: {args.start_frame} -> {args.end_frame} split={args.split_frame}",
        "",
        "target_band_results:",
    ]
    for row in out_rows:
        summary_lines.append(
            f"  {row['label']}: exact_ratio={row['exact_post_pre_ratio']}  exact_persistent={row['exact_persistent_post_frames']}  first_hit={row['exact_first_hit_frame']}  best_run={row['exact_best_run_start_frame']}->{row['exact_best_run_end_frame']}  neighborhood_ratio={row['neighborhood_post_pre_ratio']}"
        )

    if best_proxy_frame is not None:
        summary_lines.extend(
            [
                "",
                f"earliest_shared_upper_proxy_frame: {best_proxy_frame}",
                f"earliest_shared_upper_proxy_token: {best_proxy_token}",
                f"earliest_shared_upper_proxy_weight: {best_proxy_weight:.9f}",
            ]
        )

    summary_lines.extend(
        [
            "",
            f"verdict: {verdict}",
            "verdict_notes:",
            "  - Direct string fundamental support requires a real post-split rise on the exact string probe band, not only on shared field anchors.",
            "  - If exact string bands do not rise, but a new stable upper-band proxy appears, the entrant is present but first becomes measurable through shared support geometry.",
            "  - In that case the correct entry point for further separation is the earliest stable shared proxy frame, not the nominal MIDI string fundamental itself.",
        ]
    )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "track_name",
                "reference_note_token",
                "target_hz",
                "exact_probe_index",
                "exact_probe_band_token",
                "exact_probe_hz",
                "exact_pre_mean",
                "exact_post_mean",
                "exact_post_max",
                "exact_post_pre_ratio",
                "exact_threshold",
                "exact_persistent_post_frames",
                "exact_first_hit_frame",
                "exact_best_run_start_frame",
                "exact_best_run_end_frame",
                "exact_best_run_len",
                "neighborhood_pre_mean",
                "neighborhood_post_mean",
                "neighborhood_post_pre_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    Path(args.out_summary_txt).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    Path(args.out_meta_json).write_text(
        json.dumps(
            {
                "stage": "ave_maria_direct_sustained_entrant_fundamental_detector",
                "result": {
                    "direct_string_supported": direct_string_supported,
                    "best_proxy_frame": best_proxy_frame,
                    "best_proxy_token": best_proxy_token,
                    "verdict": verdict,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
