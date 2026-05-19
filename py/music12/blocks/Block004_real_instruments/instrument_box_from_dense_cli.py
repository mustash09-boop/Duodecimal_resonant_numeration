from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.median(xs))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def normalize_letters(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("А", "A").replace("В", "B").replace("С", "C")
    s = s.replace("а", "A").replace("в", "B").replace("с", "C")
    return s


def bij12_to_int(s: str) -> int:
    s = normalize_letters(s).upper()
    if not s or any(ch not in _VAL12 for ch in s):
        raise ValueError(f"Bad bij12 number: {s!r}")
    n = 0
    for ch in s:
        n = n * 12 + _VAL12[ch]
    return n


def int_to_bij12(n: int) -> str:
    n = int(n)
    if n <= 0:
        raise ValueError("int_to_bij12 expects n >= 1")
    out: list[str] = []
    while n > 0:
        n, r = divmod(n - 1, 12)
        out.append(_CH12[r + 1])
    return "".join(reversed(out))


def int_to_base12_digit(i0: int) -> str:
    i0 = int(i0)
    if not 0 <= i0 < 12:
        raise ValueError("int_to_base12_digit expects 0..11")
    return _CH12[i0 + 1]


def parse_base_note_token(tok: str) -> tuple[str, str]:
    tok = normalize_letters(tok).upper().strip()
    tok = tok.replace("’-", "'-").replace("'", "")
    tok = tok.rstrip("-")

    if "." not in tok:
        raise ValueError(f"Bad note token: {tok!r}")

    oct_s, step = tok.split(".", 1)
    step = step[:1]
    if not oct_s or any(ch not in _VAL12 for ch in oct_s):
        raise ValueError(f"Bad octave in token: {tok!r}")
    if step not in _VAL12:
        raise ValueError(f"Bad step in token: {tok!r}")
    return oct_s, step


def token_to_abs_step(token: str) -> int:
    oct_s, step = parse_base_note_token(token)
    oct0 = bij12_to_int(oct_s) - 1
    step0 = _VAL12[step] - 1
    return oct0 * 12 + step0


def abs_step_to_token(abs_step: int, micro: str = "-") -> str:
    abs_step = int(abs_step)
    if abs_step < 0:
        raise ValueError("abs_step must be >= 0")
    oct0, step0 = divmod(abs_step, 12)
    oct_s = int_to_bij12(oct0 + 1)
    step = int_to_base12_digit(step0)
    if micro:
        return f"{oct_s}.{step}'{micro}"
    return f"{oct_s}.{step}"


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str = "9.A-",
    anchor_hz: float = 440.0,
    micro_steps_per_semitone: int = 12,
    exact_mark: bool = True,
) -> str:
    if freq_hz <= 0:
        return ""

    abs_anchor = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest_semitone = int(round(semitone_offset))
    residual = semitone_offset - nearest_semitone

    abs_note = abs_anchor + nearest_semitone
    base_token = abs_step_to_token(abs_note, micro="")

    micro_float = residual * micro_steps_per_semitone
    micro_rounded = int(round(micro_float))

    if micro_rounded == 0:
        return f"{base_token}'-" if exact_mark else base_token

    sign = "i" if micro_rounded > 0 else "a"
    magnitude = abs(micro_rounded)

    while magnitude >= micro_steps_per_semitone:
        if sign == "i":
            abs_note += 1
        else:
            abs_note -= 1
        magnitude -= micro_steps_per_semitone

    if magnitude == 0:
        base_token = abs_step_to_token(abs_note, micro="")
        return f"{base_token}'-" if exact_mark else base_token

    digit = int_to_base12_digit(magnitude)
    base_token = abs_step_to_token(abs_note, micro="")
    return f"{base_token}'{sign}{digit}"


def _note_from_folder_name(folder_name: str) -> str:
    s = _safe_str(folder_name)
    if "_piano_midi_" in s:
        return s.split("_piano_midi_", 1)[1]
    parts = s.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return s


def _root_hz_from_note_token(token: str, anchor_token: str = "9.A-", anchor_hz: float = 440.0) -> float:
    try:
        semitone_delta = token_to_abs_step(token) - token_to_abs_step(anchor_token)
        return float(anchor_hz * (2.0 ** (semitone_delta / 12.0)))
    except Exception:
        return 0.0


def _range_zone(root_hz: float) -> str:
    if root_hz <= 0:
        return "unknown"
    if root_hz < 110.0:
        return "low"
    if root_hz < 880.0:
        return "mid"
    return "high"


def cents_error(observed_hz: float, target_hz: float) -> float:
    if observed_hz <= 0 or target_hz <= 0:
        return 1e9
    return 1200.0 * math.log2(observed_hz / target_hz)


def _scan_dense_csvs(reports_root: Path) -> list[Path]:
    return sorted(reports_root.glob("*/*__dense.csv"))


def _load_dense_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "time_sec": _safe_float(row.get("time_sec", 0.0)),
                    "freq_hz": _safe_float(row.get("freq_hz", 0.0)),
                    "amplitude": _safe_float(row.get("amplitude", 0.0)),
                    "phase_rad": _safe_float(row.get("phase_rad", 0.0)),
                    "frame_index": _safe_int(row.get("frame_index", 0)),
                    "peak_index": _safe_int(row.get("peak_index", 0)),
                }
            )
    return rows


def _group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["frame_index"]].append(row)
    return out


def _best_frame(rows: list[dict[str, Any]]) -> tuple[int, float, list[dict[str, Any]]]:
    by_frame = _group_by_frame(rows)
    if not by_frame:
        return 0, 0.0, []

    best_key = None
    best_score = None
    for frame_index, group in by_frame.items():
        score = sum(max(0.0, _safe_float(r["amplitude"], 0.0)) for r in group)
        if best_score is None or score > best_score:
            best_score = score
            best_key = frame_index

    group = by_frame[best_key]
    time_sec = _safe_float(group[0]["time_sec"], 0.0) if group else 0.0
    return int(best_key), float(time_sec), list(group)


def _cluster_freqs(
    frame_rows: list[dict[str, Any]],
    *,
    cluster_cents: float,
    anchor_token: str,
    anchor_hz: float,
) -> list[dict[str, Any]]:
    """
    Cluster close spectral peaks inside one note/frame.
    """
    usable = [r for r in frame_rows if _safe_float(r["freq_hz"], 0.0) > 0]
    usable.sort(key=lambda r: _safe_float(r["freq_hz"], 0.0))

    clusters: list[list[dict[str, Any]]] = []

    for row in usable:
        f = _safe_float(row["freq_hz"], 0.0)
        if not clusters:
            clusters.append([row])
            continue

        last_cluster = clusters[-1]
        ref = _mean([_safe_float(x["freq_hz"], 0.0) for x in last_cluster])

        if abs(cents_error(f, ref)) <= cluster_cents:
            last_cluster.append(row)
        else:
            clusters.append([row])

    out: list[dict[str, Any]] = []
    for cluster in clusters:
        freqs = [_safe_float(x["freq_hz"], 0.0) for x in cluster]
        amps = [_safe_float(x["amplitude"], 0.0) for x in cluster]
        phases = [_safe_float(x["phase_rad"], 0.0) for x in cluster]

        center_hz = _mean(freqs)
        sum_amp = float(sum(amps))
        mean_amp = _mean(amps)
        token = hz_to_token_with_micro(center_hz, anchor_token=anchor_token, anchor_hz=anchor_hz)

        out.append(
            {
                "cluster_center_hz": center_hz,
                "token": token,
                "sum_amplitude": sum_amp,
                "mean_amplitude": mean_amp,
                "mean_phase_rad": _mean(phases),
                "count_peaks": len(cluster),
            }
        )

    return out


def _collect_dense_box_data(
    reports_root: Path,
    *,
    anchor_token: str,
    anchor_hz: float,
    cluster_cents: float,
) -> tuple[int, dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[float, dict[str, Any]]]:
    """
    Returns:
      total_notes
      global_presence_by_token
      range_presence_by_token
      frequency_clusters_global
    """
    dense_paths = _scan_dense_csvs(reports_root)
    total_notes = 0

    global_presence_by_token: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "notes": set(),
            "sum_amplitudes": [],
            "mean_amplitudes": [],
            "phases": [],
        }
    )

    range_presence_by_token: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "notes": set(),
            "sum_amplitudes": [],
            "mean_amplitudes": [],
        }
    )

    frequency_clusters_global: dict[float, dict[str, Any]] = defaultdict(
        lambda: {
            "notes": set(),
            "tokens": [],
            "sum_amplitudes": [],
            "mean_amplitudes": [],
        }
    )

    for dense_path in dense_paths:
        folder_name = dense_path.parent.name
        expected_note = _note_from_folder_name(folder_name)
        root_hz = _root_hz_from_note_token(expected_note, anchor_token=anchor_token, anchor_hz=anchor_hz)
        zone = _range_zone(root_hz)

        rows = _load_dense_rows(dense_path)
        _, _, best_group = _best_frame(rows)
        if not best_group:
            continue

        total_notes += 1

        clustered = _cluster_freqs(
            best_group,
            cluster_cents=cluster_cents,
            anchor_token=anchor_token,
            anchor_hz=anchor_hz,
        )

        # normalize amplitudes inside one note by max cluster amplitude
        max_sum_amp = max((c["sum_amplitude"] for c in clustered), default=1.0)
        if max_sum_amp <= 0:
            max_sum_amp = 1.0

        for c in clustered:
            token = _safe_str(c["token"])
            center_hz = float(c["cluster_center_hz"])
            sum_amp = float(c["sum_amplitude"])
            mean_amp = float(c["mean_amplitude"])
            phase = float(c["mean_phase_rad"])
            rel_amp = sum_amp / max_sum_amp

            gp = global_presence_by_token[token]
            gp["notes"].add(expected_note)
            gp["sum_amplitudes"].append(sum_amp)
            gp["mean_amplitudes"].append(mean_amp)
            gp["phases"].append(phase)

            rp = range_presence_by_token[(zone, token)]
            rp["notes"].add(expected_note)
            rp["sum_amplitudes"].append(sum_amp)
            rp["mean_amplitudes"].append(mean_amp)

            # cluster hz globally in rough 1/8-semitone bins via token-like quantization
            freq_bucket = round(center_hz, 1)
            fg = frequency_clusters_global[freq_bucket]
            fg["notes"].add(expected_note)
            fg["tokens"].append(token)
            fg["sum_amplitudes"].append(sum_amp)
            fg["mean_amplitudes"].append(mean_amp)
            fg["relative_amplitudes"] = fg.get("relative_amplitudes", [])
            fg["relative_amplitudes"].append(rel_amp)

    return total_notes, global_presence_by_token, range_presence_by_token, frequency_clusters_global


def _write_dense_global_presence_csv(
    out_csv: Path,
    *,
    total_notes: int,
    global_presence_by_token: dict[str, dict[str, Any]],
) -> None:
    rows = []

    for token, data in global_presence_by_token.items():
        note_count = len(data["notes"])
        percent_notes = (100.0 * note_count / total_notes) if total_notes > 0 else 0.0

        rows.append(
            {
                "token": token,
                "count_notes": note_count,
                "percent_notes": percent_notes,
                "mean_sum_amplitude": _mean(data["sum_amplitudes"]),
                "median_sum_amplitude": _median(data["sum_amplitudes"]),
                "std_sum_amplitude": _std(data["sum_amplitudes"]),
                "mean_peak_amplitude": _mean(data["mean_amplitudes"]),
                "median_peak_amplitude": _median(data["mean_amplitudes"]),
                "std_peak_amplitude": _std(data["mean_amplitudes"]),
                "mean_phase_rad": _mean(data["phases"]),
                "note_examples": " | ".join(sorted(list(data["notes"]))[:12]),
            }
        )

    rows.sort(key=lambda r: (-r["count_notes"], -r["mean_sum_amplitude"], r["token"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "count_notes",
                "percent_notes",
                "mean_sum_amplitude",
                "median_sum_amplitude",
                "std_sum_amplitude",
                "mean_peak_amplitude",
                "median_peak_amplitude",
                "std_peak_amplitude",
                "mean_phase_rad",
                "note_examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_dense_range_presence_csv(
    out_csv: Path,
    *,
    range_presence_by_token: dict[tuple[str, str], dict[str, Any]],
) -> None:
    rows = []

    for (zone, token), data in sorted(range_presence_by_token.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "range_zone": zone,
                "token": token,
                "count_notes": len(data["notes"]),
                "mean_sum_amplitude": _mean(data["sum_amplitudes"]),
                "median_sum_amplitude": _median(data["sum_amplitudes"]),
                "std_sum_amplitude": _std(data["sum_amplitudes"]),
                "mean_peak_amplitude": _mean(data["mean_amplitudes"]),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "range_zone",
                "token",
                "count_notes",
                "mean_sum_amplitude",
                "median_sum_amplitude",
                "std_sum_amplitude",
                "mean_peak_amplitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_dense_frequency_clusters_csv(
    out_csv: Path,
    *,
    total_notes: int,
    frequency_clusters_global: dict[float, dict[str, Any]],
) -> None:
    rows = []

    for center_hz, data in frequency_clusters_global.items():
        note_count = len(data["notes"])
        percent_notes = (100.0 * note_count / total_notes) if total_notes > 0 else 0.0

        token_mode = ""
        if data["tokens"]:
            counts: dict[str, int] = defaultdict(int)
            for t in data["tokens"]:
                counts[t] += 1
            token_mode = max(counts.items(), key=lambda x: x[1])[0]

        rows.append(
            {
                "cluster_center_hz": center_hz,
                "count_notes": note_count,
                "percent_notes": percent_notes,
                "dominant_token": token_mode,
                "mean_sum_amplitude": _mean(data["sum_amplitudes"]),
                "median_sum_amplitude": _median(data["sum_amplitudes"]),
                "std_sum_amplitude": _std(data["sum_amplitudes"]),
                "mean_relative_amplitude": _mean(data.get("relative_amplitudes", [])),
                "note_examples": " | ".join(sorted(list(data["notes"]))[:12]),
            }
        )

    rows.sort(key=lambda r: (-r["count_notes"], -r["mean_relative_amplitude"], r["cluster_center_hz"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_center_hz",
                "count_notes",
                "percent_notes",
                "dominant_token",
                "mean_sum_amplitude",
                "median_sum_amplitude",
                "std_sum_amplitude",
                "mean_relative_amplitude",
                "note_examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_dense_summary_txt(
    out_txt: Path,
    *,
    reports_root: Path,
    total_notes: int,
    global_presence_by_token: dict[str, dict[str, Any]],
    range_presence_by_token: dict[tuple[str, str], dict[str, Any]],
    frequency_clusters_global: dict[float, dict[str, Any]],
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("INSTRUMENT BOX FROM DENSE SCANS")
    lines.append("=" * 100)
    lines.append(f"reports_root: {reports_root}")
    lines.append(f"total_notes : {total_notes}")
    lines.append("")

    top_tokens = []
    for token, data in global_presence_by_token.items():
        top_tokens.append(
            (
                len(data["notes"]),
                _mean(data["sum_amplitudes"]),
                token,
            )
        )
    top_tokens.sort(reverse=True)

    lines.append("TOP REPEATED TOKENS FROM RAW DENSE SCANS")
    lines.append("-" * 100)
    for count_notes, mean_amp, token in top_tokens[:25]:
        percent = (100.0 * count_notes / total_notes) if total_notes > 0 else 0.0
        lines.append(
            f"{token:12s}  notes={count_notes:4d}  percent={percent:7.2f}  mean_sum_amp={mean_amp:.6f}"
        )

    lines.append("")
    lines.append("TOP FREQUENCY CLUSTERS FROM RAW DENSE SCANS")
    lines.append("-" * 100)
    top_freqs = []
    for center_hz, data in frequency_clusters_global.items():
        top_freqs.append(
            (
                len(data["notes"]),
                _mean(data.get("relative_amplitudes", [])),
                center_hz,
                max(data["tokens"], key=data["tokens"].count) if data["tokens"] else "",
            )
        )
    top_freqs.sort(reverse=True)
    for count_notes, mean_rel_amp, center_hz, dom_token in top_freqs[:25]:
        percent = (100.0 * count_notes / total_notes) if total_notes > 0 else 0.0
        lines.append(
            f"{center_hz:10.1f} Hz  token={dom_token:12s}  notes={count_notes:4d}  percent={percent:7.2f}  mean_rel_amp={mean_rel_amp:.6f}"
        )

    lines.append("")
    lines.append("RANGE DISTRIBUTION SNAPSHOT")
    lines.append("-" * 100)
    zone_counts: dict[str, int] = defaultdict(int)
    for (zone, _token), data in range_presence_by_token.items():
        zone_counts[zone] += len(data["notes"])
    for zone in sorted(zone_counts):
        lines.append(f"{zone:8s} total_token_hits={zone_counts[zone]}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build instrument box report from raw __dense.csv files (before chain)."
    )
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--out_dense_global_presence_csv", required=True)
    ap.add_argument("--out_dense_range_presence_csv", required=True)
    ap.add_argument("--out_dense_frequency_clusters_csv", required=True)
    ap.add_argument("--out_dense_summary_txt", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--cluster_cents", type=float, default=25.0)
    args = ap.parse_args()

    reports_root = Path(args.reports_root).resolve()

    (
        total_notes,
        global_presence_by_token,
        range_presence_by_token,
        frequency_clusters_global,
    ) = _collect_dense_box_data(
        reports_root,
        anchor_token=str(args.anchor_token),
        anchor_hz=float(args.anchor_hz),
        cluster_cents=float(args.cluster_cents),
    )

    _write_dense_global_presence_csv(
        Path(args.out_dense_global_presence_csv).resolve(),
        total_notes=total_notes,
        global_presence_by_token=global_presence_by_token,
    )
    _write_dense_range_presence_csv(
        Path(args.out_dense_range_presence_csv).resolve(),
        range_presence_by_token=range_presence_by_token,
    )
    _write_dense_frequency_clusters_csv(
        Path(args.out_dense_frequency_clusters_csv).resolve(),
        total_notes=total_notes,
        frequency_clusters_global=frequency_clusters_global,
    )
    _write_dense_summary_txt(
        Path(args.out_dense_summary_txt).resolve(),
        reports_root=reports_root,
        total_notes=total_notes,
        global_presence_by_token=global_presence_by_token,
        range_presence_by_token=range_presence_by_token,
        frequency_clusters_global=frequency_clusters_global,
    )

    print({
        "reports_root": str(reports_root),
        "total_notes": total_notes,
        "out_dense_global_presence_csv": str(Path(args.out_dense_global_presence_csv).resolve()),
        "out_dense_range_presence_csv": str(Path(args.out_dense_range_presence_csv).resolve()),
        "out_dense_frequency_clusters_csv": str(Path(args.out_dense_frequency_clusters_csv).resolve()),
        "out_dense_summary_txt": str(Path(args.out_dense_summary_txt).resolve()),
        "anchor_token": str(args.anchor_token),
        "anchor_hz": float(args.anchor_hz),
        "cluster_cents": float(args.cluster_cents),
    })


if __name__ == "__main__":
    main()