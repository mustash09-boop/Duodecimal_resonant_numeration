from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DIGITS12 = "123456789ABC"
_VAL12 = {ch: i + 1 for i, ch in enumerate(DIGITS12)}
_CH12 = {i + 1: ch for i, ch in enumerate(DIGITS12)}


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


def step_index0(step: str) -> int:
    step = normalize_letters(step).upper()
    if step not in _VAL12:
        raise ValueError(f"Bad step digit: {step!r}")
    return _VAL12[step] - 1


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
    step0 = step_index0(step)
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
        raise ValueError("freq_hz must be > 0")

    abs_anchor = token_to_abs_step(anchor_token)
    semitone_offset = 12.0 * math.log2(freq_hz / anchor_hz)

    nearest_semitone = int(round(semitone_offset))
    residual = semitone_offset - nearest_semitone

    abs_note = abs_anchor + nearest_semitone
    base_token = abs_step_to_token(abs_note, micro="")

    if micro_steps_per_semitone <= 0:
        return base_token

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


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def expected_note_from_dirname(name: str) -> str:
    m = re.match(r"^\d+_piano_midi_(.+)$", name)
    if m:
        return m.group(1)
    m = re.match(r"^\d+__[^_]+(?:_[^_]+)*__(.+)$", name)
    if m:
        return m.group(1)
    m = re.match(r"^\d+_(.+)$", name)
    if m:
        return m.group(1)
    return ""


def load_theoretical_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = row["expected_note"]
            out.setdefault(note, []).append(row)
    for note in out:
        out[note].sort(key=lambda r: safe_int(r["harmonic_index"]))
    return out


def load_chain_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hits = json.loads(row.get("hits_json", "[]") or "[]")
            except Exception:
                hits = []
            row["_hits"] = hits
            rows.append(row)
    return rows


def choose_best_chain(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = max(rows, key=lambda r: safe_float(r.get("chain_score", 0.0)))
    return best


def find_hit_for_harmonic(best_chain: dict[str, Any], harmonic_index: int) -> dict[str, Any] | None:
    for hit in best_chain.get("_hits", []):
        if safe_int(hit.get("harmonic_index", 0)) == harmonic_index:
            return hit
    return None


def classify_status(observed_hz: float, lower_hz: float, upper_hz: float, found: bool) -> str:
    if not found:
        return "MISSING"
    if lower_hz <= observed_hz <= upper_hz:
        return "MATCH"
    if observed_hz < lower_hz:
        return "SHIFTED_DOWN"
    if observed_hz > upper_hz:
        return "SHIFTED_UP"
    return "UNKNOWN"


def build_rows(
    reports_root: Path,
    theoretical: dict[str, list[dict[str, Any]]],
    *,
    anchor_token: str,
    anchor_hz: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for folder in sorted(reports_root.iterdir()):
        if not folder.is_dir():
            continue

        name = folder.name
        expected_note = expected_note_from_dirname(name)
        if not expected_note:
            continue

        chain_candidates_csv = folder / f"{name}__chain_candidates.csv"
        chain_summary_json = folder / f"{name}__chain_summary.json"

        if not chain_candidates_csv.exists():
            continue
        if expected_note not in theoretical:
            continue

        rows = load_chain_candidates(chain_candidates_csv)
        best_chain = choose_best_chain(rows)
        if best_chain is None:
            continue

        detected_root_note = str(best_chain.get("root_note_token", "")).strip()
        detected_root_hz = safe_float(best_chain.get("root_hz", 0.0))
        chain_score = safe_float(best_chain.get("chain_score", 0.0))

        for th in theoretical[expected_note]:
            harmonic_index = safe_int(th["harmonic_index"])
            theoretical_hz = safe_float(th["theoretical_hz"])
            lower_hz = safe_float(th["lower_hz_tolerance"])
            upper_hz = safe_float(th["upper_hz_tolerance"])
            theoretical_token = th["theoretical_token"]
            lower_token = th["lower_token_tolerance"]
            upper_token = th["upper_token_tolerance"]

            hit = find_hit_for_harmonic(best_chain, harmonic_index)

            if hit is None:
                observed_hz = ""
                observed_token = ""
                observed_probe_index = ""
                observed_response = ""
                delta_hz = ""
                delta_cents = ""
                status = "MISSING"
            else:
                observed_hz_f = safe_float(hit.get("matched_hz", 0.0))
                observed_token_s = hz_to_token_with_micro(
                    observed_hz_f,
                    anchor_token=anchor_token,
                    anchor_hz=anchor_hz,
                    micro_steps_per_semitone=12,
                    exact_mark=True,
                )
                delta_hz_f = observed_hz_f - theoretical_hz
                if observed_hz_f > 0 and theoretical_hz > 0:
                    delta_cents_f = 1200.0 * math.log2(observed_hz_f / theoretical_hz)
                else:
                    delta_cents_f = 0.0

                observed_hz = observed_hz_f
                observed_token = observed_token_s
                observed_probe_index = safe_int(hit.get("matched_probe_index", 0))
                observed_response = safe_float(hit.get("matched_response_value", 0.0))
                delta_hz = delta_hz_f
                delta_cents = delta_cents_f
                status = classify_status(observed_hz_f, lower_hz, upper_hz, True)

            out.append(
                {
                    "folder_name": name,
                    "expected_note": expected_note,
                    "detected_root_note": detected_root_note,
                    "detected_root_hz": detected_root_hz,
                    "chain_score": chain_score,
                    "harmonic_index": harmonic_index,
                    "theoretical_token": theoretical_token,
                    "theoretical_hz": theoretical_hz,
                    "lower_token_tolerance": lower_token,
                    "upper_token_tolerance": upper_token,
                    "lower_hz_tolerance": lower_hz,
                    "upper_hz_tolerance": upper_hz,
                    "observed_token": observed_token,
                    "observed_hz": observed_hz,
                    "observed_probe_index": observed_probe_index,
                    "observed_response": observed_response,
                    "delta_hz": delta_hz,
                    "delta_cents": delta_cents,
                    "status": status,
                    "chain_candidates_csv": str(chain_candidates_csv),
                    "chain_summary_json": str(chain_summary_json),
                }
            )

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "folder_name",
        "expected_note",
        "detected_root_note",
        "detected_root_hz",
        "chain_score",
        "harmonic_index",
        "theoretical_token",
        "theoretical_hz",
        "lower_token_tolerance",
        "upper_token_tolerance",
        "lower_hz_tolerance",
        "upper_hz_tolerance",
        "observed_token",
        "observed_hz",
        "observed_probe_index",
        "observed_response",
        "delta_hz",
        "delta_cents",
        "status",
        "chain_candidates_csv",
        "chain_summary_json",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("OBSERVED VS THEORETICAL HARMONICS REPORT")
    lines.append("=" * 140)
    lines.append("")

    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    current_folder = None
    for row in rows:
        if row["folder_name"] != current_folder:
            if current_folder is not None:
                lines.append("")
            current_folder = row["folder_name"]
            lines.append(f"[{row['folder_name']}]")
            lines.append(f"expected_note     : {row['expected_note']}")
            lines.append(f"detected_root_note: {row['detected_root_note']}")
            lines.append(f"detected_root_hz  : {row['detected_root_hz']}")
            lines.append(f"chain_score       : {row['chain_score']}")
            lines.append("")

        lines.append(
            f"h{row['harmonic_index']}: "
            f"theory={row['theoretical_token']} ({row['theoretical_hz']:.6f})  "
            f"obs={row['observed_token'] or '-'} ({row['observed_hz'] if row['observed_hz'] != '' else '-'})  "
            f"delta_hz={row['delta_hz'] if row['delta_hz'] != '' else '-'}  "
            f"delta_cents={row['delta_cents'] if row['delta_cents'] != '' else '-'}  "
            f"status={row['status']}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare observed chain harmonics with theoretical harmonic table."
    )
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--theoretical_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    args = ap.parse_args()

    reports_root = Path(args.reports_root).resolve()
    theoretical_csv = Path(args.theoretical_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()

    theoretical = load_theoretical_csv(theoretical_csv)
    rows = build_rows(
        reports_root=reports_root,
        theoretical=theoretical,
        anchor_token=str(args.anchor_token),
        anchor_hz=float(args.anchor_hz),
    )

    write_csv(out_csv, rows)
    write_txt(out_txt, rows)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote TXT: {out_txt}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()