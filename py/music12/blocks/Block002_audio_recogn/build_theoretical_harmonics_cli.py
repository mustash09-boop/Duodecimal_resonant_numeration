from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


# ============================================================
# LOCAL 12-RADIX HELPERS
# ------------------------------------------------------------
# Здесь мы не трогаем старую спорную семантику notation12.py,
# а строим эталон аккуратно и отдельно.
# ============================================================

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
    """
    Принимает что-то вроде:
      9.A-
      9.A
      8.C-
    и возвращает:
      (octave, step)
    """
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


def hz_from_token_with_anchor(token: str, anchor_token: str = "9.A-", anchor_hz: float = 440.0) -> float:
    """
    Теоретическая частота ноты по равномерной 12-ричной сетке
    относительно anchor_token = anchor_hz.
    """
    abs_anchor = token_to_abs_step(anchor_token)
    abs_target = token_to_abs_step(token)
    delta = abs_target - abs_anchor
    return float(anchor_hz * (2.0 ** (delta / 12.0)))


def hz_to_token_with_micro(
    freq_hz: float,
    *,
    anchor_token: str = "9.A-",
    anchor_hz: float = 440.0,
    micro_steps_per_semitone: int = 12,
    exact_mark: bool = True,
) -> str:
    """
    Перевод частоты в 12-ричный токен с микроотклонением:
      base + '-
      base + 'iX
      base + 'aX

    Здесь micro_steps_per_semitone = 12
    => один micro-шаг = 1/12 полутона.
    """
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

    # если перелезли через 12 микро-шагов, двигаем базовую ступень
    while magnitude >= micro_steps_per_semitone:
        if sign == "i":
            abs_note += 1
        else:
            abs_note -= 1
        magnitude -= micro_steps_per_semitone

    if magnitude == 0:
        base_token = abs_step_to_token(abs_note, micro="")
        return f"{base_token}'-" if exact_mark else base_token

    # magnitude 1..11 -> 12-ричная цифра 2..C по нашей текущей логике проекта:
    # 1 шаг = i2 / a2, 2 шага = i3 / a3, ..., 11 шагов = iC / aC
    # Это согласовано с тем, как сейчас у тебя пишутся probe_coords.
    digit = int_to_base12_digit(magnitude)
    base_token = abs_step_to_token(abs_note, micro="")
    return f"{base_token}'{sign}{digit}"


# ============================================================
# REPORT BUILDERS
# ============================================================

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def read_expected_notes_from_collected_txt(path: Path) -> list[str]:
    """
    Берём expected_note из уже собранного отчёта,
    чтобы идти в том же порядке, что и диапазон папок.
    """
    notes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("expected_note"):
            _, value = line.split(":", 1)
            note = value.strip()
            if note:
                notes.append(note)
    return notes


def build_rows(
    expected_notes: list[str],
    *,
    max_harmonic: int,
    anchor_token: str,
    anchor_hz: float,
    tolerance_micro_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for note in expected_notes:
        root_hz = hz_from_token_with_anchor(note, anchor_token=anchor_token, anchor_hz=anchor_hz)

        for h in range(1, max_harmonic + 1):
            harmonic_hz = root_hz * h

            theoretical_token = hz_to_token_with_micro(
                harmonic_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
                micro_steps_per_semitone=12,
                exact_mark=True,
            )

            # допуск вверх/вниз как частотный сдвиг на tolerance_micro_steps
            # в 1/12 полутона
            micro_semitones = tolerance_micro_steps / 12.0
            lower_hz = harmonic_hz * (2.0 ** (-micro_semitones / 12.0))
            upper_hz = harmonic_hz * (2.0 ** (+micro_semitones / 12.0))

            lower_token = hz_to_token_with_micro(
                lower_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
                micro_steps_per_semitone=12,
                exact_mark=True,
            )
            upper_token = hz_to_token_with_micro(
                upper_hz,
                anchor_token=anchor_token,
                anchor_hz=anchor_hz,
                micro_steps_per_semitone=12,
                exact_mark=True,
            )

            rows.append(
                {
                    "expected_note": note,
                    "root_hz": root_hz,
                    "harmonic_index": h,
                    "theoretical_hz": harmonic_hz,
                    "theoretical_token": theoretical_token,
                    "lower_hz_tolerance": lower_hz,
                    "upper_hz_tolerance": upper_hz,
                    "lower_token_tolerance": lower_token,
                    "upper_token_tolerance": upper_token,
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "expected_note",
        "root_hz",
        "harmonic_index",
        "theoretical_hz",
        "theoretical_token",
        "lower_hz_tolerance",
        "upper_hz_tolerance",
        "lower_token_tolerance",
        "upper_token_tolerance",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("THEORETICAL HARMONICS REPORT")
    lines.append("=" * 120)
    lines.append("")

    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    current_note = None
    for row in rows:
        note = row["expected_note"]
        if note != current_note:
            if current_note is not None:
                lines.append("")
            current_note = note
            lines.append(f"[{note}]")
            lines.append(f"root_hz: {row['root_hz']}")
            lines.append("")

        lines.append(
            f"h{row['harmonic_index']}: "
            f"{row['theoretical_token']}  "
            f"hz={row['theoretical_hz']:.6f}  "
            f"tol=[{row['lower_token_tolerance']} .. {row['upper_token_tolerance']}]"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build theoretical harmonics table for the full note range."
    )
    ap.add_argument("--collected_txt", required=True, help="Collected chain results TXT to read expected_note order from")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--max_harmonic", type=int, default=12)
    ap.add_argument("--anchor_token", default="9.A-")
    ap.add_argument("--anchor_hz", type=float, default=440.0)
    ap.add_argument("--tolerance_micro_steps", type=int, default=4, help="Tolerance in 1/12 semitone microsteps")
    args = ap.parse_args()

    collected_txt = Path(args.collected_txt).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_txt = Path(args.out_txt).resolve()

    expected_notes = read_expected_notes_from_collected_txt(collected_txt)

    rows = build_rows(
        expected_notes=expected_notes,
        max_harmonic=int(args.max_harmonic),
        anchor_token=str(args.anchor_token),
        anchor_hz=float(args.anchor_hz),
        tolerance_micro_steps=int(args.tolerance_micro_steps),
    )

    write_csv(out_csv, rows)
    write_txt(out_txt, rows)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote TXT: {out_txt}")
    print(f"Notes: {len(expected_notes)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()