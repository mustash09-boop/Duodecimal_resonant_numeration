from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration")
REPORTS_ROOT = PROJECT_ROOT / r"Block004_data\RealPiano_1\00_sources\reports"

# Диапазоны можно потом подстроить по фактической статистике
LOW_MAX_RADIAL = 6.0
MID_MAX_RADIAL = 9.0

PHASE_TOL_DEG = 1e-6
RADIAL_TOL = 1e-6


@dataclass(frozen=True)
class SimpleNote:
    octave: int
    degree: int
    micro: str = ""

    @property
    def phase_deg(self) -> float:
        return float(self.degree) * 30.0

    @property
    def radial_level(self) -> float:
        return float(self.octave) + float(self.degree) / 12.0

    @property
    def core_label(self) -> str:
        return f"{self.octave}.{self.degree}"

    def to_display(self) -> str:
        return self.core_label + self.micro


def parse_note_token(token: str) -> Optional[SimpleNote]:
    """
    Примеры:
      6.4
      6.4-
      6.4'i4
      7.4'a3
      11.1-
      C.C-
    """
    if token is None:
        return None

    token = str(token).strip().replace(" ", "")
    if not token:
        return None

    m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", token, flags=re.IGNORECASE)
    if not m:
        return None

    octave_str = m.group(1).upper()
    degree_str = m.group(2).upper()

    octave = duodecimal_str_to_int(octave_str)
    degree = duodecimal_str_to_int(degree_str)

    micro = token[m.end():]
    return SimpleNote(octave=octave, degree=degree, micro=micro)


def duodecimal_digit_to_int(ch: str) -> int:
    ch = ch.strip().upper()
    mapping = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
    }
    if ch not in mapping:
        raise ValueError(f"Unsupported duodecimal digit: {ch}")
    return mapping[ch]


def duodecimal_str_to_int(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty duodecimal string")

    value = 0
    for ch in s:
        value = value * 12 + duodecimal_digit_to_int(ch)
    return value


def int_to_duodecimal_digit(v: int) -> str:
    mapping = {
        1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
        7: "7", 8: "8", 9: "9", 10: "A", 11: "B", 12: "C",
    }
    if v not in mapping:
        raise ValueError(f"Unsupported duodecimal int: {v}")
    return mapping[v]


def parse_expected_note_from_filename(path: Path) -> SimpleNote:
    """
    Ожидаем имя:
      007__RealPiano_1__6.4-__stabilized__with_phase.csv
      088__RealPiano_1__11.1-__stabilized__with_phase.csv
      087__RealPiano_1__C.C-__stabilized__with_phase.csv
    """
    name = path.name

    m = re.search(
        r"__([1-9ABC]+)\.([1-9ABC]+)[^_]*__stabilized__with_phase\.csv$",
        name,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Cannot parse expected note from filename: {name}")

    octave_str = m.group(1).upper()
    degree_str = m.group(2).upper()

    octave = duodecimal_str_to_int(octave_str)
    degree = duodecimal_str_to_int(degree_str)

    return SimpleNote(octave=octave, degree=degree)


def safe_float(x: str) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def range_regime(radial: float) -> str:
    if radial < LOW_MAX_RADIAL:
        return "low"
    if radial < MID_MAX_RADIAL:
        return "mid"
    return "high"


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def classify_error(
    expected: SimpleNote,
    final_note: Optional[SimpleNote],
) -> str:
    if final_note is None:
        return "no_decision"

    phase_match = approx_equal(expected.phase_deg, final_note.phase_deg, PHASE_TOL_DEG)
    radial_diff = final_note.radial_level - expected.radial_level

    if phase_match and approx_equal(radial_diff, 0.0, RADIAL_TOL):
        return "exact_match"
    if phase_match and approx_equal(radial_diff, 1.0, RADIAL_TOL):
        return "phase_correct_radius_plus_1"
    if phase_match and approx_equal(radial_diff, -1.0, RADIAL_TOL):
        return "phase_correct_radius_minus_1"
    if phase_match:
        return "phase_correct_radius_other"
    return "phase_wrong"


def longest_run(values: list[str]) -> tuple[Optional[str], int]:
    if not values:
        return None, 0

    best_value = values[0]
    best_len = 1

    current_value = values[0]
    current_len = 1

    for v in values[1:]:
        if v == current_value:
            current_len += 1
        else:
            if current_len > best_len:
                best_len = current_len
                best_value = current_value
            current_value = v
            current_len = 1

    if current_len > best_len:
        best_len = current_len
        best_value = current_value

    return best_value, best_len


def phase_key(phase_deg: Optional[float]) -> Optional[str]:
    if phase_deg is None:
        return None
    return f"{phase_deg:.6f}"


def radial_key(radial: Optional[float]) -> Optional[str]:
    if radial is None:
        return None
    return f"{radial:.6f}"


def compute_phase_runs(rows: list[dict]) -> tuple[Optional[float], int, float]:
    values = []
    for r in rows:
        p = safe_float(r.get("phase_deg"))
        values.append(phase_key(p) if p is not None else "__NONE__")

    best_key, best_len = longest_run(values)
    if best_key in (None, "__NONE__"):
        return None, 0, 0.0

    best_phase = float(best_key)
    hold_ratio = best_len / len(rows) if rows else 0.0
    return best_phase, best_len, hold_ratio


def compute_radial_runs_within_phase(
    rows: list[dict],
    target_phase: Optional[float],
) -> tuple[Optional[float], int, float]:
    if target_phase is None:
        return None, 0, 0.0

    filtered = []
    for r in rows:
        p = safe_float(r.get("phase_deg"))
        rr = safe_float(r.get("radial_level"))
        if p is not None and rr is not None and approx_equal(p, target_phase, PHASE_TOL_DEG):
            filtered.append(radial_key(rr))

    if not filtered:
        return None, 0, 0.0

    best_key, best_len = longest_run(filtered)
    if best_key is None:
        return None, 0, 0.0

    best_radial = float(best_key)
    hold_ratio = best_len / len(filtered) if filtered else 0.0
    return best_radial, best_len, hold_ratio


def choose_final_note_from_chain(
    rows: list[dict],
    dominant_phase: Optional[float],
    dominant_radial: Optional[float],
) -> Optional[SimpleNote]:
    if dominant_phase is None:
        return None

    # Сначала ищем лучший кандидат точно в выбранной фазе и выбранном радиусе
    candidates = []
    for r in rows:
        note = parse_note_token(r.get("representative_rc_note", ""))
        if note is None:
            continue

        if not approx_equal(note.phase_deg, dominant_phase, PHASE_TOL_DEG):
            continue

        chain_score = safe_float(r.get("rc_chain_score")) or 0.0
        stab_score = safe_float(r.get("stabilization_score")) or 0.0
        support_hits = safe_float(r.get("support_hits")) or 0.0

        radial_penalty = 0.0
        if dominant_radial is not None:
            radial_penalty = abs(note.radial_level - dominant_radial)

        score = chain_score + 0.15 * stab_score + 0.5 * support_hits - 2.0 * radial_penalty
        candidates.append((score, note))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def summarize_file(csv_path: Path) -> dict:
    expected_note = parse_expected_note_from_filename(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    chosen_notes = []
    chosen_core_notes = []

    exact_match_count = 0
    phase_match_count = 0
    plus1_count = 0
    minus1_count = 0

    chain_score_sum = 0.0
    support_hits_sum = 0.0
    stabilization_score_sum = 0.0

    for r in rows:
        note = parse_note_token(r.get("representative_rc_note", ""))
        if note is not None:
            chosen_notes.append(note.to_display())
            chosen_core_notes.append(note.core_label)

            if approx_equal(note.phase_deg, expected_note.phase_deg, PHASE_TOL_DEG):
                phase_match_count += 1

            radial_diff = note.radial_level - expected_note.radial_level
            if approx_equal(note.phase_deg, expected_note.phase_deg, PHASE_TOL_DEG) and approx_equal(radial_diff, 0.0, RADIAL_TOL):
                exact_match_count += 1
            if approx_equal(note.phase_deg, expected_note.phase_deg, PHASE_TOL_DEG) and approx_equal(radial_diff, 1.0, RADIAL_TOL):
                plus1_count += 1
            if approx_equal(note.phase_deg, expected_note.phase_deg, PHASE_TOL_DEG) and approx_equal(radial_diff, -1.0, RADIAL_TOL):
                minus1_count += 1

        chain_score_sum += safe_float(r.get("rc_chain_score")) or 0.0
        support_hits_sum += safe_float(r.get("support_hits")) or 0.0
        stabilization_score_sum += safe_float(r.get("stabilization_score")) or 0.0

    row_count = len(rows)
    expected_phase = expected_note.phase_deg
    expected_radial = expected_note.radial_level

    dominant_phase, phase_run_length_max, phase_hold_ratio = compute_phase_runs(rows)
    dominant_radial, radial_run_length_max, radial_hold_ratio = compute_radial_runs_within_phase(rows, dominant_phase)

    final_note = choose_final_note_from_chain(rows, dominant_phase, dominant_radial)
    error_type = classify_error(expected_note, final_note)

    dominant_core_mode = Counter(chosen_core_notes).most_common(1)
    dominant_core_note = dominant_core_mode[0][0] if dominant_core_mode else ""

    exact_match_ratio = exact_match_count / row_count if row_count else 0.0
    phase_match_ratio = phase_match_count / row_count if row_count else 0.0
    plus1_ratio = plus1_count / row_count if row_count else 0.0
    minus1_ratio = minus1_count / row_count if row_count else 0.0

    chain_score_mean = chain_score_sum / row_count if row_count else 0.0
    support_hits_mean = support_hits_sum / row_count if row_count else 0.0
    stabilization_score_mean = stabilization_score_sum / row_count if row_count else 0.0

    # Сводный stable chain score
    stable_chain_score = (
        0.35 * phase_hold_ratio
        + 0.25 * radial_hold_ratio
        + 0.20 * min(chain_score_mean / 20.0, 1.0)
        + 0.10 * min(support_hits_mean / 8.0, 1.0)
        + 0.10 * min(stabilization_score_mean / 40.0, 1.0)
    )

    if error_type == "exact_match" and stable_chain_score >= 0.55:
        chain_status = "stable_fundamental"
    elif error_type == "phase_correct_radius_plus_1" and stable_chain_score >= 0.45:
        chain_status = "stable_harmonic_bias_up"
    elif error_type == "phase_correct_radius_minus_1" and stable_chain_score >= 0.45:
        chain_status = "stable_harmonic_bias_down"
    elif phase_match_ratio > 0.0 and exact_match_ratio == 0.0:
        chain_status = "phase_found_radial_unstable"
    elif stable_chain_score < 0.25:
        chain_status = "fragmented"
    else:
        chain_status = "mixed"

    summary = {
        "file": str(csv_path),
        "expected_note": expected_note.core_label,
        "expected_phase_deg": round(expected_phase, 6),
        "expected_radial_level": round(expected_radial, 6),
        "range_regime": range_regime(expected_radial),

        "row_count": row_count,

        "dominant_phase_deg": "" if dominant_phase is None else round(dominant_phase, 6),
        "dominant_radial_level": "" if dominant_radial is None else round(dominant_radial, 6),
        "dominant_core_note_mode": dominant_core_note,
        "final_note_candidate": "" if final_note is None else final_note.core_label,

        "phase_run_length_max": phase_run_length_max,
        "phase_hold_ratio": round(phase_hold_ratio, 6),
        "radial_run_length_max": radial_run_length_max,
        "radial_hold_ratio": round(radial_hold_ratio, 6),

        "exact_match_count": exact_match_count,
        "exact_match_ratio": round(exact_match_ratio, 6),
        "phase_match_count": phase_match_count,
        "phase_match_ratio": round(phase_match_ratio, 6),
        "phase_correct_radius_plus_1_count": plus1_count,
        "phase_correct_radius_plus_1_ratio": round(plus1_ratio, 6),
        "phase_correct_radius_minus_1_count": minus1_count,
        "phase_correct_radius_minus_1_ratio": round(minus1_ratio, 6),

        "chain_score_mean": round(chain_score_mean, 6),
        "support_hits_mean": round(support_hits_mean, 6),
        "stabilization_score_mean": round(stabilization_score_mean, 6),
        "stable_chain_score": round(stable_chain_score, 6),

        "chain_status": chain_status,
        "error_type": error_type,
    }

    return summary


def main() -> None:
    files = sorted(REPORTS_ROOT.rglob("*__stabilized__with_phase.csv"))
    if not files:
        print("No *__stabilized__with_phase.csv files found.")
        return

    summaries = [summarize_file(p) for p in files]

    out_csv = REPORTS_ROOT / "RealPiano_1__stable_chain_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    # Ещё одна маленькая сводка по типам ошибок
    error_counter = Counter(s["error_type"] for s in summaries)
    chain_counter = Counter(s["chain_status"] for s in summaries)
    regime_counter = Counter((s["range_regime"], s["error_type"]) for s in summaries)

    out_txt = REPORTS_ROOT / "RealPiano_1__stable_chain_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("STABLE CHAIN SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"files_analyzed: {len(summaries)}\n\n")

        f.write("ERROR TYPES\n")
        for k, v in error_counter.most_common():
            f.write(f"  {k}: {v}\n")

        f.write("\nCHAIN STATUS\n")
        for k, v in chain_counter.most_common():
            f.write(f"  {k}: {v}\n")

        f.write("\nRANGE REGIME x ERROR TYPE\n")
        for k, v in sorted(regime_counter.items()):
            f.write(f"  {k[0]} | {k[1]}: {v}\n")

    print(f"DONE: {out_csv}")
    print(f"DONE: {out_txt}")


if __name__ == "__main__":
    main()