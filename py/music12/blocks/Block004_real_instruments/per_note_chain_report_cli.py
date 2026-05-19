from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration")
REPORTS_ROOT = PROJECT_ROOT / r"Block004_data\RealPiano_1\00_sources\reports"

PHASE_TOL = 1e-6
RADIAL_TOL = 1e-6


# =========================
# NOTE MODEL
# =========================

@dataclass
class SimpleNote:
    octave: int
    degree: int
    micro: str
    original: str

    @property
    def phase_deg(self) -> float:
        return self.degree * 30.0

    @property
    def radial(self) -> float:
        return self.octave + self.degree / 12.0

    @property
    def core(self) -> str:
        return f"{int_to_duodecimal_str(self.octave)}.{int_to_duodecimal_str(self.degree)}"

    @property
    def full(self) -> str:
        return self.original


# =========================
# PARSING
# =========================

def duodecimal_digit_to_int(ch: str) -> int:
    mapping = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
    }
    ch = ch.upper()
    if ch not in mapping:
        raise ValueError(f"Unsupported duodecimal digit: {ch}")
    return mapping[ch]


def int_to_duodecimal_digit(v: int) -> str:
    mapping = {
        1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
        7: "7", 8: "8", 9: "9", 10: "A", 11: "B", 12: "C",
    }
    if v not in mapping:
        raise ValueError(f"Unsupported duodecimal int: {v}")
    return mapping[v]


def duodecimal_str_to_int(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty duodecimal string")

    val = 0
    for ch in s:
        val = val * 12 + duodecimal_digit_to_int(ch)
    return val


def int_to_duodecimal_str(n: int) -> str:
    if n <= 0:
        raise ValueError("Only positive integers are supported in this notation")

    if n <= 12:
        return int_to_duodecimal_digit(n)

    digits: list[str] = []
    value = n

    while value > 0:
        value, rem = divmod(value, 12)
        if rem == 0:
            value -= 1
            rem = 12
        digits.append(int_to_duodecimal_digit(rem))

    return "".join(reversed(digits))


def parse_note(token: str) -> Optional[SimpleNote]:
    if not token:
        return None

    token = token.strip().replace(" ", "")
    m = re.match(r"^([1-9ABC]+)\.([1-9ABC]+)", token, flags=re.IGNORECASE)
    if not m:
        return None

    octave = duodecimal_str_to_int(m.group(1))
    degree = duodecimal_str_to_int(m.group(2))
    micro = token[m.end():]

    return SimpleNote(
        octave=octave,
        degree=degree,
        micro=micro,
        original=token,
    )


def parse_expected(path: Path) -> SimpleNote:
    name = path.name

    m = re.search(
        r"__([1-9ABC]+)\.([1-9ABC]+)([^_]*)__stabilized__with_phase\.csv$",
        name,
        flags=re.IGNORECASE,
    )

    if not m:
        raise ValueError(f"Cannot parse expected note from filename: {name}")

    octave = duodecimal_str_to_int(m.group(1))
    degree = duodecimal_str_to_int(m.group(2))
    micro = m.group(3)

    return SimpleNote(
        octave=octave,
        degree=degree,
        micro=micro,
        original=f"{m.group(1)}.{m.group(2)}{micro}",
    )


# =========================
# HELPERS
# =========================

def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def safe_float(x) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


# =========================
# MAIN ANALYSIS
# =========================

def process_file(path: Path) -> None:
    expected = parse_expected(path)

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    output_rows = []

    current_chain_id = 0
    last_phase = None

    exact_count = 0
    plus1_count = 0
    minus1_count = 0
    phase_only_count = 0
    wrong_count = 0

    first_phase_contact_segment = None
    first_exact_contact_segment = None

    for r in rows:
        note = parse_note(r.get("representative_rc_note", ""))
        if note is None:
            continue

        phase = safe_float(r.get("phase_deg"))
        radial = safe_float(r.get("radial_level"))

        if phase is None or radial is None:
            continue

        segment_index = r.get("segment_index", "")

        # =========================
        # CHAIN DETECTION
        # =========================
        if last_phase is None or not approx(last_phase, phase, PHASE_TOL):
            current_chain_id += 1
        last_phase = phase

        # =========================
        # ERROR TYPES
        # =========================
        phase_match = approx(note.phase_deg, expected.phase_deg, PHASE_TOL)
        radial_diff = note.radial - expected.radial

        if phase_match and approx(radial_diff, 0.0, RADIAL_TOL):
            error = "exact"
            exact_count += 1
            if first_exact_contact_segment is None:
                first_exact_contact_segment = segment_index
        elif phase_match and approx(radial_diff, 1.0, RADIAL_TOL):
            error = "+1"
            plus1_count += 1
        elif phase_match and approx(radial_diff, -1.0, RADIAL_TOL):
            error = "-1"
            minus1_count += 1
        elif phase_match:
            error = "phase_only"
            phase_only_count += 1
        else:
            error = "wrong"
            wrong_count += 1

        if phase_match and first_phase_contact_segment is None:
            first_phase_contact_segment = segment_index

        # =========================
        # OUTPUT ROW
        # =========================
        output_rows.append({
            "segment_index": segment_index,
            "time_start": r.get("window_start_frame", ""),
            "time_end": r.get("window_end_frame", ""),

            "expected_note_full": expected.full,
            "expected_note_core": expected.core,

            "note_full": note.full,
            "note_core": note.core,

            "phase_deg": phase,
            "phase_delta": r.get("phase_delta", ""),
            "radial": radial,
            "radial_delta": r.get("radial_delta", ""),

            "chain_id": current_chain_id,

            "rc_chain_score": r.get("rc_chain_score", ""),
            "support_hits": r.get("support_hits", ""),
            "stabilization_score": r.get("stabilization_score", ""),
            "stabilization_role": r.get("stabilization_role", ""),
            "stabilization_reason": r.get("stabilization_reason", ""),

            "phase_match_expected": int(phase_match),
            "radial_diff_from_expected": round(radial_diff, 6),
            "exact_match_expected": int(error == "exact"),

            "error_type": error,
        })

    if not output_rows:
        print(f"SKIP (no valid rows): {path}")
        return

    # =========================
    # SAVE FULL TABLE
    # =========================
    out_dir = path.parent / path.stem.replace("__stabilized__with_phase", "__chain_report")
    out_dir.mkdir(exist_ok=True)

    out_csv = out_dir / "chain_full.csv"

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    # =========================
    # PASSPORT
    # =========================
    chain_lengths = defaultdict(int)
    chain_error_counts = defaultdict(lambda: defaultdict(int))

    for row in output_rows:
        cid = row["chain_id"]
        chain_lengths[cid] += 1
        chain_error_counts[cid][row["error_type"]] += 1

    longest_chain_id = None
    longest_chain = 0
    for cid, ln in chain_lengths.items():
        if ln > longest_chain:
            longest_chain = ln
            longest_chain_id = cid

    dominant_error_in_longest = ""
    if longest_chain_id is not None:
        err_counter = chain_error_counts[longest_chain_id]
        if err_counter:
            dominant_error_in_longest = max(err_counter.items(), key=lambda x: x[1])[0]

    passport = {
        "expected_note_full": expected.full,
        "expected_note_core": expected.core,
        "total_segments": len(output_rows),
        "chains_count": len(chain_lengths),
        "longest_chain_id": longest_chain_id,
        "longest_chain_length": longest_chain,
        "dominant_error_in_longest_chain": dominant_error_in_longest,
        "first_phase_contact_segment": first_phase_contact_segment,
        "first_exact_contact_segment": first_exact_contact_segment,
        "exact_count": exact_count,
        "plus1_count": plus1_count,
        "minus1_count": minus1_count,
        "phase_only_count": phase_only_count,
        "wrong_count": wrong_count,
    }

    passport_path = out_dir / "passport.txt"
    with passport_path.open("w", encoding="utf-8") as f:
        for k, v in passport.items():
            f.write(f"{k}: {v}\n")

    print("DONE:", out_dir)


# =========================
# ENTRY
# =========================

def main() -> None:
    files = sorted(REPORTS_ROOT.rglob("*__stabilized__with_phase.csv"))

    if not files:
        print("No *__stabilized__with_phase.csv files found.")
        return

    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()