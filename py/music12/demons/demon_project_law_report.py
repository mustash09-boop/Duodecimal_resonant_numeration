# -*- coding: utf-8 -*-
"""
music12.demons.demon_project_law_report

Project-law diagnostic demon for music12.

Purpose
-------
This demon does NOT execute DSP and does NOT patch code.
Its task is to inspect CSV / JSON / text-like project outputs (or generic
tabular reports) and produce a law-compliance report answering questions such as:

    - is a note being asserted without visible chain support?
    - is strongest_peak_note used as if it were already the final note?
    - is f0 fixed before chain confirmation?
    - is zero leaking into final token-like columns?

This is a first-layer regulatory demon.
It checks philosophical / ontological violations in produced data.

It is intentionally conservative:
if evidence is incomplete, the demon reports "warning" instead of pretending certainty.

Outputs
-------
- TXT report
- JSON report

Typical usage
-------------
python -m music12.demons.demon_project_law_report ^
    --input_csv "Block005_data\\...\\some_report.csv" ^
    --out_txt "_demon_logs\\project_law_report.txt" ^
    --out_json "_demon_logs\\project_law_report.json"

Optional:
    --focus block002
    --strict

Notes
-----
This demon is compatible with partial tables.
It does not require every expected column to exist.
Missing columns become "not_checked" or "warning", not fake failures.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from music12.core.project_law_guard import (
    active_laws,
    project_law_report_text,
)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

TOKEN_CANDIDATE_PATTERN = re.compile(
    r"(?P<tok>[0-9A-C]+\.[0-9A-C]'(?:-|[ia][0-9A-C]+)?)",
    flags=re.IGNORECASE,
)
DEFAULT_ENCODING = "utf-8"


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str                    # ok | warning | violation | not_checked
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemonLawReport:
    demon: str
    input_path: str
    row_count: int
    columns: List[str]
    focus: str
    strict: bool
    active_laws: Dict[str, Any]
    checks: List[CheckResult]
    summary_status: str
    summary_counts: Dict[str, int]
    guard_snapshot_text: str


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported input format: {path.suffix}")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _existing_cols(df: pd.DataFrame, names: Sequence[str]) -> List[str]:
    return [c for c in names if c in df.columns]


def _token_like_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    tokenish_markers = (
        "token",
        "note",
        "chosen_rc",
        "strongest_peak",
        "root",
        "support_h",
        "octave_mode",
        "degree12_mode",
    )
    for c in df.columns:
        lc = c.lower()
        if any(m in lc for m in tokenish_markers):
            out.append(c)
    return out


def _contains_zero_token_text(text: str) -> bool:
    s = _safe_str(text)
    if not s:
        return False
    for m in TOKEN_CANDIDATE_PATTERN.finditer(s):
        tok = m.group("tok").upper()
        if "0" in tok:
            return True
    return False


def _series_nonempty_mask(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: _safe_str(x) != "")


def _count_nonempty(series: pd.Series) -> int:
    return int(_series_nonempty_mask(series).sum())


def _row_indices_where(mask: pd.Series, limit: int = 20) -> List[int]:
    idx = list(mask[mask].index[:limit])
    return [int(i) for i in idx]


def _choose_summary_status(checks: Sequence[CheckResult], strict: bool) -> str:
    has_violation = any(c.status == "violation" for c in checks)
    has_warning = any(c.status == "warning" for c in checks)

    if has_violation:
        return "violation"
    if strict and has_warning:
        return "warning"
    if has_warning:
        return "warning"
    return "ok"


def _summary_counts(checks: Sequence[CheckResult]) -> Dict[str, int]:
    counts = {
        "ok": 0,
        "warning": 0,
        "violation": 0,
        "not_checked": 0,
    }
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    return counts


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------

def check_note_without_chain(df: pd.DataFrame) -> CheckResult:
    """
    Detect rows where final-note-like columns are present but chain evidence is absent.

    Conservative heuristic:
    - if final note candidate exists
    - and there is no chain-related column/value visible
    then raise warning/violation
    """
    final_note_cols = _existing_cols(
        df,
        [
            "final_note",
            "note",
            "note_token",
            "detected_note",
            "chosen_note",
            "chosen_rc_note",
        ],
    )
    chain_cols = _existing_cols(
        df,
        [
            "chain_id",
            "chain_score",
            "chain_status",
            "chain_confirmed",
            "matched_harmonics",
            "missing_harmonics",
            "chain_string",
            "chain_label_string",
        ],
    )

    if not final_note_cols:
        return CheckResult(
            name="note_without_chain",
            status="not_checked",
            message="No final-note-like columns found.",
        )

    final_series = df[final_note_cols[0]]
    final_mask = _series_nonempty_mask(final_series)

    if not chain_cols:
        n = int(final_mask.sum())
        if n == 0:
            return CheckResult(
                name="note_without_chain",
                status="warning",
                message="Final-note-like column exists, but no note rows are populated and no chain columns exist.",
                details={"final_note_col": final_note_cols[0]},
            )
        return CheckResult(
            name="note_without_chain",
            status="violation",
            message="Final-note-like values exist, but no chain-related columns are present.",
            details={
                "final_note_col": final_note_cols[0],
                "populated_rows": n,
                "example_rows": _row_indices_where(final_mask),
            },
        )

    chain_presence = pd.Series(False, index=df.index)
    for c in chain_cols:
        chain_presence = chain_presence | _series_nonempty_mask(df[c])

    bad_mask = final_mask & (~chain_presence)
    bad_count = int(bad_mask.sum())

    if bad_count == 0:
        return CheckResult(
            name="note_without_chain",
            status="ok",
            message="No rows found where final note is populated without any visible chain evidence.",
            details={
                "final_note_col": final_note_cols[0],
                "chain_cols": chain_cols,
            },
        )

    return CheckResult(
        name="note_without_chain",
        status="violation",
        message="Some rows contain final-note-like values without visible chain evidence.",
        details={
            "final_note_col": final_note_cols[0],
            "chain_cols": chain_cols,
            "bad_row_count": bad_count,
            "example_rows": _row_indices_where(bad_mask),
        },
    )


def check_strongest_peak_used_as_note(df: pd.DataFrame) -> CheckResult:
    """
    Detect suspicious equality between strongest_peak_note and final/chosen note.

    This is NOT always a violation, because they can legitimately coincide.
    Violation happens only if they coincide broadly while no chain evidence exists.
    Otherwise warning at most.
    """
    strongest_cols = _existing_cols(df, ["strongest_peak_note"])
    final_note_cols = _existing_cols(
        df,
        ["final_note", "note", "note_token", "detected_note", "chosen_note", "chosen_rc_note"],
    )
    chain_cols = _existing_cols(
        df,
        ["chain_id", "chain_score", "chain_status", "chain_confirmed", "matched_harmonics", "chain_string"],
    )

    if not strongest_cols or not final_note_cols:
        return CheckResult(
            name="strongest_peak_used_as_note",
            status="not_checked",
            message="Required columns not found.",
        )

    s_col = strongest_cols[0]
    f_col = final_note_cols[0]

    s = df[s_col].apply(_safe_str)
    f = df[f_col].apply(_safe_str)

    comparable = (s != "") & (f != "")
    if int(comparable.sum()) == 0:
        return CheckResult(
            name="strongest_peak_used_as_note",
            status="warning",
            message="Columns exist but no comparable populated rows were found.",
            details={"strongest_peak_col": s_col, "final_note_col": f_col},
        )

    same_mask = comparable & (s == f)
    same_count = int(same_mask.sum())
    ratio = same_count / max(int(comparable.sum()), 1)

    chain_presence = pd.Series(False, index=df.index)
    for c in chain_cols:
        chain_presence = chain_presence | _series_nonempty_mask(df[c])

    same_without_chain = same_mask & (~chain_presence)
    same_without_chain_count = int(same_without_chain.sum())

    if same_without_chain_count > 0 and ratio >= 0.5:
        return CheckResult(
            name="strongest_peak_used_as_note",
            status="violation",
            message="strongest_peak_note appears to function as final note in many rows without visible chain evidence.",
            details={
                "strongest_peak_col": s_col,
                "final_note_col": f_col,
                "same_count": same_count,
                "comparable_count": int(comparable.sum()),
                "same_ratio": ratio,
                "same_without_chain_count": same_without_chain_count,
                "example_rows": _row_indices_where(same_without_chain),
            },
        )

    if ratio >= 0.5:
        return CheckResult(
            name="strongest_peak_used_as_note",
            status="warning",
            message="strongest_peak_note often equals the final note column. This may be legitimate, but should be reviewed.",
            details={
                "strongest_peak_col": s_col,
                "final_note_col": f_col,
                "same_count": same_count,
                "comparable_count": int(comparable.sum()),
                "same_ratio": ratio,
                "chain_cols": chain_cols,
            },
        )

    return CheckResult(
        name="strongest_peak_used_as_note",
        status="ok",
        message="No broad pattern suggesting strongest_peak_note is being directly used as final note.",
        details={
            "strongest_peak_col": s_col,
            "final_note_col": f_col,
            "same_count": same_count,
            "comparable_count": int(comparable.sum()),
            "same_ratio": ratio,
        },
    )


def check_f0_before_chain(df: pd.DataFrame) -> CheckResult:
    """
    Detect rows where root/f0-like fields are populated but chain evidence is absent.

    Again conservative: if root hypothesis exists, that's allowed.
    We flag only rows suggesting finalized root assignment without chain support.
    """
    root_cols = _existing_cols(
        df,
        [
            "f0_note",
            "root_token",
            "root_note",
            "chosen_rc_note",
            "final_root",
        ],
    )
    chain_cols = _existing_cols(
        df,
        [
            "chain_id",
            "chain_score",
            "chain_status",
            "chain_confirmed",
            "matched_harmonics",
            "chain_string",
        ],
    )

    if not root_cols:
        return CheckResult(
            name="f0_before_chain",
            status="not_checked",
            message="No root/f0-like columns found.",
        )

    root_col = root_cols[0]
    root_mask = _series_nonempty_mask(df[root_col])

    if int(root_mask.sum()) == 0:
        return CheckResult(
            name="f0_before_chain",
            status="warning",
            message="Root/f0-like column exists but no populated rows were found.",
            details={"root_col": root_col},
        )

    if not chain_cols:
        return CheckResult(
            name="f0_before_chain",
            status="warning",
            message="Root/f0-like values exist but no chain-related columns were found; cannot distinguish hypothesis from confirmed root.",
            details={
                "root_col": root_col,
                "example_rows": _row_indices_where(root_mask),
            },
        )

    chain_presence = pd.Series(False, index=df.index)
    for c in chain_cols:
        chain_presence = chain_presence | _series_nonempty_mask(df[c])

    suspicious = root_mask & (~chain_presence)
    suspicious_count = int(suspicious.sum())

    if suspicious_count == 0:
        return CheckResult(
            name="f0_before_chain",
            status="ok",
            message="No rows found where root/f0-like values appear without visible chain evidence.",
            details={
                "root_col": root_col,
                "chain_cols": chain_cols,
            },
        )

    return CheckResult(
        name="f0_before_chain",
        status="warning",
        message="Some rows contain root/f0-like values without visible chain evidence. This may indicate early root fixation.",
        details={
            "root_col": root_col,
            "chain_cols": chain_cols,
            "suspicious_row_count": suspicious_count,
            "example_rows": _row_indices_where(suspicious),
        },
    )


def check_zero_leak(df: pd.DataFrame) -> CheckResult:
    """
    Detect zero leaking into token-like columns.
    Internal coordinates may legitimately contain 0 in some internal structures,
    but final token-like textual outputs must not leak 0.
    """
    cols = _token_like_columns(df)
    if not cols:
        return CheckResult(
            name="zero_leak",
            status="not_checked",
            message="No token-like columns found.",
        )

    violations: Dict[str, Any] = {}
    total_hits = 0

    for c in cols:
        mask = df[c].apply(lambda x: _contains_zero_token_text(_safe_str(x)))
        hits = int(mask.sum())
        if hits > 0:
            total_hits += hits
            violations[c] = {
                "hit_count": hits,
                "example_rows": _row_indices_where(mask),
                "example_values": [
                    _safe_str(v) for v in df.loc[mask, c].head(5).tolist()
                ],
            }

    if total_hits == 0:
        return CheckResult(
            name="zero_leak",
            status="ok",
            message="No zero leak detected in token-like columns.",
            details={"checked_columns": cols},
        )

    return CheckResult(
        name="zero_leak",
        status="violation",
        message="Zero leak detected in token-like textual outputs.",
        details={
            "checked_columns": cols,
            "total_hits": total_hits,
            "violations_by_column": violations,
        },
    )


def check_inference_order_signature(df: pd.DataFrame) -> CheckResult:
    """
    Heuristic check for presence of layered fields suggesting the intended order:
        observation -> curve -> chain -> note
    """
    observation_cols = _existing_cols(
        df,
        ["bin_index", "peak_hz", "energy", "amplitude", "resonance_score", "time_index"],
    )
    curve_cols = _existing_cols(
        df,
        ["curve_id", "trajectory_id", "continuity_score", "stability_score"],
    )
    chain_cols = _existing_cols(
        df,
        ["chain_id", "chain_score", "chain_status", "matched_harmonics", "chain_string"],
    )
    note_cols = _existing_cols(
        df,
        ["final_note", "note", "note_token", "detected_note", "chosen_note", "chosen_rc_note"],
    )

    present = {
        "observation": bool(observation_cols),
        "curve": bool(curve_cols),
        "chain": bool(chain_cols),
        "note": bool(note_cols),
    }

    if all(present.values()):
        return CheckResult(
            name="inference_order_signature",
            status="ok",
            message="All major inference layers are visible in the table signature.",
            details={
                "observation_cols": observation_cols,
                "curve_cols": curve_cols,
                "chain_cols": chain_cols,
                "note_cols": note_cols,
            },
        )

    return CheckResult(
        name="inference_order_signature",
        status="warning",
        message="Not all inference layers are visible in the table signature. This may be normal for an intermediate file.",
        details={
            "present_layers": present,
            "observation_cols": observation_cols,
            "curve_cols": curve_cols,
            "chain_cols": chain_cols,
            "note_cols": note_cols,
        },
    )


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

def analyze_table(df: pd.DataFrame, *, focus: str = "generic", strict: bool = False) -> DemonLawReport:
    checks = [
        check_note_without_chain(df),
        check_strongest_peak_used_as_note(df),
        check_f0_before_chain(df),
        check_zero_leak(df),
        check_inference_order_signature(df),
    ]

    summary_status = _choose_summary_status(checks, strict=strict)
    counts = _summary_counts(checks)

    return DemonLawReport(
        demon="music12.demons.demon_project_law_report",
        input_path="",
        row_count=int(len(df)),
        columns=[str(c) for c in df.columns],
        focus=str(focus),
        strict=bool(strict),
        active_laws=active_laws(),
        checks=checks,
        summary_status=summary_status,
        summary_counts=counts,
        guard_snapshot_text=project_law_report_text(),
    )


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------

def _report_to_dict(report: DemonLawReport) -> Dict[str, Any]:
    data = asdict(report)
    data["checks"] = [asdict(c) for c in report.checks]
    return data


def _render_txt(report: DemonLawReport) -> str:
    lines: List[str] = []
    lines.append("music12 demon project law report")
    lines.append("================================")
    lines.append(f"input_path: {report.input_path}")
    lines.append(f"row_count: {report.row_count}")
    lines.append(f"focus: {report.focus}")
    lines.append(f"strict: {report.strict}")
    lines.append(f"summary_status: {report.summary_status}")
    lines.append("")

    lines.append("summary_counts:")
    for k, v in report.summary_counts.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("checks:")
    for chk in report.checks:
        lines.append(f"- {chk.name}")
        lines.append(f"  status: {chk.status}")
        lines.append(f"  message: {chk.message}")
        if chk.details:
            lines.append("  details:")
            for dk, dv in chk.details.items():
                lines.append(f"    {dk}: {dv}")
        lines.append("")

    lines.append("project_law_guard:")
    lines.append(report.guard_snapshot_text)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="music12 project law diagnostic demon")
    p.add_argument("--input_csv", required=True, help="Input CSV/TSV/TXT/XLSX/XLS table to inspect")
    p.add_argument("--out_txt", required=True, help="Output TXT report path")
    p.add_argument("--out_json", required=True, help="Output JSON report path")
    p.add_argument("--focus", default="generic", help="Context hint, e.g. generic | block002 | block005")
    p.add_argument("--strict", action="store_true", help="Treat warnings more seriously in summary")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    out_txt = Path(args.out_txt)
    out_json = Path(args.out_json)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = _read_table(input_path)
    report = analyze_table(df, focus=args.focus, strict=bool(args.strict))
    report.input_path = str(input_path)

    txt = _render_txt(report)
    js = _report_to_dict(report)

    out_txt.write_text(txt, encoding=DEFAULT_ENCODING)
    out_json.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding=DEFAULT_ENCODING)

    print(f"DONE: {out_txt}")
    print(f"DONE: {out_json}")


if __name__ == "__main__":
    main()
