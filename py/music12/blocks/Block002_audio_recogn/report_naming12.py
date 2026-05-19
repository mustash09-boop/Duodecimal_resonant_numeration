from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoteIdentity:
    index: str          # "001"
    instrument: str     # "RealPiano_1"
    note_token: str     # "5.A-"

    @property
    def prefix(self) -> str:
        return f"{self.index}__{self.instrument}__{self.note_token}"


def note_report_dir(reports_root: str | Path, note: NoteIdentity) -> Path:
    return Path(reports_root) / note.prefix


def report_file(report_dir: str | Path, note: NoteIdentity, suffix: str) -> Path:
    """
    suffix examples:
      "__probe_matrix.csv"
      "__framewise.csv"
      "__chain_candidates.csv"
    """
    return Path(report_dir) / f"{note.prefix}{suffix}"


def lab_file(lab_root: str | Path, subdir: str, note: NoteIdentity, suffix: str) -> Path:
    """
    suffix examples:
      "__target_root_convergence.csv"
      "__harmonic_frequency.csv"
      "__note_emergence_signature.json"
    """
    return Path(lab_root) / subdir / f"{note.prefix}{suffix}"