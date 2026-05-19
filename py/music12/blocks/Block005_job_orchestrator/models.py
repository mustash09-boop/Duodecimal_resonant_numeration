from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any


@dataclass
class JobSpec:
    job_id: str
    task_kind: str
    wav_path: str
    out_dir: str

    time_start: Optional[float] = None
    time_end: Optional[float] = None

    octave_min: str = "5"
    octave_max: str = "C"

    detail_depth: int = 2
    projection_depth: int = 2
    time_step_seconds: float = 1.0 / 60.0
    window_seconds: float = 0.08
    harmonic_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03]
    )

    logdir: str = "_demon_logs"
    maxwell_tag: Optional[str] = None

    extra_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobStatus:
    job_id: str
    state: str  # queued / running / done / failed / timeout / submitted
    provider: str = "local"
    provider_job_id: Optional[str] = None

    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    message: str = ""
    return_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobResult:
    job_id: str
    state: str
    probe_matrix_csv: Optional[str] = None
    probe_meta_json: Optional[str] = None
    probe_times_csv: Optional[str] = None
    probe_coords_csv: Optional[str] = None
    maxwell_report_json: Optional[str] = None
    maxwell_report_txt: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobManifestRow:
    job_id: str
    task_kind: str
    wav_path: str
    out_dir: str
    state: str
    provider: str = "local"
    provider_job_id: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)