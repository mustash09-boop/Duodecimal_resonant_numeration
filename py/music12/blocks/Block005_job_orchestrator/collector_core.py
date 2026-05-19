from __future__ import annotations

from pathlib import Path
from typing import List

from .manifest_io import job_paths, read_json
from .models import JobManifestRow, JobResult, JobStatus


def collect_rows(jobs_root: str, manifest_rows: List[JobManifestRow]) -> List[JobManifestRow]:
    updated: List[JobManifestRow] = []

    for row in manifest_rows:
        paths = job_paths(jobs_root, row.job_id)
        if Path(paths["status"]).exists():
            status = read_json(paths["status"], JobStatus)
            row.state = status.state
            row.provider = status.provider
            row.provider_job_id = status.provider_job_id or ""
            if status.message:
                row.notes = status.message
        updated.append(row)

    return updated