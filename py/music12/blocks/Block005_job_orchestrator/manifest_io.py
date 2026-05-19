from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Type, TypeVar

from .models import JobSpec, JobStatus, JobResult, JobManifestRow

T = TypeVar("T")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj.to_dict(), f, ensure_ascii=False, indent=2)


def read_json(path: str | Path, cls: Type[T]) -> T:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return cls(**data)


def write_manifest_csv(path: str | Path, rows: Iterable[JobManifestRow]) -> None:
    p = Path(path)
    ensure_parent(p)
    rows = list(rows)
    fieldnames = list(JobManifestRow.__dataclass_fields__.keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row.to_dict())


def read_manifest_csv(path: str | Path) -> List[JobManifestRow]:
    p = Path(path)
    out: List[JobManifestRow] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(JobManifestRow(**row))
    return out


def job_paths(jobs_root: str | Path, job_id: str) -> dict:
    root = Path(jobs_root) / job_id
    return {
        "root": root,
        "spec": root / "job.json",
        "status": root / "status.json",
        "result": root / "result.json",
        "stdout": root / "stdout.txt",
        "stderr": root / "stderr.txt",
    }