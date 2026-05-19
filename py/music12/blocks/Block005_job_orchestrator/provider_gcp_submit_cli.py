from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .manifest_io import job_paths, read_json, read_manifest_csv, write_json, write_manifest_csv
from .models import JobManifestRow, JobSpec, JobStatus
from .provider_gcp_batch import GcpBatchProvider, GcpBatchRuntimeConfig


def _build_provider_from_args(args: argparse.Namespace) -> GcpBatchProvider:
    cfg = GcpBatchRuntimeConfig(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket_name,
        bucket_mount_path=args.bucket_mount_path,
        container_image=args.container_image,
        machine_type=args.machine_type,
        provisioning_model=args.provisioning_model,
        service_account_email=args.service_account_email,
        cpu_milli=args.cpu_milli,
        memory_mib=args.memory_mib,
        max_run_duration_sec=args.max_run_duration_sec,
        max_retry_count=args.max_retry_count,
        logs_to_cloud_logging=not args.no_cloud_logging,
        jobs_prefix=args.jobs_prefix,
        results_prefix=args.results_prefix,
        env_label=args.env_label,
        type_label=args.type_label,
    )
    return GcpBatchProvider(cfg)


def _iter_job_ids_from_manifest_rows(rows: list[JobManifestRow]) -> Iterable[str]:
    for row in rows:
        yield row.job_id


def _load_spec(jobs_root: str, job_id: str) -> JobSpec:
    paths = job_paths(jobs_root, job_id)
    return read_json(paths["spec"], JobSpec)


def _write_status_submitted(
    jobs_root: str,
    job_id: str,
    provider_job_id: str,
    message: str,
) -> None:
    paths = job_paths(jobs_root, job_id)
    status = JobStatus(
        job_id=job_id,
        state="submitted",
        provider="gcp_batch",
        provider_job_id=provider_job_id,
        created_at=None,
        started_at=None,
        finished_at=None,
        message=message,
        return_code=None,
    )
    write_json(paths["status"], status)


def _write_status_polled(
    jobs_root: str,
    job_id: str,
    provider_job_id: str,
    polled_state: str,
    message: str,
) -> None:
    paths = job_paths(jobs_root, job_id)

    # Normalize GCP Batch states into our project states.
    state_map = {
        "QUEUED": "submitted",
        "SCHEDULED": "submitted",
        "RUNNING": "running",
        "SUCCEEDED": "done",
        "FAILED": "failed",
        "DELETION_IN_PROGRESS": "cancelled",
        "STATE_UNSPECIFIED": "unknown",
        "UNKNOWN": "unknown",
    }
    normalized = state_map.get(polled_state, polled_state.lower())

    status = JobStatus(
        job_id=job_id,
        state=normalized,
        provider="gcp_batch",
        provider_job_id=provider_job_id,
        created_at=None,
        started_at=None,
        finished_at=None,
        message=message,
        return_code=None,
    )
    write_json(paths["status"], status)


def _update_manifest_row_submit(
    row: JobManifestRow,
    provider_job_id: str,
) -> JobManifestRow:
    row.state = "submitted"
    row.provider = "gcp_batch"
    row.provider_job_id = provider_job_id
    row.notes = "submitted to GCP Batch"
    return row


def _update_manifest_row_poll(
    row: JobManifestRow,
    polled_state: str,
) -> JobManifestRow:
    state_map = {
        "QUEUED": "submitted",
        "SCHEDULED": "submitted",
        "RUNNING": "running",
        "SUCCEEDED": "done",
        "FAILED": "failed",
        "DELETION_IN_PROGRESS": "cancelled",
        "STATE_UNSPECIFIED": "unknown",
        "UNKNOWN": "unknown",
    }
    row.state = state_map.get(polled_state, polled_state.lower())
    row.provider = "gcp_batch"
    row.notes = f"gcp_state={polled_state}"
    return row


def cmd_submit(args: argparse.Namespace) -> None:
    provider = _build_provider_from_args(args)
    rows = read_manifest_csv(args.manifest_csv)

    selected_ids: set[str] | None = None
    if args.job_id:
        selected_ids = {args.job_id}
    elif args.only_queued:
        selected_ids = {row.job_id for row in rows if row.state == "queued"}

    submitted = 0
    skipped = 0
    updated_rows: list[JobManifestRow] = []

    for row in rows:
        if selected_ids is not None and row.job_id not in selected_ids:
            updated_rows.append(row)
            skipped += 1
            continue

        if row.state not in {"queued", "failed", "timeout"} and not args.force:
            updated_rows.append(row)
            skipped += 1
            continue

        spec = _load_spec(args.jobs_root, row.job_id)
        provider_job_id = provider.submit(spec)

        _write_status_submitted(
            jobs_root=args.jobs_root,
            job_id=row.job_id,
            provider_job_id=provider_job_id,
            message="submitted to GCP Batch",
        )
        updated_rows.append(_update_manifest_row_submit(row, provider_job_id))
        submitted += 1

        print(f"submitted job_id={row.job_id} provider_job_id={provider_job_id}")

    write_manifest_csv(args.manifest_csv, updated_rows)
    print(f"submitted={submitted}")
    print(f"skipped={skipped}")


def cmd_poll(args: argparse.Namespace) -> None:
    provider = _build_provider_from_args(args)
    rows = read_manifest_csv(args.manifest_csv)

    polled = 0
    updated_rows: list[JobManifestRow] = []

    for row in rows:
        if row.provider != "gcp_batch" or not row.provider_job_id:
            updated_rows.append(row)
            continue

        if args.job_id and row.job_id != args.job_id:
            updated_rows.append(row)
            continue

        polled_state = provider.poll(row.provider_job_id)
        _write_status_polled(
            jobs_root=args.jobs_root,
            job_id=row.job_id,
            provider_job_id=row.provider_job_id,
            polled_state=polled_state,
            message=f"gcp_state={polled_state}",
        )
        updated_rows.append(_update_manifest_row_poll(row, polled_state))
        polled += 1

        print(f"polled job_id={row.job_id} gcp_state={polled_state}")

    write_manifest_csv(args.manifest_csv, updated_rows)
    print(f"polled={polled}")


def cmd_cancel(args: argparse.Namespace) -> None:
    provider = _build_provider_from_args(args)
    rows = read_manifest_csv(args.manifest_csv)

    cancelled = 0
    updated_rows: list[JobManifestRow] = []

    for row in rows:
        if row.provider != "gcp_batch" or not row.provider_job_id:
            updated_rows.append(row)
            continue

        if args.job_id and row.job_id != args.job_id:
            updated_rows.append(row)
            continue

        provider.cancel(row.provider_job_id)

        row.state = "cancelled"
        row.notes = "cancel requested in GCP Batch"
        updated_rows.append(row)

        paths = job_paths(args.jobs_root, row.job_id)
        status = JobStatus(
            job_id=row.job_id,
            state="cancelled",
            provider="gcp_batch",
            provider_job_id=row.provider_job_id,
            created_at=None,
            started_at=None,
            finished_at=None,
            message="cancel requested in GCP Batch",
            return_code=None,
        )
        write_json(paths["status"], status)

        cancelled += 1
        print(f"cancelled job_id={row.job_id}")

    write_manifest_csv(args.manifest_csv, updated_rows)
    print(f"cancelled={cancelled}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Submit/poll/cancel Block005 jobs in Google Cloud Batch."
    )

    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--jobs_root", required=True)
        p.add_argument("--manifest_csv", required=True)

        p.add_argument("--project_id", required=True)
        p.add_argument("--region", required=True)
        p.add_argument("--bucket_name", required=True)
        p.add_argument("--bucket_mount_path", default="/mnt/disks/jobshare")
        p.add_argument("--container_image", required=True)

        p.add_argument("--machine_type", default="e2-standard-4")
        p.add_argument("--provisioning_model", default="STANDARD")
        p.add_argument("--service_account_email", default="")
        p.add_argument("--cpu_milli", type=int, default=2000)
        p.add_argument("--memory_mib", type=int, default=4096)
        p.add_argument("--max_run_duration_sec", type=int, default=7200)
        p.add_argument("--max_retry_count", type=int, default=0)
        p.add_argument("--no_cloud_logging", action="store_true")

        p.add_argument("--jobs_prefix", default="jobs")
        p.add_argument("--results_prefix", default="results")
        p.add_argument("--env_label", default="research")
        p.add_argument("--type_label", default="music12")

        p.add_argument("--job_id", default="")

    p_submit = sub.add_parser("submit", help="Submit jobs to GCP Batch")
    add_common(p_submit)
    p_submit.add_argument("--only_queued", action="store_true")
    p_submit.add_argument("--force", action="store_true")
    p_submit.set_defaults(func=cmd_submit)

    p_poll = sub.add_parser("poll", help="Poll submitted jobs from GCP Batch")
    add_common(p_poll)
    p_poll.set_defaults(func=cmd_poll)

    p_cancel = sub.add_parser("cancel", help="Cancel submitted jobs in GCP Batch")
    add_common(p_cancel)
    p_cancel.set_defaults(func=cmd_cancel)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()