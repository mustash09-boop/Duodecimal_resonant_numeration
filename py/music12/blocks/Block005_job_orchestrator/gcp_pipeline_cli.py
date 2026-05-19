from __future__ import annotations

import argparse
import time
from pathlib import Path

from .provider_gcp_storage_cli import (
    cmd_upload_jobs,
    cmd_download_results,
    cmd_sync_manifest,
)
from .provider_gcp_submit_cli import (
    cmd_submit,
    cmd_poll,
)
from .manifest_io import read_manifest_csv


def _count_states(manifest_csv: str) -> dict[str, int]:
    rows = read_manifest_csv(manifest_csv)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1
    return counts


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(f"\n=== {title} ===")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")


def _all_terminal(manifest_csv: str) -> bool:
    rows = read_manifest_csv(manifest_csv)
    terminal = {"done", "failed", "timeout", "cancelled", "unknown"}
    if not rows:
        return True
    return all(row.state in terminal for row in rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="End-to-end GCP pipeline for Block005 jobs: upload -> submit -> poll -> download -> sync"
    )

    ap.add_argument("--jobs_root", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--results_root", required=True)

    ap.add_argument("--project_id", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--bucket_name", required=True)
    ap.add_argument("--bucket_mount_path", default="/mnt/disks/jobshare")
    ap.add_argument("--container_image", required=True)

    ap.add_argument("--machine_type", default="e2-standard-4")
    ap.add_argument("--provisioning_model", default="STANDARD")
    ap.add_argument("--service_account_email", default="")
    ap.add_argument("--cpu_milli", type=int, default=2000)
    ap.add_argument("--memory_mib", type=int, default=4096)
    ap.add_argument("--max_run_duration_sec", type=int, default=7200)
    ap.add_argument("--max_retry_count", type=int, default=0)
    ap.add_argument("--no_cloud_logging", action="store_true")

    ap.add_argument("--jobs_prefix", default="jobs")
    ap.add_argument("--results_prefix", default="results")
    ap.add_argument("--env_label", default="research")
    ap.add_argument("--type_label", default="music12")

    ap.add_argument("--job_id", default="")
    ap.add_argument("--only_queued", action="store_true")
    ap.add_argument("--force_submit", action="store_true")
    ap.add_argument("--with_csv", action="store_true")

    ap.add_argument("--poll_interval_sec", type=int, default=60)
    ap.add_argument("--max_poll_cycles", type=int, default=120)

    ap.add_argument("--skip_upload", action="store_true")
    ap.add_argument("--skip_submit", action="store_true")
    ap.add_argument("--skip_poll", action="store_true")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_sync", action="store_true")

    return ap


def _make_storage_upload_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.jobs_root = args.jobs_root
    ns.manifest_csv = args.manifest_csv
    ns.bucket_name = args.bucket_name
    ns.jobs_prefix = args.jobs_prefix
    ns.job_id = args.job_id
    return ns


def _make_submit_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.jobs_root = args.jobs_root
    ns.manifest_csv = args.manifest_csv

    ns.project_id = args.project_id
    ns.region = args.region
    ns.bucket_name = args.bucket_name
    ns.bucket_mount_path = args.bucket_mount_path
    ns.container_image = args.container_image

    ns.machine_type = args.machine_type
    ns.provisioning_model = args.provisioning_model
    ns.service_account_email = args.service_account_email
    ns.cpu_milli = args.cpu_milli
    ns.memory_mib = args.memory_mib
    ns.max_run_duration_sec = args.max_run_duration_sec
    ns.max_retry_count = args.max_retry_count
    ns.no_cloud_logging = args.no_cloud_logging

    ns.jobs_prefix = args.jobs_prefix
    ns.results_prefix = args.results_prefix
    ns.env_label = args.env_label
    ns.type_label = args.type_label

    ns.job_id = args.job_id
    ns.only_queued = args.only_queued
    ns.force = args.force_submit
    return ns


def _make_poll_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.jobs_root = args.jobs_root
    ns.manifest_csv = args.manifest_csv

    ns.project_id = args.project_id
    ns.region = args.region
    ns.bucket_name = args.bucket_name
    ns.bucket_mount_path = args.bucket_mount_path
    ns.container_image = args.container_image

    ns.machine_type = args.machine_type
    ns.provisioning_model = args.provisioning_model
    ns.service_account_email = args.service_account_email
    ns.cpu_milli = args.cpu_milli
    ns.memory_mib = args.memory_mib
    ns.max_run_duration_sec = args.max_run_duration_sec
    ns.max_retry_count = args.max_retry_count
    ns.no_cloud_logging = args.no_cloud_logging

    ns.jobs_prefix = args.jobs_prefix
    ns.results_prefix = args.results_prefix
    ns.env_label = args.env_label
    ns.type_label = args.type_label

    ns.job_id = args.job_id
    return ns


def _make_download_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.manifest_csv = args.manifest_csv
    ns.results_root = args.results_root
    ns.bucket_name = args.bucket_name
    ns.results_prefix = args.results_prefix
    ns.job_id = args.job_id
    ns.with_csv = args.with_csv
    return ns


def _make_sync_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.manifest_csv = args.manifest_csv
    ns.results_root = args.results_root
    ns.job_id = args.job_id
    return ns


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest_csv = str(Path(args.manifest_csv).resolve())

    if not args.skip_upload:
        print("\n[1/5] upload-jobs")
        cmd_upload_jobs(_make_storage_upload_args(args))
        _print_counts("manifest after upload", _count_states(manifest_csv))

    if not args.skip_submit:
        print("\n[2/5] submit")
        cmd_submit(_make_submit_args(args))
        _print_counts("manifest after submit", _count_states(manifest_csv))

    if not args.skip_poll:
        print("\n[3/5] poll loop")
        poll_args = _make_poll_args(args)

        for cycle in range(1, args.max_poll_cycles + 1):
            print(f"\n--- poll cycle {cycle}/{args.max_poll_cycles} ---")
            cmd_poll(poll_args)
            counts = _count_states(manifest_csv)
            _print_counts("manifest after poll", counts)

            if _all_terminal(manifest_csv):
                print("all jobs are in terminal states")
                break

            if cycle < args.max_poll_cycles:
                time.sleep(args.poll_interval_sec)

    if not args.skip_download:
        print("\n[4/5] download-results")
        cmd_download_results(_make_download_args(args))

    if not args.skip_sync:
        print("\n[5/5] sync-manifest")
        cmd_sync_manifest(_make_sync_args(args))
        _print_counts("final manifest", _count_states(manifest_csv))


if __name__ == "__main__":
    main()