from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from google.cloud import storage

from .manifest_io import job_paths, read_manifest_csv, write_manifest_csv, read_json
from .models import JobManifestRow, JobStatus


def _client() -> storage.Client:
    return storage.Client()


def _bucket_blob_name(prefix: str, job_id: str, filename: str) -> str:
    prefix = prefix.strip("/").replace("\\", "/")
    return f"{prefix}/{job_id}/{filename}"


def _upload_file(
    *,
    client: storage.Client,
    bucket_name: str,
    local_path: Path,
    remote_blob_name: str,
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_blob_name)
    blob.upload_from_filename(str(local_path))


def _download_file(
    *,
    client: storage.Client,
    bucket_name: str,
    remote_blob_name: str,
    local_path: Path,
) -> bool:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_blob_name)
    if not blob.exists(client):
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return True


def _iter_selected_rows(
    rows: list[JobManifestRow],
    job_id: str,
) -> Iterable[JobManifestRow]:
    for row in rows:
        if job_id and row.job_id != job_id:
            continue
        yield row


def cmd_upload_jobs(args: argparse.Namespace) -> None:
    client = _client()
    rows = read_manifest_csv(args.manifest_csv)

    uploaded = 0
    missing = 0

    for row in _iter_selected_rows(rows, args.job_id):
        paths = job_paths(args.jobs_root, row.job_id)
        spec_path = Path(paths["spec"])

        if not spec_path.exists():
            print(f"missing spec for job_id={row.job_id}: {spec_path}")
            missing += 1
            continue

        remote_blob = _bucket_blob_name(args.jobs_prefix, row.job_id, "job.json")
        _upload_file(
            client=client,
            bucket_name=args.bucket_name,
            local_path=spec_path,
            remote_blob_name=remote_blob,
        )
        uploaded += 1
        print(f"uploaded job_id={row.job_id} -> gs://{args.bucket_name}/{remote_blob}")

    print(f"uploaded={uploaded}")
    print(f"missing={missing}")


def cmd_download_results(args: argparse.Namespace) -> None:
    client = _client()
    rows = read_manifest_csv(args.manifest_csv)

    downloaded = 0
    checked = 0

    filenames = [
        "status.json",
        "result.json",
        "probe_meta.json",
    ]
    if args.with_csv:
        filenames.extend(
            [
                "probe_matrix.csv",
                "probe_times.csv",
                "probe_coords.csv",
                "stdout.txt",
                "stderr.txt",
            ]
        )

    for row in _iter_selected_rows(rows, args.job_id):
        local_result_dir = Path(args.results_root) / row.job_id
        got_any = False

        for filename in filenames:
            remote_blob = _bucket_blob_name(args.results_prefix, row.job_id, filename)
            ok = _download_file(
                client=client,
                bucket_name=args.bucket_name,
                remote_blob_name=remote_blob,
                local_path=local_result_dir / filename,
            )
            checked += 1
            if ok:
                got_any = True
                downloaded += 1
                print(f"downloaded job_id={row.job_id} <- gs://{args.bucket_name}/{remote_blob}")

        if not got_any:
            print(f"no remote result files found for job_id={row.job_id}")

    print(f"downloaded_files={downloaded}")
    print(f"checked_paths={checked}")


def cmd_sync_manifest(args: argparse.Namespace) -> None:
    rows = read_manifest_csv(args.manifest_csv)
    updated_rows: list[JobManifestRow] = []

    updated = 0
    missing = 0

    for row in rows:
        if args.job_id and row.job_id != args.job_id:
            updated_rows.append(row)
            continue

        status_path = Path(args.results_root) / row.job_id / "status.json"
        if not status_path.exists():
            updated_rows.append(row)
            missing += 1
            continue

        status = read_json(status_path, JobStatus)
        row.state = status.state
        row.provider = status.provider
        row.provider_job_id = status.provider_job_id or ""
        row.notes = status.message or row.notes
        updated_rows.append(row)
        updated += 1

        print(f"synced job_id={row.job_id} state={row.state}")

    write_manifest_csv(args.manifest_csv, updated_rows)
    print(f"updated={updated}")
    print(f"missing_status={missing}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Upload Block005 job specs to GCS and download/sync results."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upload-jobs", help="Upload local job.json files to GCS")
    p_up.add_argument("--jobs_root", required=True)
    p_up.add_argument("--manifest_csv", required=True)
    p_up.add_argument("--bucket_name", required=True)
    p_up.add_argument("--jobs_prefix", default="jobs")
    p_up.add_argument("--job_id", default="")
    p_up.set_defaults(func=cmd_upload_jobs)

    p_down = sub.add_parser("download-results", help="Download result files from GCS")
    p_down.add_argument("--manifest_csv", required=True)
    p_down.add_argument("--results_root", required=True)
    p_down.add_argument("--bucket_name", required=True)
    p_down.add_argument("--results_prefix", default="results")
    p_down.add_argument("--job_id", default="")
    p_down.add_argument("--with_csv", action="store_true")
    p_down.set_defaults(func=cmd_download_results)

    p_sync = sub.add_parser("sync-manifest", help="Sync local manifest.csv from downloaded status.json")
    p_sync.add_argument("--manifest_csv", required=True)
    p_sync.add_argument("--results_root", required=True)
    p_sync.add_argument("--job_id", default="")
    p_sync.set_defaults(func=cmd_sync_manifest)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()