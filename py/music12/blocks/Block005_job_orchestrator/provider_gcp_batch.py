from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Optional

from google.api_core.exceptions import NotFound
from google.cloud import batch_v1

from .models import JobSpec
from .provider_base import JobProvider


@dataclass
class GcpBatchRuntimeConfig:
    project_id: str
    region: str
    bucket_name: str
    bucket_mount_path: str = "/mnt/disks/jobshare"

    # Container image must already contain:
    # - Python
    # - project code
    # - music12 package
    # - cloud entrypoint module
    container_image: str = ""

    machine_type: str = "e2-standard-4"
    provisioning_model: str = "STANDARD"  # or "SPOT"
    service_account_email: Optional[str] = None

    cpu_milli: int = 2000
    memory_mib: int = 4096
    max_run_duration_sec: int = 7200
    max_retry_count: int = 0

    logs_to_cloud_logging: bool = True

    # Layout inside bucket
    jobs_prefix: str = "jobs"
    results_prefix: str = "results"

    # Optional labels
    env_label: str = "research"
    type_label: str = "music12"


class GcpBatchProvider(JobProvider):
    """
    Real first GCP Batch provider.

    Assumptions:
    1) A Cloud Storage bucket already exists.
    2) The Batch service account can mount/read that bucket.
    3) The container image already contains your project code and can run:
         python -m music12.blocks.Block005_job_orchestrator.cloud_entrypoint_cli
    4) Block005 uploads/writes job.json into:
         gs://<bucket>/<jobs_prefix>/<job_id>/job.json

    This provider does:
    - submit(job)
    - poll(job)
    - cancel(job) via delete
    """

    def __init__(self, config: GcpBatchRuntimeConfig) -> None:
        if not config.project_id:
            raise ValueError("project_id is required")
        if not config.region:
            raise ValueError("region is required")
        if not config.bucket_name:
            raise ValueError("bucket_name is required")
        if not config.container_image:
            raise ValueError(
                "container_image is required. "
                "It must contain the music12 project and cloud_entrypoint_cli."
            )
        if not config.bucket_mount_path.startswith("/mnt/disks/"):
            raise ValueError(
                "bucket_mount_path must start with /mnt/disks/ for GCP Batch bucket mount"
            )

        self.config = config
        self.client = batch_v1.BatchServiceClient()

    @property
    def parent(self) -> str:
        return f"projects/{self.config.project_id}/locations/{self.config.region}"

    def _sanitize_job_id(self, raw: str) -> str:
        """
        GCP Batch job_id:
        - max 63 chars
        - lowercase, digits, hyphen
        - cannot start or end with hyphen
        """
        x = raw.lower()
        x = re.sub(r"[^a-z0-9-]+", "-", x)
        x = re.sub(r"-{2,}", "-", x)
        x = x.strip("-")
        if not x:
            x = "job"
        if len(x) > 50:
            x = x[:50].rstrip("-")
        suffix = uuid.uuid4().hex[:8]
        x = f"{x}-{suffix}"
        x = x[:63].rstrip("-")
        if not x:
            x = f"job-{suffix}"
        return x

    def _gcs_remote_path_for_job(self, spec: JobSpec) -> str:
        # Batch bucket volume remotePath must start with bucket name,
        # e.g. "my-bucket/subdir".
        return f"{self.config.bucket_name}"

    def _job_json_mount_path(self, spec: JobSpec) -> str:
        return (
            f"{self.config.bucket_mount_path}/"
            f"{self.config.jobs_prefix}/{spec.job_id}/job.json"
        )

    def _results_mount_dir(self, spec: JobSpec) -> str:
        return (
            f"{self.config.bucket_mount_path}/"
            f"{self.config.results_prefix}/{spec.job_id}"
        )

    def _build_container_runnable(self, spec: JobSpec) -> batch_v1.Runnable:
        """
        The container executes cloud_entrypoint_cli inside the image.

        cloud_entrypoint_cli should:
        - read mounted job.json
        - run the appropriate Block005 local/cloud runner
        - write outputs into mounted results dir
        """
        job_json_path = self._job_json_mount_path(spec)
        results_dir = self._results_mount_dir(spec)

        container = batch_v1.Runnable.Container()
        container.image_uri = self.config.container_image
        container.entrypoint = "python"
        container.commands = [
            "-m",
            "music12.blocks.Block005_job_orchestrator.cloud_entrypoint_cli",
            "--job_spec",
            job_json_path,
            "--results_dir",
            results_dir,
        ]

        runnable = batch_v1.Runnable()
        runnable.container = container
        return runnable

    def _build_job(self, spec: JobSpec) -> batch_v1.Job:
        runnable = self._build_container_runnable(spec)

        resources = batch_v1.ComputeResource()
        resources.cpu_milli = self.config.cpu_milli
        resources.memory_mib = self.config.memory_mib

        volume = batch_v1.Volume()
        volume.gcs = batch_v1.GCS()
        volume.gcs.remote_path = self._gcs_remote_path_for_job(spec)
        volume.mount_path = self.config.bucket_mount_path

        task = batch_v1.TaskSpec()
        task.runnables = [runnable]
        task.compute_resource = resources
        task.max_retry_count = self.config.max_retry_count
        task.max_run_duration = {"seconds": self.config.max_run_duration_sec}
        task.volumes = [volume]

        task_group = batch_v1.TaskGroup()
        task_group.task_count = 1
        task_group.task_spec = task

        policy = batch_v1.AllocationPolicy.InstancePolicy()
        policy.machine_type = self.config.machine_type
        if self.config.provisioning_model.upper() == "SPOT":
            policy.provisioning_model = (
                batch_v1.AllocationPolicy.ProvisioningModel.SPOT
            )
        else:
            policy.provisioning_model = (
                batch_v1.AllocationPolicy.ProvisioningModel.STANDARD
            )

        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instances.policy = policy

        allocation_policy = batch_v1.AllocationPolicy()
        allocation_policy.instances = [instances]

        if self.config.service_account_email:
            allocation_policy.service_account = batch_v1.ServiceAccount()
            allocation_policy.service_account.email = self.config.service_account_email

        logs_policy = batch_v1.LogsPolicy()
        if self.config.logs_to_cloud_logging:
            logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

        job = batch_v1.Job()
        job.task_groups = [task_group]
        job.allocation_policy = allocation_policy
        job.logs_policy = logs_policy
        job.labels = {
            "env": self.config.env_label,
            "type": self.config.type_label,
            "taskkind": spec.task_kind[:63],
        }
        return job

    def submit(self, spec: JobSpec) -> str:
        provider_job_id = self._sanitize_job_id(spec.job_id)
        job = self._build_job(spec)

        request = batch_v1.CreateJobRequest()
        request.parent = self.parent
        request.job = job
        request.job_id = provider_job_id
        request.request_id = str(uuid.uuid4())

        response = self.client.create_job(request=request)
        return response.name

    def poll(self, provider_job_id: str) -> str:
        """
        Accepts either:
        - full job resource name: projects/.../locations/.../jobs/...
        - or just the short job id
        Returns:
        - raw status.state name if available
        """
        if provider_job_id.startswith("projects/"):
            job_name = provider_job_id
        else:
            job_name = f"{self.parent}/jobs/{provider_job_id}"

        job = self.client.get_job(name=job_name)
        state = getattr(job.status, "state", None)
        if state is None:
            return "UNKNOWN"

        try:
            return batch_v1.JobStatus.State(state).name
        except Exception:
            return str(state)

    def cancel(self, provider_job_id: str) -> None:
        """
        Batch Python client exposes delete_job; for our orchestration this is
        enough as a cancellation primitive.
        """
        if provider_job_id.startswith("projects/"):
            job_name = provider_job_id
        else:
            job_name = f"{self.parent}/jobs/{provider_job_id}"

        try:
            self.client.delete_job(name=job_name)
        except NotFound:
            return