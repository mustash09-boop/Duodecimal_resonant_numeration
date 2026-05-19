from __future__ import annotations

from abc import ABC, abstractmethod

from .models import JobSpec


class JobProvider(ABC):
    @abstractmethod
    def submit(self, spec: JobSpec) -> str:
        raise NotImplementedError

    @abstractmethod
    def poll(self, provider_job_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, provider_job_id: str) -> None:
        raise NotImplementedError