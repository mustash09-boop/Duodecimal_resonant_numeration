from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MaxwellVerdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL_RUNTIME = "FAIL_RUNTIME"
    FAIL_PRINCIPLE = "FAIL_PRINCIPLE"
    FAIL_RESULT = "FAIL_RESULT"
    FAIL_UNKNOWN = "FAIL_UNKNOWN"
    CRITICAL = "CRITICAL"


class FailureClass(str, Enum):
    NONE = "none"
    IMPORT_FAILURE = "import_failure"
    IO_FAILURE = "io_failure"
    NAME_FAILURE = "name_failure"
    KEY_FAILURE = "key_failure"
    VALUE_FAILURE = "value_failure"
    NOTATION_FAILURE = "notation_failure"
    TIME60_FAILURE = "time60_failure"
    ANCHOR_FAILURE = "anchor_failure"
    RESULT_SUSPICION = "result_suspicion"
    RUNTIME_EXCEPTION = "runtime_exception"
    UNKNOWN_FAILURE = "unknown_failure"


class TaskClass(str, Enum):
    MODULE_RUN = "module_run"
    CODE_SCAN = "code_scan"
    PRINCIPLE_CHECK = "principle_check"
    REPORT_ANALYSIS = "report_analysis"
    AUDIO_ANALYSIS = "audio_analysis"
    VERIFY_ANALYSIS = "verify_analysis"
    INSTRUMENT_ANALYSIS = "instrument_analysis"


@dataclass(frozen=True)
class DemonSpec:
    demon_id: str
    title: str
    entrypoint: str
    task_classes: List[str] = field(default_factory=list)
    failure_classes: List[str] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # kind of subordinate demon
    # code_scan | result_report | generic_module
    demon_kind: str = "code_scan"

    # name of arg builder function in demon_maxwell_core.py
    arg_builder_name: str = ""

    def matches(self, *, task_class: str, failure_class: str) -> bool:
        if not self.enabled:
            return False
        task_ok = (not self.task_classes) or (task_class in self.task_classes)
        fail_ok = (not self.failure_classes) or (failure_class in self.failure_classes)
        return task_ok and fail_ok


@dataclass
class DemonRunResult:
    demon_id: str
    title: str
    status: str
    matched_failure_class: str
    log_json: Optional[str] = None
    log_txt: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaxwellContext:
    task_class: str
    target_module: str
    argv: List[str]
    project_root: Optional[str] = None
    logdir: str = "_demon_logs"
    tag: str = "maxwell"
    failure_class: str = FailureClass.NONE.value
    exception_type: str = ""
    exception_message: str = ""
    traceback_text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaxwellReport:
    status: str
    verdict: str
    task_class: str
    target_module: str
    argv: List[str]
    failure_class: str
    exception_type: str
    exception_message: str
    trusted_demons_selected: List[str] = field(default_factory=list)
    trusted_demons_run: List[DemonRunResult] = field(default_factory=list)
    discovered_demons_selected: List[str] = field(default_factory=list)
    discovered_demons_run: List[DemonRunResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)