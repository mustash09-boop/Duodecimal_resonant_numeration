from __future__ import annotations

from typing import Dict, List

from music12.demons.demon_types import DemonSpec, FailureClass, TaskClass


class DemonRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, DemonSpec] = {}

    def register(self, spec: DemonSpec) -> None:
        if spec.demon_id in self._specs:
            raise ValueError(f"Duplicate demon_id: {spec.demon_id}")
        self._specs[spec.demon_id] = spec

    def has(self, demon_id: str) -> bool:
        return demon_id in self._specs

    def all_specs(self) -> List[DemonSpec]:
        return sorted(self._specs.values(), key=lambda x: (x.priority, x.demon_id))

    def select(self, *, task_class: str, failure_class: str) -> List[DemonSpec]:
        out = [
            spec
            for spec in self._specs.values()
            if spec.matches(task_class=task_class, failure_class=failure_class)
        ]
        return sorted(out, key=lambda x: (x.priority, x.demon_id))


def build_default_registry() -> DemonRegistry:
    reg = DemonRegistry()

    reg.register(
        DemonSpec(
            demon_id="notation_alphabet12",
            title="Notation Alphabet 12 Consistency",
            entrypoint="music12.demons.demon_notation_alphabet12_consistency",
            task_classes=[
                TaskClass.MODULE_RUN.value,
                TaskClass.CODE_SCAN.value,
                TaskClass.PRINCIPLE_CHECK.value,
            ],
            failure_classes=[
                FailureClass.NOTATION_FAILURE.value,
                FailureClass.UNKNOWN_FAILURE.value,
            ],
            priority=10,
            tags=["notation", "alphabet12", "principle"],
            demon_kind="code_scan",
            arg_builder_name="build_code_scan_args",
        )
    )

    reg.register(
        DemonSpec(
            demon_id="time60_consistency",
            title="Time60 Consistency",
            entrypoint="music12.demons.demon_time60_consistency",
            task_classes=[
                TaskClass.MODULE_RUN.value,
                TaskClass.CODE_SCAN.value,
                TaskClass.PRINCIPLE_CHECK.value,
                TaskClass.AUDIO_ANALYSIS.value,
            ],
            failure_classes=[
                FailureClass.TIME60_FAILURE.value,
                FailureClass.UNKNOWN_FAILURE.value,
            ],
            priority=20,
            tags=["time60", "principle", "time"],
            demon_kind="code_scan",
            arg_builder_name="build_code_scan_args",
        )
    )

    reg.register(
        DemonSpec(
            demon_id="anchor_9a_consistency",
            title="Anchor 9.A Consistency",
            entrypoint="music12.demons.demon_anchor_9a_consistency",
            task_classes=[
                TaskClass.MODULE_RUN.value,
                TaskClass.CODE_SCAN.value,
                TaskClass.PRINCIPLE_CHECK.value,
                TaskClass.AUDIO_ANALYSIS.value,
            ],
            failure_classes=[
                FailureClass.ANCHOR_FAILURE.value,
                FailureClass.NOTATION_FAILURE.value,
                FailureClass.UNKNOWN_FAILURE.value,
            ],
            priority=30,
            tags=["anchor", "9.A", "principle"],
            demon_kind="code_scan",
            arg_builder_name="build_code_scan_args",
        )
    )

    reg.register(
        DemonSpec(
            demon_id="project_law_report",
            title="Project Law Report",
            entrypoint="music12.demons.demon_project_law_report",
            task_classes=[
                TaskClass.REPORT_ANALYSIS.value,
                TaskClass.AUDIO_ANALYSIS.value,
                TaskClass.VERIFY_ANALYSIS.value,
                TaskClass.INSTRUMENT_ANALYSIS.value,
            ],
            failure_classes=[
                FailureClass.RESULT_SUSPICION.value,
                FailureClass.UNKNOWN_FAILURE.value,
            ],
            priority=35,
            tags=[
                "project_law",
                "ontology",
                "chain",
                "note",
                "result_check",
            ],
            description=(
                "Checks project-law compliance in produced tables/reports: "
                "note without chain, strongest_peak_note misuse, early f0 fixation, "
                "zero leak, and inference-order signature."
            ),
            demon_kind="result_report",
            arg_builder_name="build_project_law_report_args",
        )
    )

    reg.register(
        DemonSpec(
            demon_id="resonance_scan_report",
            title="Resonance Scan Report",
            entrypoint="music12.demons.demon_resonance_scan_report",
            task_classes=[
                TaskClass.REPORT_ANALYSIS.value,
                TaskClass.AUDIO_ANALYSIS.value,
            ],
            failure_classes=[
                FailureClass.RESULT_SUSPICION.value,
                FailureClass.UNKNOWN_FAILURE.value,
            ],
            priority=100,
            tags=["result", "resonance", "report"],
            demon_kind="result_report",
            arg_builder_name="build_resonance_report_args",
        )
    )

    return reg