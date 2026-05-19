from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from music12.demons.demon_discovery import discover_unregistered_demons
from music12.demons.demon_registry import DemonRegistry, build_default_registry
from music12.demons.demon_types import (
    DemonRunResult,
    DemonSpec,
    FailureClass,
    MaxwellContext,
    MaxwellReport,
    MaxwellVerdict,
    TaskClass,
)
from music12.demons.demon_wrap import run_module_under_demon


def classify_failure(exc_type: str, exc_message: str) -> str:
    t = (exc_type or "").strip()
    m = (exc_message or "").strip().lower()

    if t in {"ImportError", "ModuleNotFoundError"}:
        return FailureClass.IMPORT_FAILURE.value
    if t == "FileNotFoundError":
        return FailureClass.IO_FAILURE.value
    if t == "NameError":
        return FailureClass.NAME_FAILURE.value
    if t == "KeyError":
        return FailureClass.KEY_FAILURE.value
    if t == "ValueError":
        if "9.a" in m or "anchor" in m or "440" in m:
            return FailureClass.ANCHOR_FAILURE.value
        if "time60" in m or "1/60" in m or "0.01" in m:
            return FailureClass.TIME60_FAILURE.value
        if "token" in m or "notation" in m or "alphabet" in m:
            return FailureClass.NOTATION_FAILURE.value
        return FailureClass.VALUE_FAILURE.value

    if t:
        return FailureClass.RUNTIME_EXCEPTION.value

    return FailureClass.NONE.value


def run_target_with_capture(
    *,
    target_module: str,
    argv: List[str],
    logdir: str,
    tag: str,
) -> Tuple[str, str, str, str, str, str]:
    payload = run_module_under_demon(
        logdir=logdir,
        tag=tag,
        module=target_module,
        module_args=argv,
    )
    return (
        str(payload.get("status", "")),
        str(payload.get("error_type") or ""),
        str(payload.get("error_message") or ""),
        str(payload.get("traceback") or ""),
        str(payload.get("log_json") or ""),
        str(payload.get("log_txt") or ""),
    )


def _merge_verdict(
    target_status: str,
    failure_class: str,
    trusted: List[DemonRunResult],
    discovered: List[DemonRunResult],
) -> str:
    if target_status == "ok":
        if trusted or discovered:
            return MaxwellVerdict.PASS_WITH_WARNINGS.value
        return MaxwellVerdict.PASS.value

    if failure_class in {
        FailureClass.NOTATION_FAILURE.value,
        FailureClass.TIME60_FAILURE.value,
        FailureClass.ANCHOR_FAILURE.value,
    }:
        return MaxwellVerdict.FAIL_PRINCIPLE.value

    if failure_class in {
        FailureClass.IMPORT_FAILURE.value,
        FailureClass.IO_FAILURE.value,
        FailureClass.NAME_FAILURE.value,
        FailureClass.KEY_FAILURE.value,
        FailureClass.VALUE_FAILURE.value,
        FailureClass.RUNTIME_EXCEPTION.value,
    }:
        return MaxwellVerdict.FAIL_RUNTIME.value

    if failure_class == FailureClass.RESULT_SUSPICION.value:
        return MaxwellVerdict.FAIL_RESULT.value

    return MaxwellVerdict.FAIL_UNKNOWN.value


def build_code_scan_args(
    *,
    spec: DemonSpec,
    ctx: MaxwellContext,
    out_base: Path,
) -> list[str]:
    root_dir = Path(ctx.project_root or ".").resolve() / "py"
    return [
        "--root", str(root_dir),
        "--out_txt", str(out_base.with_suffix(".txt")),
        "--out_json", str(out_base.with_suffix(".json")),
    ]


def build_resonance_report_args(
    *,
    spec: DemonSpec,
    ctx: MaxwellContext,
    out_base: Path,
) -> list[str]:
    """
    Requires ctx.extra:
      matrix_csv
      times_csv (optional)
      coords_csv (optional)
      detail_depth (optional)
      top_k (optional)
      source_name (optional)
    """
    extra = ctx.extra or {}

    matrix_csv = extra.get("matrix_csv", "")
    if not matrix_csv:
        raise ValueError("build_resonance_report_args requires ctx.extra['matrix_csv']")

    args = [
        "--matrix_csv", str(matrix_csv),
        "--out_txt", str(out_base.with_suffix(".txt")),
        "--out_json", str(out_base.with_suffix(".json")),
    ]

    if extra.get("times_csv"):
        args.extend(["--times_csv", str(extra["times_csv"])])
    if extra.get("coords_csv"):
        args.extend(["--coords_csv", str(extra["coords_csv"])])
    if extra.get("detail_depth") is not None:
        args.extend(["--detail_depth", str(extra["detail_depth"])])
    if extra.get("top_k") is not None:
        args.extend(["--top_k", str(extra["top_k"])])
    if extra.get("source_name"):
        args.extend(["--source_name", str(extra["source_name"])])

    return args


def build_project_law_report_args(
    *,
    spec: DemonSpec,
    ctx: MaxwellContext,
    out_base: Path,
) -> list[str]:
    """
    Supports both:
      - input_csv
      - matrix_csv

    We reuse matrix_csv as a generic table input because demon_maxwell_cli
    already supports it.
    """
    extra = ctx.extra or {}

    input_csv = extra.get("input_csv") or extra.get("matrix_csv")
    if not input_csv:
        raise ValueError(
            "build_project_law_report_args requires ctx.extra['input_csv'] "
            "or ctx.extra['matrix_csv']"
        )

    args = [
        "--input_csv", str(input_csv),
        "--out_txt", str(out_base.with_suffix(".txt")),
        "--out_json", str(out_base.with_suffix(".json")),
    ]

    if extra.get("focus"):
        args.extend(["--focus", str(extra["focus"])])
    if extra.get("strict"):
        args.append("--strict")

    return args


ARG_BUILDERS = {
    "build_code_scan_args": build_code_scan_args,
    "build_resonance_report_args": build_resonance_report_args,
    "build_project_law_report_args": build_project_law_report_args,
}


def _run_subordinate_demon(
    spec: DemonSpec,
    ctx: MaxwellContext,
) -> DemonRunResult:
    out_base = Path(ctx.logdir).resolve() / f"{ctx.tag}__{spec.demon_id}"

    builder = ARG_BUILDERS.get(spec.arg_builder_name)
    if builder is None:
        return DemonRunResult(
            demon_id=spec.demon_id,
            title=spec.title,
            status="builder_missing",
            matched_failure_class=ctx.failure_class,
            details={
                "entrypoint": spec.entrypoint,
                "arg_builder_name": spec.arg_builder_name,
            },
        )

    try:
        demon_args = builder(spec=spec, ctx=ctx, out_base=out_base)
    except Exception as e:
        return DemonRunResult(
            demon_id=spec.demon_id,
            title=spec.title,
            status="arg_build_error",
            matched_failure_class=ctx.failure_class,
            details={
                "entrypoint": spec.entrypoint,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

    payload = run_module_under_demon(
        logdir=ctx.logdir,
        tag=f"{ctx.tag}__{spec.demon_id}",
        module=spec.entrypoint,
        module_args=demon_args,
    )

    return DemonRunResult(
        demon_id=spec.demon_id,
        title=spec.title,
        status=str(payload.get("status", "")),
        matched_failure_class=ctx.failure_class,
        log_json=str(payload.get("log_json") or ""),
        log_txt=str(payload.get("log_txt") or ""),
        details={
            "entrypoint": spec.entrypoint,
            "argv": demon_args,
            "exit_code": payload.get("exit_code"),
            "error_type": payload.get("error_type"),
            "error_message": payload.get("error_message"),
        },
    )


def select_with_fallback(
    *,
    registry: DemonRegistry,
    ctx: MaxwellContext,
    demons_dir: str | Path,
) -> tuple[list[DemonSpec], list[DemonSpec], list[str]]:
    notes: list[str] = []

    trusted = registry.select(
        task_class=ctx.task_class,
        failure_class=ctx.failure_class,
    )

    if trusted:
        notes.append("Trusted registry produced matching demons.")
        return trusted, [], notes

    notes.append("Trusted registry found no direct matches. Entering discovery fallback.")

    discovered_all = discover_unregistered_demons(
        demons_dir=demons_dir,
        registry=registry,
    )

    discovered_selected = [
        spec for spec in discovered_all
        if spec.matches(task_class=ctx.task_class, failure_class=ctx.failure_class)
    ]

    if discovered_selected:
        notes.append("Discovery fallback found matching unregistered demons.")
    else:
        notes.append("Discovery fallback found no matching unregistered demons.")

    return trusted, discovered_selected, notes


def write_maxwell_report(
    report: MaxwellReport,
    *,
    logdir: str | Path,
    tag: str,
) -> tuple[str, str]:
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    json_path = logdir / f"{tag}_maxwell_report.json"
    txt_path = logdir / f"{tag}_maxwell_report.txt"

    json_payload = asdict(report)
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("MAXWELL DEMON REPORT")
    lines.append("=" * 80)
    lines.append(f"status           : {report.status}")
    lines.append(f"verdict          : {report.verdict}")
    lines.append(f"task_class       : {report.task_class}")
    lines.append(f"target_module    : {report.target_module}")
    lines.append(f"failure_class    : {report.failure_class}")
    lines.append(f"exception_type   : {report.exception_type}")
    lines.append(f"exception_msg    : {report.exception_message}")
    lines.append("")

    lines.append("Trusted demons selected:")
    for x in report.trusted_demons_selected:
        lines.append(f"  - {x}")
    if not report.trusted_demons_selected:
        lines.append("  (none)")
    lines.append("")

    lines.append("Discovered demons selected:")
    for x in report.discovered_demons_selected:
        lines.append(f"  - {x}")
    if not report.discovered_demons_selected:
        lines.append("  (none)")
    lines.append("")

    if report.notes:
        lines.append("Notes:")
        for n in report.notes:
            lines.append(f"  - {n}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(json_path), str(txt_path)


def run_maxwell(
    *,
    target_module: str,
    argv: List[str],
    task_class: str = TaskClass.MODULE_RUN.value,
    project_root: str | None = None,
    logdir: str = "_demon_logs",
    tag: str = "maxwell",
    demons_dir: str | Path = "py/music12/demons",
    extra: dict | None = None,
) -> MaxwellReport:
    registry = build_default_registry()

    status, exc_type, exc_msg, tb, target_log_json, target_log_txt = run_target_with_capture(
        target_module=target_module,
        argv=argv,
        logdir=logdir,
        tag=f"{tag}__target",
    )

    failure_class = classify_failure(exc_type, exc_msg) if status != "ok" else FailureClass.NONE.value

    ctx = MaxwellContext(
        task_class=task_class,
        target_module=target_module,
        argv=list(argv),
        project_root=project_root,
        logdir=logdir,
        tag=tag,
        failure_class=failure_class,
        exception_type=exc_type,
        exception_message=exc_msg,
        traceback_text=tb,
        extra=dict(extra or {}),
    )

    trusted_specs, discovered_specs, notes = select_with_fallback(
        registry=registry,
        ctx=ctx,
        demons_dir=demons_dir,
    )

    trusted_runs = [_run_subordinate_demon(spec, ctx) for spec in trusted_specs]
    discovered_runs = [_run_subordinate_demon(spec, ctx) for spec in discovered_specs]

    verdict = _merge_verdict(
        target_status=status,
        failure_class=failure_class,
        trusted=trusted_runs,
        discovered=discovered_runs,
    )

    notes = list(notes)
    notes.append(f"target_log_json={target_log_json}")
    notes.append(f"target_log_txt={target_log_txt}")

    report = MaxwellReport(
        status=status,
        verdict=verdict,
        task_class=task_class,
        target_module=target_module,
        argv=list(argv),
        failure_class=failure_class,
        exception_type=exc_type,
        exception_message=exc_msg,
        trusted_demons_selected=[s.demon_id for s in trusted_specs],
        trusted_demons_run=trusted_runs,
        discovered_demons_selected=[s.demon_id for s in discovered_specs],
        discovered_demons_run=discovered_runs,
        notes=notes,
    )

    write_maxwell_report(report, logdir=logdir, tag=tag)
    return report