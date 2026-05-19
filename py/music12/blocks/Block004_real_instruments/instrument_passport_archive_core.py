from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PY_ROOT = PROJECT_ROOT / "py"

KEYED_RE = re.compile(
    r"^(?P<idx>\d{3})_(?P<source_tag>[A-Za-z0-9_]+)_(?P<token>[1-9ABC]+[.][1-9ABC][ia0-9ABC'\-]*)[.]wav$",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(
    r"^(?P<index>\d+)_(?P<note12>[1-9ABC]+\.[1-9ABC]+'?(?:[ia][1-9ABC]+)?-?)_(?P<label>.+?)\.wav$",
    re.IGNORECASE,
)

Logger = Callable[[str], None]


@dataclass
class ArchiveBuildConfig:
    source_dir: str
    archive_root: str
    instrument_name: str
    library_kind: str = "pitched"  # pitched | percussion
    manifest_mode: str = "auto"  # auto | keyed | token
    fix_cyrillic_abc: bool = True
    use_maxwell: bool = True
    maxwell_logdir: str = "_demon_logs"
    maxwell_tag_prefix: str = "archive_program"
    ffmpeg_path: str = ""
    include_archive_index: bool = True
    include_archive_audit: bool = True


@dataclass
class ArchiveBuildSummary:
    status: str
    started_at: str
    finished_at: str
    instrument_name: str
    library_kind: str
    source_dir: str
    archive_root: str
    instrument_root: str
    manifest_mode_requested: str
    manifest_mode_selected: str
    wav_files_copied: int
    outputs: dict


def safe_tag(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    text = text.strip("._-")
    return text or "stage"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    old = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(PY_ROOT) if not old else str(PY_ROOT) + os.pathsep + old
    return env


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_library_kind(value: str) -> str:
    kind = (value or "pitched").strip().lower()
    if kind not in {"pitched", "percussion"}:
        raise ValueError(f"Unsupported library kind: {value}")
    return kind


def count_matches(audio_dir: Path, pattern: re.Pattern[str]) -> int:
    count = 0
    for path in sorted(audio_dir.glob("*.wav")):
        if pattern.match(path.name):
            count += 1
    return count


def detect_manifest_mode(audio_dir: Path) -> str:
    keyed = count_matches(audio_dir, KEYED_RE)
    token = count_matches(audio_dir, TOKEN_RE)

    if keyed == 0 and token == 0:
        raise RuntimeError(
            "Could not recognize the WAV filename format. "
            "Expected either NNN_SOURCE_TAG_NOTE12.wav or NNN_NOTE12_label.wav."
        )

    if keyed >= token:
        return "keyed"
    return "token"


def stream_process(cmd: list[str], logger: Logger, cwd: Optional[Path] = None) -> None:
    logger("RUN:")
    logger(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        env=build_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        logger(line.rstrip())

    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}")


def run_module(
    *,
    module: str,
    module_args: list[str],
    logger: Logger,
    use_maxwell: bool,
    task_class: str,
    tag: str,
    maxwell_logdir: str,
) -> None:
    if use_maxwell:
        cmd = [
            sys.executable,
            "-m",
            "music12.demons.demon_maxwell_cli",
            "-m",
            module,
            "--task-class",
            task_class,
            "--project-root",
            str(PROJECT_ROOT),
            "--logdir",
            maxwell_logdir,
            "--tag",
            safe_tag(tag),
            "--",
            *module_args,
        ]
    else:
        cmd = [sys.executable, "-m", module, *module_args]

    stream_process(cmd, logger=logger, cwd=PROJECT_ROOT)


def run_filename_guard(audio_dir: Path, logger: Logger, fix: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "music12.demons.demon_filename_alphabet_guard_cli",
        "--audio_dir",
        str(audio_dir),
    ]
    if fix:
        cmd.append("--fix")

    stream_process(cmd, logger=logger, cwd=PROJECT_ROOT)


def collect_audio_files(source_dir: Path, suffix: str) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix.lower()
    )


def ensure_unique_target_names(paths: list[Path], suffix_override: str | None = None) -> None:
    seen: dict[str, Path] = {}
    collisions: dict[str, list[Path]] = {}

    for path in paths:
        target_name = path.name if suffix_override is None else f"{path.stem}{suffix_override}"
        key = target_name.lower()
        if key in seen:
            collisions.setdefault(key, [seen[key]]).append(path)
        else:
            seen[key] = path

    if not collisions:
        return

    lines = [
        "Duplicate target filenames detected while preparing the library.",
        "The archive builder keeps a flat WAV directory, so basenames must be unique.",
        "",
    ]
    for target_name, dup_paths in sorted(collisions.items()):
        lines.append(f"{target_name}:")
        for dup_path in dup_paths:
            lines.append(f"  - {dup_path}")
    raise RuntimeError("\n".join(lines))


def resolve_ffmpeg_path(config: ArchiveBuildConfig) -> str:
    ffmpeg_path = (config.ffmpeg_path or "").strip()
    if ffmpeg_path:
        discovered = shutil.which(ffmpeg_path)
        if discovered:
            return discovered
        return ffmpeg_path
    return shutil.which("ffmpeg") or ""


def convert_mp3_files(
    *,
    mp3_files: list[Path],
    target_audio_dir: Path,
    ffmpeg_path: str,
    logger: Logger,
) -> int:
    ffmpeg = Path(ffmpeg_path)
    if not ffmpeg.exists():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")

    converted = 0
    total = len(mp3_files)

    for index, src in enumerate(mp3_files, 1):
        out_wav = target_audio_dir / f"{src.stem}.wav"
        logger(f"[MP3 {index}/{total}] {src.name} -> {out_wav.name}")

        result = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-i",
                str(src),
                "-ar",
                "44100",
                "-ac",
                "1",
                str(out_wav),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-2000:].strip()
            raise RuntimeError(
                f"FFmpeg failed for {src.name} (return code {result.returncode}).\n{stderr_tail}"
            )

        converted += 1

    logger(f"MP3 files converted: {converted}")
    return converted


def prepare_audio_library(config: ArchiveBuildConfig, target_audio_dir: Path, logger: Logger) -> int:
    source_dir = Path(config.source_dir)
    ensure_dir(target_audio_dir)

    wav_files = collect_audio_files(source_dir, ".wav")
    mp3_files = collect_audio_files(source_dir, ".mp3")

    if not wav_files and not mp3_files:
        raise RuntimeError("No WAV or MP3 files were found in the selected source folder.")

    ensure_unique_target_names(wav_files)
    ensure_unique_target_names(mp3_files, suffix_override=".wav")

    target_name_map: dict[str, Path] = {}
    for path in wav_files:
        target_name_map[path.name.lower()] = path
    for path in mp3_files:
        key = f"{path.stem}.wav".lower()
        if key in target_name_map:
            raise RuntimeError(
                "Target name collision between WAV and MP3 sources: "
                f"{target_name_map[key].name} <-> {path.name}"
            )
        target_name_map[key] = path

    copied = 0
    for src in wav_files:
        dst = target_audio_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied += 1
    if wav_files:
        logger(f"WAV files copied: {copied}")

    converted = 0
    if mp3_files:
        ffmpeg_path = resolve_ffmpeg_path(config)
        if not ffmpeg_path:
            raise RuntimeError(
                "MP3 files were found, but ffmpeg was not provided. "
                "Select ffmpeg.exe or make ffmpeg available in PATH."
            )
        converted = convert_mp3_files(
            mp3_files=mp3_files,
            target_audio_dir=target_audio_dir,
            ffmpeg_path=ffmpeg_path,
            logger=logger,
        )

    wav_count = len(list(target_audio_dir.glob("*.wav")))
    logger(f"Prepared WAV library size: {wav_count}")
    return copied + converted


def build_manifest(
    *,
    config: ArchiveBuildConfig,
    audio_dir: Path,
    manifest_csv: Path,
    manifest_mode: str,
    logger: Logger,
) -> None:
    if manifest_mode == "keyed":
        module = "music12.blocks.Block004_real_instruments.manifest12_from_piano_midi_filename_cli"
        module_args = [
            "--audio_dir", str(audio_dir),
            "--out_csv", str(manifest_csv),
            "--instrument_name", config.instrument_name,
        ]
    elif manifest_mode == "token":
        module = "music12.blocks.Block004_real_instruments.manifest12_from_token_filename_cli"
        module_args = [
            "--audio_dir", str(audio_dir),
            "--out_csv", str(manifest_csv),
        ]
    else:
        raise ValueError(f"Unsupported manifest mode: {manifest_mode}")

    run_module(
        module=module,
        module_args=module_args,
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__{config.instrument_name}__manifest",
        maxwell_logdir=config.maxwell_logdir,
    )


def build_pipeline(
    *,
    config: ArchiveBuildConfig,
    audio_dir: Path,
    manifest_csv: Path,
    reports_root: Path,
    range_dir: Path,
    note_box_dir: Path,
    spiral3d_dir: Path,
    logger: Logger,
) -> None:
    module_args = [
        "--instrument_name", config.instrument_name,
        "--audio_dir", str(audio_dir),
        "--manifest_csv", str(manifest_csv),
        "--reports_root", str(reports_root),
        "--layer", "all",
        "--box_out_dir", str(range_dir),
        "--passport_out_dir", str(range_dir),
        "--note_box_out_dir", str(note_box_dir),
        "--spiral3d_out_dir", str(spiral3d_dir),
    ]

    if config.use_maxwell:
        module_args.extend([
            "--use_maxwell",
            "--maxwell_logdir", config.maxwell_logdir,
            "--maxwell_task_class", "instrument_analysis",
            "--maxwell_tag_prefix", f"{config.maxwell_tag_prefix}__{config.instrument_name}",
        ])

    run_module(
        module="music12.blocks.Block004_real_instruments.instrument_pipeline_runner_cli",
        module_args=module_args,
        logger=logger,
        use_maxwell=False,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__{config.instrument_name}__pipeline",
        maxwell_logdir=config.maxwell_logdir,
    )


def build_percussion_manifest(
    *,
    config: ArchiveBuildConfig,
    audio_dir: Path,
    manifest_csv: Path,
    logger: Logger,
) -> Path:
    core_list = manifest_csv.with_name(f"{manifest_csv.stem}__core_list.txt")

    run_module(
        module="music12.blocks.Block004_real_instruments.percussion_manifest12_cli",
        module_args=[
            "--input_dir", str(audio_dir),
            "--out_csv", str(manifest_csv),
            "--out_core_list", str(core_list),
            "--instrument_family", "percussion",
        ],
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__{config.instrument_name}__percussion_manifest",
        maxwell_logdir=config.maxwell_logdir,
    )
    return core_list


def build_percussion_pipeline(
    *,
    config: ArchiveBuildConfig,
    manifest_csv: Path,
    reports_root: Path,
    logger: Logger,
) -> None:
    run_module(
        module="music12.blocks.Block004_real_instruments.percussion_event_pipeline_cli",
        module_args=[
            "--manifest_csv", str(manifest_csv),
            "--reports_root", str(reports_root),
        ],
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__{config.instrument_name}__percussion_pipeline",
        maxwell_logdir=config.maxwell_logdir,
    )


def build_percussion_spiral3d(
    *,
    config: ArchiveBuildConfig,
    reports_root: Path,
    spiral3d_dir: Path,
    logger: Logger,
) -> None:
    run_module(
        module="music12.blocks.Block004_real_instruments.percussion_spiral3d_builder_cli",
        module_args=[
            "--instrument_name", config.instrument_name,
            "--reports_root", str(reports_root),
            "--out_dir", str(spiral3d_dir),
        ],
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__{config.instrument_name}__percussion_spiral3d",
        maxwell_logdir=config.maxwell_logdir,
    )


def build_percussion_passports(
    *,
    config: ArchiveBuildConfig,
    reports_root: Path,
    passports_dir: Path,
    logger: Logger,
) -> None:
    run_module(
        module="music12.blocks.Block004_real_instruments.percussion_passport_builder_cli",
        module_args=[
            "--reports_root", str(reports_root),
            "--out_dir", str(passports_dir),
        ],
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__{config.instrument_name}__percussion_passports",
        maxwell_logdir=config.maxwell_logdir,
    )


def build_archive_index(config: ArchiveBuildConfig, logger: Logger, archive_root: Path) -> Path:
    out_dir = archive_root / "_archive_indexes"
    ensure_dir(out_dir)
    out_csv = out_dir / "instrument_note_file_index.csv"

    run_module(
        module="music12.blocks.Block004_real_instruments.instrument_note_file_index_cli",
        module_args=[
            "--block004_root", str(archive_root),
            "--out_csv", str(out_csv),
        ],
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__archive_index",
        maxwell_logdir=config.maxwell_logdir,
    )
    return out_csv


def build_archive_audit(config: ArchiveBuildConfig, logger: Logger, archive_root: Path) -> Path:
    out_dir = archive_root / "_archive_audit"
    ensure_dir(out_dir)

    run_module(
        module="music12.blocks.Block004_real_instruments.block004_source_manifest_report_audit_cli",
        module_args=[
            "--block004_root", str(archive_root),
            "--out_dir", str(out_dir),
        ],
        logger=logger,
        use_maxwell=config.use_maxwell,
        task_class="instrument_analysis",
        tag=f"{config.maxwell_tag_prefix}__archive_audit",
        maxwell_logdir=config.maxwell_logdir,
    )
    return out_dir


def build_instrument_passport_archive(
    config: ArchiveBuildConfig,
    logger: Optional[Logger] = None,
) -> ArchiveBuildSummary:
    logger = logger or (lambda message: None)

    started_at = utc_now()
    library_kind = normalize_library_kind(config.library_kind)
    archive_root = Path(config.archive_root)
    instrument_root = archive_root / config.instrument_name

    audio_subdir = "audio_notes_wav" if library_kind == "pitched" else "audio_events_wav"
    audio_dir = instrument_root / "00_sources" / audio_subdir
    manifest_dir = instrument_root / "20_manifest"
    reports_root = instrument_root / "10_reports"
    range_dir = instrument_root / "20_range_research"
    note_box_dir = instrument_root / "30_note_box_profiles"
    passports_dir = instrument_root / "40_passports"
    spiral3d_dir = instrument_root / "50_spiral3d"
    program_log_dir = instrument_root / "99_program_logs"

    dirs_to_create = [audio_dir, manifest_dir, reports_root, spiral3d_dir, program_log_dir]
    if library_kind == "pitched":
        dirs_to_create.extend([range_dir, note_box_dir])
    else:
        dirs_to_create.append(passports_dir)

    for path in dirs_to_create:
        ensure_dir(path)

    logger(f"Instrument archive root: {instrument_root}")
    logger(f"Library kind: {library_kind}")

    wav_files_copied = prepare_audio_library(config, audio_dir, logger)
    run_filename_guard(audio_dir, logger=logger, fix=config.fix_cyrillic_abc)

    manifest_csv = manifest_dir / f"{config.instrument_name}_manifest_12.csv"

    outputs = {
        "audio_dir": str(audio_dir),
        "manifest_csv": str(manifest_csv),
        "reports_root": str(reports_root),
        "spiral3d_dir": str(spiral3d_dir),
        "archive_index_csv": "",
        "archive_audit_dir": "",
    }

    if library_kind == "pitched":
        manifest_mode = config.manifest_mode
        if manifest_mode == "auto":
            manifest_mode = detect_manifest_mode(audio_dir)
            logger(f"Manifest mode detected: {manifest_mode}")
        else:
            logger(f"Manifest mode selected: {manifest_mode}")

        build_manifest(
            config=config,
            audio_dir=audio_dir,
            manifest_csv=manifest_csv,
            manifest_mode=manifest_mode,
            logger=logger,
        )

        build_pipeline(
            config=config,
            audio_dir=audio_dir,
            manifest_csv=manifest_csv,
            reports_root=reports_root,
            range_dir=range_dir,
            note_box_dir=note_box_dir,
            spiral3d_dir=spiral3d_dir,
            logger=logger,
        )

        outputs.update({
            "range_dir": str(range_dir),
            "note_box_dir": str(note_box_dir),
            "passport_json": str(range_dir / f"{config.instrument_name}__instrument_passport.json"),
            "passport_md": str(range_dir / f"{config.instrument_name}__instrument_passport.md"),
            "passport_notes_csv": str(range_dir / f"{config.instrument_name}__instrument_passport_notes.csv"),
        })
    else:
        manifest_mode = "percussion_event_manifest"
        logger("Manifest mode selected: percussion_event_manifest")

        core_list = build_percussion_manifest(
            config=config,
            audio_dir=audio_dir,
            manifest_csv=manifest_csv,
            logger=logger,
        )
        build_percussion_pipeline(
            config=config,
            manifest_csv=manifest_csv,
            reports_root=reports_root,
            logger=logger,
        )
        build_percussion_spiral3d(
            config=config,
            reports_root=reports_root,
            spiral3d_dir=spiral3d_dir,
            logger=logger,
        )
        build_percussion_passports(
            config=config,
            reports_root=reports_root,
            passports_dir=passports_dir,
            logger=logger,
        )

        outputs.update({
            "passports_dir": str(passports_dir),
            "percussion_core_list": str(core_list),
            "family_passport_json": str(passports_dir / "percussion__family_passport.json"),
            "family_passport_md": str(passports_dir / "percussion__family_passport.md"),
            "event_summary_csv": str(reports_root / "percussion__event_pipeline_summary.csv"),
            "event_summary_json": str(reports_root / "percussion__event_pipeline_summary.json"),
            "event_failed_csv": str(reports_root / "percussion__event_pipeline_failed.csv"),
        })

    if library_kind == "pitched" and config.include_archive_index:
        outputs["archive_index_csv"] = str(
            build_archive_index(config, logger=logger, archive_root=archive_root)
        )
    elif library_kind == "percussion" and config.include_archive_index:
        logger("Archive index skipped: current indexer is note-library specific.")

    if library_kind == "pitched" and config.include_archive_audit:
        outputs["archive_audit_dir"] = str(
            build_archive_audit(config, logger=logger, archive_root=archive_root)
        )
    elif library_kind == "percussion" and config.include_archive_audit:
        logger("Archive audit skipped: current audit is note-library specific.")

    finished_at = utc_now()
    summary = ArchiveBuildSummary(
        status="ok",
        started_at=started_at,
        finished_at=finished_at,
        instrument_name=config.instrument_name,
        library_kind=library_kind,
        source_dir=str(Path(config.source_dir).resolve()),
        archive_root=str(archive_root.resolve()),
        instrument_root=str(instrument_root.resolve()),
        manifest_mode_requested=config.manifest_mode,
        manifest_mode_selected=manifest_mode,
        wav_files_copied=wav_files_copied,
        outputs=outputs,
    )

    summary_json = program_log_dir / f"{config.instrument_name}__archive_build_summary.json"
    summary_txt = program_log_dir / f"{config.instrument_name}__archive_build_summary.txt"

    summary_json.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "BLOCK004 INSTRUMENT PASSPORT ARCHIVE SUMMARY",
        "=" * 80,
        f"status                 : {summary.status}",
        f"started_at             : {summary.started_at}",
        f"finished_at            : {summary.finished_at}",
        f"instrument_name        : {summary.instrument_name}",
        f"library_kind           : {summary.library_kind}",
        f"source_dir             : {summary.source_dir}",
        f"archive_root           : {summary.archive_root}",
        f"instrument_root        : {summary.instrument_root}",
        f"manifest_mode_requested: {summary.manifest_mode_requested}",
        f"manifest_mode_selected : {summary.manifest_mode_selected}",
        f"wav_files_copied       : {summary.wav_files_copied}",
        "",
        "OUTPUTS",
    ]
    for key, value in summary.outputs.items():
        lines.append(f"{key:22}: {value}")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger(f"Summary written: {summary_json}")
    return summary
