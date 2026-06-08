import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
PY_ROOT = PROJECT_ROOT / "py"


def run(cmd, cwd: Path):
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(PY_ROOT)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def safe_tag(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def main():
    ap = argparse.ArgumentParser(
        description="Run targeted notation/time60 demons on selected Python files and project-law demon on selected CSV outputs."
    )
    ap.add_argument("--py-file", action="append", default=[])
    ap.add_argument("--csv", action="append", default=[])
    ap.add_argument("--focus", default="block002")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "docs" / "reports"),
    )
    ap.add_argument(
        "--tmp-scan-root",
        default=str(PROJECT_ROOT / "_demon_logs" / "tmp_targeted_scan"),
    )
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    tmp_scan_root = Path(args.tmp_scan_root)
    if tmp_scan_root.exists():
        shutil.rmtree(tmp_scan_root)
    tmp_scan_root.mkdir(parents=True, exist_ok=True)

    for file_str in args.py_file:
        src = Path(file_str)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, tmp_scan_root / src.name)

    if args.py_file:
        run(
            [
                sys.executable,
                "-m",
                "music12.demons.demon_notation_alphabet12_consistency",
                "--root",
                str(tmp_scan_root),
                "--out_txt",
                str(report_dir / "targeted_notation12.txt"),
                "--out_json",
                str(report_dir / "targeted_notation12.json"),
            ],
            PROJECT_ROOT,
        )
        run(
            [
                sys.executable,
                "-m",
                "music12.demons.demon_time60_consistency",
                "--root",
                str(tmp_scan_root),
                "--out_txt",
                str(report_dir / "targeted_time60.txt"),
                "--out_json",
                str(report_dir / "targeted_time60.json"),
            ],
            PROJECT_ROOT,
        )

    for csv_str in args.csv:
        csv_path = Path(csv_str)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        tag = safe_tag(csv_path.stem)
        cmd = [
            sys.executable,
            "-m",
            "music12.demons.demon_project_law_report",
            "--input_csv",
            str(csv_path),
            "--focus",
            args.focus,
            "--out_txt",
            str(report_dir / f"{tag}_project_law.txt"),
            "--out_json",
            str(report_dir / f"{tag}_project_law.json"),
        ]
        if args.strict:
            cmd.append("--strict")
        run(cmd, PROJECT_ROOT)

    print(f"reports_dir={report_dir}")


if __name__ == "__main__":
    main()
