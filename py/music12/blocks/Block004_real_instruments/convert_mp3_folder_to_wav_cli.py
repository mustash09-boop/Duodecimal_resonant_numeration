from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MP3 folder to WAV and log broken files.")
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ffmpeg_path", required=True)
    ap.add_argument("--sample_rate", default="44100")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--bad_report_csv", default="")
    ap.add_argument("--bad_report_txt", default="")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    ffmpeg = Path(args.ffmpeg_path)

    if not ffmpeg.exists():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")

    out_dir.mkdir(parents=True, exist_ok=True)

    bad_csv = Path(args.bad_report_csv) if args.bad_report_csv else out_dir / "_bad_mp3_files.csv"
    bad_txt = Path(args.bad_report_txt) if args.bad_report_txt else out_dir / "_bad_mp3_files.txt"

    files = sorted(src_dir.glob("*.mp3"))
    print(f"MP3 files found: {len(files)}")

    converted = 0
    skipped = 0
    failed = []

    for i, p in enumerate(files, 1):
        out_wav = out_dir / f"{p.stem}.wav"

        if out_wav.exists() and not args.overwrite:
            print(f"[SKIP] {out_wav.name} already exists")
            skipped += 1
            continue

        print(f"[{i}/{len(files)}] {p.name} -> {out_wav.name}", flush=True)

        cmd = [
            str(ffmpeg),
            "-y" if args.overwrite else "-n",
            "-i", str(p),
            "-ar", str(args.sample_rate),
            "-ac", "1",
            str(out_wav),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            print(f"[BAD MP3] {p.name}")
            failed.append({
                "source_mp3": str(p),
                "target_wav": str(out_wav),
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-2000:].replace("\r", ""),
            })
            continue

        converted += 1

    with bad_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["source_mp3", "target_wav", "returncode", "stderr_tail"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(failed)

    lines = [
        "MP3 TO WAV CONVERSION REPORT",
        "=" * 80,
        f"src_dir   : {src_dir}",
        f"out_dir   : {out_dir}",
        f"total     : {len(files)}",
        f"converted : {converted}",
        f"skipped   : {skipped}",
        f"failed    : {len(failed)}",
        "",
        "FAILED FILES",
        "-" * 80,
    ]

    for item in failed:
        lines.append(item["source_mp3"])
        lines.append(f"returncode: {item['returncode']}")
        lines.append(item["stderr_tail"])
        lines.append("-" * 80)

    bad_txt.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("DONE")
    print(f"converted: {converted}")
    print(f"skipped  : {skipped}")
    print(f"failed   : {len(failed)}")
    print(f"bad csv  : {bad_csv}")
    print(f"bad txt  : {bad_txt}")


if __name__ == "__main__":
    main()