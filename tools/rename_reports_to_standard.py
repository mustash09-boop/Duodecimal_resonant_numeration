from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser(description="Rename report files to standard format")
    ap.add_argument("--report_dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. 001__RealPiano_1__5.A-")
    args = ap.parse_args()

    report_dir = Path(args.report_dir).resolve()
    prefix = args.prefix.strip()

    if not report_dir.exists():
        raise ValueError(f"Directory not found: {report_dir}")

    for file in report_dir.iterdir():
        if not file.is_file():
            continue

        name = file.name

        # уже нормализован — пропускаем
        if name.startswith(prefix):
            continue

        # начинаются с __
        if name.startswith("__"):
            new_name = f"{prefix}{name}"
        else:
            # fallback
            new_name = f"{prefix}__{name}"

        new_path = file.with_name(new_name)

        print(f"{name} -> {new_name}")
        file.rename(new_path)


if __name__ == "__main__":
    main()