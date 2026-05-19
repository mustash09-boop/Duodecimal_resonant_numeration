from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()

TARGET_FOLDERS = [
    "py/music12/blocks",
    "py/music12/tools",
]


def run_demon(module: str, tag: str, root: str):
    cmd = [
        sys.executable,
        "-m",
        "music12.demons.demon_wrap",
        "--logdir",
        "_demon_logs",
        "--tag",
        tag,
        "-m",
        module,
        "--",
        "--root",
        root,
        "--out_txt",
        f"docs/reports/{tag}.txt",
        "--out_json",
        f"docs/reports/{tag}.json",
    ]

    print("\nRUNNING:", " ".join(cmd), "\n")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all principle-consistency demons for selected project folders"
    )

    args = parser.parse_args()

    for folder in TARGET_FOLDERS:

        print("\n" + "=" * 70)
        print("SCANNING:", folder)
        print("=" * 70)

        run_demon(
            "music12.demons.demon_notation_alphabet12_consistency",
            "notation_alphabet12_consistency",
            folder,
        )

        run_demon(
            "music12.demons.demon_time60_consistency",
            "time60_consistency",
            folder,
        )

    print("\nAll principle demons finished.\n")


if __name__ == "__main__":
    main()