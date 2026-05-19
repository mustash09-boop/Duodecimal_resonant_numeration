from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from music12.demons.demon_maxwell_core import run_maxwell


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Supreme Maxwell Demon for routing subordinate demons."
    )
    p.add_argument("-m", "--module", required=True, help="Target module to run")
    p.add_argument("--task-class", default="module_run", help="Task class")
    p.add_argument("--project-root", default=None, help="Project root path")
    p.add_argument("--logdir", default="_demon_logs", help="Log directory")
    p.add_argument("--tag", default="maxwell", help="Tag for report files")
    p.add_argument(
        "--demons-dir",
        default="py/music12/demons",
        help="Directory to scan for unregistered demons",
    )

    # optional context for result-report demons
    p.add_argument("--matrix_csv", default=None, help="Matrix CSV for result-report demons")
    p.add_argument("--times_csv", default=None, help="Times CSV for result-report demons")
    p.add_argument("--coords_csv", default=None, help="Coords CSV for result-report demons")
    p.add_argument("--detail_depth", type=int, default=None, help="Detail depth for result-report demons")
    p.add_argument("--top_k", type=int, default=None, help="Top-K for result-report demons")
    p.add_argument("--source_name", default=None, help="Source name for result-report demons")

    p.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="Arguments for target module after '--'",
    )
    return p


def main() -> int:
    parser = build_parser()
    ns = parser.parse_args()

    module_args = list(ns.module_args)
    if module_args and module_args[0] == "--":
        module_args = module_args[1:]

    extra = {
        "matrix_csv": ns.matrix_csv,
        "times_csv": ns.times_csv,
        "coords_csv": ns.coords_csv,
        "detail_depth": ns.detail_depth,
        "top_k": ns.top_k,
        "source_name": ns.source_name,
    }

    report = run_maxwell(
        target_module=ns.module,
        argv=module_args,
        task_class=ns.task_class,
        project_root=ns.project_root,
        logdir=ns.logdir,
        tag=ns.tag,
        demons_dir=ns.demons_dir,
        extra=extra,
    )

    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())