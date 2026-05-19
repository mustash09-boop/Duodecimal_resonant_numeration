from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional

from music12.demons.demon_registry import DemonRegistry
from music12.demons.demon_types import DemonSpec


def _module_name_from_file(py_path: Path, package_root: str = "music12.demons") -> Optional[str]:
    name = py_path.stem
    if not name.endswith(".py") and py_path.suffix != ".py":
        return None
    if name.startswith("_"):
        return None
    return f"{package_root}.{name}"


def discover_unregistered_demons(
    *,
    demons_dir: str | Path,
    registry: DemonRegistry,
    package_root: str = "music12.demons",
) -> List[DemonSpec]:
    demons_dir = Path(demons_dir)
    out: List[DemonSpec] = []

    if not demons_dir.exists():
        return out

    for py_file in sorted(demons_dir.glob("*.py")):
        if py_file.stem.startswith("_"):
            continue
        if py_file.stem in {
            "demon_types",
            "demon_registry",
            "demon_discovery",
            "demon_maxwell_core",
            "demon_maxwell_cli",
            "demon_wrap",
        }:
            continue

        mod_name = _module_name_from_file(py_file, package_root=package_root)
        if not mod_name:
            continue

        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        spec = getattr(mod, "DEMON_SPEC", None)
        if spec is None:
            getter = getattr(mod, "get_demon_spec", None)
            if callable(getter):
                try:
                    spec = getter()
                except Exception:
                    spec = None

        if not isinstance(spec, DemonSpec):
            continue

        if registry.has(spec.demon_id):
            continue

        out.append(spec)

    return sorted(out, key=lambda x: (x.priority, x.demon_id))