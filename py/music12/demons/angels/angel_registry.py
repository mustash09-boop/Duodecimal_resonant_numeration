from __future__ import annotations

from typing import Dict, Any, List, Optional, Type

from music12.demons.angels.angel_base import AngelBase
from music12.demons.angels.angel_repair_cli import CliNotationRepairAngel


ANGEL_REGISTRY: List[Type[AngelBase]] = [
    CliNotationRepairAngel,
]


def select_angel(report: Dict[str, Any]) -> Optional[AngelBase]:
    for angel_cls in ANGEL_REGISTRY:
        angel = angel_cls()
        if angel.can_handle(report):
            return angel
    return None