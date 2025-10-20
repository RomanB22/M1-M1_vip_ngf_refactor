
from typing import Dict, Callable, Iterable
from pathlib import Path
from m1_model.cells.base import CellProvider
from m1_model.cells.types.pv_reduced import PVReducedFromHoc

def build_registry(ctx) -> Dict[str, Callable[[], CellProvider]]:
    proj_dir = Path(ctx.project_root)
    scale_pas = getattr(ctx.cfg, "pv_scale_pas_g", None)
    return {
        "PV_reduced": lambda: PVReducedFromHoc(proj_dir, scale_pas_g=scale_pas),
    }

def get_enabled_cells(config: dict, ctx) -> Iterable[CellProvider]:
    reg = build_registry(ctx)
    enabled = config.get("enabled_cells")
    for name, factory in reg.items():
        if enabled and name not in enabled:
            continue
        yield factory()
