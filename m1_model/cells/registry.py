
from typing import Dict, Callable, Iterable
from pathlib import Path
from m1_model.cells.base import CellProvider
from m1_model.cells.types.pv_simple import PVSimpleFromHoc

def build_registry(ctx) -> Dict[str, Callable[[], CellProvider]]:
    sim_dir = Path(ctx.sim_dir)
    scale_pas = getattr(ctx.cfg, "pv_scale_pas_g", None)  # optional cfg knob
    return {
        "PV_simple": lambda: PVSimpleFromHoc(sim_dir, scale_pas_g=scale_pas),
        # add more cells here as you migrate
    }

def get_enabled_cells(config: dict, ctx) -> Iterable[CellProvider]:
    reg = build_registry(ctx)
    enabled = config.get("enabled_cells")
    for name, factory in reg.items():
        if enabled and name not in enabled:
            continue
        yield factory()
