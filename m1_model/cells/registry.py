
from typing import Dict, Callable, Iterable
from pathlib import Path
from m1_model.cells.base import CellProvider
from m1_model.cells.types.pv_reduced import PVReducedFromHoc
from m1_model.cells.types.som_reduced import SOMReducedFromHoc
from m1_model.cells.types.vip_reduced import VIPReducedFromHoc
from m1_model.cells.types.ngf_reduced import NGFReducedFromHoc
from m1_model.cells.types.it5a_full import IT5AFullFromPy
from m1_model.cells.types.pt5b_full import PT5BFullFromHoc
from m1_model.cells.types.pt5b_tim import PT5BFullTimFromPy
from m1_model.cells.types.reduced_6comp_single import SingleSixCompReduced

def build_registry(ctx) -> Dict[str, Callable[[], CellProvider]]:
    proj_dir = Path(ctx.project_root)

    # Pull layer bounds directly from cfg (source of truth)
    cfg_layers = getattr(getattr(ctx, "cfg", object()), "layer", None)
    if not cfg_layers:
        raise RuntimeError("cfg.layer is missing; required for reduced 6-comp models.")

    # Other cfg-derived knobs
    scale_pas = getattr(ctx.cfg, "pv_scale_pas_g", None)
    ynorm_5a = cfg_layers.get("5A", getattr(ctx, "ynorm_5a", 0.5))
    use_tim = getattr(ctx.cfg, "pt5b_variant", "standard") == "tim"

    return {
        # Existing explicit entries
        "PV_reduced":  lambda: PVReducedFromHoc(proj_dir, scale_pas_g=scale_pas),
        "SOM_reduced": lambda: SOMReducedFromHoc(ctx.project_root),
        "VIP_reduced": lambda: VIPReducedFromHoc(ctx.project_root),
        "NGF_reduced": lambda: NGFReducedFromHoc(ctx.project_root),
        "IT5A_full":   lambda: IT5AFullFromPy(ctx.project_root, ynorm_5a=ynorm_5a),
        "PT5B_full": (lambda: PT5BFullTimFromPy(ctx.project_root)) if use_tim
                      else (lambda: PT5BFullFromHoc(ctx.project_root)),

        # Explicit 6-comp reduced models (pass layers_override)
        "IT2_reduced":  lambda: SingleSixCompReduced(proj_dir, "IT2_reduced",  layers_override=cfg_layers),
        "IT4_reduced":  lambda: SingleSixCompReduced(proj_dir, "IT4_reduced",  layers_override=cfg_layers),
        "IT5A_reduced": lambda: SingleSixCompReduced(proj_dir, "IT5A_reduced", layers_override=cfg_layers),
        "IT5B_reduced": lambda: SingleSixCompReduced(proj_dir, "IT5B_reduced", layers_override=cfg_layers),
        "PT5B_reduced": lambda: SingleSixCompReduced(proj_dir, "PT5B_reduced", layers_override=cfg_layers),
        "IT6_reduced":  lambda: SingleSixCompReduced(proj_dir, "IT6_reduced",  layers_override=cfg_layers),
        "CT6_reduced":  lambda: SingleSixCompReduced(proj_dir, "CT6_reduced",  layers_override=cfg_layers),
    }

def get_enabled_cells(config: dict, ctx) -> Iterable[CellProvider]:
    reg = build_registry(ctx)
    enabled = config.get("enabled_cells")
    for name, factory in reg.items():
        if enabled and name not in enabled:
            continue
        yield factory()