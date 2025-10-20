
from pathlib import Path
from m1_model.cells.base import CellProvider, ImportSpec
from m1_model.utils.mech_scaler import make_scale_postfn

class PVSimpleFromHoc(CellProvider):
    """PV cell (3-compartment) imported from FS3.hoc/FScell1."""
    def __init__(self, sim_dir: Path, scale_pas_g: float | None = None):
        self.sim_dir = sim_dir
        self.scale_pas_g = scale_pas_g

    def import_spec(self, ctx) -> ImportSpec:
        hoc_file = self.sim_dir / "cells" / "FS3.hoc"
        weight_norm = self.sim_dir / "conn" / "PV_simple_weightNorm.pkl"
        saved_pkl  = self.sim_dir / "cells" / "PV_simple_cellParams.pkl"

        post_fn = None
        if self.scale_pas_g and self.scale_pas_g != 1.0:
            post_fn = make_scale_postfn(
                cell_label="PV_simple",
                mech="pas",
                param="g",
                factor=self.scale_pas_g,
                sections=["soma","dend"],
            )

        return ImportSpec(
            label="PV_simple",
            conds={"cellType": "PV", "cellModel": "HH_simple"},
            kind="hoc",
            file=hoc_file,
            cell_name="FScell1",
            kwargs={"cellInstance": True},
            post_patch={"secLists": {"spiny": ["soma", "dend"]}},
            weight_norm_pkl=weight_norm,
            save_to_pkl=saved_pkl,
            load_from_pkl=saved_pkl,
            post_fn=post_fn,
        )
