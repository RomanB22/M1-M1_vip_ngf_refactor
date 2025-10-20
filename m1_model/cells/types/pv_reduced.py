
from pathlib import Path
from m1_model.cells.base import CellProvider, ImportSpec
from m1_model.utils.mech_scaler import make_scale_postfn

class PVReducedFromHoc(CellProvider):
    def __init__(self, project_root: Path, scale_pas_g: float | None = None):
        self.project_root = project_root
        self.scale_pas_g = scale_pas_g

    def import_spec(self, ctx) -> ImportSpec:
        # NOTE: resolve from project root (no "sim/" prefix)
        hoc_file = self.project_root / "cells" / "FS3.hoc"
        weight_norm = self.project_root / "conn" / "PV_reduced_weightNorm.pkl"
        saved_pkl  = self.project_root / "cells" / "PV_reduced_cellParams.pkl"

        post_fn = None
        if self.scale_pas_g and self.scale_pas_g != 1.0:
            post_fn = make_scale_postfn(
                cell_label="PV_reduced",
                mech="pas",
                param="g",
                factor=self.scale_pas_g,
                sections=["soma","dend"],
            )

        return ImportSpec(
            label="PV_reduced",
            conds={"cellType": "PV", "cellModel": "HH_reduced"},
            kind="hoc",
            file=hoc_file,                  # -> <project>/cells/FS3.hoc
            cell_name="FScell1",
            kwargs={"cellInstance": True},
            post_patch={"secLists": {"spiny": ["soma", "dend"]}},
            weight_norm_pkl=weight_norm,
            save_to_pkl=saved_pkl,
            load_from_pkl=saved_pkl,
            post_fn=post_fn,
        )