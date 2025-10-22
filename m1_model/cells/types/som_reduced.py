# m1_model/cells/types/som_reduced.py
from pathlib import Path
from m1_model.cells.base import ImportSpec, CellProvider

class SOMReducedFromHoc(CellProvider):
    """SOM (3-compartment) from LTS3.hoc/LTScell1."""
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def import_spec(self, ctx) -> ImportSpec:
        label = "SOM_reduced"
        hoc_file   = self.project_root / "cells" / "LTS3.hoc"                # ← matches your snippet
        pkl_path   = self.project_root / "cells" / f"{label}_cellParams.pkl"
        weight_pkl = self.project_root / "conn"  / f"{label}_weightNorm.pkl"

        return ImportSpec(
            label=label,
            conds={"cellType": "SOM", "cellModel": "HH_reduced"},
            kind="hoc",
            file=hoc_file,
            cell_name="LTScell1",
            kwargs={"cellInstance": True},
            post_patch={"secLists": {"spiny": ["soma", "dend"]}},
            weight_norm_pkl=weight_pkl,
            save_to_pkl=pkl_path,
            load_from_pkl=pkl_path,  # adapter will load if label ∈ ctx.loadCellParams
        )
