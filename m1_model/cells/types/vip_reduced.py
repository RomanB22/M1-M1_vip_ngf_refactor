# m1_model/cells/types/vip_reduced.py
from pathlib import Path
from m1_model.cells.base import ImportSpec, CellProvider

class VIPReducedFromHoc(CellProvider):
    """VIP (reduced) from vipcr_cell.hoc / VIPCRCell_EDITED."""
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def import_spec(self, ctx) -> ImportSpec:
        label = "VIP_reduced"
        hoc_file   = self.project_root / "cells" / "vipcr_cell.hoc"
        pkl_path   = self.project_root / "cells" / f"{label}_cellParams.pkl"
        weight_pkl = self.project_root / "conn"  / f"{label}_weightNorm.pkl"

        return ImportSpec(
            label=label,
            conds={"cellType": "VIP", "cellModel": "HH_reduced"},
            kind="hoc",
            file=hoc_file,
            cell_name="VIPCRCell_EDITED",
            kwargs={"cellInstance": False, "importSynMechs": True},
            post_patch={"secLists": {"spiny": ["soma", "rad1", "rad2", "ori1", "ori2"]}},
            weight_norm_pkl=weight_pkl,
            save_to_pkl=pkl_path,
            load_from_pkl=pkl_path,  # loaded if label ∈ ctx.loadCellParams
        )
