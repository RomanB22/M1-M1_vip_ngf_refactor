# m1_model/cells/types/ngf_reduced.py
from pathlib import Path
from m1_model.cells.base import ImportSpec, CellProvider

class NGFReducedFromHoc(CellProvider):
    """NGF (reduced) from ngf_cell.hoc / ngfcell; includes 2×1.5 = 2.25 scaling of soma.weightNorm[0]."""
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def _weightnorm_boost(self, netParams):
        rule = netParams.cellParams["NGF_reduced"]
        try:
            wn = rule["secs"]["soma"]["weightNorm"]
            if wn and isinstance(wn[0], (int, float)):
                wn[0] *= 2.25  # two successive 1.5× multiplies
        except Exception:
            # If fields are missing, do nothing (keeps semantics safe)
            pass

    def import_spec(self, ctx) -> ImportSpec:
        label = "NGF_reduced"
        hoc_file   = self.project_root / "cells" / "ngf_cell.hoc"
        pkl_path   = self.project_root / "cells" / f"{label}_cellParams.pkl"
        weight_pkl = self.project_root / "conn"  / f"{label}_weightNorm.pkl"

        return ImportSpec(
            label=label,
            conds={"cellType": "NGF", "cellModel": "HH_reduced"},
            kind="hoc",
            file=hoc_file,
            cell_name="ngfcell",
            kwargs={"cellInstance": False, "importSynMechs": True},
            post_patch={"secLists": {"spiny": ["soma", "dend"]}},
            weight_norm_pkl=weight_pkl,
            save_to_pkl=pkl_path,
            load_from_pkl=pkl_path,   # loaded if label ∈ ctx.loadCellParams
            post_fn=self._weightnorm_boost,  # apply the 2.25× after import/load
        )
