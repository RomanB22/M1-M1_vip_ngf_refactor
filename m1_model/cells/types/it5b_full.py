# m1_model/cells/types/it5b_full.py
from pathlib import Path
from typing import Any, Dict
from m1_model.cells.base import ImportSpec, CellProvider

class IT5BFullFromPy(CellProvider):
    """
    IT5B full (BS1579) imported from cells/ITcell.py: class ITcell.
    Post: perisomatic secList, alldend/apicdend/spiny lists, weight norm, save PKL.
    """
    def __init__(self, project_root: Path, ynorm_5b: float):
        self.project_root = project_root
        self.ynorm_5b = ynorm_5b

    # ---- post-import helpers ----
    def _post_lists(self, netParams):
        rule = netParams.cellParams["IT5B_full"]

        # perisomatic (within 50 µm of soma)
        netParams.addCellParamsSecList(
            label="IT5B_full",
            secListName="perisom",
            somaDist=[0, 50],
        )

        # build lists from section names
        sec_names = list(rule["secs"].keys())
        alldend = [s for s in sec_names if ("dend" in s or "apic" in s)]
        apicdend = [s for s in sec_names if "apic" in s]
        spiny = [s for s in alldend if s not in ["apic_0", "apic_1"]]

        rule.setdefault("secLists", {})
        rule["secLists"]["alldend"] = alldend
        rule["secLists"]["apicdend"] = apicdend
        rule["secLists"]["spiny"] = spiny

    def import_spec(self, ctx) -> ImportSpec:
        label = "IT5B_full"
        file_py   = self.project_root / "cells" / "ITcell.py"
        pkl_path  = self.project_root / "cells" / f"{label}_cellParams.pkl"
        wnorm_pkl = self.project_root / "conn"  / "IT_full_BS1579_weightNorm.pkl"

        conds: Dict[str, Any] = {"cellType": "IT", "cellModel": "HH_full", "ynorm": self.ynorm_5b}

        return ImportSpec(
            label=label,
            conds=conds,
            kind="python",
            file=file_py,
            cell_name="ITcell",
            kwargs={
                "cellInstance": False,              # keep instance import like your HOC cases
                "cellArgs": {"params": "BS1579"},  # ← matches your snippet
                "somaAtOrigin": True,
            },
            # (no static post_patch needed; lists are computed)
            weight_norm_pkl=wnorm_pkl,
            save_to_pkl=pkl_path,
            load_from_pkl=pkl_path,                # will load if label ∈ ctx.loadCellParams
            post_fn=self._post_lists,              # compute perisom/alldend/apicdend/spiny
        )
