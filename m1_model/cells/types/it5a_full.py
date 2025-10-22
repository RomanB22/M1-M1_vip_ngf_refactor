# m1_model/cells/types/it5a_full.py
from pathlib import Path
from typing import Any, Dict
from m1_model.cells.base import ImportSpec, CellProvider

class IT5AFullFromPy(CellProvider):
    """
    IT5A full (BS1579) imported from cells/ITcell.py : class ITcell.
    Post: rename soma_0→soma, add weightNorm, perisom, alldend/apicdend/spiny, save PKL.
    """
    def __init__(self, project_root: Path, ynorm_5a: float):
        self.project_root = project_root
        self.ynorm_5a = ynorm_5a

    def _post(self, netParams):
        label = "IT5A_full"
        # 1) rename soma_0 -> soma
        netParams.renameCellParamsSec(label=label, oldSec="soma_0", newSec="soma")

        # 2) weight norm (explicitly here to control ordering)
        netParams.addCellParamsWeightNorm(
            label, str(self.project_root / "conn" / "IT_full_BS1579_weightNorm.pkl"),
            threshold=getattr(getattr(self, "ctx", None), "cfg", None).weightNormThreshold
            if getattr(self, "ctx", None) and hasattr(self.ctx, "cfg") else None
        )

        # 3) perisomatic secList (0–50 µm)
        netParams.addCellParamsSecList(label=label, secListName="perisom", somaDist=[0, 50])

        # 4) build secLists from names
        rule = netParams.cellParams[label]
        names = list(rule["secs"].keys())
        alldend = [s for s in names if ("dend" in s or "apic" in s)]
        apicdend = [s for s in names if "apic" in s]
        spiny = [s for s in alldend if s not in ["apic_0", "apic_1"]]

        rule.setdefault("secLists", {})
        rule["secLists"]["alldend"] = alldend
        rule["secLists"]["apicdend"] = apicdend
        rule["secLists"]["spiny"] = spiny

    def import_spec(self, ctx) -> ImportSpec:
        # keep ctx for threshold in _post (optional)
        self.ctx = ctx

        label = "IT5A_full"
        file_py  = self.project_root / "cells" / "ITcell.py"
        pkl_path = self.project_root / "cells" / f"{label}_cellParams.pkl"

        conds: Dict[str, Any] = {"cellType": "IT", "cellModel": "HH_full", "ynorm": self.ynorm_5a}

        return ImportSpec(
            label=label,
            conds=conds,
            kind="python",
            file=file_py,
            cell_name="ITcell",
            kwargs={
                "cellInstance": False,              # per your note
                "cellArgs": {"params": "BS1579"},
                "somaAtOrigin": True,
            },
            # weight norm handled in post_fn to ensure it's after rename
            save_to_pkl=pkl_path,
            load_from_pkl=pkl_path,                # will load if label ∈ ctx.loadCellParams
            post_fn=self._post,
        )
