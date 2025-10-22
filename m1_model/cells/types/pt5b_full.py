# m1_model/cells/types/pt5b_full.py
from pathlib import Path
from typing import Any, Dict
from m1_model.cells.base import ImportSpec, CellProvider

class PT5BFullFromHoc(CellProvider):
    """
    PT5B full imported from cells/PTcell.hoc : template PTcell.
    Matches original: secLists (perisom, below_soma), nonSpiny removal, Ih tuning, Na tweaks, TTX, weightNorm, save.
    """
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ctx = None

    def _post(self, netParams):
        cfg = self.ctx.cfg
        label = "PT5B_full"
        rule = netParams.cellParams[label]

        # secLists
        netParams.addCellParamsSecList(label=label, secListName="perisom", somaDist=[0, 50])
        netParams.addCellParamsSecList(label=label, secListName="below_soma", somaDistY=[-600, 0])
        rule.setdefault("secLists", {})
        nonSpiny = ["apic_0", "apic_1"]
        # remove nonSpiny from perisom if present
        perisom = rule["secLists"].get("perisom", [])
        rule["secLists"]["perisom"] = [s for s in perisom if s not in nonSpiny]

        # build dend lists
        names = list(rule["secs"].keys())
        alldend = [s for s in names if ("dend" in s or "apic" in s)]
        apicdend = [s for s in names if "apic" in s]
        spiny = [s for s in alldend if s not in nonSpiny]
        rule["secLists"]["alldend"] = alldend
        rule["secLists"]["apicdend"] = apicdend
        rule["secLists"]["spiny"] = spiny

        # Ih adaptations
        ih_mechs = {"ih", "h", "h15", "hd"}
        for secName, sec in rule["secs"].items():
            for mechName, mech in sec.get("mechs", {}).items():
                if mechName in ih_mechs:
                    gbar = mech.get("gbar")
                    if isinstance(gbar, list):
                        mech["gbar"] = [g * cfg.ihGbar for g in gbar]
                    elif gbar is not None:
                        mech["gbar"] = gbar * cfg.ihGbar
                    if cfg.ihModel == "migliore":
                        mech["clk"] = cfg.ihlkc
                        mech["elk"] = cfg.ihlke
                    if secName.startswith("dend"):
                        if "gbar" in mech: mech["gbar"] *= cfg.ihGbarBasal
                        if "clk" in mech: mech["clk"] *= cfg.ihlkcBasal
                    if secName in rule["secLists"]["below_soma"]:
                        if "clk" in mech: mech["clk"] *= cfg.ihlkcBelowSoma

        # Dend/soma/axon Na adjustments
        nax_base = 0.0153130368342
        for secName in rule["secLists"]["alldend"]:
            rule["secs"][secName]["mechs"]["nax"]["gbar"] = nax_base * cfg.dendNa
        rule["secs"]["soma"]["mechs"]["nax"]["gbar"] = nax_base * cfg.somaNa
        rule["secs"]["axon"]["mechs"]["nax"]["gbar"] = nax_base * cfg.axonNa
        rule["secs"]["axon"]["geom"]["Ra"] = 137.494564931 * cfg.axonRa

        # Weight norm at the end
        netParams.addCellParamsWeightNorm(
            label,
            str(self.project_root / "conn" / "PT5B_full_weightNorm.pkl"),
            threshold=getattr(cfg, "weightNormThreshold", None),
        )

    def import_spec(self, ctx) -> ImportSpec:
        self.ctx = ctx
        label = "PT5B_full"
        file_hoc = self.project_root / "cells" / "PTcell.hoc"
        pkl_path = self.project_root / "cells" / f"{label}_cellParams.pkl"

        ihMod2str = {"harnett": 1, "kole": 2, "migliore": 3}
        cell_args = [ihMod2str[ctx.cfg.ihModel], ctx.cfg.ihSlope]

        conds: Dict[str, Any] = {"cellType": "PT", "cellModel": "HH_full"}

        return ImportSpec(
            label=label,
            conds=conds,
            kind="hoc",
            file=file_hoc,
            cell_name="PTcell",
            kwargs={
                "cellInstance": False,       # (omit/False is fine for a rule)
                "cellArgs": cell_args,       # [ih model code, ihSlope]
                "somaAtOrigin": True,
            },
            save_to_pkl=pkl_path,
            load_from_pkl=pkl_path,         # loaded if label ∈ ctx.loadCellParams
            post_fn=self._post,             # all post-processing here
        )
