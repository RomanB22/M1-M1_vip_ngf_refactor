# m1_model/cells/types/pt5b_tim.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from m1_model.cells.base import ImportSpec, CellProvider


class PT5BFullTimFromPy(CellProvider):
    """
    Additional PT5B model (“Tim” variant) under the canonical label PT5B_full.

    Pipeline:
      1) Load mutant params CSV selected by cfg.variant (or 'WT') and write the JSON
         file that the Na12 model expects.
      2) importCellParams from Na12HMMModel_TF.py (class Na12Model_TF), soma at origin.
      3) Post-processing:
         - Rename soma_0 -> soma
         - Set spikeGenLoc on axon_0
         - Inject pt3d points for axon_0 and axon_1 (keeps SectionLists robust)
         - Reset/compute secLists: perisom, below_soma, alldend, apicdend, spiny
         - Heterozygous / blockNa toggles (na12/na12mut gbar)
         - Ih scaling via cfg.ihGbar (+ cfg.ihGbarBasal on dend*), skip axon_*
         - Reduce apical (dendritic) Na via cfg.dendNa
         - Add weight normalization; optional save to JSON
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ctx = None  # set in import_spec

    def _resolve_model_json(self, ctx) -> Path:
        """
        Resolve the Tim JSON rule file.
        Priority:
          1) cfg.pt5b_tim_json (if provided)
          2) Current canonical cells/ JSON
          3) Historical fallback paths
        """
        variant = ctx.cfg.variant if getattr(ctx.cfg, "loadmutantParams", False) else "WT"
        configured = getattr(ctx.cfg, "pt5b_tim_json", None)

        candidates = []
        if configured:
            cfg_path = Path(str(configured))
            if not cfg_path.is_absolute():
                cfg_path = self.project_root / cfg_path
            candidates.append(cfg_path)

        candidates.extend(
            [
                self.project_root / "cells" / "Na12HH16HH_TF_Feb18th2026_NoWeightNorm.json",
                self.project_root / "wscale_PT5B_Tim" / "Na12HH16HH_TF_Feb18th2026_NoWeightNorm.json",
                self.project_root / "wscale_PT5B_Tim" / "Na12HH16HH_TF_May29th2025_NoWeightNorm.json",
                self.project_root / "cells" / "UCDavisCells_Heterozygous" / f"Na12HH16HH_{variant}_11242025.json",
            ]
        )

        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    # -------------------------- Post-processing hook --------------------------

    def _post(self, netParams):
        cfg = self.ctx.cfg
        label = "PT5B_full"
        rule = netParams.cellParams[label]

        # 7) Ih scaling (skip axon_*)
        if hasattr(cfg, "ihGbar"):
            for secName, sec in rule["secs"].items():
                if secName in ("axon_0", "axon_1"):
                    continue
                Ih = sec.get("mechs", {}).get("Ih")
                if Ih and "gIhbar" in Ih:
                    g = Ih["gIhbar"]
                    scaled = [v * cfg.ihGbar for v in g] if isinstance(g, list) else g * cfg.ihGbar
                    if secName.startswith("dend") and hasattr(cfg, "ihGbarBasal"):
                        scaled = [v * cfg.ihGbarBasal for v in scaled] if isinstance(scaled, list) else scaled * cfg.ihGbarBasal
                    Ih["gIhbar"] = scaled

        # 8) Reduce dendritic Na on apical sections
        if hasattr(cfg, "dendNa"):
            for secName, sec in rule["secs"].items():
                if secName.startswith("apic"):
                    mechs = sec.get("mechs", {})
                    for mname in ("na12", "na12mut"):
                        if mname in mechs and "gbar" in mechs[mname]:
                            g = mechs[mname]["gbar"]
                            mechs[mname]["gbar"] = [v * cfg.dendNa for v in g] if isinstance(g, list) else g * cfg.dendNa

        # 9) Weight normalization
        netParams.addCellParamsWeightNorm(
            label,
            str(self.project_root / "conn" / "PT5B_full_weightNorm_TIM.pkl"),
            threshold=getattr(cfg, "weightNormThreshold", None),
        )

        # if getattr(cfg, "saveCellParams", False):
        #     netParams.saveCellParamsRule(
        #         label=label,
        #         fileName=str(self.project_root / "cells" / "WeightNorm_Na12HH16HH_WT_11242025.json"),
        #     )

    # ------------------------------ ImportSpec --------------------------------

    def import_spec(self, ctx) -> ImportSpec:
        """
        Prepare the ImportSpec for PT5B_full (Tim variant):

          - Import the cell directly from a saved JSON NetPyNE cellParams rule.
        """
        self.ctx = ctx

        label = "PT5B_full"
        model_json_rule = self._resolve_model_json(ctx)

        # 2) Build ImportSpec: **only** load from JSON, never run the Python file
        conds: Dict[str, Any] = {"cellType": "PT", "cellModel": "HH_full"}

        return ImportSpec(
            label=label,
            conds=conds,
            # IMPORTANT: this tells your loader to read the cellParams rule
            # directly from JSON rather than importing a Python class.
            kind="json",
            file=model_json_rule,          # primary JSON file
            load_from_pkl=model_json_rule, # optional, if you want same path reused
            post_fn=self._post,
        )
