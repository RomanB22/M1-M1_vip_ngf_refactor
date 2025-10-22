# m1_model/cells/types/reduced_6comp_single.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from m1_model.cells.base import ImportSpec, CellProvider

# Mapping for the 6-comp reduced family
_FAMILY: Dict[str, Dict[str, Optional[str]]] = {
    "IT2_reduced":  {"layer": "2",  "cname": "CSTR6", "carg": "BS1578"},
    "IT4_reduced":  {"layer": "4",  "cname": "CSTR6", "carg": "BS1578"},
    "IT5A_reduced": {"layer": "5A", "cname": "CSTR6", "carg": "BS1579"},
    "IT5B_reduced": {"layer": "5B", "cname": "CSTR6", "carg": "BS1579"},
    "PT5B_reduced": {"layer": "5B", "cname": "SPI6",  "carg": None},
    "IT6_reduced":  {"layer": "6",  "cname": "CSTR6", "carg": "BS1579"},
    "CT6_reduced":  {"layer": "6",  "cname": "CSTR6", "carg": "BS1578"},
}

# Standard secLists for these templates
_SECLISTS: Dict[str, list] = {
    "alldend":  ["Adend1", "Adend2", "Adend3", "Bdend"],
    "spiny":    ["Adend1", "Adend2", "Adend3", "Bdend"],
    "apicdend": ["Adend1", "Adend2", "Adend3"],
    "perisom":  ["soma"],
}


def _midpoint(bounds) -> float:
    """Return midpoint of [ymin, ymax]."""
    return bounds[0] + (bounds[1] - bounds[0]) / 2.0


def _resolve_layer_bounds(
    ctx: Any, layer_key: str, layers_override: Optional[Dict[Any, Any]] = None
):
    """
    Resolve layer bounds from:
      1) layers_override (preferred; passed from registry)
      2) ctx.cfg.layers (fallback)

    Supports keys like '2', 2, 'L2', '5A', '5a'.
    """
    # Choose the source of truth
    layers = layers_override or getattr(getattr(ctx, "cfg", object()), "layers", {}) or {}
    if not layers:
        return None

    # Direct hit
    if layer_key in layers:
        return layers[layer_key]

    # Try integer variants and 'L#'
    try:
        as_int = int(layer_key)
        if as_int in layers:
            return layers[as_int]
        lkey = f"L{as_int}"
        if lkey in layers:
            return layers[lkey]
    except (TypeError, ValueError):
        pass

    # Case-insensitive (for 5A/5B)
    up = str(layer_key).upper()
    if up in layers:
        return layers[up]
    low = str(layer_key).lower()
    if low in layers:
        return layers[low]

    # 'L5A' style aliases
    for prefix in ("L", "l"):
        cand = f"{prefix}{layer_key}"
        if cand in layers:
            return layers[cand]

    return None


class SingleSixCompReduced(CellProvider):
    """
    Provider for a single 6-comp reduced model:
      One of IT2/IT4/IT5A/IT5B/PT5B/IT6/CT6 (label ends with '_reduced').

    Pass `layers_override` from registry to make ynorm/length math independent
    of any ctx-side hydration.
    """

    def __init__(self, project_root: Path, label: str, layers_override: Optional[Dict[Any, Any]] = None):
        assert label in _FAMILY, f"Unknown reduced label: {label}"
        self.project_root = project_root
        self.label = label
        self.layers_override = layers_override

    # --------- internal post-step that mirrors the legacy loop ----------
    def _post_fn(self, layer_key: str, ctx):
        # MUST use cfg.sizeY, with a safe fallback to ctx.sizeY
        sizeY = float(getattr(getattr(ctx, "cfg", object()), "sizeY", getattr(ctx, "sizeY", 1.0)))
        threshold = getattr(ctx, "weightNormThreshold", None)
        weight_pkl = self.project_root / "conn" / f"{self.label}_weightNorm.pkl"

        def _post(netParams):
            rule = netParams.cellParams[self.label]

            # 1) adapt dend L based on layer midpoint * cfg.sizeY
            bounds = _resolve_layer_bounds(ctx, layer_key, self.layers_override)
            if bounds is not None:
                dendL = _midpoint(bounds) * sizeY
                for secName in ["Adend1", "Adend2", "Adend3", "Bdend"]:
                    if secName in rule["secs"]:
                        rule["secs"][secName]["geom"]["L"] = float(dendL) / 3.0

            # 2) secLists (filter to existing sections)
            rule.setdefault("secLists", {})
            for k, v in _SECLISTS.items():
                rule["secLists"][k] = [s for s in v if s in rule["secs"]]

            # 3) ensure weightNorms (fallback if not already attached)
            try:
                any_sec = next(iter(rule["secs"].values()))
                has_wn = "weightNorm" in any_sec
            except StopIteration:
                has_wn = False
            if not has_wn:
                try:
                    netParams.addCellParamsWeightNorm(self.label, str(weight_pkl), threshold=threshold)
                except Exception:
                    pass

            # 4) 3D geometry stacking (matches original loop)
            offset, prevL = 0.0, 0.0
            secs = rule["secs"]
            somaL = secs.get("soma", {}).get("geom", {}).get("L", 0.0)
            for secName, sec in secs.items():
                g = sec["geom"]
                g["pt3d"] = []  # reset
                diam = float(g.get("diam", 1.0))
                L = float(g.get("L", 0.0))

                if secName in ["soma", "Adend1", "Adend2", "Adend3"]:
                    g["pt3d"].append([offset, prevL, 0.0, diam])
                    prevL += L
                    g["pt3d"].append([offset, prevL, 0.0, diam])
                elif secName == "Bdend":
                    g["pt3d"].append([offset, somaL, 0.0, diam])
                    g["pt3d"].append([offset + L, somaL, 0.0, diam])
                elif secName == "axon":
                    g["pt3d"].append([offset, 0.0, 0.0, diam])
                    g["pt3d"].append([offset, -L, 0.0, diam])

        return _post

    # --------------------- ImportSpec construction ----------------------
    def import_spec(self, ctx) -> ImportSpec:
        meta = _FAMILY[self.label]
        file = self.project_root / "cells" / f"{meta['cname']}.py"
        pkl = self.project_root / "cells" / f"{self.label}_cellParams.pkl"
        weight_pkl = self.project_root / "conn" / f"{self.label}_weightNorm.pkl"

        # Set ynorm in conds (from layers_override or cfg.layers)
        conds: Dict[str, Any] = {"cellType": self.label[0:2], "cellModel": "HH_reduced"}
        bounds = _resolve_layer_bounds(ctx, meta["layer"], self.layers_override)
        if bounds is not None:
            conds["ynorm"] = bounds  # e.g., [0.1, 0.29]

        # kwargs for import; include cellArgs when needed
        kwargs: Dict[str, Any] = {"cellInstance": True}
        if meta["carg"]:
            kwargs["cellArgs"] = {"params": meta["carg"]}

        return ImportSpec(
            label=self.label,
            conds=conds,
            kind="python",
            file=file,
            cell_name=meta["cname"],
            kwargs=kwargs,            # includes cellArgs when present
            post_patch={},            # all mutations happen in post_fn
            save_to_pkl=pkl,
            load_from_pkl=pkl,        # load if present in ctx.loadCellParams
            weight_norm_pkl=weight_pkl,  # primary path to attach per-sec weightNorm
            post_fn=self._post_fn(meta["layer"], ctx),
        )
