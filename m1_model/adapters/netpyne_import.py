# m1_model/adapters/netpyne_import.py
from typing import Iterable
from m1_model.cells.base import CellProvider, ImportSpec
from pathlib import Path

def add_cells_via_import(netParams, cells: Iterable[CellProvider], ctx) -> None:
    load_labels = set(getattr(ctx, "loadCellParams", []) or [])

    for provider in cells:
        spec: ImportSpec = provider.import_spec(ctx)

        # 0) sanity: file must exist
        f = Path(spec.file)
        if not f.exists():
            raise FileNotFoundError(f"Cell '{spec.label}': file not found: {f}")

        # 1) import or load
        if spec.label in load_labels and spec.load_from_pkl:
            netParams.loadCellParamsRule(label=spec.label, fileName=str(spec.load_from_pkl))
            cell_rule = netParams.cellParams.get(spec.label)
        else:
            kwargs = dict(label=spec.label, conds=spec.conds, fileName=str(f))
            if spec.cell_name: kwargs["cellName"] = spec.cell_name
            if spec.kwargs:    kwargs.update(spec.kwargs)
            cell_rule = netParams.importCellParams(**kwargs)  # capture returned rule

        # 2) verify import actually produced a rule
        if not cell_rule or spec.label not in netParams.cellParams:
            # extra hint: wrong cellName or incompatible HOC
            raise RuntimeError(
                f"Cell '{spec.label}' was not added to netParams.cellParams. "
                f"Check 'cellName' ('{spec.cell_name}') and the HOC template."
            )

        # 3) post-merge
        if spec.post_patch:
            _deep_merge(cell_rule, spec.post_patch)

        # 4) syn mechs
        if spec.syn_mechs:
            for name, mech in spec.syn_mechs.items():
                netParams.synMechParams[name] = mech

        # 5) weight norm
        thr = getattr(ctx.cfg, "weightNormThreshold", None)
        if spec.weight_norm_pkl and thr is not None:
            netParams.addCellParamsWeightNorm(spec.label, str(spec.weight_norm_pkl), threshold=thr)

        # 6) save rule
        if getattr(ctx.cfg, "saveCellParams", False) and spec.save_to_pkl:
            netParams.saveCellParamsRule(label=spec.label, fileName=str(spec.save_to_pkl))

        # 7) post-fn hook
        if spec.post_fn:
            spec.post_fn(netParams)

def _deep_merge(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v

