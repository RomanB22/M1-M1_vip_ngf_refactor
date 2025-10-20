
from typing import Iterable
from m1_model.cells.base import CellProvider, ImportSpec

def add_cells_via_import(netParams, cells: Iterable[CellProvider], ctx) -> None:
    load_labels = set(getattr(ctx, "loadCellParams", []) or [])

    for provider in cells:
        spec: ImportSpec = provider.import_spec(ctx)

        # load from pkl if requested
        if spec.label in load_labels and spec.load_from_pkl:
            netParams.loadCellParamsRule(label=spec.label, fileName=str(spec.load_from_pkl))
        else:
            kwargs = dict(
                label=spec.label,
                conds=spec.conds,
                fileName=str(spec.file),
            )
            if spec.cell_name:
                kwargs["cellName"] = spec.cell_name
            if spec.kwargs:
                kwargs.update(spec.kwargs)
            netParams.importCellParams(**kwargs)

        # post-merge
        if spec.post_patch:
            _deep_merge(netParams.cellParams[spec.label], spec.post_patch)

        # syn mechs
        if spec.syn_mechs:
            for name, mech in spec.syn_mechs.items():
                netParams.synMechParams[name] = mech

        # weight norm
        if spec.weight_norm_pkl:
            thr = getattr(ctx.cfg, "weightNormThreshold", None)
            if thr is not None:
                netParams.addCellParamsWeightNorm(spec.label, str(spec.weight_norm_pkl), threshold=thr)

        # save rule
        if getattr(ctx.cfg, "saveCellParams", False) and spec.save_to_pkl:
            netParams.saveCellParamsRule(label=spec.label, fileName=str(spec.save_to_pkl))

        # post-fn hook
        if spec.post_fn:
            spec.post_fn(netParams)

def _deep_merge(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
