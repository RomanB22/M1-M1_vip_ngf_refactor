# m1_model/adapters/netpyne_import.py
from typing import Iterable
from m1_model.cells.base import CellProvider, ImportSpec
from pathlib import Path
import json

def _wrap_if_plain_dict(netParams, label: str):
    """If netParams.cellParams[label] is a plain dict, re-add it via addCellParams to get the wrapped rule."""
    rule = netParams.cellParams.get(label)
    if isinstance(rule, dict):
        # remove the plain dict and re-add (wrapped)
        del netParams.cellParams[label]
        netParams.addCellParams(label, rule)
    return netParams.cellParams.get(label)

def add_cells_via_import(netParams, cells: Iterable[CellProvider], ctx) -> None:
    load_labels = set(getattr(ctx, "loadCellParams", []) or [])

    for provider in cells:
        spec: ImportSpec = provider.import_spec(ctx)

        # Decide whether we're loading a pre-saved rule or importing from source
        want_load = (spec.label in load_labels) and bool(spec.load_from_pkl)
        load_path = Path(spec.load_from_pkl) if spec.load_from_pkl else None

        cell_rule = None

        try:
            if want_load and load_path and load_path.exists():
                # ---------- Load path ----------
                if load_path.suffix.lower() == ".json":
                    # Read JSON manually, then WRAP via addCellParams so NetPyNE helpers work
                    with load_path.open() as f:
                        rule_dict = json.load(f)
                    # Ensure conds is present (JSON from NetPyNE usually has it already)
                    if spec.conds:
                        rule_dict.setdefault("conds", spec.conds)
                    netParams.addCellParams(spec.label, rule_dict)
                    cell_rule = netParams.cellParams[spec.label]
                else:
                    # Let NetPyNE load (e.g., PKL). Some versions store a dict—wrap if needed.
                    netParams.loadCellParamsRule(label=spec.label, fileName=str(load_path))
                    cell_rule = _wrap_if_plain_dict(netParams, spec.label)
            else:
                # ---------- Import path ----------
                # Only check source file when we actually import
                if not spec.file:
                    raise FileNotFoundError(
                        f"Cell '{spec.label}': no source file provided and no loadable artifact found."
                    )
                f = Path(spec.file)
                if not f.exists():
                    raise FileNotFoundError(f"Cell '{spec.label}': file not found: {f}")

                kwargs = dict(label=spec.label, conds=spec.conds or {}, fileName=str(f))
                if spec.cell_name:
                    kwargs["cellName"] = spec.cell_name
                if spec.kwargs:
                    kwargs.update(spec.kwargs)

                cell_rule = netParams.importCellParams(**kwargs)

                # Persist rule if requested
                if getattr(ctx.cfg, "saveCellParams", False) and spec.save_to_pkl:
                    netParams.saveCellParamsRule(label=spec.label, fileName=str(spec.save_to_pkl))

            # ---------- Verify we actually have a rule ----------
            if not cell_rule or spec.label not in netParams.cellParams:
                raise RuntimeError(
                    f"Cell '{spec.label}' was not added to netParams.cellParams. "
                    f"Check 'cellName' ('{spec.cell_name}') and the source/template."
                )

            # ---------- post_patch (deep merge) ----------
            if spec.post_patch:
                _deep_merge(netParams.cellParams[spec.label], spec.post_patch)

            # ---------- syn mechs ----------
            if spec.syn_mechs:
                for name, mech in spec.syn_mechs.items():
                    netParams.synMechParams[name] = mech

            # ---------- weight norm ----------
            thr = getattr(ctx.cfg, "weightNormThreshold", None)
            if spec.weight_norm_pkl and thr is not None:
                netParams.addCellParamsWeightNorm(spec.label, str(spec.weight_norm_pkl), threshold=thr)

            # ---------- post-fn hook ----------
            if spec.post_fn:
                spec.post_fn(netParams)

        except Exception as e:
            # Add cell label context to any error
            raise type(e)(f"[{spec.label}] {e}") from e


def _deep_merge(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v

