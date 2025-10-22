# m1_model/utils/mutations.py
import re
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

Op = Literal["set", "scale", "add"]

@dataclass(frozen=True)
class Mutation:
    label: str                     # cell label in netParams.cellParams
    mech: str                      # e.g. "pas", "nax", "ih"
    param: str                     # e.g. "g", "gbar", "e"
    op: Op                         # "set" | "scale" | "add"
    value: float                   # number to set/scale/add
    sections: Optional[Union[Iterable[str], str]] = None
    on_missing: Literal["skip","create","error"] = "skip"
    init_value: Optional[float] = None  # used when on_missing="create"
    only_if_present: Optional[Dict[str, Any]] = None   # optional guard, e.g. {"mech": "nax"}

def _select_sections(rule: Dict[str, Any], sel: Optional[Union[Iterable[str], str]]) -> List[str]:
    secs = rule.get("secs", {})
    if sel is None or (isinstance(sel, str) and sel.upper() == "ALL"):
        return list(secs.keys())
    if isinstance(sel, str) and sel.startswith("regex:"):
        pat = re.compile(sel.split("regex:", 1)[1])
        return [s for s in secs if pat.search(s)]
    if isinstance(sel, str) and sel == "*":
        return list(secs.keys())
    return [s for s in sel if s in secs]

def _apply_op(val: Any, op: Op, x: float):
    if isinstance(val, Number):
        if op == "set":   return float(x)
        if op == "scale": return float(val) * x
        if op == "add":   return float(val) + x
    elif isinstance(val, (list, tuple)):
        out = []
        for v in val:
            if isinstance(v, Number):
                if op == "set":   out.append(float(x))
                if op == "scale": out.append(float(v) * x)
                if op == "add":   out.append(float(v) + x)
            else:
                out.append(v)
        return type(val)(out)
    return val  # unmodified for unsupported types

def apply_mutations(netParams, mutations: Iterable[Mutation], dry_run: bool=False) -> Dict[str, Any]:
    """
    Apply a list of mutations to netParams.cellParams.
    Returns a report dict. If dry_run=True, nothing is changed.
    """
    report = {"mutations": [], "errors": []}

    for m in mutations:
        try:
            rule = netParams.cellParams[m.label]
        except KeyError:
            report["errors"].append(f"[{m.label}] rule not found")
            continue

        targets = _select_sections(rule, m.sections)
        changed, skipped, created = {}, [], []

        for sname in targets:
            sec = rule["secs"][sname]
            mechs = sec.setdefault("mechs", {})
            if m.mech not in mechs or m.param not in mechs.get(m.mech, {}):
                if m.on_missing == "skip":
                    skipped.append(sname); continue
                if m.on_missing == "error":
                    report["errors"].append(f"[{m.label}.{sname}] {m.mech}.{m.param} missing"); continue
                if m.on_missing == "create":
                    if m.init_value is None:
                        report["errors"].append(f"[{m.label}.{sname}] init_value required to create {m.mech}.{m.param}")
                        continue
                    mechs.setdefault(m.mech, {})[m.param] = m.init_value
                    created.append(sname)

            # optional guard
            if m.only_if_present:
                guard_mech = m.only_if_present.get("mech")
                if guard_mech and guard_mech not in mechs:
                    skipped.append(sname); continue

            old = mechs[m.mech][m.param]
            new = _apply_op(old, m.op, m.value)
            changed[sname] = {"old": old, "new": new}

            if not dry_run:
                mechs[m.mech][m.param] = new

        report["mutations"].append({
            "label": m.label, "mech": m.mech, "param": m.param, "op": m.op, "value": m.value,
            "sections": list(targets), "changed": changed, "created": created, "skipped": skipped
        })

    return report