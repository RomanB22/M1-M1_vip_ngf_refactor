from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

Number = Union[int, float]

# ---------------------------- helpers ----------------------------

def _apply_op(old: Any, op: str, value: Number) -> Any:
    """Apply set/scale/add to scalars or lists; ignore None."""
    if old is None:
        return old
    if isinstance(old, (int, float)):
        return value if op == "set" else (old * value if op == "scale" else old + value)
    if isinstance(old, list):
        if op == "set":
            return [value for _ in old]
        if op == "scale":
            return [x * value for x in old]
        if op == "add":
            return [x + value for x in old]
    return old  # leave dicts/other types unchanged


def _ensure_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _matches_celltype(rule: Mapping[str, Any], targets: Sequence[str]) -> bool:
    ct = (rule.get("conds") or {}).get("cellType")
    return ct in targets


# ----------------- intrinsic (per-cell mechanism) drugs -----------------

def apply_cell_mech_drugs(
    netParams,
    treatments: Iterable[Mapping[str, Any]],
    *,
    dry_run: bool = False,
) -> List[str]:
    """
    Each treatment row (e.g. from Excel) is a dict with keys:
      - cell_types: list|str         # e.g. ["PV","SOM","VIP"]
      - mech: str                    # e.g. "hd", "nax", "na12", "catcb"
      - param: str                   # e.g. "gbar", "gmax", "vhalfl"
      - op: str                      # one of {"set","scale","add"}
      - value: number
      - sections: list|str|"ALL"     # optional; default "ALL"
    Returns a list of human-readable changes (or would-change if dry_run).
    """
    log: List[str] = []
    for t in treatments:
        targets_ct = _ensure_list(t.get("cell_types"))
        mech = t["mech"]
        param = t["param"]
        op = t.get("op", "set")
        value: Number = t["value"]
        sections = t.get("sections", "ALL")
        sections = None if sections == "ALL" else _ensure_list(sections)

        for label, rule in netParams.cellParams.items():
            if targets_ct and not _matches_celltype(rule, targets_ct):
                continue

            secs = rule.get("secs", {})
            for sec_name, sec in secs.items():
                if sections is not None and sec_name not in sections:
                    continue
                mechs = sec.get("mechs", {})
                m = mechs.get(mech)
                if not isinstance(m, dict):
                    continue
                if param not in m:
                    continue

                before = m[param]
                after = _apply_op(before, op, value)
                if before == after:
                    continue

                if not dry_run:
                    m[param] = after
                log.append(f"[{label}:{sec_name}:{mech}.{param}] {op} {value} | {before} -> {after}")
    return log


# ----------------- synaptic (global synMechParams) drugs -----------------

def apply_syn_drugs(
    netParams,
    treatments: Iterable[Mapping[str, Any]],
    *,
    dry_run: bool = False,
) -> List[str]:
    """
    Each syn-treatment row (e.g. from Excel) is a dict with keys:
      - syn_mechs: list|str                  # e.g. ["AMPA","NMDA","GABAA","GABAB", ...]
      - param: str                           # e.g. "tau1","tau2","e"
      - op: str                              # {"set","scale","add"}
      - value: number
    NOTE: This updates netParams.synMechParams *definitions* (global).
    """
    log: List[str] = []
    for t in treatments:
        syn_mechs = _ensure_list(t.get("syn_mechs"))
        param = t["param"]
        op = t.get("op", "set")
        value: Number = t["value"]

        for name in syn_mechs:
            mech_def = netParams.synMechParams.get(name)
            if not isinstance(mech_def, dict):
                continue
            if param not in mech_def:
                continue

            before = mech_def[param]
            after = _apply_op(before, op, value)
            if before == after:
                continue

            if not dry_run:
                mech_def[param] = after
            log.append(f"[syn:{name}.{param}] {op} {value} | {before} -> {after}")
    return log
