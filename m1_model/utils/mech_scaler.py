
import re
from numbers import Number
from typing import Iterable, Optional, Union, Dict, Any

def scale_cell_mech(
    netParams,
    cell_label: str,
    mech: str,
    param: str,
    factor: float,
    sections: Optional[Union[Iterable[str], str]] = None,
    on_missing: str = "skip",
    init_value: Optional[float] = None,
) -> Dict[str, Any]:
    if factor is None or not isinstance(factor, (int, float)):
        raise ValueError("factor must be a number")
    rule = netParams.cellParams[cell_label]
    secs = rule.get("secs", {})
    if isinstance(sections, str) and sections.upper() == "ALL":
        sections = None

    # Build target section list
    if sections is None:
        target_names = list(secs.keys())
    elif isinstance(sections, str) and sections.startswith("regex:"):
        pat = re.compile(sections.split("regex:", 1)[1])
        target_names = [s for s in secs if pat.search(s)]
    else:
        target_names = [s for s in sections if s in secs]

    updated = {}
    skipped = []
    created = []

    for sname in target_names:
        s = secs[sname]
        mechs = s.setdefault("mechs", {})
        if mech not in mechs or param not in mechs.get(mech, {}):
            if on_missing == "skip":
                skipped.append(sname)
                continue
            elif on_missing == "error":
                raise KeyError(f"{cell_label}.{sname}.mechs.{mech}.{param} missing")
            elif on_missing == "create":
                if init_value is None:
                    raise ValueError("init_value must be provided when on_missing='create'")
                mechs.setdefault(mech, {})[param] = init_value
                created.append(sname)

        val = mechs[mech][param]
        if isinstance(val, Number):
            old = float(val)
            new = old * factor
            mechs[mech][param] = new
            updated[sname] = {"old": old, "new": new}
        elif isinstance(val, (list, tuple)):
            old = list(val)
            scaled = []
            for x in val:
                if isinstance(x, Number):
                    scaled.append(float(x) * factor)
                else:
                    scaled.append(x)
            mechs[mech][param] = type(val)(scaled)
            updated[sname] = {"old": old, "new": scaled}
        else:
            skipped.append(sname)

    return {
        "cell": cell_label,
        "mech": mech,
        "param": param,
        "factor": factor,
        "updated": updated,
        "created": created,
        "skipped": skipped,
    }

def make_scale_postfn(
    cell_label: str,
    mech: str,
    param: str,
    factor: float,
    sections: Optional[Union[Iterable[str], str]] = None,
    on_missing: str = "skip",
    init_value: Optional[float] = None,
):
    def _post(netParams):
        scale_cell_mech(
            netParams,
            cell_label=cell_label,
            mech=mech,
            param=param,
            factor=factor,
            sections=sections,
            on_missing=on_missing,
            init_value=init_value,
        )
    return _post
