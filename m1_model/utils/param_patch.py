# m1_model/utils/param_patch.py
import json
from typing import Iterable, Mapping, Any

def apply_na_paramfile_to_rule(rule: Mapping[str, Any],
                               param_path: str,
                               target_mechs: Iterable[str] = ("na12","na12mut","na16","na16mut")) -> int:
    """
    Read the (JSON) param file and assign any matching keys into the listed Na mechs.
    Returns the number of assignments made.
    """
    with open(param_path) as f:
        params = json.load(f)

    n_changes = 0
    for sec in rule.get("secs", {}).values():
        mechs = sec.get("mechs", {})
        for mname in target_mechs:
            m = mechs.get(mname)
            if not isinstance(m, dict):
                continue
            for k, v in params.items():
                if k in m:
                    # handle scalar vs list gbar
                    if k == "gbar" and isinstance(m[k], list):
                        m[k] = [float(v)] * len(m[k])
                    else:
                        m[k] = float(v)
                    n_changes += 1
    return n_changes
