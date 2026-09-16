"""Build optional mutation recipes selected by flags in ``cfg.py``."""

from __future__ import annotations

from typing import Any


def build_mutation_list(cfg: Any) -> list[dict[str, Any]]:
    """Return custom mutations plus the enabled named presets."""

    mutations = list(getattr(cfg, "mutations", []))

    if bool(getattr(cfg, "heterozygous", False)):
        mutations.append(
            {
                "label": "PT5B_full",
                "mech": "na12mut",
                "param": "gbar",
                "op": "set",
                "value": 0.0,
                "sections": "ALL",
                "only_if_present": {"mech": "na12mut"},
            }
        )

    if bool(getattr(cfg, "blockNa", False)):
        for mechanism in ("na12", "na12mut", "nax"):
            mutations.append(
                {
                    "label": "PT5B_full",
                    "mech": mechanism,
                    "param": "gbar",
                    "op": "set",
                    "value": 0.0,
                    "sections": "ALL",
                    "only_if_present": {"mech": mechanism},
                }
            )

    if bool(getattr(cfg, "KCNT1", False)):
        mutations.extend(
            [
                {
                    "label": "PT5B_full",
                    "mech": "kBK",
                    "param": "gpeak",
                    "op": "scale",
                    "value": 2.0,
                    "sections": "ALL",
                    "only_if_present": {"mech": "kBK"},
                },
                {
                    "label": "PT5B_full",
                    "mech": "pas",
                    "param": "g",
                    "op": "scale",
                    "value": 1.86,
                    "sections": "ALL",
                    "only_if_present": {"mech": "pas"},
                },
                {
                    "label": "PT5B_full",
                    "mech": "hd",
                    "param": "gbar",
                    "op": "scale",
                    "value": 0.15,
                    "sections": "ALL",
                    "only_if_present": {"mech": "hd"},
                },
                {
                    "label": "PV_reduced",
                    "mech": "IKsin",
                    "param": "gKsbar",
                    "op": "scale",
                    "value": 3.0,
                    "sections": "ALL",
                    "only_if_present": {"mech": "IKsin"},
                },
            ]
        )

    return mutations


__all__ = ["build_mutation_list"]
