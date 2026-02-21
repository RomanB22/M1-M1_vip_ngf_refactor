"""
Utilities for deriving active populations and models from config/cells.yml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Iterable

import yaml


EXC_TYPES_ORDER = ["IT", "CT", "PT"]
INH_TYPES_ORDER = ["PV", "SOM", "VIP", "NGF"]


@dataclass(frozen=True)
class PopTemplate:
    name: str
    cell_type: str
    layer: str
    density_key: tuple[str, str]
    density_index: int
    density_factor: float = 1.0
    model_cfg_key: str | None = None
    fixed_model: str | None = None

    def resolve_cell_model(self, cellmod: Mapping[str, str]) -> str:
        if self.model_cfg_key:
            return str(cellmod[self.model_cfg_key])
        if self.fixed_model:
            return self.fixed_model
        raise KeyError(f"No model defined for population template {self.name}")


LOCAL_POP_TEMPLATES: tuple[PopTemplate, ...] = (
    PopTemplate("NGF1", "NGF", "1", ("M1", "nonVIP"), 0, fixed_model="HH_reduced"),
    PopTemplate("IT2", "IT", "2", ("M1", "E"), 1, model_cfg_key="IT2"),
    PopTemplate("SOM2", "SOM", "2", ("M1", "SOM"), 1, fixed_model="HH_reduced"),
    PopTemplate("PV2", "PV", "2", ("M1", "PV"), 1, fixed_model="HH_reduced"),
    PopTemplate("VIP2", "VIP", "2", ("M1", "VIP"), 1, fixed_model="HH_reduced"),
    PopTemplate("NGF2", "NGF", "2", ("M1", "nonVIP"), 1, fixed_model="HH_reduced"),
    PopTemplate("IT4", "IT", "4", ("M1", "E"), 2, model_cfg_key="IT4"),
    PopTemplate("SOM4", "SOM", "4", ("M1", "SOM"), 2, fixed_model="HH_reduced"),
    PopTemplate("PV4", "PV", "4", ("M1", "PV"), 2, fixed_model="HH_reduced"),
    PopTemplate("VIP4", "VIP", "4", ("M1", "VIP"), 2, fixed_model="HH_reduced"),
    PopTemplate("NGF4", "NGF", "4", ("M1", "nonVIP"), 2, fixed_model="HH_reduced"),
    PopTemplate("IT5A", "IT", "5A", ("M1", "E"), 3, model_cfg_key="IT5A"),
    PopTemplate("SOM5A", "SOM", "5A", ("M1", "SOM"), 3, fixed_model="HH_reduced"),
    PopTemplate("PV5A", "PV", "5A", ("M1", "PV"), 3, fixed_model="HH_reduced"),
    PopTemplate("VIP5A", "VIP", "5A", ("M1", "VIP"), 3, fixed_model="HH_reduced"),
    PopTemplate("NGF5A", "NGF", "5A", ("M1", "nonVIP"), 3, fixed_model="HH_reduced"),
    PopTemplate("IT5B", "IT", "5B", ("M1", "E"), 4, density_factor=0.5, model_cfg_key="IT5B"),
    PopTemplate("PT5B", "PT", "5B", ("M1", "E"), 4, density_factor=0.5, model_cfg_key="PT5B"),
    PopTemplate("SOM5B", "SOM", "5B", ("M1", "SOM"), 4, fixed_model="HH_reduced"),
    PopTemplate("PV5B", "PV", "5B", ("M1", "PV"), 4, fixed_model="HH_reduced"),
    PopTemplate("VIP5B", "VIP", "5B", ("M1", "VIP"), 4, fixed_model="HH_reduced"),
    PopTemplate("NGF5B", "NGF", "5B", ("M1", "nonVIP"), 4, fixed_model="HH_reduced"),
    PopTemplate("IT6", "IT", "6", ("M1", "E"), 5, density_factor=0.5, model_cfg_key="IT6"),
    PopTemplate("CT6", "CT", "6", ("M1", "E"), 5, density_factor=0.5, model_cfg_key="CT6"),
    PopTemplate("SOM6", "SOM", "6", ("M1", "SOM"), 5, fixed_model="HH_reduced"),
    PopTemplate("PV6", "PV", "6", ("M1", "PV"), 5, fixed_model="HH_reduced"),
    PopTemplate("VIP6", "VIP", "6", ("M1", "VIP"), 1, fixed_model="HH_reduced"),
    PopTemplate("NGF6", "NGF", "6", ("M1", "nonVIP"), 1, fixed_model="HH_reduced"),
)


_DEFAULT_INH_LAYERS = ("2", "4", "5A", "5B", "6")


LABEL_TO_SIGNATURES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "PV_reduced": tuple(("PV", "HH_reduced", layer) for layer in _DEFAULT_INH_LAYERS),
    "SOM_reduced": tuple(("SOM", "HH_reduced", layer) for layer in _DEFAULT_INH_LAYERS),
    "VIP_reduced": tuple(("VIP", "HH_reduced", layer) for layer in _DEFAULT_INH_LAYERS),
    "NGF_reduced": (
        ("NGF", "HH_reduced", "1"),
        *(("NGF", "HH_reduced", layer) for layer in _DEFAULT_INH_LAYERS),
    ),
    "IT2_reduced": (("IT", "HH_reduced", "2"),),
    "IT4_reduced": (("IT", "HH_reduced", "4"),),
    "IT5A_reduced": (("IT", "HH_reduced", "5A"),),
    "IT5B_reduced": (("IT", "HH_reduced", "5B"),),
    "PT5B_reduced": (("PT", "HH_reduced", "5B"),),
    "IT6_reduced": (("IT", "HH_reduced", "6"),),
    "CT6_reduced": (("CT", "HH_reduced", "6"),),
    "IT5A_full": (("IT", "HH_full", "5A"),),
    "PT5B_full": (("PT", "HH_full", "5B"),),
}


def load_cells_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r") as file_obj:
        return yaml.safe_load(file_obj) or {}


def enabled_labels_from_config(cell_cfg: Mapping[str, object] | None) -> set[str]:
    labels = (cell_cfg or {}).get("enabled_cells")
    if not labels:
        return set()
    return {str(label) for label in labels}


def resolve_cellmod(
    base_cellmod: Mapping[str, str],
    enabled_labels: set[str],
    cell_cfg: Mapping[str, object] | None = None,
) -> dict[str, str]:
    resolved = {k: str(v) for k, v in dict(base_cellmod).items()}

    overrides = (cell_cfg or {}).get("population_models")
    if isinstance(overrides, dict):
        for pop_name, model_name in overrides.items():
            pop_key = str(pop_name)
            model = str(model_name)
            if pop_key in resolved and model in {"HH_reduced", "HH_full"}:
                resolved[pop_key] = model

    if enabled_labels:
        if "IT5A_full" in enabled_labels and "IT5A_reduced" not in enabled_labels:
            resolved["IT5A"] = "HH_full"
        if "IT5A_reduced" in enabled_labels and "IT5A_full" not in enabled_labels:
            resolved["IT5A"] = "HH_reduced"

        if "PT5B_full" in enabled_labels and "PT5B_reduced" not in enabled_labels:
            resolved["PT5B"] = "HH_full"
        if "PT5B_reduced" in enabled_labels and "PT5B_full" not in enabled_labels:
            resolved["PT5B"] = "HH_reduced"

    return resolved


def signatures_for_enabled_labels(enabled_labels: set[str]) -> set[tuple[str, str, str]]:
    signatures: set[tuple[str, str, str]] = set()
    for label in enabled_labels:
        signatures.update(LABEL_TO_SIGNATURES.get(label, ()))
    return signatures


def resolve_enabled_populations(enabled_labels: set[str], cellmod: Mapping[str, str]) -> list[str]:
    if not enabled_labels:
        return [template.name for template in LOCAL_POP_TEMPLATES]

    enabled_signatures = signatures_for_enabled_labels(enabled_labels)
    if not enabled_signatures:
        return []

    selected: list[str] = []
    for template in LOCAL_POP_TEMPLATES:
        signature = (template.cell_type, template.resolve_cell_model(cellmod), template.layer)
        if signature in enabled_signatures:
            selected.append(template.name)
    return selected


def build_local_population_specs(
    cfg,
    density: Mapping[tuple[str, str], list[float]],
    enabled_labels: set[str],
) -> dict[str, dict]:
    enabled_pops = set(resolve_enabled_populations(enabled_labels, cfg.cellmod))
    include_all = not enabled_labels

    pop_params: dict[str, dict] = {}
    for template in LOCAL_POP_TEMPLATES:
        if not include_all and template.name not in enabled_pops:
            continue

        cell_model = template.resolve_cell_model(cfg.cellmod)
        density_value = density[template.density_key][template.density_index] * template.density_factor

        pop_params[template.name] = {
            "cellModel": cell_model,
            "cellType": template.cell_type,
            "ynormRange": cfg.layer[template.layer],
            "density": density_value,
        }

    return pop_params


def active_cell_types_from_pop_params(pop_params: Mapping[str, Mapping[str, object]]) -> tuple[list[str], list[str]]:
    local_types = {
        str(spec.get("cellType"))
        for spec in pop_params.values()
        if isinstance(spec, Mapping) and isinstance(spec.get("cellType"), str)
    }

    exc_types = [cell_type for cell_type in EXC_TYPES_ORDER if cell_type in local_types]
    inh_types = [cell_type for cell_type in INH_TYPES_ORDER if cell_type in local_types]
    return exc_types, inh_types


def filter_existing_pops(candidates: Iterable[str], available_pops: set[str]) -> list[str]:
    return [candidate for candidate in candidates if candidate in available_pops]


def first_available(candidates: Iterable[str], available_pops: set[str], fallback: str = "None") -> str:
    for candidate in candidates:
        if candidate in available_pops:
            return candidate
    return fallback
