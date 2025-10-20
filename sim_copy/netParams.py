
from pathlib import Path
import yaml

# NetPyNE import occurs in your actual runtime environment
from netpyne import specs

from m1_model.cells.registry import get_enabled_cells
from m1_model.adapters.netpyne_import import add_cells_via_import

# Your existing cfg module/object should already exist in this repo
try:
    from cfg import cfg  # use your project's cfg
except Exception:
    class _Cfg:
        weightNormThreshold = 1.0
        saveCellParams = False
        loadCellParams = []
        pv_scale_pas_g = None
    cfg = _Cfg()

netParams = specs.NetParams()

class Ctx:
    sim_dir = Path(__file__).resolve().parent              # sim/
    project_root = Path(__file__).resolve().parents[1]     # <project>/
    cfg = cfg
    loadCellParams = set(getattr(cfg, "loadCellParams", []) or [])

cfg_path = Path(__file__).resolve().parents[1] / "config" / "cells.yml"
cell_cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}

ctx = Ctx()
cells = list(get_enabled_cells(cell_cfg or {}, ctx))
add_cells_via_import(netParams, cells, ctx)

print("Providers:", [p.import_spec(None).label for p in cells])  # did we build PV_simple?
print("Before:", netParams.cellParams.keys())
# after add_cells_via_import:
print("After:", netParams.cellParams.keys())
print("PV_reduced rule present:", "PV_reduced" in netParams.cellParams)

# ... continue with pops/conns/simConfig as in your project ...
