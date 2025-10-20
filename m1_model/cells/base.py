
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Protocol, Literal, Callable

ImportKind = Literal["hoc", "python", "swc", "json"]

@dataclass(frozen=True)
class ImportSpec:
    label: str
    conds: Dict[str, Any]
    kind: ImportKind
    file: Path
    cell_name: Optional[str] = None
    kwargs: Dict[str, Any] = None            # e.g., {"cellInstance": True}
    post_patch: Dict[str, Any] = None        # dict to deep-merge after import/load
    syn_mechs: Dict[str, Dict[str, Any]] = None
    weight_norm_pkl: Optional[Path] = None
    save_to_pkl: Optional[Path] = None
    load_from_pkl: Optional[Path] = None
    post_fn: Optional[Callable[["specs.NetParams"], None]] = None  # run after import/load

class CellProvider(Protocol):
    def import_spec(self, ctx) -> ImportSpec: ...
