# m1_model/utils/param_patch.py
import json, os, time, random
from typing import Iterable, Mapping, Any, Dict

# ---- Optional POSIX file locking (works on Linux/NFSv4). If not available, we still retry. ----
try:
    import fcntl  # type: ignore
    _HAVE_FCNTL = True
except Exception:
    _HAVE_FCNTL = False

# Per-process cache so we don't hammer the filesystem from each worker
_PARAM_CACHE: Dict[str, Dict[str, float]] = {}  # path -> params
_PARAM_CACHE_MTIME: Dict[str, float] = {}       # path -> last seen mtime

import tempfile

def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX & Windows (Python 3.3+)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass

def _robust_load_json(path: str,
                      attempts: int = 10,
                      base_sleep: float = 0.025) -> Dict[str, float]:
    """
    Robustly load JSON from `path` with:
      - optional shared lock (POSIX)
      - retry with jitter on empty/partial reads
    Returns a dict[str, float] (your params look numeric).
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            # quick guard: zero-length files on distributed FS can appear transiently
            try:
                size = os.path.getsize(path)
                if size == 0:
                    raise ValueError("zero-length file")
            except FileNotFoundError:
                # propagate to outer except to retry
                raise

            with open(path, "r") as f:
                if _HAVE_FCNTL:
                    # Shared lock for reading
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)

                data = f.read()
                if _HAVE_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            if not data.strip():
                raise ValueError("empty read")

            params = json.loads(data)

            # Coerce values to float once here (keeps rest of code simple)
            out: Dict[str, float] = {}
            for k, v in params.items():
                # allow numeric strings / ints
                out[k] = float(v)
            return out

        except Exception as e:
            last_exc = e
            # exponential backoff with jitter
            sleep = base_sleep * (2 ** (i - 1))
            sleep *= 0.5 + random.random()  # jitter in [0.5x, 1.5x]
            # cap sleep so we don't stall too long
            time.sleep(min(sleep, 0.3))

    # if we’re here, all attempts failed
    raise RuntimeError(f"Failed to load JSON from {path} after {attempts} attempts: {last_exc}") from last_exc


def _get_params_cached(path: str) -> Dict[str, float]:
    """
    Load params with a small per-process cache that invalidates
    if the file's mtime changes (so edits are picked up).
    """
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Param file not found: {path}") from e

    if (path not in _PARAM_CACHE) or (_PARAM_CACHE_MTIME.get(path) != mtime):
        params = _robust_load_json(path)
        _PARAM_CACHE[path] = params
        _PARAM_CACHE_MTIME[path] = mtime
    return _PARAM_CACHE[path]


def apply_na_paramfile_to_rule(rule: Mapping[str, Any],
                               param_path: str,
                               target_mechs: Iterable[str] = ("na12","na12mut","na16","na16mut")) -> int:
    """
    Read the (JSON) param file and assign any matching keys into the listed Na mechs.
    Returns the number of assignments made.
    """
    params = _get_params_cached(param_path)

    n_changes = 0
    for sec in rule.get("secs", {}).values():
        mechs = sec.get("mechs", {})
        for mname in target_mechs:
            m = mechs.get(mname)
            if not isinstance(m, dict):
                continue
            for k, v in params.items():
                if k in m:
                    if k == "gbar" and isinstance(m[k], list):
                        m[k] = [float(v)] * len(m[k])
                    else:
                        m[k] = float(v)
                    n_changes += 1
    return n_changes
