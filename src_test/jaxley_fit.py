from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Jaxley M1 surrogate with gradient descent.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--outdir", type=Path, default=Path("optimization") / "jaxley")
    parser.add_argument("--init-from", type=Path, default=None)
    return parser.parse_args()


def load_init_vector(path: Path, model) -> object:
    payload = json.loads(path.read_text())
    if "loss" in payload:
        payload = {k: v for k, v in payload.items() if k != "loss"}
    return model.unconstrained_from_bounded(model.dict_to_vector(payload))


def main() -> int:
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", args.device)

    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:
        raise ImportError(
            "The Jaxley training path requires `jax`, `jaxley`, and `optax`. "
            "Install them before running src_test/jaxley_fit.py."
        ) from exc

    from m1_model.jaxley_m1 import M1

    jax.config.update("jax_platform_name", args.device)

    model = M1()
    if model.defaults.add_in_vivo_thalamus:
        raise NotImplementedError("The Jaxley M1 path does not support addInVivoThalamus=True")

    if args.init_from is not None:
        params_unconstrained = load_init_vector(args.init_from, model)
    else:
        params_unconstrained = model.unconstrained_from_bounded(model.default_bounded_vector(), xp=jnp)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = optimizer.init(params_unconstrained)

    def loss_fn(unconstrained_vector):
        bounded = model.bounded_from_unconstrained(unconstrained_vector, xp=jnp)
        return jnp.asarray(model.loss(bounded, backend="jax"), dtype=jnp.float32)

    loss_and_grad = jax.value_and_grad(loss_fn)
    best_loss = float("inf")
    best_params = None
    best_step = -1
    stagnant_steps = 0
    history: list[dict[str, float]] = []

    for step in range(args.steps):
        loss_value, grads = loss_and_grad(params_unconstrained)
        updates, opt_state = optimizer.update(grads, opt_state, params_unconstrained)
        params_unconstrained = optax.apply_updates(params_unconstrained, updates)

        current_loss = float(loss_value)
        history.append({"step": float(step), "loss": current_loss})

        if current_loss + 1e-9 < best_loss:
            best_loss = current_loss
            best_params = jnp.asarray(model.bounded_from_unconstrained(params_unconstrained, xp=jnp))
            best_step = step
            stagnant_steps = 0
        else:
            stagnant_steps += 1
        if stagnant_steps >= 50:
            break

    if best_params is None:
        best_params = jnp.asarray(model.bounded_from_unconstrained(params_unconstrained, xp=jnp))

    final_params = model.bounded_from_unconstrained(params_unconstrained, xp=jnp)
    final_loss = float(model.loss(final_params, backend="jax"))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_params_dict = model.vector_to_dict(np.asarray(best_params))
    metrics = {
        "best_loss": best_loss,
        "final_loss": final_loss,
        "steps_completed": len(history),
        "seed": args.seed,
        "duration_ms": model.duration_ms,
        "best_step": best_step,
    }

    (run_dir / "best_params.json").write_text(json.dumps(best_params_dict, indent=2, sort_keys=True))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    payload = {**best_params_dict, "loss": best_loss}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
