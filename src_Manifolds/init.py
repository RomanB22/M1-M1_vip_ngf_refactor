"""Run one NetPyNE simulation and return its calibration objectives to BatchTK."""

import json
import os

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

from netpyne import sim

from calibration.objective_pipeline import (
    evaluate_objectives,
    validate_instantiated_objectives,
    validate_objective_setup,
)
from netParams import cfg, netParams
from simulation.spike_guard import (
    blockade_penalty,
    guard_summary_for_results,
    record_local_spike_guard,
    summarize_simulation_spike_guard,
)


def main() -> None:
    validate_objective_setup(cfg, strict_files=True)

    sim.initialize(simConfig=cfg, netParams=netParams)
    sim.net.createPops()
    sim.net.createCells()
    validate_instantiated_objectives(sim, cfg)
    sim.net.connectCells()
    sim.net.addStims()
    sim.setupRecording()
    sim.runSim()

    record_local_spike_guard(sim, cfg)
    sim.gatherData()

    if sim.rank == 0:
        guard_summary = summarize_simulation_spike_guard(sim, cfg)
        results = evaluate_objectives(
            sim,
            cfg,
            guard_summary=guard_summary,
            guard_penalty=(
                blockade_penalty(guard_summary, cfg.spikeGuard)
                if guard_summary is not None
                else 0.0
            ),
        )
        if guard_summary is not None:
            results["spike_guard"] = guard_summary_for_results(guard_summary)

        payload = {**cfg.get_mappings(), **results}
        print(
            "objective results:",
            {name: payload[name] for name in cfg.objectives if name in payload},
        )
        sim.send(json.dumps(payload))

    if cfg.plotSimResults:
        sim.saveData()
        sim.analysis.plotData()


if __name__ == "__main__":
    main()
