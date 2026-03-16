NEURON {
    POINT_PROCESS SpikeGuard
    RANGE detector_v
    RANGE candidateStartMv, minPeakMv, minProminenceMv, minDvdtMvPerMs
    RANGE refractoryMs, plateauMv
    RANGE accepted_count, rejected_count, plateau_max_ms
}

UNITS {
    (mV) = (millivolt)
}

PARAMETER {
    candidateStartMv = -20 (mV)
    minPeakMv = 10 (mV)
    minProminenceMv = 20 (mV)
    minDvdtMvPerMs = 10 (mV/ms)
    refractoryMs = 2 (ms)
    plateauMv = -40 (mV)
}

ASSIGNED {
    v (mV)
    dt (ms)
    detector_v
    dvdt (mV/ms)
    last_v (mV)
    last_dvdt (mV/ms)
    recent_trough (mV)
    candidate_trough (mV)
    candidate_peak (mV)
    candidate_max_dvdt (mV/ms)
    in_candidate
    last_accept_t (ms)
    plateau_run_ms (ms)
    plateau_max_ms (ms)
    accepted_count
    rejected_count
}

INITIAL {
    detector_v = 0
    dvdt = 0
    last_v = v
    last_dvdt = 0
    recent_trough = v
    candidate_trough = v
    candidate_peak = v
    candidate_max_dvdt = 0
    in_candidate = 0
    last_accept_t = -1e9
    plateau_run_ms = 0
    plateau_max_ms = 0
    accepted_count = 0
    rejected_count = 0
}

BREAKPOINT {
    detector_v = 0
    if (dt <= 0) {
        dvdt = 0
    } else {
        dvdt = (v - last_v) / dt
    }

    if (v >= plateauMv) {
        plateau_run_ms = plateau_run_ms + dt
        if (plateau_run_ms > plateau_max_ms) {
            plateau_max_ms = plateau_run_ms
        }
    } else {
        plateau_run_ms = 0
    }

    if (in_candidate > 0.5) {
        if (v > candidate_peak) {
            candidate_peak = v
        }
        if (dvdt > candidate_max_dvdt) {
            candidate_max_dvdt = dvdt
        }

        if (dvdt <= 0) {
            if ((t - last_accept_t) >= refractoryMs &&
                candidate_peak >= minPeakMv &&
                (candidate_peak - candidate_trough) >= minProminenceMv &&
                candidate_max_dvdt >= minDvdtMvPerMs) {
                detector_v = 1
                accepted_count = accepted_count + 1
                last_accept_t = t
            } else {
                rejected_count = rejected_count + 1
            }
            in_candidate = 0
            recent_trough = v
        }
    } else {
        if (dvdt <= 0) {
            recent_trough = v
        } else if (v < recent_trough) {
            recent_trough = v
        }

        if (dvdt > 0 &&
            ((v >= candidateStartMv && last_v < candidateStartMv) ||
             (last_v >= candidateStartMv && last_dvdt <= 0))) {
            in_candidate = 1
            candidate_trough = recent_trough
            candidate_peak = v
            candidate_max_dvdt = dvdt
        }
    }

    last_v = v
    last_dvdt = dvdt
}
