def get_batch_params(minChg, maxChg):
    return {
        'weightLong.TPO': (0.1 * minChg, 0.8 * maxChg),
        'weightLong.TVL': (0.1 * minChg, 0.8 * maxChg),
        'weightLong.S1': (0.1 * minChg, 0.8 * maxChg),
        'weightLong.S2': (0.1 * minChg, 0.8 * maxChg),
        'weightLong.cM1': (0.1 * minChg, 0.8 * maxChg),
        'weightLong.M2': (0.1 * minChg, 0.8 * maxChg),
        'weightLong.OC': (0.1 * minChg, 0.8 * maxChg),
        'EEGain': (0.5 * minChg, 4.0 * maxChg),
        'IEweights.0': (0.5 * minChg, 2.0 * maxChg),  # L2/3+4
        'IEweights.1': (0.5 * minChg, 2.0 * maxChg),  # L5
        'IEweights.2': (0.5 * minChg, 2.0 * maxChg),  # L6
        'IIweights.0': (0.5 * minChg, 2.0 * maxChg),  # L2/3+4
        'IIweights.1': (0.5 * minChg, 2.0 * maxChg),  # L5
        'IIweights.2': (0.5 * minChg, 2.0 * maxChg),  # L6
        'EICellTypeGain.PV': (1.0 * minChg, 4.0 * maxChg),
        'EICellTypeGain.SOM': (1.0 * minChg, 4.0 * maxChg),
        'EICellTypeGain.VIP': (1.0 * minChg, 4.0 * maxChg),
        'EICellTypeGain.NGF': (1.0 * minChg, 4.0 * maxChg),
        # 'scaleDensity': (0.15)
    }
