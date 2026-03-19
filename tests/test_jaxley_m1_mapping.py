from __future__ import annotations

from m1_model.jaxley_data import build_feature_set, cortical_population_specs, long_range_populations


def test_population_to_bin_mapping_matches_expected_layers():
    specs = {spec.name: spec for spec in cortical_population_specs()}
    assert specs["NGF1"].layer_bin == 0
    assert specs["PV2"].layer_bin == 0
    assert specs["PV5A"].layer_bin == 1
    assert specs["SOM5B"].layer_bin == 1
    assert specs["PV6"].layer_bin == 2
    assert specs["NGF6"].layer_bin == 2


def test_family_grouping_matches_population_type():
    specs = {spec.name: spec for spec in cortical_population_specs()}
    assert specs["IT2"].family == "exc"
    assert specs["PT5B"].family == "exc"
    assert specs["CT6"].family == "exc"
    assert specs["PV2"].family == "inh"
    assert specs["VIP5B"].family == "inh"


def test_long_range_source_mapping_and_feature_keys_exist():
    feature_set = build_feature_set()
    for long_name in long_range_populations():
        key = f"weightLong.{long_name}"
        assert key in feature_set.external_features
        assert feature_set.external_features[key].shape == (len(feature_set.population_specs),)
