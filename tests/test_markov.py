import numpy as np
from itw_rd.core import aggregate_joint, mutual_information
from itw_rd.generators import make_hierarchical_reversible_chain


def test_hierarchical_chain_stationary_reversible():
    c=make_hierarchical_reversible_chain(macros=3,micros_per_macro=2,states_per_micro=3,seed=4)
    assert np.max(np.abs(c.stationary@c.transition-c.stationary))<1e-10
    assert np.max(np.abs(c.joint-c.joint.T))<1e-10
    assert np.all(c.transition>0)


def test_label_aggregation_mass():
    c=make_hierarchical_reversible_chain(macros=2,micros_per_macro=2,states_per_micro=2,seed=2)
    J=aggregate_joint(c.joint,c.macro_labels)
    assert np.isclose(J.sum(),1) and mutual_information(J)>=0


def test_experiment2_seed_list_and_legacy_seed_config():
    from itw_rd.experiment2 import Experiment2Config

    cfg = Experiment2Config.from_dict({"seeds": [1, 3, 5]})
    assert cfg.seeds == (1, 3, 5)

    legacy = Experiment2Config.from_dict({"seed": 7})
    assert legacy.seeds == (7,)
