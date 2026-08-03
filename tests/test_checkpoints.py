import numpy as np

from itw_rd.generators import make_generic_channel
from itw_rd.infonce import train_tabular_infonce, train_tabular_infonce_checkpoints


def test_checkpoint_training_matches_single_final_run():
    problem = make_generic_channel(n=8, seed=12, matched=False)
    kwargs = dict(K=4, batch_size=64, learning_rate=0.03, seed=99)
    snapshots = train_tabular_infonce_checkpoints(
        problem.p_x,
        problem.channel,
        checkpoint_steps=(10, 25, 40),
        **kwargs,
    )
    final = train_tabular_infonce(
        problem.p_x,
        problem.channel,
        steps=40,
        **kwargs,
    )
    assert set(snapshots) == {10, 25, 40}
    assert snapshots[10].positive_pairs_seen == 10 * 64
    assert snapshots[25].positive_pairs_seen == 25 * 64
    assert np.allclose(snapshots[40].score, final.score)
