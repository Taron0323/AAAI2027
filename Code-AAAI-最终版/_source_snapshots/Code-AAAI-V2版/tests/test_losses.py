import numpy as np

from foreact.training.losses import forward_kl_stopgrad


def test_forward_kl_direction_penalizes_wrong_prediction():
    target = np.array([0.9, 0.1])
    good = np.array([0.8, 0.2])
    bad = np.array([0.1, 0.9])
    assert forward_kl_stopgrad(target, good) < forward_kl_stopgrad(target, bad)

