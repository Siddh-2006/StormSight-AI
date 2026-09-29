import numpy as np

from app.core.config import CHANNELS
from app.ml.source_dropout import apply_source_dropout, full_availability_mask


def test_full_availability_mask_shape():
    mask = full_availability_mask()
    assert mask.shape == (len(CHANNELS),)
    assert np.all(mask == 1.0)


def test_source_dropout_zeroes_dropped_channels_single_frame():
    stack = np.random.default_rng(0).uniform(0.2, 1.0, size=(len(CHANNELS), 8, 8)).astype(np.float32)
    masked, mask = apply_source_dropout(stack, drop_prob=1.0, min_available=1, rng=np.random.default_rng(0))
    # With drop_prob=1.0, everything WOULD be dropped, but min_available=1
    # forces at least one channel to survive.
    assert int(mask.sum()) == 1
    kept_idx = int(np.argmax(mask))
    for i in range(len(CHANNELS)):
        if i == kept_idx:
            assert np.allclose(masked[i], stack[i])
        else:
            assert np.allclose(masked[i], 0.0)


def test_source_dropout_never_drops_below_min_available():
    stack = np.ones((len(CHANNELS), 4, 4), dtype=np.float32)
    for trial in range(20):
        _, mask = apply_source_dropout(stack, drop_prob=0.9, min_available=2, rng=np.random.default_rng(trial))
        assert int(mask.sum()) >= 2


def test_source_dropout_applies_same_mask_across_time_sequence():
    t, c, h, w = 4, len(CHANNELS), 5, 5
    stack = np.random.default_rng(2).uniform(0, 1, size=(t, c, h, w)).astype(np.float32)
    masked, mask = apply_source_dropout(stack, drop_prob=0.5, rng=np.random.default_rng(2))
    dropped_channels = np.where(mask == 0.0)[0]
    for ch in dropped_channels:
        assert np.allclose(masked[:, ch], 0.0)  # dropped for EVERY time step, not flickering


def test_source_dropout_zero_prob_keeps_everything():
    stack = np.random.default_rng(3).uniform(0, 1, size=(len(CHANNELS), 6, 6)).astype(np.float32)
    masked, mask = apply_source_dropout(stack, drop_prob=0.0, rng=np.random.default_rng(3))
    assert np.all(mask == 1.0)
    assert np.allclose(masked, stack)
