import numpy as np

from app.core.config import CHANNELS, GRID_H, GRID_W
from app.data.synthetic_generator import StormGenerator, stack_channels


def test_step_shapes_and_range():
    gen = StormGenerator(rng_seed=42)
    gen.reset(seed=42, num_cells=3)
    frame = gen.step()
    assert set(frame.keys()) == set(CHANNELS)
    for name, arr in frame.items():
        assert arr.shape == (GRID_H, GRID_W)
        assert arr.dtype == np.float32
        assert np.all(arr >= 0) and np.all(arr <= 1), f"{name} out of [0,1] range"


def test_run_produces_sequence():
    gen = StormGenerator(rng_seed=1)
    gen.reset(seed=1, num_cells=2)
    frames = gen.run(10)
    assert len(frames) == 10
    stacked = stack_channels(frames[0])
    assert stacked.shape == (len(CHANNELS), GRID_H, GRID_W)


def test_lightning_correlates_with_radar_intensity():
    """Sanity: over many steps with cells forced intense, some lightning should
    eventually occur (i.e. the trigger logic isn't dead code)."""
    gen = StormGenerator(rng_seed=7)
    gen.reset(seed=7, num_cells=3)
    for cell in gen.cells:
        cell.peak_intensity = 1.0
        cell.lifespan = 40
    total_lightning = 0.0
    for _ in range(30):
        frame = gen.step()
        total_lightning += frame["lightning_density"].sum()
    assert total_lightning > 0, "expected at least some lightning activity with forced high-intensity cells"
