"""
Smoke test for the OGBench data pipeline.

Loads a small slice of an OGBench dataset and verifies that:
  1. sample_states() returns the right keys and shape.
  2. sample_trajectories() returns the right number of episodes.
  3. StatesDataset encodes states through a fresh mlpCL without error.

Run from the project root:
    python tests/ogbench/test_ogbench.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import ogbench

from src.utils.sampling_states import sample_states, sample_trajectories
from src.data.StatesDataset import StatesDataset
from src.models.cl_model import mlpCL

# Use the smallest available navigate dataset.
DATASET_NAME = "antmaze-medium-navigate-v0"
# Only keep the first N steps so the test stays fast.
SUBSET_SIZE = 2_000


def make_subset(dataset: dict, n: int) -> dict:
    """Return a copy of dataset sliced to the first n steps, with at least one terminal."""
    sub = {k: v[:n] for k, v in dataset.items()}
    # Guarantee at least one terminal so sample_trajectories has episodes to pick.
    if not np.any(sub["terminals"]):
        sub["terminals"][-1] = 1
    return sub


def test_sample_states():
    _, og_dataset, _ = ogbench.make_env_and_datasets(DATASET_NAME)
    dataset = make_subset(og_dataset, SUBSET_SIZE)

    result = sample_states(dataset, num_states=500)

    assert "states" in result, "sample_states must return a 'states' key"
    assert "trajectory_idx" in result, "sample_states must return a 'trajectory_idx' key"
    assert len(result["states"]) == 500, f"Expected 500 states, got {len(result['states'])}"
    assert result["states"].ndim == 2, "States should be 2-D (N, obs_dim)"
    print(f"  sample_states: shape={result['states'].shape}  PASS")


def test_sample_states_clamps_to_total():
    _, og_dataset, _ = ogbench.make_env_and_datasets(DATASET_NAME)
    dataset = make_subset(og_dataset, SUBSET_SIZE)

    result = sample_states(dataset, num_states=10_000_000)
    assert len(result["states"]) == SUBSET_SIZE, (
        f"Expected {SUBSET_SIZE} states (clamped), got {len(result['states'])}"
    )
    print(f"  sample_states clamp: returned {len(result['states'])} states  PASS")


def test_sample_trajectories():
    _, og_dataset, _ = ogbench.make_env_and_datasets(DATASET_NAME)
    dataset = make_subset(og_dataset, SUBSET_SIZE)

    trajs = sample_trajectories(dataset, n_episodes=2, ep_len=SUBSET_SIZE + 1)

    assert len(trajs) == 2, f"Expected 2 trajectories, got {len(trajs)}"
    for traj in trajs:
        assert isinstance(traj, np.ndarray), "Each trajectory should be a numpy array"
        assert traj.ndim == 2, "Each trajectory should be 2-D (T, obs_dim)"
    print(f"  sample_trajectories: got {len(trajs)} episodes, "
          f"lengths={[len(t) for t in trajs]}  PASS")


def test_states_dataset():
    _, og_dataset, _ = ogbench.make_env_and_datasets(DATASET_NAME)
    dataset = make_subset(og_dataset, SUBSET_SIZE)

    result = sample_states(dataset, num_states=256)
    states = result["states"]

    obs_dim = states.shape[1]
    cl_model = mlpCL(input_dim=obs_dim)
    cl_model.eval()

    ds = StatesDataset(cl_model=cl_model, data=states)

    assert len(ds) == 256, f"Expected dataset length 256, got {len(ds)}"
    z = ds[0]
    assert z.shape == (32,), f"Expected z-dim 32, got {z.shape}"
    print(f"  StatesDataset: len={len(ds)}, z_dim={z.shape[0]}  PASS")


if __name__ == "__main__":
    tests = [
        test_sample_states,
        test_sample_states_clamps_to_total,
        test_sample_trajectories,
        test_states_dataset,
    ]
    passed = 0
    for t in tests:
        print(f"Running {t.__name__}...")
        t()
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")
