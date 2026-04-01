import numpy as np
from src.data.trajectories import TrajectorySet


def save_trajectories(tset: TrajectorySet, path: str) -> None:
    """
    Save all trajectories in tset to a compressed .npz file.

    Trajectories must be stored as {'states': ndarray, 'actions': ndarray} dicts.

    On-disk keys:
        n_trajectories          — scalar int
        traj_{i}_states         — ndarray [T_i, state_dim]
        traj_{i}_actions        — ndarray [T_i, action_dim]
    """
    arrays = {"n_trajectories": np.array(tset.get_num_trajectories())}
    for i in range(tset.get_num_trajectories()):
        traj, _ = tset.get_trajectory(i)
        arrays[f"traj_{i}_states"] = traj["states"]
        arrays[f"traj_{i}_actions"] = traj["actions"]
    np.savez_compressed(path, **arrays)


def load_trajectories(path: str) -> TrajectorySet:
    """
    Load trajectories from a .npz file into a TrajectorySet.

    Returns a TrajectorySet populated with {'states', 'actions'} dicts,
    ready to pass to Sampler.
    """
    data = np.load(path)
    n = int(data["n_trajectories"])
    tset = TrajectorySet()
    for i in range(n):
        tset.add_trajectory({
            "states":  data[f"traj_{i}_states"],
            "actions": data[f"traj_{i}_actions"],
        })
    return tset


def ogbench_to_trajectory_set(dataset: dict) -> TrajectorySet:
    """
    Convert an OGBench dataset dict into a TrajectorySet.

    Splits the flat observations/actions arrays into per-episode dicts
    using the 'terminals' array to find episode boundaries.
    """
    observations = dataset["observations"]
    actions = dataset["actions"]
    terminals = dataset["terminals"]

    end_indices = np.where(terminals == 1)[0]
    starts = np.concatenate([[0], end_indices[:-1] + 1])
    ends   = end_indices + 1

    # Include any trailing steps after the last terminal as a final episode.
    if len(ends) == 0 or ends[-1] < len(observations):
        starts = np.concatenate([starts, [ends[-1] if len(ends) > 0 else 0]])
        ends   = np.concatenate([ends,   [len(observations)]])

    tset = TrajectorySet()
    for start, end in zip(starts, ends):
        tset.add_trajectory({
            "states":  observations[start:end],
            "actions": actions[start:end],
        })
    return tset
