import numpy as np
from src.data.TrajectorySet import TrajectorySet
from src.data.SyntheticTrajectorySet import SyntheticTrajectorySet


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


def load_trajectories(path: str) -> SyntheticTrajectorySet:
    """
    Load trajectories from a .npz file into a SyntheticTrajectorySet.

    Returns a SyntheticTrajectorySet populated with {'states', 'actions'} dicts,
    ready to pass to Sampler.
    """
    data = np.load(path)
    n = int(data["n_trajectories"])
    tset = SyntheticTrajectorySet(n_trajectories=n)
    for i in range(n):
        tset.add_trajectory({
            "states":  data[f"traj_{i}_states"],
            "actions": data[f"traj_{i}_actions"],
        })
    return tset
