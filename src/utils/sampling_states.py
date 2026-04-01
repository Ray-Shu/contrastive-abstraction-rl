import numpy as np

def sample_states(dataset, num_states: int = None, save_n_trajectories: int = None) -> dict:
    """
    Samples a number of states (observations) from an OGBench dataset.

    Args:
        dataset: An OGBench dataset dict with 'observations' and 'terminals' keys.
        num_states: The number of states to sample.
        save_n_trajectories: If set, records the cumulative end indices of this many trajectories.

    Returns:
        A dictionary with:
            "trajectory_idx": list of cumulative end indices for saved trajectories
            "states": np.ndarray of sampled states
    """
    observations = dataset['observations']
    terminals = dataset['terminals']
    total_steps = len(observations)

    if num_states is None or num_states > total_steps:
        num_states = total_steps

    d = {"trajectory_idx": [], "states": []}

    if save_n_trajectories is not None and save_n_trajectories > 0:
        end_indices = np.where(terminals == 1)[0]
        for idx in end_indices[:save_n_trajectories]:
            d["trajectory_idx"].append(int(idx) + 1)

    d["states"] = observations[:num_states]
    return d


def sample_trajectories(dataset, n_episodes: int = 2, ep_len: int = 200):
    """
    Returns a list of n_episodes state arrays, each shorter than ep_len steps.

    Args:
        dataset: An OGBench dataset dict with 'observations' and 'terminals' keys.
        n_episodes: Number of episodes to return.
        ep_len: Maximum episode length (exclusive).
    """
    observations = dataset['observations']
    terminals = dataset['terminals']

    end_indices = np.where(terminals == 1)[0]
    starts = np.concatenate([[0], end_indices[:-1] + 1])

    trajs = []
    for i in np.random.permutation(len(end_indices)):
        start, end = int(starts[i]), int(end_indices[i]) + 1
        ep_obs = observations[start:end]
        if len(ep_obs) < ep_len:
            trajs.append(ep_obs)
            if len(trajs) == n_episodes:
                break
    return trajs
