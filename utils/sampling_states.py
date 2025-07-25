import numpy as np
import minari

def sample_states(dataset, num_states: int = None, save_n_trajectories: int = None) -> dict: 
    """
    Samples a number of states (observations) and returns a list of those corresponding states. 

    Args: 
        dataset: The minari dataset to use.  
        num_states: The number of states to sample. 
        save_trajectories: An integer number of how many trajectories to "keep track of". 
    
    Returns:
        A dictionary containing the indices of where a trajectory ends, and the list of states. 
    """ 

    assert dataset != None, "Must have a minari dataset specified!"
    assert num_states > 0, "Must have positive non-zero integer to determine the number of states sampled."

    if num_states > dataset.total_steps: 
        num_states = dataset.total_steps 

    length_counter = 0 
    d = {
        "trajectory_idx": [], 
        "states": []
    } 

    total_eps = dataset.total_episodes

    eps = dataset.sample_episodes(n_episodes=total_eps)

    d["states"] = eps[0].observations["observation"]

    # stack all states vertically so the states array has shape: [N, 4], where N is the total number of states
    for i in range(1, total_eps): 
        traj = eps[i].observations["observation"]

        if save_n_trajectories > 0: 
            traj_len = length_counter + len(traj)
            d["trajectory_idx"].append(traj_len)

            # update counters
            length_counter = traj_len
            save_n_trajectories -= 1

        d["states"] = np.vstack((d["states"], traj))
        if len(d["states"]) > num_states: 
            d["states"] = d["states"][:num_states]
            return d

    return d