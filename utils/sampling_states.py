import numpy as np
import minari

def sample_states(dataset, num_states: int = None) -> list: 
    """
    Samples a number of states (observations) and returns a list of those corresponding states. 

    Args: 
        dataset: The minari dataset to use.  
        num_states: The number of states to sample. 
    
    Returns:
        list[states]: A numpy list of states of shape [num_states, 4]
    """ 

    assert dataset != None, "Must have a minari dataset specified!"
    assert num_states > 0, "Must have positive non-zero integer to determine the number of states sampled."

    if num_states > dataset.total_steps: 
        num_states = dataset.total_steps 

    total_eps = dataset.total_episodes

    eps = dataset.sample_episodes(n_episodes=total_eps)
    states = eps[0].observations["observation"]

    if len(states) > num_states: 
        return states[:num_states]
    else: 
        # stack all states vertically so the states array has shape: [N, 4], where N is the total number of states
        for i in range(1, total_eps): 
            states = np.vstack((states, eps[i].observations["observation"]))
            if len(states) > num_states: 
                return states[:num_states]

    return states