from data.TrajectorySet import TrajectorySet

import torch 


class Sampler(): 
    def __init__(self, T: TrajectorySet, dist="g"): 
        """
        T: The Trajectory Set class 
        dist: The distribution used for centering over the anchor state. 
            ['u', 'g', 'l', 'e'] - uniform, gaussian, laplace, exponential
        """

        self.T = T 
        self.dist = dist

    def sample_anchor_state(self, t: list) -> tuple[list, int]: 
        """
        Given a trajectory, we sample the anchor state s_i uniformly. 

        Args: 
            t: The given trajectory we sample from. 

        Returns: 
            A tuple containing [s_i, idx]
            s_i: The state that is sampled, represented as a list of (x,y) coordinates and velocities. 
            idx: The time step of s_i. 
        """
        idx = torch.randint(low=0, high=len(t), size=(1,)).item()
        s_i = t[idx] 
        return [s_i, idx]

    def sample_positive_pair(self, t: list, anchor_state: tuple[list, int]) -> tuple[list, int]: 
        """
        Given the same trajectory that s_i was sampled from, 
        center a gaussian distribution around s_i to get obtain its positve pair: s_j. 
        
        Args: 
            t: The given trajectory, which must be the same as the trajectory that was used to sample the anchor state. 
            anchor_state: The anchor state; a tuple containing [s_i, idx].
            s_i: The state itself.
            idx: The time step of s_i.
            
        
        Return: 
            A tuple containing [s_j, idx]
            s_j: The state that is sampled, represented as a list of (x,y) coordinates and velocities. 
            idx: The time step of s_j.    
        """
        std = 15     # we use 15 to replicate the paper's hyperparams 
        b = 15       # laplace scale hyper param
        gamma = 0.99 # exponential hyper param 

        _, si_idx = anchor_state

        while True: 
            if self.dist == "u": 
                # uniform 
                sj_idx = torch.randint(low=0, high=len(t), size=(1,))
            elif self.dist == "g": 
                # gaussian 
                sj_idx = torch.normal(mean=si_idx, std=std, size=(1,))
            elif self.dist == "l": 
                # laplacian
                sj_idx = torch.distributions.laplace.Laplace(loc=si_idx, scale=b).sample() 
            elif self.dist == "e": 
                # exponential 
                i = int(torch.distributions.exponential.Exponential(rate=gamma).sample()) + 1   # +1 so we don't get an offset of 0
                sj_idx = si_idx + i 
            else: 
                # default to gaussian
                sj_idx = torch.normal(mean=si_idx, std=std, size=(1,))

            sj_idx = int(sj_idx) 

            # Ensures we don't choose an index out of range or the same state. 
            if (sj_idx < len(t)) and (sj_idx > 0) and (sj_idx != si_idx): 
                break 
        
        s_j = t[sj_idx] 

        return [s_j, sj_idx]
    
    def sample_batch(self, batch_size=1024, k=2) -> list[tuple]: 
        """ 
        Creates a batch of anchor states, their positive pairs, and negative pairs. 
        There will be 2(batch_size - 1) amount of negative examples per positive pair.

        Args: 
            batch_size: The size of the batch to be generated.
            k: A hyperparameter that dictates the average number of 
                positive pairs sampled from the same trajectory. The 
                lower the number, the lesser the chance of false negatives. 
        
        Returns: 
            A list of tuples containing the anchor_state and its positive pair. 
            The list is the same length as batch_size. 
        """ 

        batch = [] 

        # Generate trajectory set 
        n_trajectories = batch_size // k
        self.T.generate_trajectories(n_trajectories= n_trajectories)

        for _ in range(batch_size): 
            # Sample anchor state 
            rng = torch.randint(low=0, high=n_trajectories, size=(1,)).item() 
            t = self.T.get_trajectory(index=rng)[0]
            
            anchor_state = self.sample_anchor_state(t) 

            # Sample positive pair 
            positive_pair = self.sample_positive_pair(t, anchor_state=anchor_state)

            # Retrieve states; time-steps aren't necessary. 
            s_i = anchor_state[0]
            s_j = positive_pair[0]

            batch.append([s_i, s_j]) 

        return batch 