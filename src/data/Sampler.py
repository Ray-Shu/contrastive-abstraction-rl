from data.TrajectorySet import TrajectorySet
from utils.truncated_distributions import truncated_normal
from utils.truncated_distributions import truncated_laplace
from utils.truncated_distributions import truncated_exponential

import torch 
import numpy as np

class Sampler(): 
    def __init__(self, T: TrajectorySet, dist="g", sigma = 15, b = 15, rate = 0.99): 
        """
        T: The Trajectory Set class 
        dist: The distribution used for centering over the anchor state. 
            ['u', 'g', 'l', 'e'] - uniform, gaussian, laplace, exponential
        """

        self.T = T 
        self.dist = dist
        self.total_episodes = T.get_total_episodes() 
        self.T.generate_trajectories(n_trajectories= self.total_episodes)
        
        # Hyperparameters
        self.sigma = sigma
        self.b = b 
        self.rate = rate  


    def sample_anchor_state(self, t_idx: int) -> tuple[list, int]: 
        """
        Given a trajectory, we sample the anchor state s_i uniformly. 

        Args: 
            t: The index of the specific trajectory to sample from.

        Returns: 
            A tuple containing [s_i, idx]
            s_i: The state that is sampled, represented as a list of (x,y) coordinates and velocities. 
            idx: The time step of s_i. 
        """
        trajectory = self.T.get_trajectory(index=t_idx)[0]
        idx = torch.randint(low=0, high=len(trajectory), size=(1,)).item()
        s_i = trajectory[idx] 
        return [s_i, idx]


    def sample_positive_pair(self, t_idx: int, anchor_state: tuple[list, int]) -> tuple[list, int]: 
        """
        Given the same trajectory that s_i was sampled from, 
        center a gaussian distribution around s_i to get obtain its positve pair: s_j. 
        
        Args: 
            t_idx: The index to locate a specific trajectory, which must be the same as the trajectory that was used to sample the anchor state. 
            anchor_state: The anchor state; a tuple containing [s_i, idx].
            s_i: The state itself.
            idx: The time step of s_i.
            
        Return: 
            Returns the positive pair's state and state index.
        """

        _, si_idx = anchor_state
        trajectory = self.T.get_trajectory(index=t_idx)[0]

        if self.dist == "u": 
            # uniform 
            sj_idx = torch.randint(low=0, high=len(trajectory), size=(1,))

        elif self.dist == "g": 
            # gaussian 
            p = truncated_normal(len(trajectory), mu=si_idx, sigma=self.sigma) 
            sj_idx = np.random.choice(a=len(trajectory), p=p)
            
        elif self.dist == "l": 
            # laplacian
            p = truncated_laplace(len=len(trajectory), mu=si_idx, b=self.b)
            sj_idx = np.random.choice(a=len(trajectory), p=p)

        elif self.dist == "e": 
            # exponential 
            p = truncated_exponential(len=len(trajectory), anchor_state_index=si_idx, rate=self.rate)
            sj_idx = np.random.choice(a=len(trajectory), p=p) 

        else: 
            # default to gaussian
            p = truncated_normal(len(trajectory), mu=si_idx, sigma=self.sigma) 
            sj_idx = np.random.choice(a=len(trajectory), p=p)
        
        s_j = trajectory[sj_idx]
        return [s_j, sj_idx]
    

    def sample_batch(self, batch_size=1024,) -> list[tuple]: 
        """ 
        Creates a batch of anchor states, their positive pairs, and negative pairs. 
        There will be 2(batch_size - 1) amount of negative examples per positive pair.

        Args: 
            batch_size: The size of the batch to be generated.
        
        Returns: 
            A list of tuples containing the anchor_state and its positive pair. 
            The list is the same length as batch_size. 
        """ 

        batch = [] 
    
        for _ in range(batch_size): 
            # Sample anchor state 
            t_idx = torch.randint(low=0, high=self.total_episodes, size=(1,)).item() 
            
            anchor_state = self.sample_anchor_state(t_idx) 

            # Sample positive pair 
            positive_pair = self.sample_positive_pair(t_idx, anchor_state=anchor_state)

            # Retrieve states; time-steps aren't necessary. 
            s_i = anchor_state[0]
            s_j = positive_pair[0]

            batch.append([s_i, s_j]) 

        return batch 

    def sample_states(self, batch_size=1024) -> list[tuple]: 
        """ 
        Creates a batch of anchor states, and its corresponding trajectory to use to sample positive pairs.. 
        There will be 2(batch_size - 1) amount of negative examples per positive pair.

        Args: 
            batch_size: The size of the batch to be generated.
            k: A hyperparameter that dictates the average number of 
                positive pairs sampled from the same trajectory. The 
                lower the number, the lesser the chance of false negatives. 
        
        Returns: 
            A list of tuples containing the anchor_state and its corresponding trajectory. 
            The list is the same length as batch_size. 
        """ 

        batch = [] 
            

        for _ in range(batch_size): 
            # Sample anchor state 
            t_idx = torch.randint(low=0, high=self.total_episodes, size=(1,)).item() 
        
            s_i = self.sample_anchor_state(t_idx) 
            batch.append((s_i, t_idx)) 

        return batch