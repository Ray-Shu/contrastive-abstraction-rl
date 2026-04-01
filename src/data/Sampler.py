from src.data.TrajectorySet import TrajectorySet
from src.utils.truncated_distributions import truncated_normal
from src.utils.truncated_distributions import truncated_laplace
from src.utils.truncated_distributions import truncated_exponential

import torch
import numpy as np

class Sampler():
    def __init__(self, T: TrajectorySet, dist="g", sigma = 15, b = 15, rate = 0.99, add_action = False):
        """
        T: The Trajectory Set class
        dist: The distribution used for centering over the anchor state.
            ['u', 'g', 'l', 'e'] - uniform, gaussian, laplace, exponential
        add_action: If True, each sampled element is a concatenation of [state, action].
        """

        self.T = T
        self.dist = dist
        self.add_action = add_action
        self.total_episodes = T.get_num_trajectories()
        self.T.generate_trajectories(n_trajectories=self.total_episodes, add_action=add_action)

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
            s_i: The state (or state-action concatenation) that is sampled.
            idx: The time step of s_i.
        """
        trajectory = self.T.get_trajectory(index=t_idx)[0]

        states = trajectory['states']
        if self.add_action:
            actions = trajectory['actions']
            idx = torch.randint(low=0, high=len(actions), size=(1,)).item()
            s_i = np.concatenate([states[idx], actions[idx]])
        else:
            idx = torch.randint(low=0, high=len(states), size=(1,)).item()
            s_i = states[idx]

        return [s_i, idx]


    def sample_positive_pair(self, t_idx: int, anchor_state: tuple[list, int]) -> tuple[list, int]:
        """
        Given the same trajectory that s_i was sampled from,
        center a distribution around s_i to obtain its positive pair: s_j.

        Args:
            t_idx: The index to locate a specific trajectory, which must be the same as the trajectory that was used to sample the anchor state.
            anchor_state: The anchor state; a tuple containing [s_i, idx].

        Return:
            Returns the positive pair's state (or state-action concatenation) and state index.
        """

        _, si_idx = anchor_state
        trajectory = self.T.get_trajectory(index=t_idx)[0]

        if self.add_action:
            traj_len = len(trajectory['actions'])
        else:
            traj_len = len(trajectory['states'])

        if self.dist == "u":
            # uniform
            sj_idx = torch.randint(low=0, high=traj_len, size=(1,))

        elif self.dist == "g":
            # gaussian
            p = truncated_normal(traj_len, mu=si_idx, sigma=self.sigma)
            sj_idx = np.random.choice(a=traj_len, p=p)

        elif self.dist == "l":
            # laplacian
            p = truncated_laplace(len=traj_len, mu=si_idx, b=self.b)
            sj_idx = np.random.choice(a=traj_len, p=p)

        elif self.dist == "e":
            # exponential
            p = truncated_exponential(len=traj_len, anchor_state_index=si_idx, rate=self.rate)
            sj_idx = np.random.choice(a=traj_len, p=p)

        else:
            # default to gaussian
            p = truncated_normal(traj_len, mu=si_idx, sigma=self.sigma)
            sj_idx = np.random.choice(a=traj_len, p=p)

        if self.add_action:
            s_j = np.concatenate([trajectory['states'][sj_idx], trajectory['actions'][sj_idx]])
        else:
            s_j = trajectory['states'][sj_idx]

        return [s_j, sj_idx]


    def sample_batch(self, batch_size=1024,) -> list[tuple]:
        """
        Creates a batch of anchor states and their positive pairs.
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
        Creates a batch of anchor states, and its corresponding trajectory to use to sample positive pairs.

        Args:
            batch_size: The size of the batch to be generated.

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
