class TrajectorySet: 
    def __init__(self, dataset): 
        """
        dataset: The minari dataset to use. 
        trajectories: a dictionary housing all of the trajectories. The dictionary structure is: 
            {
                1: [trajectory, length of trajectory]
                2: [ ... ]
                etc...
            } 

        num_trajectories: the number of trajectories currently in the set. 
        """
        self.dataset = dataset
        self.total_episodes = dataset.total_episodes 

        self.trajectories = {} 
        self.num_trajectories = 0 
    
    def add_trajectory(self, trajectory):
        if isinstance(trajectory, dict):
            length = len(trajectory['actions'])
        else:
            length = len(trajectory)
        self.trajectories[self.num_trajectories] = [trajectory, length]
        self.num_trajectories += 1
    
    def get_total_episodes(self): 
        return self.total_episodes 
    
    def get_num_trajectories(self):
        return self.num_trajectories

    def get_trajectory(self, index): 
        assert index < self.num_trajectories, "Specified index is too large."
        return self.trajectories[index]
    
    def get_trajectory_set(self): 
        return self.trajectories
    
    def get_total_states(self): 
        sum = 0
        for _, v in self.trajectories.items(): 
            sum += v[1]
        return sum 

    def generate_trajectories(self, n_trajectories: int = 2, add_action: bool = False):
        """
        Generates a specified number of trajectories and saves them into the TrajectorySet class.

        This runs the scripted agent, where the agent uses a PD controller to follow a
        path of waypoints generated with QIteration until it reaches the goal.

        Args:
            n_trajectories: The number of trajectories to generate.
            add_action: If True, stores both states and actions as a dict {'states': ..., 'actions': ...}.
                        State indices are aligned to actions (shape [T, action_dim]), so T valid (state, action) pairs exist.
        """
        ep_data = self.dataset.sample_episodes(n_episodes=n_trajectories) # sample trajectories

        # adds all of the sampled trajectories into the TrajectorySet
        for i in range(len(ep_data)):
            ep = ep_data[i]

            if add_action:
                self.add_trajectory({'states': ep.observations["observation"], 'actions': ep.actions})
            else:
                # Note: only saving states since we only need state representations in the encoder
                self.add_trajectory(ep.observations["observation"])
