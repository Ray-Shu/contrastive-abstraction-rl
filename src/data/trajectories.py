class TrajectorySet:
    def __init__(self):
        """
        trajectories: a dictionary housing all of the trajectories. The dictionary structure is:
            {
                1: [trajectory, length of trajectory]
                2: [ ... ]
                etc...
            }

        num_trajectories: the number of trajectories currently in the set.
        """
        self.trajectories = {}
        self.num_trajectories = 0

    def add_trajectory(self, trajectory):
        if isinstance(trajectory, dict):
            length = len(trajectory['actions'])
        else:
            length = len(trajectory)
        self.trajectories[self.num_trajectories] = [trajectory, length]
        self.num_trajectories += 1

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

