from src.data.TrajectorySet import TrajectorySet


class SyntheticTrajectorySet(TrajectorySet):
    """
    A TrajectorySet that holds pre-loaded trajectory data.

    Trajectories must be added via add_trajectory() before passing to Sampler.
    generate_trajectories() is overridden as a no-op because Sampler.__init__
    calls it unconditionally; we don't want it to overwrite already-loaded data.

    Invariant: add_trajectory() must be called exactly n_trajectories times so
    that num_trajectories equals the expected count (Sampler uses num_trajectories
    as the upper bound for random trajectory selection).
    """

    def __init__(self, n_trajectories: int):
        super().__init__()

    def generate_trajectories(self, n_trajectories: int = 2, add_action: bool = False) -> None:
        pass  # no-op: data is pre-loaded via add_trajectory()
