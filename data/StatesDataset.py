from data.Sampler import Sampler 
import torch

class RandomSamplingDataset(torch.utils.data.Dataset): 
    def __init__(self, sampler: Sampler, num_states: int):
        """
        Creates a dataset with randomly sampled states from the minari dataset specified by the TrajectorySet class. 
        """

        self.states = sampler.sample_states(batch_size=num_states) 

    def __len__(self): 
        return len(self.states) 
    
    def __getitem__(self, index):
        val = self.states[index]
        return torch.tensor(val, dtype=torch.float32)
