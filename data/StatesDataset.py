from data.Sampler import Sampler 
import torch

class StatesDataset(torch.utils.data.Dataset): 
    def __init__(self, cl_model,  sampler: Sampler, num_states: int):
        """
        Creates a dataset with randomly sampled states from the minari dataset specified by the TrajectorySet class. 
        """
        self.cl_model = cl_model 

        states = torch.tensor(sampler.sample_states(batch_size=num_states), dtype=torch.float32)

        with torch.no_grad():
            self.z = cl_model(states)

    def __len__(self): 
        return len(self.z) 
    
    def __getitem__(self, index):
        return self.z[index]
