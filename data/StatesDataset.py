from data.Sampler import Sampler 
import torch
import numpy as np
import minari

class RandomSamplingDataset(torch.utils.data.Dataset): 
    def __init__(self, cl_model, sampler: Sampler = None,  minari_dataset_id = "D4RL/pointmaze/large-v2", num_states: int = None, iterate_thru_dataset: bool = False):
        """
        Creates a dataset with randomly sampled states from the minari dataset specified by the TrajectorySet class. 
        """
        self.cl_model = cl_model 
        self.minari_dataset = minari.load_dataset(minari_dataset_id)
        self.sampler = sampler 

        if iterate_thru_dataset: 
            self.states = self.__get_all_states()
            self.states = torch.as_tensor(self.states, dtype=torch.float32)

        else: 
            assert num_states != None, "If iterate_thru_dataset is False, must set num_states to a positive nonzero integer."
            assert sampler != None, "If iterate_thru_dataset is False, must have a sampler to sample states."
            self.states = torch.tensor(sampler.sample_states(batch_size=num_states), dtype=torch.float32)

        self.z = cl_model(self.states)

    def __len__(self): 
        return len(self.z) 
    
    def __getitem__(self, index):
        return self.z[index]

    def __get_all_states(self): 
        """
        Gets all of the states from the minari dataset and returns all states in Tensor form. 
        """
        total_eps = self.minari_dataset.total_episodes
        print(total_eps)

        eps = self.minari_dataset.sample_episodes(n_episodes=total_eps) 
        states = eps[0].observations["observation"]

        # stack all states vertically so the states array has shape: [N, 4], where N is the total number of states
        for i in range(1, total_eps): 
            states = np.vstack((states, eps[i].observations["observation"]))

        return states

