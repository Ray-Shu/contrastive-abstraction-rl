from data.Sampler import Sampler 
import torch
import numpy as np
import minari

class RandomSamplingDataset(torch.utils.data.Dataset): 
    def __init__(self, cl_model,  minari_dataset, iterate_thru_dataset: bool = False, data = None):
        """
        Creates a dataset with randomly sampled states from the minari dataset specified by the TrajectorySet class. 

        Args: 
            cl_model: The contrastive learning model that changes x to z representations. 
            minari_dataset: The minari dataset to use. 
            iterate_thru_dataset: If true, gets all of the states (observations) from the specified minari dataset. 
            data: If there is existing to data, convert it to this dataset to use with the PyTorch dataloader. Data should be a numpy array. 
        """

        self.cl_model = cl_model 
        self.minari_dataset = minari_dataset

        if data is not None: 
            self.states = torch.as_tensor(data, dtype=torch.float32)

        elif iterate_thru_dataset: 
            self.states = self.__get_all_states()
            self.states = torch.as_tensor(self.states, dtype=torch.float32)

        with torch.no_grad(): 
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


