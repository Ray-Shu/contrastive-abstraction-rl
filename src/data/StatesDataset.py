from src.data.Sampler import Sampler
import torch
import numpy as np

class StatesDataset(torch.utils.data.Dataset):
    def __init__(self, cl_model, data):
        """
        Creates a dataset of z-representations from pre-sampled states.

        Args:
            cl_model: The contrastive learning model that maps x to z representations.
            data: A numpy array of states to encode.
        """
        self.cl_model = cl_model
        self.states = torch.as_tensor(data, dtype=torch.float32)

        with torch.no_grad():
            self.z = cl_model(self.states)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        return self.z[index]
