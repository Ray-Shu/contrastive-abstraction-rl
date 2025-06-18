import torch 
from utils.tensor_utils import convert_batch_to_tensor
class DatasetCL(torch.utils.data.Dataset): 
    def __init__(self, sampler, batch_size: int, k: int = 2): 
        """
        sampler: The Sampler class to sample batches. 
        batch_size: The size of the batch (ie. the number of state pairs)
        k: A hyperparameter that dictates the average number of 
                positive pairs sampled from the same trajectory. The 
                lower the number, the lesser the chance of false negatives. 
        """
        self.sampler = sampler 
        self.batch_size = batch_size 
        self.k = k 

        anchor, positive = convert_batch_to_tensor(self.sampler.sample_batch(batch_size=self.batch_size, k=self.k))
        self.pairs = list(zip(anchor, positive))

    def __len__(self): 
        return len(self.pairs)
    
    def __getitem__(self, index):
        s_i, s_j = self.pairs[index] 
        return (torch.as_tensor(s_i, dtype=torch.float32), torch.as_tensor(s_j, dtype=torch.float32))
    
    def get_batch(self):
        return self.pairs