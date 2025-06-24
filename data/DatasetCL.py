import torch 
from utils.tensor_utils import convert_batch_to_tensor
class DatasetCL(torch.utils.data.Dataset): 
    def __init__(self, sampler = None, num_state_pairs: int = None, k: int = 2, data=None): 
        """
        sampler: The Sampler class to sample batches. 
        num_state_pairs: The number of state pairs
        k: A hyperparameter that dictates the average number of 
                positive pairs sampled from the same trajectory. The 
                lower the number, the lesser the chance of false negatives. 
        data: If there is an already existing data, and needs to be converted to this datasetCL type. 
        """
        if data: 
            self.pairs = data 
            self.num_state_pairs = len(self.pairs)
            self.k = k 
        else: 
            assert sampler != None, "Must have a sampler if you don't have a dataset inputted."
            assert num_state_pairs != None, "Must have a sampled pairs amount if you don't have a dataset inputted."

            self.sampler = sampler 
            self.num_state_pairs = num_state_pairs 
            self.k = k 
            anchor, positive = convert_batch_to_tensor(self.sampler.sample_batch(batch_size=self.num_state_pairs, k=self.k))
            self.pairs = list(zip(anchor, positive))

    def __len__(self): 
        return len(self.pairs)
    
    def __getitem__(self, index):
        s_i, s_j = self.pairs[index] 
        return (torch.as_tensor(s_i, dtype=torch.float32), torch.as_tensor(s_j, dtype=torch.float32))
    
    def get_batch(self):
        return self.pairs