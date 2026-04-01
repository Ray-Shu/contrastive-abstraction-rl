import torch
import numpy as np

def convert_batch_to_tensor(batch: list[list, list]) -> tuple[torch.tensor, torch.tensor]: 
    """
    Converts the batch to a tuple of tensors. 
    The first tensor corresponds to the anchor states.
    The second tensor corresponds to their corresponding positive pair. 
    i.e. i'th anchor state in the first tensor will have its positive pair be in the i'th state in the second tensor. 
    """

    #unzips the batch into two tuples
    a, b = zip(*batch)  

    # stack arrays row-wise and then convert to tensor of dtype float (to be compatible w/ model weights)
    a_t = torch.tensor(np.stack(a, axis=0), dtype= torch.float32)
    b_t = torch.tensor(np.stack(b, axis=0), dtype= torch.float32)

    return (a_t, b_t)


def split_data(data:list, split_val: float = 0.7): 
    """
    Splits a dataset according to the split_val value. 
    ie. split_val = 0.7 means 70% of the dataset becomes train, 30% of the dataset becomes validation. 
    """
    train_len = round(len(data) * split_val) 
    return data[:train_len], data[train_len:]