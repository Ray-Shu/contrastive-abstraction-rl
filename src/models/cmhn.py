import os
import sys

import faiss 

from PIL import Image 
import torch 
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt 

class cmhn(): 
    def __init__(self, max_iter = 100, threshold = 0.95, topk = 512, device="cpu"):
        """
        Continuous Modern Hopfield Network 

        Args: 
            update_steps: The number of iterations the cmhn will do. (Usually just one).
            topk: Using faiss, only the top k most similar patterns will be used. (more efficient in batch-wise updates) 
            use_gpu: Tells faiss if we use faiss-cpu or faiss-gpu for behind the scenes calculations. 
            device: The device that torch will use. 
        """
        self.max_iter = max_iter 
        self.threshold = threshold
        self.topk = topk

        self.device = torch.device(device)
        self.index = None 

    def build_index(self, X, d): 
        """
        Builds a faiss index (an object) for efficient searching of top-k patterns from X (on cpu). 
        """
        X_np = X.detach().cpu().numpy().astype("float32") # convert X from tensor to numpy 

        self.index = faiss.IndexFlatIP(d)
        
        self.index.add(X_np)
    
    def __update(self, X, xi, beta): 
        """
        The update rule for a continuous modern hopfield network. 

        Args: 
            X: The stored patterns. X is of size [N, d], where N is the number of patterns, and d the size of the patterns. 
            xi: The state pattern (ie. the current pattern being updated). xi is of size [d, 1]. 
            beta: The scalar inverse-temperature hyperparamater. Controls the number of metastable states that occur in the energy landscape. 
                - High beta corresponds to low temp, more separation between patterns.  
                - Low beta corresponds to high temp, less separation (more metastable states). 
        """
        X_norm = F.normalize(X, p=2, dim=1)
        xi_norm = F.normalize(xi, p=2, dim=0)
        sims = X_norm @ xi_norm  # simularity between stored patterns and current pattern 
        p = F.softmax(beta * sims, dim=0, dtype=torch.float32)  # softmax dist along patterns (higher probability => more likely to be that stored pattern)
        # p of size [N, 1] 

        X_T = X_norm.transpose(0, 1) 
        xi_new = X_T @ p  # xi_new, the updated state pattern; size [d, 1]
        return xi_new

    def __run_batch(self, X, queries, beta=None): 
        """
        Runs the mhn batch-wise for efficient computation. 

        Args: 
            X: Stored patterns, size [N, d].
            queries: Input queries, size [N, d].
            beta: The beta value per sample, size [N].
        """        
        
        assert beta != None, "Must have a value for beta." 

        # normalize for cos sim calcs
        X_norm = F.normalize(X, p=2, dim=-1)
        queries_norm = F.normalize(queries, p=2, dim=-1)

        with torch.no_grad():
            queries_np = queries_norm.detach().cpu().numpy().astype("float32")
            _, indices = self.index.search(queries_np, self.topk)
            indices = torch.from_numpy(indices).to(X.device) # indices of shape [N, topk]

        topk_X = X_norm[indices] # size [N, topk, d] 
        topk_q = queries_norm.unsqueeze(1) # change queries from [N, d] to [N, 1, d] for broadcasting
        
        # dot product of x_ij * q_i along "d dim" to obtain tensor of [N, topk]
        # q_i represents the i'th query
        # x_ij represents the corresponding i'th query and j'th pattern, where j is among the topk 
        # then sum over d to obtain the similarity between row i and col j. 
        # sims = torch.sum(topk_X * topk_q, dim=-1) 

        # USE torch.bmm instead of the above comments for more efficient computation (they do the same thing tho) 
        sims = torch.bmm(topk_q, topk_X.transpose(1,2)).squeeze(1)

        # removing beta broadcasting
        #beta = beta.view(-1, 1)  # beta: [N, 1], broadcasting beta. 
        logits = beta * sims       # sims * beta: [N, topk]
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        probs = F.softmax(logits - logits_max.detach(), dim=-1)   # calculate probs along patterns (NOT queries) ie. along topk --> [N, topk]

        # weighted sum over topk_X: x_ij * probs_i
        xi_new = torch.sum(probs.unsqueeze(-1) * topk_X, dim=1)

        return xi_new
    
    def __has_converged(self, old_xi, new_xi): 
        """ 
        Checks whether or not the hopfield network has converged. Convergence is measured through taking the average cosine similarity 
        between old_xi and new_xi. If this average meets the threshold (ie. avg_cos_sim >= threshold), then we say that old_xi and 
        new_xi are the same and that the hopfield network has converged. 

        old_xi and new_xi are shapes: [N, d]

        Args: 
            old_xi xi before running the udpate rule.
            new_xi: xi after running the update rule.
        
        Returns:
            True: if the average cosine similarity between old_xi and new_xi is meets the threshold.
            False: if the average cosine similarity between old_xi and new_xi is below the threshold.
        """
        converged = False 

        old_norm = F.normalize(old_xi, p=2, dim=-1)  # normalize along rows
        new_norm = F.normalize(new_xi, p=2, dim=-1)

        cos_sim = torch.sum(old_norm * new_norm, dim=1)  # [N], similarity for each query

        min_cos_sim = cos_sim.min().item()
        if min_cos_sim >= self.threshold:
            converged = True

        return converged

    def run(self, X, xi, beta=None, run_as_batch=False): 
        """
        Runs the network. 

        Args: 
            X: The stored patterns. X is of size [N, d], where B is the batches, N is the number of patterns, and d the size of the patterns. 
            xi: The state pattern (ie. the current pattern being updated). xi is of size [d, 1]. xi can also be a batch of queries [N, d].
            beta: The scalar inverse-temperature hyperparamater. Controls the number of metastable states that occur in the energy landscape. 
                - High beta corresponds to low temp, more separation between patterns.  
                - Low beta corresponds to high temp, less separation (more metastable states). 
        """
        assert beta != None, "Must have a value for beta."

        if not isinstance(beta, torch.Tensor):
           beta = torch.as_tensor(beta, dtype=torch.float32)

        X = X.to(self.device)
        xi = xi.to(self.device)
        beta = beta.to(self.device)

        if run_as_batch: 
            if xi.dim() == 1: 
                raise ValueError("Query shape should be [N, d] when updating as a batch.")
            for _ in range(self.max_iter): 
                old_xi = xi.clone()
                xi = self.__run_batch(X, xi, beta)

                if self.__has_converged(old_xi=old_xi, new_xi=xi): 
                    break 
            return xi

        else:
            # if xi is of size [d], then change to [d, 1] 
            if xi.dim() == 1: 
                xi = xi.unsqueeze(1) #[d, 1]
            elif xi.dim() == 2 and xi.size(1) != 1: 
                raise ValueError("Query shape should be [d] or [d, 1].") 

            for _ in range(self.max_iter): 
                xi = self.__update(X, xi, beta)
            return xi 