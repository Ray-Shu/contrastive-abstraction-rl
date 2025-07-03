import os
import sys

from PIL import Image 
import torch 
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt 

class cmhn(): 
    def __init__(self, update_steps = 1):
        """
        Continuous Modern Hopfield Network 

        Args: 
            update_steps: The number of iterations the cmhn will do. (Usually just one).
        """
        self.update_steps = update_steps 
    
    def update(self, X, xi, beta): 
        """
        The update rule for a continuous modern hopfield network. 

        Args: 
            X: The stored patterns. X is of size [N, d], where N is the number of patterns, and d the size of the patterns. 
            xi: The state pattern (ie. the current pattern being updated). xi is of size [d, 1]. 
            beta: The scalar inverse-temperature hyperparamater. Controls the number of metastable states that occur in the energy landscape. 
                - High beta corresponds to low temp, more separation between patterns.  
                - Low beta corresponds to high temp, less separation (more metastable states). 
        """
        sims = X @ xi  # simularity between stored patterns and current pattern 
        p = F.softmax(beta * sims, dim=0, dtype=torch.float32)  # softmax dist along patterns (higher probability => more likely to be that stored pattern)
        # p of size [N, 1] 

        X_T = X.transpose(0, 1) 
        xi_new = X_T @ p  # xi_new, the updated state pattern; size [d, 1]
        return xi_new

    def run(self, X, xi, beta=None): 
        """
        Runs the network. 
        
        Args: 
            X: The stored patterns. X is of size [N, d], where N is the number of patterns, and d the size of the patterns. 
            xi: The state pattern (ie. the current pattern being updated). xi is of size [d, 1]. 
            beta: The scalar inverse-temperature hyperparamater. Controls the number of metastable states that occur in the energy landscape. 
                - High beta corresponds to low temp, more separation between patterns.  
                - Low beta corresponds to high temp, less separation (more metastable states). 
        """
        assert beta != None, "Must have a value for beta."

        for _ in range(self.update_steps): 
            xi = self.update(X, xi, beta)
        return xi 
