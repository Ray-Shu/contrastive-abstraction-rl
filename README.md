# contrastive-abstraction-RL
This github repo aims to reproduce the results from [Contrastive Abstraction for Reinforcement Learning](https://arxiv.org/pdf/2410.00704). 

## Overview
This project implements the contrastive model, continuous hopfield network, and beta model. The goal is to: 
1. Learn temporally correlated representations using the InfoNCE loss.
2. Cluster embeddings to reduce the state-space of a given MDP using a continuous Hopfield network with a beta network.
The implementation uses Minari's D4RL `pointmaze-large` [dataset](https://minari.farama.org/datasets/D4RL/pointmaze/large-v2/).

## Installation 
Clone repo and install dependencies. 
```bash
git clone https://github.com/Ray-Shu/contrastive-abstraction-rl.git
pip install -r requirements.txt
```

## Running the Models
To run the contrastive learning model: 
```bash
python cl_main.py 
```
The contrastive model has many flags that can be adjusted: 
```bash
--distribution  # the statistical distribution used to sample positive pairs: [l, g, e, u] correspond to laplace, gaussian, exponential, uniform respectively (default is laplace). 
--num_states    # the total number of states the model is trained on (default is one million).
--lr            # learning rate (default is 1e-3).
--weight_decay  # default is 1e-5.
--temperature   # tunes the sharpness of the softmax distribution in the InfoNCE loss; smaller temp means sharper distribution and vice versa (default is 30).
--max_epochs    # number of epochs to train the model for (default is 1000).
--filename      # the name of the model.
--device        # the device to use when running computations.
--minibatch     # size of the minibatch (default is 4096).
```

To run the beta model: 
```bash
python beta_main.py
```
The beta model also has many flags that can be adjusted: 
```bash
--num_states    # the total number of states the model is trained on (default is one million).
--lr            # learning rate (default is 1e-3).
--temperature   # tunes the sharpness of the softmax distribution in the InfoNCE loss; smaller temp means sharper distribution and vice versa (default is 1).
--weight_decay  # default is 1e-5.
--masking_ratio # introduces noise in learned representations z to allow the hopfield network to produce more robust clusters (default is 0.3).
--beta_max      # the maximum beta range. ie. beta_max = 200 corresponds to a beta in the range [0,200] (default is 200).
--max_epochs    # number of epochs to train the model for (default is 100).
--filename      # the name of the model.
--device        # the device to use when running computations.
--minibatch     # size of the minibatch (default is 4096).
--cl_model_distribution  #which model to use: [l, g, e, u] (default is the laplace model).
```
