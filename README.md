# contrastive-abstraction-RL
This github repo aims to reproduce the results from [Contrastive Abstraction for Reinforcement Learning](https://arxiv.org/pdf/2410.00704). 

## Overview
This project implements the contrastive model, continuous hopfield network, and beta model. The goal is to: 
1. Learn temporally correlated representations using the InfoNCE loss.
2. Cluster embeddings to reduce the state-space of a given MDP using a continuous Hopfield network with a beta network.
The implementation uses Minari's D4RL `pointmaze-large` [dataset](https://minari.farama.org/datasets/D4RL/pointmaze/large-v2/).

## Installation 
Clone repo and install dependencies. Note that we used python version 3.9.6 to run everything. 
```bash
git clone https://github.com/Ray-Shu/contrastive-abstraction-rl.git
cd contrastive-abstraction-rl
pip install -r requirements.txt
```

Creating a virtual environment to host these dependencies is recommended. This will create a conda virtual environment. Info on Conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 
```bash
conda create --name myvenv python=3.9.6
conda activate myvenv
pip install -r requirements.txt
```

## Running Visualizations: 
```bash 
python -m src.visuals.visuals
```
Note that the visualizations here are only for the Minari PointMaze dataset linked above, and the environment cannot be changed. These visualizations only serve to show what the four contrastive models (using different distributions to sample positive pairs) and what the beta model can do. However, the model corresponding to its statistical distribution can be changed. 
```bash
--distribution      # the model corresponding to the distribution that samples positive pairs; can only choose [l, g, e, u] which corresponds to laplace, gaussian, exponential and uniform (default is l). 
--subsample_size    # how many states to be shown on the plots (default is 10,000).
--total_states      # the total number of states to sample, which affects the weights of the PCA (default is one million). 
```

## Additional Info
1. The `best_models` folder has the trained models for the four distributions and the beta network.
2. The `notebooks` folder holds the prototypes and testing for everything that was written.

This downloadable [pdf](https://github.com/user-attachments/files/21882859/Reproducing_the__Contrastive_Abstraction_for_Reinforcement_Learning__Paper.pdf) contains in-depth information and mathematical formulas behind this project.
