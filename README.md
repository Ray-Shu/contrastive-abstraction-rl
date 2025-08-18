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
git clone 
```
