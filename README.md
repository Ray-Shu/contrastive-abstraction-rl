# contrastive-abstraction-RL
Implementation of [Contrastive Abstraction for Reinforcement Learning](https://arxiv.org/pdf/2410.00704) based on manuscript and author's codebase. Learns temporally correlated state representations via InfoNCE, then clusters them into abstract states using a beta network + continuous Hopfield network. 

## Installation
```bash
git clone https://github.com/Ray-Shu/contrastive-abstraction-rl.git
cd contrastive-abstraction-rl
uv sync
```

## Training

**Stage 1 — Contrastive learning model:**
```bash
python src/train_cl.py --data_path /path/to/trajectories.npz --distribution l
# distributions: l=Laplace, g=Gaussian, e=Exponential, u=Uniform
```

**Stage 2 — Beta model:**
```bash
python src/train_beta.py
```

## Visualization
```bash
python -m src.visualize --distribution l --subsample_size 10000 --total_states 1000000
```

## Testing
```bash
python tests/fourroom/test_fourroom.py
```
Trains both models end-to-end on a synthetic 9×9 four-room grid and outputs PCA/UMAP plots to `tests/fourroom/plots/`.

## Additional Info
- Pre-trained checkpoints for all four distributions and the beta model are in `trained_models/`.
- See this [pdf](https://github.com/user-attachments/files/21882859/Reproducing_the__Contrastive_Abstraction_for_Reinforcement_Learning__Paper.pdf) for in-depth math and background.
