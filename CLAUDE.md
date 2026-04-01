# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Contrastive Abstraction for Reinforcement Learning** (arXiv:2410.00704). It learns temporally correlated state representations via InfoNCE contrastive loss, then clusters them into abstract states using a learned beta network + continuous Hopfield network. The main benchmark is the D4RL `pointmaze-large` dataset from Minari.

## Environment Setup

```bash
conda env create -f env.yml
conda activate <env_name>
# or
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `minari`, `hopfield-layers` (`hflayers`), `gymnasium`, `wandb`.

## Training Commands

**Stage 1 — Train contrastive learning (CL) model:**
```bash
python src/main_cl.py --data_path /path/to/trajectories.npz --distribution l
# distributions: l=Laplace, g=Gaussian, e=Exponential, u=Uniform
# saves checkpoint to cl_model/<filename>.ckpt
```

**Stage 2 — Train beta model (requires pre-trained CL model):**
```bash
python src/main_beta.py
# loads trained_models/laplace_cos_sim-v1.ckpt by default
# saves checkpoint to beta_models/<filename>.ckpt
```

**Visualization:**
```bash
python -m src.visuals.visuals --distribution l --subsample_size 10000 --total_states 1000000
```

## Running Tests

```bash
python tests/fourroom/test_fourroom.py
```

This is the main integration test — it generates a synthetic 9×9 four-room grid environment, trains both models end-to-end, and outputs PCA/UMAP plots to `tests/fourroom/plots/` showing room-level state clustering.

## Architecture

### Two-Stage Training Pipeline

```
Trajectories → Sampler (anchor-positive pairs) → DatasetCL → mlpCL (InfoNCE)
                                                                     ↓
                                                          32-dim CL embedding z
                                                                     ↓
StatesDataset (embeds raw states via frozen CL) → LearnedBetaModel
    ├── Beta network: z → β ∈ [0,1] (MLP, sigmoid)
    ├── Scaled query: z_scaled = z × β
    └── Hopfield network: z_scaled → abstract state u
```

### Key Components

**`src/models/cl_model.py` — `mlpCL`**: 4-layer MLP (256→128→64→32) trained with InfoNCE. Temperature-scaled cosine similarity over anchor-positive pairs sampled from the same trajectory. Outputs L2-normalized 32-dim embeddings.

**`src/models/beta_model.py` — `LearnedBetaModel`**: Beta network gates CL embeddings before feeding them as queries to the Hopfield network. The Hopfield network uses trajectory CL embeddings as stored patterns and retrieves abstract states `u`. Trained with `ContrastiveHopfieldObjective` (InfoNCE between `u` and dropout-augmented `z`).

**`src/models/beta_objectives.py`**: Separates the training objective from the model. `ContrastiveHopfieldObjective` computes InfoNCE with a shared projection head between the Hopfield-retrieved abstract states and masked-dropout augmented CL embeddings.

**`src/data/Sampler.py`**: Core data construction. Samples anchor states uniformly from trajectories, then samples positive pairs by drawing from a distribution (Laplace/Gaussian/Exponential/Uniform) centered at the anchor's trajectory index. `add_action=True` concatenates action vectors to states.

**`src/data/TrajectorySet.py`**: Wraps Minari datasets or `.npz` files into a uniform trajectory interface. `StatesDataset` pre-encodes all states through a trained CL model for beta training.

### Checkpoints & Pre-trained Models

Pre-trained CL checkpoints in `trained_models/` for all four distributions. Load via:
```python
mlpCL.load_from_checkpoint("trained_models/laplace_cos_sim-v1.ckpt")
LearnedBetaModel.load_from_checkpoint("trained_models/beta_model.ckpt")
```

## Preferences

- Prioritize readability and navigable code organization over feature richness
- Before adding new features or keeping existing ones that add complexity, ask whether they can be removed

## Experiment Tracking

Training logs to Weights & Biases project `"Contrastive Learning RL"` automatically. Pass `--filename` to name checkpoints and runs.
