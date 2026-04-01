# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Contrastive Abstraction for Reinforcement Learning** (arXiv:2410.00704). It learns temporally correlated state representations via InfoNCE contrastive loss, then clusters them into abstract states using a learned beta network + continuous Hopfield network. The main benchmark is OGBench.

## Environment Setup

```bash
uv sync
```

Key dependencies: `torch`, `pytorch-lightning`, `hopfield-layers` (`hflayers`), `gymnasium`, `wandb`, `ogbench`.

## Training Commands

**Stage 1 — Train contrastive learning (CL) model:**
```bash
# From a .npz file:
python src/train_cl.py --data_path /path/to/trajectories.npz --distribution l
# From an OGBench dataset:
python src/train_cl.py --og_dataset_name antmaze-large-navigate-v0 --distribution l
# distributions: l=Laplace, g=Gaussian, e=Exponential, u=Uniform
# saves checkpoint to cl_model/<filename>.ckpt
```

**Stage 2 — Train beta model (requires pre-trained CL model):**
```bash
python src/train_beta.py
# loads trained_models/laplace_cos_sim-v1.ckpt by default
# saves checkpoint to beta_models/<filename>.ckpt
```

**Visualization:**
```bash
python -m src.visualize \
  --cl_model_path cl_model/cl_model.ckpt \
  --beta_model_path beta_models/beta_model.ckpt \
  --subsample_size 10000 --total_states 1000000
```

## Running Tests

```bash
python tests/fourroom/test_fourroom.py
```

This is the main integration test — it generates a synthetic 9×9 four-room grid environment, trains both models end-to-end, and outputs PCA/UMAP plots to `tests/fourroom/plots/` showing room-level state clustering.

## Architecture

### Two-Stage Training Pipeline

```
Trajectories → Sampler (anchor-positive pairs) → cl_dataset → mlpCL (InfoNCE)
                                                                     ↓
                                                              CL embedding z
                                                                     ↓
latent_dataset (embeds raw states via frozen CL) → LearnedBetaModel
    ├── Beta network: z → β ∈ [0,1] (MLP, sigmoid)
    ├── Scaled query: z_scaled = z × β
    └── Hopfield network: z_scaled → abstract state u
```

### Key Components

**`src/models/cl_model.py` — `mlpCL`**: MLP trained with InfoNCE. Temperature-scaled cosine similarity over anchor-positive pairs sampled from the same trajectory. Outputs L2-normalized embeddings.

**`src/models/beta_model.py` — `LearnedBetaModel`**: Beta network gates CL embeddings before feeding them as queries to the Hopfield network. The Hopfield network uses ou peCL embeddings as stored patterns and retrieves abstract states `u`. Trained with `ContrastiveHopfieldObjective` (InfoNCE between `u` and dropout-augmented `z`).

**`src/models/beta_objective.py`**: Separates the training objective from the model. `ContrastiveHopfieldObjective` computes InfoNCE with a shared projection head between the Hopfield-retrieved abstract states and masked-dropout augmented CL embeddings.

**`src/data/sampler.py`**: Core data construction. Samples anchor states uniformly from trajectories, then samples positive pairs by drawing from a distribution (Laplace/Gaussian/Exponential/Uniform) centered at the anchor's trajectory index. `add_action=True` concatenates action vectors to states.

**`src/data/trajectories.py`**: Defines `TrajectorySet` (base class) and `SyntheticTrajectorySet` (holds pre-loaded data). `latent_dataset.py` pre-encodes all states through a trained CL model for beta training.

### Checkpoints & Pre-trained Models

Pre-trained CL checkpoints in `trained_models/` for all four distributions. Load via:
```python
mlpCL.load_from_checkpoint("trained_models/laplace_cos_sim-v1.ckpt")
LearnedBetaModel.load_from_checkpoint("trained_models/beta_model.ckpt")
```

## Preferences

- Prioritize readability and logical code organization 
- Prefer grouping similar functions and classes into one script

## Experiment Tracking

Training logs to Weights & Biases project `"Contrastive Learning RL"` automatically. Pass `--filename` to name checkpoints and runs.
