"""
End-to-end integration test for the OGBench pipeline.

Loads a small slice of antmaze-medium-navigate-v0, trains both models,
and writes plots to tests/ogbench/plots/.

Run from the project root:
    python tests/ogbench/test_ogbench.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger

import ogbench

from src.data.sampler import Sampler
from src.data.cl_dataset import DatasetCL
from src.data.latent_dataset import StatesDataset
from src.models.cl_model import mlpCL
from src.models.beta_model import LearnedBetaModel
from src.models.beta_objective import ContrastiveHopfieldObjective
from src.trainers.cl_trainer import train_cl
from src.trainers.beta_trainer import train_beta_model
from src.utils.trajectory_io import ogbench_to_trajectory_set
from src.utils.sampling import sample_states, sample_trajectories
from src.utils.tensor import split_data
from src.utils import pca
from src.utils.remove_dupes import remove_dupes

# =============================================================================
# CONFIG
# =============================================================================

DATASET_NAME = "antmaze-medium-navigate-v0"

# Steps to keep for CL trajectory data
CL_SUBSET_SIZE = 5_000
# States for beta training and visualization
NUM_STATES = 2_000

# CL model
CL_EPOCHS       = 3
CL_BATCH        = 256
CL_LR           = 1e-3
CL_WEIGHT_DECAY = 1e-5
CL_TEMPERATURE  = 30
CL_TRAIN_PAIRS  = 1_600
CL_VAL_PAIRS    = 400

# Beta model
BETA_EPOCHS         = 3
BETA_BATCH          = 256
BETA_LR             = 1e-3
BETA_WEIGHT_DECAY   = 1e-5
BETA_TEMPERATURE    = 0.03796
BETA_MASKING_RATIO  = 0.3
BETA_HOPFIELD_SCALE = 500.0

# =============================================================================


def make_subset(dataset: dict, n: int) -> dict:
    """Slice dataset to first n steps."""
    return {k: v[:n] for k, v in dataset.items()}


def train_beta(cl_model, states, checkpoint_path, logger, device):
    """Train LearnedBetaModel on CL latents from the given states array."""
    objective = ContrastiveHopfieldObjective(
        temperature=BETA_TEMPERATURE,
        masking_ratio=BETA_MASKING_RATIO,
    )

    train_states, val_states = split_data(states, split_val=0.8)
    train_ds = StatesDataset(cl_model=cl_model, data=train_states)
    val_ds   = StatesDataset(cl_model=cl_model, data=val_states)

    return train_beta_model(
        bm_model=LearnedBetaModel,
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=BETA_BATCH,
        logger=logger,
        checkpoint_path=checkpoint_path,
        max_epochs=BETA_EPOCHS,
        device=device,
        filename="best_beta_ogbench",

        # kwargs forwarded to LearnedBetaModel
        objective=objective,
        lr=BETA_LR,
        weight_decay=BETA_WEIGHT_DECAY,
        hopfield_scale=BETA_HOPFIELD_SCALE,
    )


def visualize(cl_model, beta_model, states, og_dataset, plots_dir):
    """Generate three plots: representations, trajectory overlay, cluster points."""
    cl_model.eval()
    beta_model.eval()

    # Fit PCA on all sampled states
    pca_dict = pca.process_states(states, cl_model)
    pca_states = pca_dict["pca-reps"]

    subsample_size = min(500, len(states))
    idx = np.random.choice(len(states), size=subsample_size, replace=False)
    sub_pca = pca_states[idx]

    # -- Plot 1: learned representations --
    plt.figure(figsize=(8, 6))
    plt.scatter(sub_pca[:, 0], sub_pca[:, 1], s=1, c="lightblue", alpha=0.4)
    plt.title("Learned Representations (OGBench)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "learned_representations.png"), dpi=150)
    plt.close()
    print("Saved: learned_representations.png")

    # -- Plot 2: trajectories overlaid on representations --
    trajs = sample_trajectories(og_dataset, n_episodes=2)
    pca_t1 = pca.pca_transform(trajs[0], pca_dict, model=cl_model, has_representation=False)
    pca_t2 = pca.pca_transform(trajs[1], pca_dict, model=cl_model, has_representation=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(sub_pca[:, 0], sub_pca[:, 1], s=1, c="lightblue", alpha=0.4)
    plt.scatter(pca_t1[:50, 0], pca_t1[:50, 1], s=2, c="red",   label="traj 1")
    plt.scatter(pca_t2[:50, 0], pca_t2[:50, 1], s=2, c="green", label="traj 2")
    plt.title("Trajectories Overlaid on Representations (OGBench)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "traj_overlaid_on_reps.png"), dpi=150)
    plt.close()
    print("Saved: traj_overlaid_on_reps.png")

    # -- Plot 3: cluster points from beta/Hopfield --
    sub_states = states[idx]
    with torch.no_grad():  #TODO: why not use src/data/latent_dataset here?
        z      = cl_model(torch.as_tensor(sub_states, dtype=torch.float32))
        z_norm = F.normalize(z, p=2, dim=-1)
        beta   = beta_model.get_beta(z_norm)
        u = beta_model.hopfield((   # TODO: add a script called abstract_dataset.py in src/data to compute the u's given the z's from latent_dataset.py
            z_norm.unsqueeze(0),
            (z_norm * beta).unsqueeze(0),
            z_norm.unsqueeze(0),
        )).squeeze(0)
        u_norm = F.normalize(u, p=2, dim=-1).cpu().numpy()

    k = min(1000, len(u_norm) - 1)
    unique_u = u_norm[remove_dupes(u_norm, k=k, threshold=0.5)]
    pca_u = pca.pca_transform(
        torch.as_tensor(unique_u, dtype=torch.float32),
        pca_dict, model=None, has_representation=True,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(sub_pca[:, 0], sub_pca[:, 1], s=1, c="lightblue", alpha=0.4)
    plt.scatter(pca_u[:, 0], pca_u[:, 1], s=8, c="red", label="cluster pts")
    plt.title("Cluster Points Overlaid on Representations (OGBench)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cluster_pts_overlaid_on_reps.png"), dpi=150)
    plt.close()
    print("Saved: cluster_pts_overlaid_on_reps.png")


def main():
    TESTS_DIR       = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINTS_DIR = os.path.join(TESTS_DIR, "checkpoints")
    PLOTS_DIR       = os.path.join(TESTS_DIR, "plots")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,       exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Load dataset --------------------------------------------------------
    print(f"Loading {DATASET_NAME}...")
    _, og_dataset, _ = ogbench.make_env_and_datasets(DATASET_NAME)

    obs_dim = og_dataset["observations"].shape[1]
    print(f"  obs_dim={obs_dim}, total steps={len(og_dataset['observations'])}")

    # -- Stage 1: train CL model ---------------------------------------------
    cl_ckpt = os.path.join(CHECKPOINTS_DIR, "best_cl_ogbench.ckpt")
    if os.path.exists(cl_ckpt):
        print(f"Reusing CL checkpoint: {cl_ckpt}")
        cl_model = mlpCL.load_from_checkpoint(cl_ckpt, map_location=DEVICE)
    else:
        print(f"Training CL model ({CL_EPOCHS} epochs)...")
        subset  = make_subset(og_dataset, CL_SUBSET_SIZE)
        tset    = ogbench_to_trajectory_set(subset)
        sampler = Sampler(tset, dist="l", b=15, sigma=15, add_action=False)
        train_ds = DatasetCL(sampler, num_state_pairs=CL_TRAIN_PAIRS)
        val_ds   = DatasetCL(sampler, num_state_pairs=CL_VAL_PAIRS)
        cl_model = train_cl(
            cl_model=mlpCL,
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=CL_BATCH,
            logger=CSVLogger(save_dir=TESTS_DIR, name="cl_logs"),
            checkpoint_path=CHECKPOINTS_DIR,
            max_epochs=CL_EPOCHS,
            device=DEVICE,
            filename="best_cl_ogbench",
            input_dim=obs_dim,
            lr=CL_LR,
            temperature=CL_TEMPERATURE,
            weight_decay=CL_WEIGHT_DECAY,
        )
    cl_model = cl_model.to(DEVICE)

    # -- Stage 2: train beta model -------------------------------------------
    beta_ckpt = os.path.join(CHECKPOINTS_DIR, "best_beta_ogbench.ckpt")
    if os.path.exists(beta_ckpt):
        print(f"Reusing beta checkpoint: {beta_ckpt}")
        objective = ContrastiveHopfieldObjective(
            temperature=BETA_TEMPERATURE,
            masking_ratio=BETA_MASKING_RATIO,
        )
        beta_model = LearnedBetaModel.load_from_checkpoint(beta_ckpt, objective=objective)
    else:
        print(f"Training beta model ({BETA_EPOCHS} epochs)...")
        states_dict = sample_states(og_dataset, num_states=NUM_STATES)
        states      = states_dict["states"]
        beta_model  = train_beta(
            cl_model=cl_model,
            states=states,
            checkpoint_path=CHECKPOINTS_DIR,
            logger=CSVLogger(save_dir=TESTS_DIR, name="beta_logs"),
            device=DEVICE,
        )
    beta_model = beta_model.to(DEVICE)

    # -- Stage 3: visualize --------------------------------------------------
    print("Generating plots...")
    states_dict = sample_states(og_dataset, num_states=NUM_STATES)
    visualize(cl_model, beta_model, states_dict["states"], og_dataset, PLOTS_DIR)

    print("\nDone. Outputs in tests/ogbench/")


if __name__ == "__main__":
    main()
