import os
import sys
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import ogbench
import numpy as np
import faiss

from src.utils import sampling
from src.utils import pca
from src.utils.remove_dupes import remove_dupes
from src.utils import checkpoint

from src.models.cl_model import mlpCL
from src.models.beta_model import LearnedBetaModel
from src.models.beta_objective import ContrastiveHopfieldObjective

# Resolving some weird faiss issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# Solves a faiss issue with macbooks
sys.modules['faiss.swigfaiss_avx2'] = faiss

PROJECT_ROOT = os.getcwd()
DEVICE = 'cpu'

DEFAULT_CONFIG = {
    "og_dataset_name": "antmaze-large-navigate-v0",
    "distribution": "l",
    "subsample_size": 10_000,
    "total_states": 1_000_000,
    "cl_model_path": None,
    "beta_model_path": None,
    "output_dir": "test_plots",
}

def parse_args():
    parser = argparse.ArgumentParser(description="visualize")
    parser.add_argument("--og_dataset_name", type=str, default=DEFAULT_CONFIG["og_dataset_name"])
    parser.add_argument("--distribution", type=str, default=DEFAULT_CONFIG["distribution"])
    parser.add_argument("--subsample_size", type=int, default=DEFAULT_CONFIG["subsample_size"])
    parser.add_argument("--total_states", type=int, default=DEFAULT_CONFIG["total_states"])
    parser.add_argument("--cl_model_path", type=str, default=None,
                        help="Path to CL checkpoint. Overrides --distribution lookup in trained_models/.")
    parser.add_argument("--beta_model_path", type=str, default=None,
                        help="Path to beta model checkpoint.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directory to save plots (default: test_plots/)")
    return parser.parse_args()

def main():
    args = parse_args()
    CONFIG = vars(args)

    output_dir = CONFIG["output_dir"]
    os.makedirs(name=output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load CL model
    # ------------------------------------------------------------------
    if CONFIG["cl_model_path"]:
        cl_model_file = CONFIG["cl_model_path"]
    else:
        dist_to_name = {
            "l": "laplace_cos_sim-v1.ckpt",
            "g": "gaussian_resaved.ckpt",
            "e": "exponential_resaved.ckpt",
            "u": "uniform_resaved.ckpt",
        }
        model_name = dist_to_name.get(CONFIG["distribution"], "laplace_cos_sim-v1.ckpt")
        cl_model_file = os.path.join(PROJECT_ROOT, "trained_models", model_name)

    if not os.path.isfile(cl_model_file):
        raise FileNotFoundError(f"CL model not found at {cl_model_file}")
    cl_model = mlpCL.load_from_checkpoint(cl_model_file, map_location="cpu")
    cl_model.eval()

    # ------------------------------------------------------------------
    # Load OGBench dataset
    # ------------------------------------------------------------------
    _, og_dataset, _ = ogbench.make_env_and_datasets(CONFIG["og_dataset_name"])

    # Get states from dataset
    print(f'Sampling {CONFIG["total_states"]} states.')
    states_dict = sampling.sample_states(og_dataset, CONFIG["total_states"])
    states = states_dict["states"]

    # Transform to pca
    pca_dict = pca.process_states(states, cl_model)
    pca_states = pca_dict["pca-reps"]

    # Subsample states for visualization
    subsample_size = min(CONFIG["subsample_size"], len(states))
    idx = np.random.choice(np.arange(len(states)), size=subsample_size, replace=False)
    subsampled_pca_states = pca_states[idx]

    # ------------------------------------------------------------------
    # Plot 1: Learned representations
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(x=subsampled_pca_states[:, 0], y=subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    plt.title("Learned Representations")
    plt.axis("off")
    file_path = os.path.join(output_dir, "learned_representations.png")
    plt.savefig(file_path)
    plt.close()
    print("Image 1 processed successfully.")

    # ------------------------------------------------------------------
    # Plot 2: Trajectories overlaid on representation space
    # ------------------------------------------------------------------
    trajs = sampling.sample_trajectories(og_dataset, n_episodes=2, ep_len=10_000)
    t1 = trajs[0]
    t2 = trajs[1]

    pca_t1 = pca.pca_transform(t1, pca_dict, model=cl_model, has_representation=False)
    pca_t2 = pca.pca_transform(t2, pca_dict, model=cl_model, has_representation=False)
    plt.figure(figsize=(10, 6))
    plt.scatter(x=subsampled_pca_states[:, 0], y=subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    plt.scatter(pca_t1[:, 0], pca_t1[:, 1], s=1, c="red")
    plt.scatter(pca_t2[:, 0], pca_t2[:, 1], s=1, c="green")
    plt.title("Trajectories Overlaid on Representation Space")
    plt.axis("off")
    file_path = os.path.join(output_dir, "traj_overlaid_on_reps.png")
    plt.savefig(file_path)
    plt.close()
    print("Image 2 processed successfully.")

    # ------------------------------------------------------------------
    # Load beta model
    # ------------------------------------------------------------------
    if CONFIG["beta_model_path"] is None:
        raise ValueError("--beta_model_path is required. Train one first with train_beta.py.")

    beta_model_file = CONFIG["beta_model_path"]
    if not os.path.isfile(beta_model_file):
        raise FileNotFoundError(f"Beta model not found at {beta_model_file}")

    objective = ContrastiveHopfieldObjective()
    beta_model = LearnedBetaModel(objective=objective)
    checkpoint.load_lightning_checkpoint(beta_model, beta_model_file)
    beta_model.eval()

    # ------------------------------------------------------------------
    # Plot 3: Cluster points overlaid on representation space
    # ------------------------------------------------------------------
    subsampled_states = states[idx]
    with torch.no_grad():
        z = cl_model(torch.as_tensor(subsampled_states, dtype=torch.float32))
        z_norm = F.normalize(z, p=2, dim=-1)
        BETA = beta_model.get_beta(z_norm)
        u = beta_model.hopfield((
            z_norm.unsqueeze(0),
            (z_norm * BETA).unsqueeze(0),
            z_norm.unsqueeze(0),
        )).squeeze(0)
        u_norm = F.normalize(u, p=2, dim=-1).cpu().numpy()

    k_dupes = min(1000, len(u_norm) - 1)
    unique_mask = remove_dupes(u_norm, k=k_dupes, threshold=0.5)
    unique_u_norm = u_norm[unique_mask]

    pca_u = pca.pca_transform(
        torch.as_tensor(unique_u_norm, dtype=torch.float32),
        pca_dict, model=None, has_representation=True,
    )
    plt.figure(figsize=(10, 6))
    plt.scatter(x=subsampled_pca_states[:, 0], y=subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    plt.scatter(pca_u[:, 0], pca_u[:, 1], s=5, c="red")
    plt.title("Cluster Points Overlaid on Representation Space")
    plt.axis("off")
    file_path = os.path.join(output_dir, "cluster_pts_overlaid_on_reps.png")
    plt.savefig(file_path)
    plt.close()
    print("Image 3 processed successfully.")

if __name__ == "__main__":
    main()
