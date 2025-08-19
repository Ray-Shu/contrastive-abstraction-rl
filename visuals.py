import os 
import sys 
import argparse

import matplotlib.pyplot as plt 
import torch  
import minari 
import numpy as np
import faiss

from utils import sampling_states
from utils import pca
from utils.remove_dupes import remove_dupes
from utils import visualizations

from models.cl_model import mlpCL
from models.cmhn import cmhn
from models.beta_model import LearnedBetaModel

from data.TrajectorySet import TrajectorySet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

MINARI_DATASET = minari.load_dataset("D4RL/pointmaze/large-v2")
PROJECT_ROOT = os.getcwd()
FOLDER_PATH = "test_plots"
DEVICE = 'cpu'
BETA_MODEL_NAME = "beta_model.ckpt"
MINARI_POINTMAZE_LARGE_MAP = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
     [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], 
     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
     [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], 
     [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], 
     [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1], 
     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
)
ENV_HEIGHT, ENV_WIDTH = 480, 480 # this comes from env.shape when recreating the maze 2d environment.

# agent coordinate bounds from the minari pointmaze env
X_BOUND = (-5,5)
Y_BOUND = (-3.5, 3.5)

DEFAULT_CONFIG = {
    "distribution": "l", 
    "subsample_size": 10_000,
    "total_states": 1_000_000
}

def parse_args(): 
    parser = argparse.ArgumentParser(description="visualize")
    parser.add_argument("--distribution", type=str, default=DEFAULT_CONFIG["distribution"])
    parser.add_argument("--subsample_size", type=int, default=DEFAULT_CONFIG["subsample_size"])
    parser.add_argument("--total_states", type=int, default=DEFAULT_CONFIG["total_states"])
    return parser.parse_args()

def main(): 
    args = parse_args() 
    CONFIG = vars(args)
    
    # Load in contrastive model  
    model_name = ""
    if CONFIG["distribution"] == "l": 
        model_name = "laplace.ckpt"
    elif CONFIG["distribution"] == "g": 
        model_name = "gaussian.ckpt"
    elif CONFIG["distribution"] == "e":
        model_name = "exponential.ckpt"
    elif CONFIG["distribution"] == "u": 
        model_name = "uniform.ckpt" 

    pretrained_model_file = os.path.join(PROJECT_ROOT+ "/best_models", model_name) 

    if os.path.isfile(pretrained_model_file): 
        print(f"Found pretrained model at {pretrained_model_file}, loading...") 
        model = mlpCL.load_from_checkpoint(pretrained_model_file, map_location=torch.device(DEVICE))
    else: 
        print("Model not found...")

    # Get states from dataset
    print(f'Sampling {CONFIG["total_states"]} states.')
    states_dict = sampling_states.sample_states(MINARI_DATASET, CONFIG["total_states"],)
    states = states_dict["states"]

    # Transform to pca 
    pca_dict = pca.process_states(states, model)
    pca_states = pca_dict["pca-reps"]

    # Subsample states for visualization
    idx = np.random.choice(np.arange(CONFIG["total_states"]), size=CONFIG["subsample_size"], replace=False)
    subsampled_pca_states = pca_states[idx]

    # Plot learned representations
    plt.plot(figsize=(10,6))
    plt.scatter(x=subsampled_pca_states[:, 0], y=subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    plt.title("Learned Representations")
    plt.axis("off")
    # create and save image to folder path
    os.makedirs(name=FOLDER_PATH, exist_ok=True)
    file_path = os.path.join(FOLDER_PATH, "learned_representations.png")
    plt.savefig(file_path)
    plt.close()
    print("Image 1 processed succesfully.")

    # Overlay 2 trajectories onto the representation space
    trajs = sampling_states.sample_trajectories(MINARI_DATASET, n_episodes=2, ep_len=200)
    t1 = trajs[0][0].observations["observation"]
    t2 = trajs[1][0].observations["observation"]

    pca_t1 = pca.pca_transform(t1, pca_dict,  model=model, has_representation=False)
    pca_t2 = pca.pca_transform(t2, pca_dict,  model=model, has_representation=False)
    plt.plot(figsize=(10,6))
    plt.scatter(x=subsampled_pca_states[:, 0], y=subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    plt.scatter(pca_t1[:, 0], pca_t1[:, 1], s=1, c= "red")
    plt.scatter(pca_t2[:, 0], pca_t2[:, 1], s=1, c= "green")

    plt.title("Trajectories Overlaid on Representation Space")
    plt.axis("off")
    file_path = os.path.join(FOLDER_PATH, "traj_overlaid_on_reps.png")
    plt.savefig(file_path)
    plt.close()
    print("Image 2 processed succesfully.")

    #----------------------------------------------------------------------------------------------------------
    # Clustering the representations
    #----------------------------------------------------------------------------------------------------------
    mhn = cmhn(update_steps=1, device=DEVICE)

    # Get beta model
    pretrained_model_file = os.path.join(PROJECT_ROOT+ "/best_models", BETA_MODEL_NAME) 

    if os.path.isfile(pretrained_model_file): 
        print(f"Found pretrained model at {pretrained_model_file}, loading...") 
        beta_model = LearnedBetaModel.load_from_checkpoint(
            pretrained_model_file, map_location=torch.device(DEVICE),
            cmhn = mhn, 
            device=DEVICE,
        )

    # Input subsampled states to get learned representations 
    subsampled_states = states[idx]
    with torch.no_grad(): 
        z_reps = model(torch.as_tensor(subsampled_states, dtype=torch.float32))
        BETA = beta_model.get_beta(z_reps)
    
    # Get the u-values (output from hopfield network)
    u, u_norm = mhn.run(z_reps, z_reps, beta=BETA, run_as_batch=True)
    
    # Remove duplicate/similar u-values to obtain cluster points (fixed points)
    unique_mask = remove_dupes(u_norm, k=1000, threshold=0.5)
    unique_u = u[unique_mask]

    pca_u = pca.pca_transform(unique_u, pca_dict,  model=model, has_representation=True)
    plt.plot(figsize=(10,6))
    plt.scatter(x=subsampled_pca_states[:, 0], y=subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    plt.scatter(pca_u[:, 0], pca_u[:, 1], s=5, c= "red")
    plt.title("Cluster Points Overlaid on Representation Space")
    plt.axis("off")
    file_path = os.path.join(FOLDER_PATH, "cluster_pts_overlaid_on_reps.png")
    plt.savefig(file_path)
    plt.close()
    print("Image 3 processed succesfully.")

    #----------------------------------------------------------------------------------------------------------
    # Visualizing Clusters on the Real Maze Environment
    #----------------------------------------------------------------------------------------------------------

    # Matching unique_u values to their closest real state (using euclidean distance as the metric)
    min_dist = float("inf")
    mask = np.zeros(shape=(10_000), dtype=bool)

    for i in range(unique_u.size(0)):
        saved_idx = 0
        min_dist = float('inf')
        for j in range(len(z_reps)):
            euclidean_dist = np.linalg.norm(unique_u[i] - z_reps[j])
            if euclidean_dist < min_dist: 
                min_dist = euclidean_dist
                saved_idx = j
        mask[saved_idx] = True

    clustered_states = subsampled_states[mask]
    clustered_states.shape

    cluster_pts = clustered_states[:, :2]
    grid_points = visualizations.xy_to_grid(cluster_pts, X_BOUND, Y_BOUND, ENV_HEIGHT, ENV_WIDTH)
    rows, cols = zip(*grid_points)

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].scatter(subsampled_pca_states[:, 0], subsampled_pca_states[:, 1], s=1, c="lightblue", alpha=0.25)
    axs[0].scatter(pca_u[:, 0], pca_u[:, 1], s=8, c= "red", alpha=0.5)
    axs[0].set_title("Learned Representation Space")
    axs[0].axis("off")

    axs[1].imshow(MINARI_POINTMAZE_LARGE_MAP, cmap="gray_r", origin="upper",
            extent=[0, ENV_WIDTH, 0, ENV_HEIGHT])
    axs[1].plot(cols, rows, "ro")  # note cols = x-axis, rows = y-axis
    axs[1].set_title("Cluster Points Overlaid on Maze (top-down)")
    axs[1].axis("off")
    output_path = os.path.join(FOLDER_PATH, "representation_maze.png")
    fig.savefig(output_path)
    plt.close(fig)
    print("Image 4 processed succesfully.")
    

if __name__ == "__main__": 
    main()