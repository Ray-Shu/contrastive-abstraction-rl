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

from models.cl_model import mlpCL
from data.TrajectorySet import TrajectorySet


from models.cmhn import cmhn
from models.beta_model import LearnedBetaModel
from utils import visualizations

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

MINARI_DATASET = minari.load_dataset("D4RL/pointmaze/large-v2")
PROJECT_ROOT = os.getcwd()
FOLDER_PATH = "test_plots"
DEVICE = 'cpu'

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

if __name__ == "__main__": 
    main()