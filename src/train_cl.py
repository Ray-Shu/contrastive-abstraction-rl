# Personal
from data.sampler import Sampler
from data.cl_dataset import DatasetCL
from models.cl_model import mlpCL
from trainers.cl_trainer import train_cl
from utils.trajectory_io import load_trajectories, ogbench_to_trajectory_set

# Misc
import os
import argparse

# Torch
import torch

# PyTorch Lightning
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger

import ogbench

PROJECT_ROOT = os.getcwd()

DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "cl_model")

PROJECT_NAME = "Contrastive Learning RL"
RUN_NAME = "cl_model"
FILENAME = RUN_NAME

DEFAULT_CONFIG = {
        "distribution": "l",
        "num_states": 1_000_000,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "temperature": 30,
        "max_epochs": 1000,
        "filename": FILENAME,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "minibatch": 4096,
        "add_action": False,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Contrastive Learning")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_path", type=str, help="Path to a .npz trajectory dataset")
    group.add_argument("--og_dataset_name", type=str, help="OGBench dataset name (e.g. antmaze-large-navigate-v0)")
    parser.add_argument("--distribution", type=str, default=DEFAULT_CONFIG["distribution"])
    parser.add_argument("--num_states", type=int, default=DEFAULT_CONFIG["num_states"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--filename", type=str, default=DEFAULT_CONFIG["filename"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--minibatch", type=int, default=DEFAULT_CONFIG["minibatch"])
    parser.add_argument("--add_action", type=lambda x: x.lower() == "true", default=DEFAULT_CONFIG["add_action"])
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints (default: cl_model/)")
    return parser.parse_args()

def main():
    args = parse_args()
    CONFIG = vars(args)

    checkpoint_path = CONFIG["checkpoint_dir"] or DEFAULT_CHECKPOINT_DIR
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb_logger = WandbLogger(
            project=PROJECT_NAME,
            name=RUN_NAME,
            save_dir=PROJECT_ROOT,
            log_model=True,
            config=CONFIG)

    if CONFIG["data_path"] is not None:
        T = load_trajectories(CONFIG["data_path"])
    else:
        _, og_dataset, _ = ogbench.make_env_and_datasets(CONFIG["og_dataset_name"])
        T = ogbench_to_trajectory_set(og_dataset)

    first_traj = T.get_trajectory(0)[0]
    state_dim = first_traj["states"].shape[1]
    action_dim = first_traj["actions"].shape[1]
    input_dim = state_dim + action_dim if CONFIG["add_action"] else state_dim

    S = Sampler(T, dist=CONFIG["distribution"], b=15, sigma=15, add_action=CONFIG["add_action"])

    split_val = 0.8
    train_batch = int(round(CONFIG["num_states"] * split_val))
    val_batch = int(round(CONFIG["num_states"] * (1 - split_val)))

    print(f'Sampling {CONFIG["num_states"]} states...')
    train_dataset = DatasetCL(S, num_state_pairs=train_batch)
    val_dataset = DatasetCL(S, num_state_pairs=val_batch)
    print("Sampling finished!")

    model = train_cl(cl_model=mlpCL,
                train_ds=train_dataset,
                val_ds=val_dataset,
                batch_size=CONFIG["minibatch"],
                logger=wandb_logger,
                checkpoint_path=checkpoint_path,

                # kwargs
                max_epochs=CONFIG["max_epochs"],
                filename=CONFIG["filename"],
                device=CONFIG["device"],
                lr=CONFIG["lr"],
                temperature=CONFIG["temperature"],
                weight_decay=CONFIG["weight_decay"],
                input_dim=input_dim,
    )

if __name__ == "__main__":
    main()
