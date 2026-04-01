import os
import sys
import argparse

import ogbench
import torch
import torch.utils.data as data

from src.models.cl_model import mlpCL
from src.models.beta_model import LearnedBetaModel
from src.models.beta_objective import ContrastiveHopfieldObjective

from src.data.latent_dataset import StatesDataset

from src.trainers.beta_trainer import train_beta_model

from src.utils.sampling import sample_states
from src.utils.tensor import split_data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

PROJECT_ROOT = os.getcwd()

DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "beta_models")
DEFAULT_CL_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "laplace_cos_sim-v1.ckpt")

PROJECT_NAME = "Learning Beta Model"
RUN_NAME = "run"
FILENAME = RUN_NAME
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIG = {
        "og_dataset_name": "antmaze-large-navigate-v0",
        "num_states": 1_000_000,
        "lr": 1e-3,
        "temperature": 0.03796123348109251,
        "weight_decay": 1e-5,
        "masking_ratio": 0.3,
        "hopfield_scale": 500.0,
        "hopfield_steps_max": 10,
        "hopfield_steps_eps": 1e-6,
        "max_epochs": 100,
        "filename": FILENAME,
        "device": DEVICE,
        "minibatch": 4096,
        "cl_model_distribution": "l"
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Beta Model")
    parser.add_argument("--og_dataset_name", type=str, default=DEFAULT_CONFIG["og_dataset_name"])
    parser.add_argument("--num_states", type=int, default=DEFAULT_CONFIG["num_states"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--masking_ratio", type=float, default=DEFAULT_CONFIG["masking_ratio"])
    parser.add_argument("--hopfield_scale", type=float, default=DEFAULT_CONFIG["hopfield_scale"])
    parser.add_argument("--hopfield_steps_max", type=int, default=DEFAULT_CONFIG["hopfield_steps_max"])
    parser.add_argument("--hopfield_steps_eps", type=float, default=DEFAULT_CONFIG["hopfield_steps_eps"])
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--filename", type=str, default=DEFAULT_CONFIG["filename"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--minibatch", type=int, default=DEFAULT_CONFIG["minibatch"])
    parser.add_argument("--cl_model_distribution", type=str, default=DEFAULT_CONFIG["cl_model_distribution"])
    parser.add_argument("--cl_model_path", type=str, default=None, help="Path to pre-trained CL model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints (default: beta_models/)")

    return parser.parse_args()

def main():
    args = parse_args()
    CONFIG = vars(args)

    checkpoint_path = CONFIG["checkpoint_dir"] or DEFAULT_CHECKPOINT_DIR
    os.makedirs(checkpoint_path, exist_ok=True)

    # Load trained CL model
    cl_model_path = CONFIG["cl_model_path"] or DEFAULT_CL_MODEL_PATH
    if not os.path.isfile(cl_model_path):
        raise FileNotFoundError(f"CL model not found at {cl_model_path}. Train one first with train_cl.py.")
    print(f"Loading CL model from {cl_model_path}...")
    cl_model = mlpCL.load_from_checkpoint(cl_model_path, map_location=torch.device(DEVICE))

    # Load OGBench dataset
    _, og_dataset, _ = ogbench.make_env_and_datasets(CONFIG["og_dataset_name"])

    # Preprocessing step to get train/val data
    print(f'Sampling {CONFIG["num_states"]} states...')
    data = sample_states(dataset=og_dataset, num_states=CONFIG["num_states"])
    states = data["states"]
    train, val = split_data(states, split_val=0.8)
    train_ds = StatesDataset(cl_model=cl_model, data=train)
    val_ds = StatesDataset(cl_model=cl_model, data=val)
    print("Sampling finished!")

    wandb_logger = WandbLogger(
            project=PROJECT_NAME,
            name=RUN_NAME,
            save_dir=PROJECT_ROOT,
            log_model=True,
            config=CONFIG)

    objective = ContrastiveHopfieldObjective(
        temperature=CONFIG["temperature"],
        masking_ratio=CONFIG["masking_ratio"],
    )

    model = train_beta_model(
        bm_model=LearnedBetaModel,
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=CONFIG["minibatch"],
        logger=wandb_logger,
        checkpoint_path=checkpoint_path,
        max_epochs=CONFIG["max_epochs"],
        device=CONFIG["device"],
        filename=CONFIG["filename"],

        # kwargs
        objective=objective,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        hopfield_scale=CONFIG["hopfield_scale"],
        hopfield_steps_max=CONFIG["hopfield_steps_max"],
        hopfield_steps_eps=CONFIG["hopfield_steps_eps"],
    )

if __name__ == "__main__":
    main()
