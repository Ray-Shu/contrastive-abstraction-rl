import os 
import sys 
import argparse

import minari
import torch 
import torch.utils.data as data
import faiss

from models.cl_model import mlpCL 
from models.cmhn import cmhn 
from models.beta_model import LearnedBetaModel

from data.StatesDataset import StatesDataset

from trainer.beta_trainer import train_beta_model

from utils.sampling_states import sample_states 
from utils.tensor_utils import split_data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Ensures that the jupyter kernel doesn't crash when running chn calculations with faiss
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# Globals
MINARI_DATASET = minari.load_dataset("D4RL/pointmaze/large-v2")
PROJECT_ROOT = os.getcwd()
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models")

PROJECT_NAME = "Learning Beta Model"
RUN_NAME = "beta_model"
FILENAME = RUN_NAME
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIG = {
        "num_states": 1_000_000,  
        "lr": 1e-3,
        "temperature": 1, 
        "weight_decay": 1e-5, 
        "masking_ratio": 0.3,
        "beta_max": 200,
        "max_epochs": 100,
        "filename": FILENAME,
        "device": DEVICE,
        "minibatch": 4096, 
        "cl_model_distribution": "l"
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Beta Model")
    parser.add_argument("--num_states", type=int, default=DEFAULT_CONFIG["num_states"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--masking_ratio", type=float, default=DEFAULT_CONFIG["masking_ratio"])
    parser.add_argument("--beta_max", type=float, default=DEFAULT_CONFIG["beta_max"])
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--filename", type=str, default=DEFAULT_CONFIG["filename"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--minibatch", type=int, default=DEFAULT_CONFIG["minibatch"])
    parser.add_argument("--cl_model_distribution", type=str, default=DEFAULT_CONFIG["cl_model_distribution"])

    return parser.parse_args()

def main(): 
    args = parse_args()
    CONFIG = vars(args)

    # Load cmhn model 
    mhn = cmhn(update_steps=1, device=DEVICE)

    # Load trained CL model 
    model_name = ""
    if CONFIG['cl_model_distribution'] == "l": 
        model_name = "laplace.ckpt"
    elif CONFIG['cl_model_distribution'] == "g": 
        model_name = "gaussian.ckpt"
    elif CONFIG["cl_model_distribution"] == "e":
        model_name = "exponential.ckpt"
    elif CONFIG["cl_model_distribution"] == "u": 
        model_name= "uniform.ckpt"

    pretrained_model_file = os.path.join(PROJECT_ROOT+ "/best_models", model_name) 

    if os.path.isfile(pretrained_model_file): 
        print(f"Found pretrained model at {pretrained_model_file}, loading...") 
        cl_model = mlpCL.load_from_checkpoint(pretrained_model_file, map_location=torch.device(DEVICE))
    else:
        print("Model not found...")

    # Preprocessing step to get train/val data
    print(f"Sampling {CONFIG["num_states"]} states...")
    data = sample_states(dataset=MINARI_DATASET, num_states=CONFIG["num_states"])
    states = data["states"]
    train, val = split_data(states, split_val=0.8) 
    train_ds = StatesDataset(cl_model=cl_model, minari_dataset=MINARI_DATASET, data=train)
    val_ds = StatesDataset(cl_model=cl_model, minari_dataset=MINARI_DATASET, data=val)
    print("Sampling finished!")

    wandb_logger = WandbLogger(
            project=PROJECT_NAME, 
            name=RUN_NAME, 
            save_dir = PROJECT_ROOT, 
            log_model=True,
            config = CONFIG) 

    model = train_beta_model(
        bm_model=LearnedBetaModel,
        cmhn=mhn, 
        train_ds=train_ds,
        val_ds = val_ds,
        batch_size=CONFIG["minibatch"], 
        logger=wandb_logger, 
        checkpoint_path=CHECKPOINT_PATH, 
        max_epochs=CONFIG["max_epochs"],
        device=CONFIG["device"], 
        filename= FILENAME,

        # kwaargs
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"], 
        masking_ratio=CONFIG["masking_ratio"], 
        beta_max=CONFIG["beta_max"],
        temperature=CONFIG["temperature"]
    )

if __name__ == "__main__": 
    main()