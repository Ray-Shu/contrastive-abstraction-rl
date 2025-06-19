# Personal 
from data.TrajectorySet import TrajectorySet
from data.Sampler import Sampler 
from data.DatasetCL import DatasetCL 
from models.cl_model import mlpCL
from trainer.cl_trainer import train_cl

# Misc
import minari 
import os

# PyTorch Lightning 
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger

MINARI_DATASET = minari.load_dataset("D4RL/pointmaze/large-v2")
PROJECT_ROOT = os.getcwd() 

CONFIG = {
        "distribution": "g",
        "batch_size": 256,
        "k": 2,
        "lr": 5e-4,
        "weight_decay": 1e-4, 
        "temperature": 0.08,
        "max_epochs": 10
    }

def main(): 
    wandb_logger = WandbLogger(
        project="Contrastive Learning RL", 
        name="test-run-new-infoNCE-loss", 
        save_dir = PROJECT_ROOT, 
        log_model=True,
        config = CONFIG
    ) 

    dist = CONFIG["distribution"]
    batch_size = CONFIG["batch_size"]
    k = CONFIG["k"]
    lr = CONFIG["lr"]
    weight_decay = CONFIG["weight_decay"]
    temperature = CONFIG["temperature"]
    max_epochs = CONFIG["max_epochs"]

    T = TrajectorySet(dataset=MINARI_DATASET)
    S = Sampler(T, dist=dist)
    train_dataset = DatasetCL(S, batch_size=batch_size, k=k)

    val_dataset = DatasetCL(S, batch_size=batch_size, k=k)

    model = train_cl(cl_model=mlpCL, 
                    train_ds=train_dataset, 
                    val_ds=val_dataset, 
                    batch_size=batch_size,
                    logger=wandb_logger, 
                    checkpoint_path=PROJECT_ROOT,
                    max_epochs=max_epochs, 
                    lr=lr, 
                    temperature=temperature, 
                    weight_decay = weight_decay)
        
if __name__ == "__main__": 
    main() 