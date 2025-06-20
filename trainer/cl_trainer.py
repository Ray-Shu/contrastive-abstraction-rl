# Torch 
import torch 
import torch.utils.data as data 

# PyTorch Lightning 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def train_cl(cl_model, train_ds, val_ds, batch_size, logger, checkpoint_path, max_epochs=1000, device="cpu", filename= "best_model", **kwargs):
    # Create model checkpoints based on the top5 metric
    filename = kwargs.pop("filename", filename) 
    
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                      filename=filename, 
                                      save_top_k=3, 
                                      save_weights_only=True, 
                                      mode="max",
                                      monitor="val/top5")
    
    trainer = pl.Trainer(
        default_root_dir=checkpoint_path, 
        logger = logger,
        accelerator= "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu", 
        devices=1, 
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback,
                   LearningRateMonitor("epoch")]) # creates a model checkpoint when a new max in val/top5 has been reached 
    train_loader = data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(dataset= val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    pl.seed_everything(10)
    model = cl_model(max_epochs=max_epochs, device=device, **kwargs) 
    trainer.fit(model, train_loader, val_loader)

    print("Best model path:", checkpoint_callback.best_model_path)
    model = cl_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    return model 