# Torch 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

# PyTorch Lightning 
import pytorch_lightning as pl

class mlpCL(pl.LightningModule): 
    def __init__(self, lr, weight_decay, temperature=30, max_epochs=1000, h1=256, h2=128, h3=64, h4=32, device = "cpu"):
        super().__init__() # inherit from LightningModule and nn.module 
        self.save_hyperparameters() # save args  
        self.device_type = device 

        self.mlp = nn.Sequential(
            nn.Linear(4, h1), 
            nn.ReLU(inplace=True), 

            nn.Linear(h1, h2), 
            nn.ReLU(inplace=True),

            nn.Linear(h2, h3), 
            nn.ReLU(inplace=True),

            nn.Linear(h3, h4), # representation z 
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), 
                                lr= self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return ([optimizer], [lr_scheduler])
    

    def info_nce_loss(self, batch, mode="train"):
        # batch is of shape: [N, D]
        x = torch.cat(batch, dim=0)  # shape: [2N, D]

        z = F.normalize(self.mlp(x), dim=1)  # [2N, h4]
        N = z.size(0) // 2

        sim = torch.matmul(z, z.T) / self.hparams.temperature  # cosine sim matrix [2N, 2N]

        # mask out self similarities
        mask = torch.eye(2 * N, device=sim.device).bool()
        sim = sim.masked_fill_(mask, -9e15)

        # positives: i-th sample matches i + N mod 2N
        pos_idx = (torch.arange(2 * N, device=sim.device) + N) % (2 * N)
        labels = pos_idx

        loss = F.cross_entropy(sim, labels)

        # metrics
        preds = sim.argmax(dim=1)
        top1 = (preds == labels).float().mean()   # top1: true positive is most similar to anchor 
        top5 = (sim.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean() # top5: true positive is atleast in the top 5 most similar to anchor 

        self.log(f"{mode}/nll_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/top1", top1, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/top5", top5, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch):
        self.info_nce_loss(batch, mode='val')

    @torch.no_grad() 
    def get_embeddings(self, dataloader):
        self.eval() 
        all_z = [] 
        for batch in dataloader: 
            x = torch.cat(batch, dim=0).to(device=self.device_type)
            z = self.mlp(x)
            z = F.normalize(z, dim=1)
            all_z.append(z)
        return torch.cat(all_z, dim=0)
    
