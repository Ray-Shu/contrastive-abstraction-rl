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
    
    def info_nce_loss(self, batch, mode='train'):
        
        x = torch.cat(batch, dim=0)

        # encode states
        z = self.mlp(x)

        # cos sim matrix
        cos_sim = F.cosine_similarity(z[:,None,:], z[None,:,:], dim=-1)

        # mask and fill diag with big neg number
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # find positive pair -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

        # infoNCE loss 
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # logging loss
        self.log(f"{mode}/nll_loss", nll)
        # get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # girst position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log(f"{mode}/top1", (sim_argsort == 0).float().mean(), on_epoch=True, prog_bar=True) # correct pair
        self.log(f"{mode}/top5", (sim_argsort < 5).float().mean(), on_epoch=True, prog_bar=True)  # in the top 5 indices for most similar to anchor
        self.log(f"{mode}/mean_pos", 1+sim_argsort.float().mean(), on_epoch=True, prog_bar=True)  # average index position 

        return nll
    
    
    
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
    
