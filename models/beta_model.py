import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import pytorch_lightning as pl


class LearnedBetaModel(pl.LightningModule): 
    def __init__(self, cmhn, beta_max, lr=1e-3, weight_decay=1e-5, masking_ratio=0.3, max_epochs=1000, input_dim=32, h1=128, h2=32, fc_h1 = 64, device="cpu"):
        super().__init__() 
        self.save_hyperparameters()
        self.cmhn = cmhn 
        self.device_type = torch.device(device=device)

        self.dropout = nn.Dropout(p=masking_ratio, inplace=False)

        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(), 

            nn.Linear(h1, h2), 
            nn.ReLU(),

            nn.Linear(h2, 1),
            nn.Sigmoid() 
        ).to(self.device_type)

        self.fc_nn = nn.Sequential( 
            nn.Linear(input_dim, fc_h1),
            nn.ReLU(), 
            nn.Linear(fc_h1, input_dim)
        ).to(self.device_type)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), 
                                lr= self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return ([optimizer], [lr_scheduler])

    def loss(self, batch, mode="train"): 
        """
        The loss function for the beta network. 

        Args: 
            batch: The batch data that the beta network will use (z representations). 
        
        Returns: 
            loss: The infoNCE loss. 
        """
        batch = batch.to(self.device_type)

        # get the trial beta 
        beta = self.beta_net(batch)

        # get abstract representation 'u' 
        U = self.cmhn.run(batch, batch, beta, run_as_batch=True) 

        # get the noisy batch, nn.Dropout uses scaling=True to maintain expected value of tensor
        z_prime = self.dropout(batch)

        # create positive pairs
        pairs = torch.cat([U, z_prime], dim=0)
      
        # put new batch pairs into fc_nn to obtain vectors in new embedding space useful for contrastive learning 
        p = self.fc_nn(pairs)


        ######################################################################
        #     use p for contrastive loss 
        ######################################################################

        N = p.size(0) // 2

        # normalize vector embedding
        p = F.normalize(p, dim=1)

        sim = torch.matmul(p, p.T) # cosine sim matrix [2N, 2N]
        #print("sim: ", sim)

        # mask diagonals to large negative numbers so we don't calculate same state similarities
        mask = torch.eye(2 * N, device=sim.device).bool()
        sim = sim.masked_fill_(mask, -9e15)

        # positives: i-th sample matches i + N mod 2N
        labels = (torch.arange(2 * N, device=sim.device) + N) % (2 * N)

        loss = F.cross_entropy(sim, labels) # over mean reduction 

        # extra statistics 
        if mode=="train": 
            with torch.no_grad(): 
                norms = torch.norm(p, dim=1)
                self.log(f"{mode}/sim_mean", sim.mean(), on_epoch=True)
                self.log(f"{mode}/sim_std", sim.std(), on_epoch=True)
                self.log(f"{mode}/p_norm_mean", norms.mean(), on_epoch=True)
                self.log(f"{mode}/p_norm_std", norms.std(), on_epoch=True)

        # metrics
        preds = sim.argmax(dim=1)
        top1 = (preds == labels).float().mean()   # top1: true positive is most similar to anchor 
        top5 = (sim.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean() # top5: true positive is atleast in the top 5 most similar to anchor 

        self.log(f"{mode}/nll_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/top1", top1, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/top5", top5, on_epoch=True, prog_bar=True)

        return loss
    
    def training_step(self, batch):
        return self.loss(batch, mode='train')

    def validation_step(self, batch):
        self.loss(batch, mode='val')

    def debugging(self): 
        for name, param in self.beta_net.named_parameters():
            print(f"{name} device:", param.device)