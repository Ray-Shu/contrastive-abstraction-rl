import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from hflayers import Hopfield
from src.models.beta_objective import BetaObjective


class LearnedBetaModel(pl.LightningModule):
    def __init__(self, objective: BetaObjective,
                 hopfield_scale=500.0, hopfield_steps_max=10, hopfield_steps_eps=1e-6,
                 lr=1e-3, weight_decay=1e-5, max_epochs=1000,
                 input_dim=32, h1=128, h2=32, device="cpu"):
        super().__init__()
        self.save_hyperparameters(ignore=['objective'])
        self.device_type = torch.device(device=device)

        self.objective = objective

        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),

            nn.Linear(h1, h2),
            nn.ReLU(),

            nn.Linear(h2, 1),
            nn.Sigmoid()
        ).to(self.device_type)

        self.hopfield = Hopfield(
            input_size=input_dim,
            scaling=hopfield_scale,
            update_steps_max=hopfield_steps_max,
            update_steps_eps=hopfield_steps_eps,
            normalize_hopfield_space=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            state_pattern_as_static=False,
            pattern_projection_as_static=True,
            stored_pattern_as_static=True,
            disable_out_projection=True,
            num_heads=1,
            dropout=False,
        ).to(self.device_type)

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return ([optimizer], [lr_scheduler])

    def loss(self, batch, mode="train"):
        batch_norm = F.normalize(batch, p=2, dim=-1)

        beta = self.beta_net(batch_norm)  # [N, 1], range [0, 1]

        loss, metrics = self.objective.loss(batch_norm, self.hopfield, beta)

        prog_bar_keys = {"nll_loss", "top1", "top5"}
        for k, v in metrics.items():
            self.log(f"{mode}/{k}", v, on_epoch=True, prog_bar=(k in prog_bar_keys))

        # beta-specific logging (lives here since beta_net belongs to this model)
        if mode == "train":
            with torch.no_grad():
                self.log(f"{mode}/beta_mean", beta.mean(), on_epoch=True)
                self.log(f"{mode}/beta_max", beta.max(), on_epoch=True)
                self.log(f"{mode}/beta_min", beta.min(), on_epoch=True)

        return loss

    def training_step(self, batch):
        return self.loss(batch, mode='train')

    def validation_step(self, batch):
        self.loss(batch, mode='val')

    def get_beta(self, batch):
        """Returns the beta value in [0, 1]. Effective Hopfield temperature = hopfield_scale * beta."""
        return self.beta_net(batch)
