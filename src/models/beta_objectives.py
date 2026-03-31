from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaObjective(ABC, nn.Module):
    """
    Abstract base class for beta learning objectives.

    A BetaObjective defines how to turn Hopfield-abstracted representations
    into a training signal for the beta network.

    Subclasses must implement loss(), which receives:
        - batch_norm: L2-normalised z-representations [N, d]
        - hopfield:   the Hopfield nn.Module from LearnedBetaModel
        - beta:       per-sample beta values [N, 1], sigmoid output in [0, 1]

    Returns: (loss_scalar, metrics_dict)
        metrics_dict keys will be logged by LearnedBetaModel under "{mode}/".
    """

    @abstractmethod
    def loss(self, batch_norm: torch.Tensor, hopfield: nn.Module, beta: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pass


class ContrastiveHopfieldObjective(BetaObjective):
    """
    Current objective: InfoNCE between the Hopfield abstract state and a
    dropout-augmented version of the original z-representation.

    Positive pair for sample i:
        anchor   = fc_nn(F.normalize(hopfield_output_i))
        positive = fc_nn(dropout(z_norm_i))

    The fc_nn projects both into a shared embedding space; similarity is
    measured geometrically (cosine).
    """

    def __init__(self, input_dim=32, fc_h1=256, fc_h2=128, fc_h3=64,
                 temperature=1.0, masking_ratio=0.3):
        super().__init__()

        self.temperature = temperature

        self.dropout = nn.Dropout(p=masking_ratio, inplace=False)

        self.fc_nn = nn.Sequential(
            nn.Linear(input_dim, fc_h1),
            nn.ReLU(),

            nn.Linear(fc_h1, fc_h2),
            nn.ReLU(),

            nn.Linear(fc_h2, fc_h3),
            nn.ReLU(),

            nn.Linear(fc_h3, input_dim),
        )

    def loss(self, batch_norm, hopfield, beta):
        # scale L2-normalised queries by beta; hflayers iterates internally
        scaled_queries = batch_norm * beta  # [N, d]

        u = hopfield((
            batch_norm.unsqueeze(0),      # stored patterns [1, N, d]
            scaled_queries.unsqueeze(0),  # queries          [1, N, d]
            batch_norm.unsqueeze(0)       # values           [1, N, d]
        )).squeeze(0)
        u_norm = F.normalize(u, p=2, dim=-1)

        # augment z with dropout to form positive pair
        z_prime = self.dropout(batch_norm)

        # project both through fc_nn
        p = self.fc_nn(torch.cat([u_norm, z_prime], dim=0))

        N = p.size(0) // 2
        p_norm = F.normalize(p, p=2, dim=-1)

        sim = torch.matmul(p_norm, p_norm.T) / self.temperature  # [2N, 2N]

        mask = torch.eye(2 * N, device=sim.device).bool()
        sim = sim.masked_fill_(mask, -9e15)

        labels = (torch.arange(2 * N, device=sim.device) + N) % (2 * N)

        loss = F.cross_entropy(sim, labels)

        preds = sim.argmax(dim=1)
        top1 = (preds == labels).float().mean()
        top5 = (sim.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean()

        metrics = {
            "nll_loss": loss.detach(),
            "top1": top1,
            "top5": top5,
            "sim_mean": sim.detach().mean(),
            "sim_std": sim.detach().std(),
            "sim_xy": torch.mean(torch.sum(batch_norm * u_norm, dim=-1)).detach(),
            "p_norm_mean": torch.norm(p_norm, dim=1).mean().detach(),
            "p_norm_std": torch.norm(p_norm, dim=1).std().detach(),
            "U_norm_mean": torch.norm(u_norm, dim=1).mean().detach(),
            "U_norm_std": torch.norm(u_norm, dim=1).std().detach(),
            "U_norm_max": torch.norm(u_norm, dim=1).max().detach(),
        }

        return loss, metrics
