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


class ContrastiveHopfieldObjective(BetaObjective): #TODO: this needs work
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


class DiscriminativeHopfieldObjective(BetaObjective):
    """
    Objective from minimo (arXiv:2410.00704 authors' codebase): a discriminative FNN
    scores all N×N pairs of (Hopfield abstract state, masked z-representation)
    and is trained with InfoNCE so that diagonal pairs (same sample) are
    positive.

    Positive pair for sample i:
        anchor   = u_i  (Hopfield output for beta-scaled query i)
        positive = masked_z_i  (z_i with a random binary keep-mask)

    The discriminative FNN receives the concatenation [u_i, masked_z_j] for
    every (i, j) pair and outputs a scalar logit.  The N×N logit matrix is
    then trained with cross-entropy, labels = arange(N) (diagonal).

    This differs from ContrastiveHopfieldObjective in that similarity is
    learned by the FNN rather than computed geometrically.
    """

    def __init__(self, input_dim=32, fnn_hidden_dims=(256, 128),
                 masking_ratio=0.3):
        """
        Args:
            input_dim:        dimensionality of z (and u) representations.
            fnn_hidden_dims:  hidden layer sizes for the discriminative FNN.
            masking_ratio:    fraction of dimensions *dropped* in the binary
                              mask applied to z to form the positive view.
        """
        super().__init__()
        self.masking_ratio = masking_ratio

        layers = []
        in_dim = input_dim * 2
        for h in fnn_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.fnn = nn.Sequential(*layers)

    def loss(self, batch_norm, hopfield, beta):
        N = batch_norm.size(0)

        scaled_queries = batch_norm * beta  # [N, d]

        u = hopfield((
            batch_norm.unsqueeze(0),       # stored patterns [1, N, d]
            scaled_queries.unsqueeze(0),   # queries          [1, N, d]
            batch_norm.unsqueeze(0)        # values           [1, N, d]
        )).squeeze(0)  # [N, d]

        # binary keep-mask applied to z to form positive views
        mask = (torch.rand(batch_norm.size(), device=batch_norm.device)
                > self.masking_ratio)
        masked_z = batch_norm * mask  # [N, d]

        # build all-pairs input: [u_i, masked_z_j] for every (i, j)
        u_expanded = u.unsqueeze(1).expand(N, N, -1)          # [N, N, d]
        z_expanded = masked_z.unsqueeze(0).expand(N, N, -1)   # [N, N, d]
        fnn_input = torch.cat([u_expanded, z_expanded], dim=2) \
                         .view(N * N, -1)                      # [N*N, 2d]

        logits = self.fnn(fnn_input).view(N, N)  # [N, N]

        labels = torch.arange(N, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=1)
        top1 = (preds == labels).float().mean()

        metrics = {
            "nll_loss": loss.detach(),
            "top1": top1,
            "logits_mean": logits.detach().mean(),
            "logits_std": logits.detach().std(),
            "beta_mean": beta.detach().mean(),
            "beta_std": beta.detach().std(),
        }

        return loss, metrics
