import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl

from src.data.trajectories import TrajectorySet
from src.data.sampler import Sampler
from src.data.cl_dataset import DatasetCL
from src.data.latent_dataset import StatesDataset
from src.models.cl_model import mlpCL
from src.models.beta_model import LearnedBetaModel
from src.models.beta_objective import ContrastiveHopfieldObjective
from src.trainers.cl_trainer import train_cl
from src.utils.trajectory_io import save_trajectories, load_trajectories
from src.utils.tensor import split_data
import umap

# =============================================================================
# CONFIG — edit these
# =============================================================================

# Data generation
N_TRAJECTORIES  = 50
N_STEPS         = 200
DATA_SEED       = 0

# CL model
CL_EPOCHS       = 100
CL_BATCH        = 256
CL_LR           = 1e-3
CL_WEIGHT_DECAY = 1e-5
CL_TEMPERATURE  = 30        # InfoNCE temperature
CL_SIGMA        = 10        # Gaussian sampler width (in timesteps)
CL_TRAIN_PAIRS  = 20_000
CL_VAL_PAIRS    = 5_000

# Beta model
BETA_EPOCHS         = 50
BETA_BATCH          = 256
BETA_LR             = 1e-3
BETA_WEIGHT_DECAY   = 1e-5
BETA_TEMPERATURE    = 0.03796   # InfoNCE temperature for Hopfield objective
BETA_MASKING_RATIO  = 0.3
BETA_HOPFIELD_SCALE = 500.0

# =============================================================================


# ---------------------------------------------------------------------------
# Four-room grid environment
# ---------------------------------------------------------------------------

class FourRoomGrid:
    """
    A tabular four-room grid on a 9x9 lattice.

    Layout:
        outer walls:           rows/cols 0 and 8
        inner horizontal wall: row 4 (except doorways)
        inner vertical wall:   col 4 (except doorways)
        doorways:              (2,4), (4,2), (4,6), (6,4)

    Actions: 0=up, 1=down, 2=left, 3=right
    State:   np.array([row, col], dtype=float32)
    """

    GRID_SIZE = 9
    DOORWAYS  = frozenset([(2, 4), (4, 2), (4, 6), (6, 4)])
    DELTAS    = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self):
        self.walls = self._build_walls()
        self.valid_states = [
            (r, c)
            for r in range(self.GRID_SIZE)
            for c in range(self.GRID_SIZE)
            if (r, c) not in self.walls
        ]

    def _build_walls(self):
        n = self.GRID_SIZE
        walls = set()
        for i in range(n):
            walls.update([(0, i), (n-1, i), (i, 0), (i, n-1)])
        for c in range(n):
            if (4, c) not in self.DOORWAYS:
                walls.add((4, c))
        for r in range(n):
            if (r, 4) not in self.DOORWAYS:
                walls.add((r, 4))
        return walls

    def step(self, state: tuple, action: int) -> tuple:
        dr, dc = self.DELTAS[action]
        nxt = (state[0] + dr, state[1] + dc)
        return nxt if nxt not in self.walls else state

    def random_valid_state(self) -> tuple:
        return self.valid_states[np.random.randint(len(self.valid_states))]

    def state_to_array(self, state: tuple) -> np.ndarray:
        return np.array([state[0], state[1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_and_save(env: FourRoomGrid, save_path: str) -> None:
    np.random.seed(DATA_SEED)
    tset = TrajectorySet()

    for _ in range(N_TRAJECTORIES):
        state   = env.random_valid_state()
        states  = np.zeros((N_STEPS, 2), dtype=np.float32)
        actions = np.zeros((N_STEPS, 1), dtype=np.float32)

        for t in range(N_STEPS):
            action     = np.random.randint(4)
            states[t]  = env.state_to_array(state)
            actions[t] = action
            state      = env.step(state, action)

        tset.add_trajectory({"states": states, "actions": actions})

    save_trajectories(tset, save_path)
    print(f"Saved {N_TRAJECTORIES} trajectories ({N_STEPS} steps each) to {save_path}")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

ROOM_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green",
               3: "tab:red",  4: "tab:purple"}
ROOM_NAMES  = {0: "Top-left", 1: "Top-right", 2: "Bottom-left",
               3: "Bottom-right", 4: "Doorway"}


def assign_room_labels(env: FourRoomGrid) -> list:
    labels = []
    for (r, c) in env.valid_states:
        if (r, c) in env.DOORWAYS:
            labels.append(4)
        elif r < 4 and c < 4:
            labels.append(0)
        elif r < 4 and c > 4:
            labels.append(1)
        elif r > 4 and c < 4:
            labels.append(2)
        else:
            labels.append(3)
    return labels


def _scatter_plot(z_2d: np.ndarray, labels: list, title: str,
                  xlabel: str, ylabel: str, save_path: str) -> None:
    labels_arr = np.array(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lid in sorted(set(labels)):
        mask = labels_arr == lid
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=ROOM_COLORS[lid], label=ROOM_NAMES[lid],
                   alpha=0.85, s=80, edgecolors="k", linewidths=0.4)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def _combined_scatter(z_2d: np.ndarray, u_2d: np.ndarray, labels: list,
                      title: str, xlabel: str, ylabel: str, save_path: str) -> None:
    """Single plot: CL latents (circles) and Hopfield abstract states (diamonds)."""
    labels_arr = np.array(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lid in sorted(set(labels)):
        mask = labels_arr == lid
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=ROOM_COLORS[lid], marker="o",
                   alpha=0.85, s=70, edgecolors="k", linewidths=0.4)
        ax.scatter(u_2d[mask, 0], u_2d[mask, 1],
                   c=ROOM_COLORS[lid], marker="D",
                   alpha=0.55, s=55, edgecolors="k", linewidths=0.4)

    # Legend: room colors + marker-type guide
    color_handles  = [mpatches.Patch(color=ROOM_COLORS[lid], label=ROOM_NAMES[lid])
                      for lid in sorted(ROOM_COLORS)]
    marker_handles = [
        mlines.Line2D([], [], marker="o", color="gray", ls="None",
                      markersize=7, label="CL latent"),
        mlines.Line2D([], [], marker="D", color="gray", ls="None",
                      markersize=6, label="Hopfield abstract"),
    ]
    ax.legend(handles=color_handles + marker_handles, loc="best", fontsize=8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def visualize_combined_pca(z: np.ndarray, u_norm: np.ndarray, labels: list,
                           save_path: str) -> None:
    """Fit PCA on z; overlay both z (circles) and u_norm (diamonds) in one plot."""
    scaler   = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    pca      = PCA(n_components=2).fit(z_scaled)
    ev       = pca.explained_variance_ratio_

    z_2d = pca.transform(z_scaled)
    u_2d = pca.transform(scaler.transform(u_norm))

    _combined_scatter(
        z_2d, u_2d, labels,
        title="CL Latents & Hopfield Abstract States — PCA (four-room)",
        xlabel=f"PC1 ({ev[0]*100:.1f}% var, fit on CL latents)",
        ylabel=f"PC2 ({ev[1]*100:.1f}% var, fit on CL latents)",
        save_path=save_path,
    )


def visualize_combined_umap(z: np.ndarray, u_norm: np.ndarray, labels: list,
                            save_path: str) -> None:
    """Fit UMAP on z; project u_norm through the same map and overlay in one plot."""
    scaler   = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    reducer  = umap.UMAP(n_components=2, random_state=42)
    z_2d     = reducer.fit_transform(z_scaled)
    u_2d     = reducer.transform(scaler.transform(u_norm))

    _combined_scatter(
        z_2d, u_2d, labels,
        title="CL Latents & Hopfield Abstract States — UMAP (four-room)",
        xlabel="UMAP-1",
        ylabel="UMAP-2",
        save_path=save_path,
    )


def plot_learning_curve(log_dir: str, name: str, title: str, save_path: str) -> None:
    """Read the most recent CSVLogger run and plot train/val NLL loss over epochs."""
    versions = sorted(glob.glob(os.path.join(log_dir, name, "version_*")))
    if not versions:
        print(f"No logs found under {os.path.join(log_dir, name)}; skipping curve.")
        return
    metrics_path = os.path.join(versions[-1], "metrics.csv")
    df = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, label, color in [
        ("train/nll_loss", "train loss", "tab:blue"),
        ("val/nll_loss",   "val loss",   "tab:orange"),
    ]:
        if metric in df.columns:
            subset = df[["epoch", metric]].dropna()
            # one point per epoch (last logged value within that epoch)
            subset = subset.groupby("epoch")[metric].last().reset_index()
            ax.plot(subset["epoch"], subset[metric], label=label, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Beta model helpers
# ---------------------------------------------------------------------------

def collect_all_states(tset: TrajectorySet) -> np.ndarray:
    """Concatenate every raw state from all trajectories. Returns [N_total, state_dim]."""
    return np.vstack([
        tset.get_trajectory(i)[0]["states"]
        for i in range(tset.get_num_trajectories())
    ])


def train_beta(cl_model: mlpCL, all_states: np.ndarray,
               checkpoint_path: str, logger, device: str) -> LearnedBetaModel:
    """
    Train LearnedBetaModel on CL latents of four-room trajectories.

    Uses an inline loop instead of beta_trainer.train_beta_model() because that
    function's load_from_checkpoint call omits `objective` (excluded from
    save_hyperparameters), causing a crash. Here we pass it explicitly on reload.
    """
    objective = ContrastiveHopfieldObjective(
        temperature=BETA_TEMPERATURE,
        masking_ratio=BETA_MASKING_RATIO,
    )

    train_states, val_states = split_data(all_states, split_val=0.8)
    train_ds = StatesDataset(cl_model=cl_model, data=train_states)
    val_ds   = StatesDataset(cl_model=cl_model, data=val_states)

    train_loader = data.DataLoader(train_ds, batch_size=BETA_BATCH, shuffle=True,  drop_last=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=BETA_BATCH, shuffle=False, drop_last=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best_beta_fourroom",
        save_top_k=1,
        save_weights_only=True,
        mode="max",
        monitor="val/top1",
    )
    trainer = pl.Trainer(
        default_root_dir=checkpoint_path,
        logger=logger,
        accelerator="mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu",
        devices=1,
        max_epochs=BETA_EPOCHS,
        callbacks=[checkpoint_callback, LearningRateMonitor("epoch")],
    )

    pl.seed_everything(10)
    beta_model = LearnedBetaModel(
        objective=objective,
        hopfield_scale=BETA_HOPFIELD_SCALE,
        lr=BETA_LR,
        weight_decay=BETA_WEIGHT_DECAY,
        max_epochs=BETA_EPOCHS,
        device=device,
    )
    trainer.fit(beta_model, train_loader, val_loader)

    print("Best beta model path:", checkpoint_callback.best_model_path)
    return LearnedBetaModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        objective=objective,
    )


def extract_latents_and_abstract_states(
        cl_model: mlpCL, beta_model: LearnedBetaModel,
        env: FourRoomGrid, all_states_np: np.ndarray,
        device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        z      — CL latents for every valid grid cell  [N_valid, 32]
        u_norm — L2-normalised Hopfield abstract states [N_valid, 32]

    Hopfield memory = all trajectory states encoded via the CL model,
    matching the distribution the beta model trained on.
    """
    dev = torch.device(device)
    cl_model.eval()
    beta_model.eval()

    with torch.no_grad():
        # Memory: encode all trajectory states
        z_mem      = cl_model(torch.as_tensor(all_states_np, dtype=torch.float32).to(dev))
        z_mem_norm = F.normalize(z_mem, p=2, dim=-1)

        # Queries: encode every valid grid cell
        valid_arr    = np.array([env.state_to_array(s) for s in env.valid_states], dtype=np.float32)
        z_valid      = cl_model(torch.as_tensor(valid_arr).to(dev))
        z_valid_norm = F.normalize(z_valid, p=2, dim=-1)
        beta         = beta_model.get_beta(z_valid_norm)        # [N_valid, 1]
        scaled       = z_valid_norm * beta

        # Hopfield retrieval: memory as stored patterns/values, valid states as queries
        u = beta_model.hopfield((
            z_mem_norm.unsqueeze(0),   # stored patterns  [1, N_traj,  32]
            scaled.unsqueeze(0),       # queries          [1, N_valid, 32]
            z_mem_norm.unsqueeze(0),   # values           [1, N_traj,  32]
        )).squeeze(0)
        u_norm = F.normalize(u, p=2, dim=-1)

    return z_valid.cpu().numpy(), u_norm.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TESTS_DIR       = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH       = os.path.join(TESTS_DIR, "data",        "fourroom.npz")
    CHECKPOINT_PATH = os.path.join(TESTS_DIR, "checkpoints")
    PLOTS_DIR       = os.path.join(TESTS_DIR, "plots")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(PLOTS_DIR,       exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Step 1: build environment -------------------------------------------
    print("Building four-room grid...")
    env = FourRoomGrid()
    print(f"  {len(env.valid_states)} valid cells")

    # -- Step 2: generate or reuse trajectories ------------------------------
    if os.path.exists(DATA_PATH):
        print(f"Reusing existing trajectory data at {DATA_PATH}")
    else:
        print("Generating random-walk trajectories...")
        generate_and_save(env, DATA_PATH)
    tset = load_trajectories(DATA_PATH)

    # -- Step 3: train or reuse CL model -------------------------------------
    cl_ckpt = os.path.join(CHECKPOINT_PATH, "best_model_fourroom.ckpt")
    if os.path.exists(cl_ckpt):
        print(f"Loading existing CL model from {cl_ckpt}")
        cl_model = mlpCL.load_from_checkpoint(cl_ckpt, map_location=torch.device(DEVICE))
    else:
        print(f"Training CL model ({CL_EPOCHS} epochs)...")
        sampler  = Sampler(tset, dist="g", sigma=CL_SIGMA, b=CL_SIGMA, add_action=False)
        train_ds = DatasetCL(sampler, num_state_pairs=CL_TRAIN_PAIRS)
        val_ds   = DatasetCL(sampler, num_state_pairs=CL_VAL_PAIRS)
        cl_model = train_cl(
            cl_model=mlpCL,
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=CL_BATCH,
            logger=CSVLogger(save_dir=TESTS_DIR, name="cl_logs"),
            checkpoint_path=CHECKPOINT_PATH,
            max_epochs=CL_EPOCHS,
            device=DEVICE,
            filename="best_model_fourroom",
            input_dim=2,
            temperature=CL_TEMPERATURE,
            lr=CL_LR,
            weight_decay=CL_WEIGHT_DECAY,
        )
    cl_model = cl_model.to(torch.device(DEVICE))

    plot_learning_curve(
        log_dir=TESTS_DIR, name="cl_logs",
        title=f"CL Model — Training Curve (four-room)",
        save_path=os.path.join(PLOTS_DIR, "cl_learning_curve.png"),
    )

    # -- Step 4: collect trajectory states (Hopfield memory) -----------------
    print("Collecting all trajectory states...")
    all_states = collect_all_states(tset)
    print(f"  Total states: {all_states.shape[0]}")

    # -- Step 5: train beta model --------------------------------------------
    print(f"Training beta model ({BETA_EPOCHS} epochs)...")
    beta_model = train_beta(
        cl_model=cl_model,
        all_states=all_states,
        checkpoint_path=CHECKPOINT_PATH,
        logger=CSVLogger(save_dir=TESTS_DIR, name="beta_logs"),
        device=DEVICE,
    )
    beta_model = beta_model.to(torch.device(DEVICE))

    plot_learning_curve(
        log_dir=TESTS_DIR, name="beta_logs",
        title=f"Beta Model — Training Curve (four-room)",
        save_path=os.path.join(PLOTS_DIR, "beta_learning_curve.png"),
    )

    # -- Step 6: extract latents and abstract states -------------------------
    print("Extracting latents and abstract states...")
    z, u_norm = extract_latents_and_abstract_states(
        cl_model=cl_model,
        beta_model=beta_model,
        env=env,
        all_states_np=all_states,
        device=DEVICE,
    )
    labels = assign_room_labels(env)
    print(f"  z shape:      {z.shape}")
    print(f"  u_norm shape: {u_norm.shape}")

    # -- Step 7: combined PCA plot -------------------------------------------
    print("Generating PCA plot...")
    visualize_combined_pca(
        z=z, u_norm=u_norm, labels=labels,
        save_path=os.path.join(PLOTS_DIR, "pca.png"),
    )

    # -- Step 8: combined UMAP plot ------------------------------------------
    print("Generating UMAP plot...")
    visualize_combined_umap(
        z=z, u_norm=u_norm, labels=labels,
        save_path=os.path.join(PLOTS_DIR, "umap.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
