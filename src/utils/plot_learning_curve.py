import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
