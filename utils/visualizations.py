from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch

def visualize_embeddings(z, method="tsne", title="Embeddings"):
    if torch.is_tensor(z): 
        z = z.detach().cpu().numpy() # change to cpu calculations 

    if method == "tsne":
        projector = TSNE(n_components=2, perplexity=30)
    elif method == "pca":
        projector = PCA(n_components=2)
    else:
        raise ValueError("Unsupported method")

    z_2d = projector.fit_transform(z)

    plt.figure(figsize=(6, 6))
    plt.scatter(z_2d[:len(z)//2, 0], z_2d[:len(z)//2, 1], label='Anchor', alpha=0.6)
    plt.scatter(z_2d[len(z)//2:, 0], z_2d[len(z)//2:, 1], label='Positive', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()