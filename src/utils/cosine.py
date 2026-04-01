import torch
import torch.nn.functional as F

def match_closest_cosine(unique_u, z_reps):
    """
    Matches each vector in unique_u to the closest vector in z_reps
    using cosine similarity, then returns the corresponding subsampled_states.

    Args:
        unique_u: Tensor of shape [N_unique, d]
        z_reps: Tensor of shape [N_z, d]
        subsampled_states: Tensor of shape [N_z, ...] (same first dim as z_reps)

    Returns:
        mask: the mask that corresponds to the z_reps closest to each unique_u
    """
    # normalize vectors to lie on hypersphere
    unique_u_norm = F.normalize(unique_u, p=2, dim=1)
    z_reps_norm = F.normalize(z_reps, p=2, dim=1)

    # cos sim of shape: [n_unique, n_z]
    cos_sim = torch.mm(unique_u_norm, z_reps_norm.T)

    # indices of closest cos sim
    best_match_indices = torch.argmax(cos_sim, dim=1)

    # create mask for selected z_reps
    mask = torch.zeros(z_reps.size(0), dtype=torch.bool, device=z_reps.device)
    mask[best_match_indices] = True

    return mask