import torch
from collections import OrderedDict

def load_lightning_checkpoint(model, ckpt_path, map_location="cpu", prefix_to_strip="model."):
    """
    Load a Lightning checkpoint into a plain PyTorch model.
    
    Args:
        model: nn.Module instance
        ckpt_path: path to the .ckpt file
        map_location: "cpu" or "cuda"
        prefix_to_strip: prefix added by Lightning (often "model.")
    
    Returns:
        model with weights loaded
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt["state_dict"]

    # Strip Lightning prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix_to_strip):
            new_k = k[len(prefix_to_strip):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    return model