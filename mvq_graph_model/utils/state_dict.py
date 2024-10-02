from collections import OrderedDict
from pathlib import Path

import torch


def get_state_dict(
    cp_path: Path, prefix_remove: str = "model.", map_location: str = ""
):
    """Loads state dict from torch checkpoint to specified device.
    Removes prefix, if specified (e.g., "module." or "model.")
    """
    if map_location:
        checkpoint = torch.load(cp_path, map_location=map_location)
    else:
        checkpoint = torch.load(cp_path)

    state_dict = checkpoint["state_dict"]

    if prefix_remove == "":
        return state_dict

    n_chars_remove = len(prefix_remove)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Removes prefix from state dict:
        name = k[n_chars_remove:] if k.startswith(prefix_remove) else k
        new_state_dict[name] = v

    return new_state_dict
