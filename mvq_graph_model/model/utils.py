from pathlib import Path

import torch
from einops import rearrange


def load_shape_template(path: Path) -> torch.Tensor:
    """Returns tensor with shape template, shape: (n_points, 3)."""
    initial_template = torch.load(path)
    return rearrange(initial_template, "n_dim n_points -> n_points n_dim", n_dim=3)
