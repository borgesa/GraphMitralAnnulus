"""Functions related to the graph of a curve."""

import torch
import torch.nn.functional as F
from einops import rearrange


def _prepare_coordinates(coordinates: torch.Tensor, n_batch: int) -> torch.Tensor:
    """Normalize coordinates to [-1, 1], flip last axis, and reshape for grid_sample.

    :param coordinates: Input coordinates tensor, assumed in [0, 1]^3, the
        expected input shape (B, N_nodes, 3) or (B, n_x, n_y, n_z, 3)
    :param n_batch: Batch size of
    :return: Reshaped coordinates tensor, normalized to [-1, 1]^3 and flipped final axis.
    """
    # Ensure the last dimension is 3 for spatial coordinates
    if not coordinates.shape[-1] == 3:
        raise ValueError(
            f"Expected the last dimension of coordinates to be 3, "
            f"but got {coordinates.shape[-1]}"
        )
    # Rescale and flip coordinates
    coordinates = (coordinates - 0.5) * 2
    coordinates = torch.flip(coordinates, [-1])

    if coordinates.ndim == 5:
        # No reshaping. Mainly checks n_dim=3:
        return rearrange(
            coordinates,
            "batch n_x n_y n_z n_dim -> batch n_x n_y n_z n_dim",
            batch=n_batch,
            n_dim=3,
        )
    elif coordinates.ndim == 3:
        return rearrange(
            coordinates,
            "batch n_points n_dim  -> batch n_points 1 1 n_dim",
            batch=n_batch,
            n_dim=3,
        )
    else:
        raise ValueError(
            f"Coordinates must have 3 or 5 dimensions, "
            f"but has {len(coordinates.shape)}."
        )


def sample_coords_from_encoder_features(
    feature_map: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Returns F.grid_sample results from 'feature_maps', sampled in 'coordinates'.
    Each point in 'coordinates' will result in one sample per feature map channel.

    :param feature_map: tensor of size (B, num_channels, N_x, N_y, N_z)
    :param coordinates: Coordinates tensor w/graph nodes. Either has
        shape (B, N_nodes, 3) or nodes in grid: (B, n_x, n_y, n_z, 3)
        (NB: n_<xyz> are not feature space dimension, N_<xyz>)
    :return: Feature tensor, shape (B, N_nodes, num_channels)
    """
    # Prep for 'grid_sample' format (coordinates, normalize to [-1, 1] and flip last axis:
    coordinates = _prepare_coordinates(coordinates, n_batch=feature_map.shape[0])
    # Sample features
    spatial_features = F.grid_sample(feature_map, coordinates, align_corners=False)

    # Rearrange tensor
    return rearrange(
        spatial_features,
        "batch num_features n_x n_y n_z -> batch (n_x n_y n_z) num_features",
    )
