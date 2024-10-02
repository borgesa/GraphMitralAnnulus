"""Calculating the 'surgical view error'."""

import torch
from einops import rearrange
from kornia.geometry.transform import get_affine_matrix3d


def apply_transform(affine_matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply transform to a batch of point grids.

    Args:
        affine_matrix: tensor (batch, 4, 4) affine matrix
        points: tensor (batch, n_nodes, 3), points before transform
    Return:
        Transformed version of 'points', shape (batch, n_nodes, 3)
    """
    batch_size, n_points, _ = points.shape
    ones = torch.ones(batch_size, n_points, 1, device=points.device)
    # Add column of ones to points for homogenous coordinates:
    points_mod = torch.cat((points, ones), dim=-1)
    # Rearrange in prep for homogeneous dimension setup:
    points_hom = rearrange(points_mod, "batch n_points dim -> batch dim n_points")
    points_transformed = torch.matmul(affine_matrix, points_hom)

    # Arrange coordinates back and drop final dimension:
    return rearrange(
        points_transformed,
        "batch dim_hom n_points -> batch n_points dim_hom",
        dim_hom=4,
        batch=batch_size,
    )[..., :3]


def get_affine_matrix(
    graph_output: torch.Tensor,
    centers: torch.Tensor | None = None,
    no_rotation: bool = False,
    no_scaling: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Returns a transformation matrix and parameters from graph layer output.

    Args:
        graph_output (torch.Tensor): The output tensor from a graph layer.
            Shape [batch_size, n_dof]. See below for valid dof dimensions.
        centers: Centers for the transformation, set to center of [0, 1]^3 if not specified.
            Shape [batch_size, 3].
        no_rotation: If True, this function will ignore rotation part of graph output.
        no_scaling: It True (and n_dof=7), will ignore scaling part of graph output.
    Returns:
        Tuple with the affine transformation matrix and the parameters used to create it..

    Notes:
    Last two arguments intended for early training process (only focus on translation).

    n_dof: Number of degrees of freedom. Valid numbers:
        n_dof=6
            graph_output[..., :3] -> Translation in x, y, z direction
            graph_output[..., 3:6] -> Rotation in x, y, z
        n_dof=7
            graph_output[..., 6] -> Scaling factor
            (remaining as above)
    """
    if not isinstance(graph_output, torch.Tensor):
        raise TypeError("graph_output must be a torch.Tensor")
    if graph_output.dim() != 2 or graph_output.size(1) not in {6, 7}:
        raise ValueError(
            "graph_output must have shape [batch_size, n_dof] with n_dof in {6, 7}"
        )

    device = graph_output.device
    dtype = graph_output.dtype
    batch_size, n_dof = graph_output.shape

    if centers is None:
        centers = torch.full(
            size=(batch_size, 3),
            fill_value=0.5,
            dtype=dtype,
            device=device,
        )

    scale_factor = (
        torch.zeros((batch_size, 1), device=device, dtype=dtype)
        if no_scaling or n_dof < 7
        else graph_output[..., 6:]
    )
    scaling = torch.exp(scale_factor)

    # Raw output interpreted as radians. Scaled to degrees for transform:
    rotation = (
        torch.zeros((batch_size, 3), device=device, dtype=dtype)
        if no_rotation
        else graph_output[..., 3:6] * 180 / torch.pi
    )

    params = {
        "translations": graph_output[..., :3],
        "angles": rotation,  # Takes degrees
        "scale": scaling,
        "center": centers,
    }
    affine_matrix = get_affine_matrix3d(**params)

    return affine_matrix, params
