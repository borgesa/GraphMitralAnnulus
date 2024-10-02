import torch


def create_uniform_3d_grid(n_points: int) -> tuple[torch.Tensor, int]:
    """Return 3d grid, uniformly sampled in [0, 1], with specified shape and total points.
    NB: Before 'F.grid_sample' is called, renormalize to [-1, 1].

    Returns:
        torch.Tensor: The 3D grid with shape (1, n_points, n_points, n_points, 3).
        int: The total number of points in the grid.
    """
    # Create 1D tensor for coordinates (x, y, z will be the same)
    coord_1d = torch.linspace(0, 1, n_points)

    # Generate a 3D grid
    x, y, z = torch.meshgrid(coord_1d, coord_1d, coord_1d, indexing="ij")
    grid_single = torch.stack((x, y, z), dim=-1)
    grid_single = grid_single.unsqueeze(0)

    total_points = n_points**3

    return grid_single, total_points
