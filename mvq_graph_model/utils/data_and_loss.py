import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger


def focal_loss(loss, gamma):
    """Focal variant of base losses."""
    return {k: torch.pow(v, gamma) for k, v in loss.items()}


def interpolate_curve(curve, points=1):
    """Interpolate curves by inserting intermediate points."""
    if points <= 1:
        return curve

    rolled = torch.roll(curve, shifts=-1, dims=1)

    stack = []
    for i in range(0, points + 1):
        alpha = i / (points + 1)
        stack.append(curve * (1 - alpha) + rolled * alpha)

    result = torch.stack(stack, dim=2)
    result = rearrange(result, "B A I R -> B (A I) R")
    return result


def curve_to_curve(curve1, curve2, interpolation_points=0, gamma=1.0, scaling=1.0):
    """Return 'curve to curve distance' between two discrete curves.

    Args:
        curve1: array of coordinates for first curve
        curve2: array of coordinat)es for first curve
        interpolation_points: number of new, intermediate interpolation points.

    The shape of the two curves is assumed to be:
        (num_planes, coord_dimension)

    Handling rows and columns separately (not requiring same number of points)
    """
    # NB: Function assumes isotropic dimensions.
    assert curve1.shape[-1] == 3 and curve2.shape[-1] == 3

    if scaling != 1.0:
        curve1 = curve1 * scaling
        curve2 = curve2 * scaling

    if interpolation_points > 0:
        curve1 = interpolate_curve(curve1, interpolation_points)
        curve2 = interpolate_curve(curve2, interpolation_points)

    dist_matrix = torch.cdist(curve1, curve2, p=2)

    row_values = dist_matrix.min(dim=-2).values
    col_values = dist_matrix.min(dim=-1).values

    if gamma > 1.0:
        row_values = torch.pow(row_values, gamma)
        col_values = torch.pow(col_values, gamma)

    results = (row_values.mean(dim=-1) + col_values.mean(dim=-1)) / 2

    return results


def perimeter(curve):
    """curve perimeter measuremet"""
    rolled = torch.roll(curve, shifts=-1, dims=-1)
    perimeter = torch.linalg.norm(curve - rolled, ord=2, dim=-1).sum(dim=1)
    return perimeter


def coordinate_loss(label, prediction, order=2, gamma=1):
    """Stable Eucledian distance norm"""
    distance = torch.linalg.norm(label - prediction, ord=order, dim=-1)
    if gamma != 1:
        return torch.pow(distance, gamma).mean()
    else:
        return distance.mean()


def smooth_loss(prediction, label, weights=None, alpha=0.1):
    z = torch.abs(label - prediction) / alpha
    if weights is not None:
        return (z * weights).mean()
    else:
        return z.mean()


def average_error(prediction, label, threshold=None):
    """Average voxelwise distance map difference.
    The volume is assumed to be normalized, the prediction shape
    is used to convert back from normalized coordinates into pixel coordinates.

    """

    if threshold is not None and not isinstance(threshold, dict):
        mask = label < (threshold / prediction.shape[-1])
        if mask.sum() == 0:
            return average_error(prediction, label, threshold=None)

        return (
            torch.abs(label - prediction.detach())[mask].mean() * prediction.shape[-1]
        )
    else:
        return torch.abs(label - prediction.detach()).mean() * prediction.shape[-1]


def max_error(prediction, label, threshold=None):
    """Average voxelwise distance map difference.
    The volume is assumed to be normalized, the prediction shape
    is used to convert back from normalized coordinates into pixel coordinates.

    """
    if threshold is not None and not isinstance(threshold, dict):
        mask = label < (threshold / prediction.shape[-1])
        if mask.sum() == 0:
            return max_error(prediction, label, threshold=None)

        return torch.abs(label - prediction.detach())[mask].max() * prediction.shape[-1]
    else:
        return torch.abs(label - prediction.detach()).max() * prediction.shape[-1]


def drop_random_3rd_dim(data, threshold=0.5):
    """
    Drop random elements from the 3rd dimension of a 5D volume based on a threshold.

    Parameters:
    - data (torch.Tensor): The input 5D volume with dimensions [N, C, D, H, W].
    - threshold (float): The probability threshold to drop elements. Higher values result in more dropping.

    Returns:
    - new_data (torch.Tensor): The augmented data with elements dropped.
    - mask (torch.Tensor): A mask indicating which elements were dropped (1) and which were kept (0).
    """
    # Copy the data to avoid modifying the original
    new_data = data.clone()

    # Initialize the mask with zeros on the same device as data
    mask = torch.zeros_like(data, device=data.device, dtype=data.dtype)

    # Generate the mask and apply the drops
    for a in range(data.size(0)):  # Loop over the batch dimension
        for b in range(data.size(1)):  # Loop over the channel dimension
            for c in range(data.size(2)):  # Loop over the depth/time dimension
                if torch.rand(1).item() > threshold:
                    mask[a, b, c, :, :].fill_(1)
                    new_data[a, b, c, :, :].fill_(0)

    return new_data, mask


class CutAndDrop(nn.Module):

    def __init__(self, keys: str, p: float, enabled: float):
        super().__init__()
        logger.debug(f"'CutAndDrop' initialized. p={p}, enabled={enabled}")

        if isinstance(keys, str):
            self.keys = [
                keys,
            ]
        else:
            self.keys = keys

        self.p = p
        self.enabled = enabled

    def forward(self, data):
        if (torch.rand(1) > self.enabled).item():
            return data
        x = data[self.keys[0]]

        B = x.shape[0]
        Nx, Ny, Nz = x.shape[-3:]
        device = x.device

        mask1 = (torch.rand(B * Nx).reshape(B, 1, -1, 1, 1) > self.p).to(device)
        mask2 = (torch.rand(B * Ny).reshape(B, 1, 1, -1, 1) > self.p).to(device)
        mask3 = (torch.rand(B * Nz).reshape(B, 1, 1, 1, -1) > self.p).to(device)

        mask = mask1 * mask2 * mask3

        result = {**data}

        for key in self.keys:
            result[key] = result[key] * mask

        return result
