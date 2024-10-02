#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def compute_distance_map(
    A: NDArray[np.float32], B: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Compute the distance map from each voxel in A to the closest point in B.

    Parameters:
    - A: NumPy array of shape (n, 3), representing n voxel coordinates in 3D space.
    - B: NumPy array of shape (m, 3), representing m arbitrary points in 3D space.

    Returns:
    - A NumPy array of the distances from each voxel in A to its closest point in B.
    """
    # Create a KD-Tree with the points in B
    tree = cKDTree(B)

    # Query the KD-Tree for the nearest neighbor in B for each point in A
    distances, _ = tree.query(A, k=1)
    distances = distances.astype(np.float32)

    return distances


def distance_map(
    volume_shape: tuple[int, int, int], points: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Generate a distance map for a 3D volume given a set of points.

    Parameters:
    - shape: A tuple of three integers (k, l, m) defining the size of the 3D volume.
    - points: A NumPy array of shape (n, 3), representing n points in 3D space, where each point is scaled according to the volume size.

    Returns:
    - A NumPy array of shape (k, l, m) representing the distance of each voxel in the volume to the closest point in 'points'.
    """
    # Rescale points according to the shape of the volume
    rescale_factor = np.array(volume_shape, dtype=np.float32).reshape(1, -1)
    scaled_points = points * rescale_factor

    # Generate all voxel coordinates in the volume
    dim1, dim2, dim3 = volume_shape
    coord1, coord2, coord3 = np.mgrid[0:dim1, 0:dim2, 0:dim3]
    coordinates = np.vstack([coord1.ravel(), coord2.ravel(), coord3.ravel()]).T

    # Compute distances from each voxel to the closest point
    distances = compute_distance_map(coordinates, scaled_points)

    return distances.reshape(volume_shape)
