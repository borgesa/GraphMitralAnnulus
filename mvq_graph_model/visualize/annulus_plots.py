"""In work. Simply meant to make annulus plots."""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.linalg import svd

from mvq_graph_model.visualize.plot_utils import add_plane, get_plane_dict_for_plot


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane to a set of points in d-dimensional space.

    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.

    Source: https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    """
    points = np.reshape(
        points, (np.shape(points)[0], -1)
    )  # Collapse trialing dimensions
    assert (
        points.shape[0] <= points.shape[1]
    ), "There are only {} points in {} dimensions.".format(
        points.shape[1], points.shape[0]
    )
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.

    svd_basis = svd(M)[0]

    return ctr, svd_basis


def add_annulus(
    ax,
    points: np.ndarray,
    color: str,
    add_fitted_plane: bool = False,
    label: str = "",
):
    """Adds single annulus to ax. points should be shape (3, n_points)."""

    if add_fitted_plane:
        label_origin, label_basis_matrix = fit_plane(points)
        plane_dict = get_plane_dict_for_plot(
            base_matrix=label_basis_matrix,
            points=points,
            plane_center=label_origin,
        )
        add_plane(ax, plane_dict, plane_color=color)

    ax.plot(points[0], points[1], points[2], color=color, label=label)


def create_annulus_plot(
    annuli: torch.Tensor | np.ndarray,
    colors: str | list[str],
    ax: Axes3D,
    labels: str | list[str] = "",
    set_limits: bool = True,
) -> Axes3D:
    """Plots annulus in 3D space.

    :param annuli: list of annulus to plot, shape (batch, n_points, 3)
    :param colors: colors for each annulus
    :param ax: Axes object to plot on. If None, creates a new one.
    :param set_limits: Whether to set axis limits or not.
    :return: ax (with the generated figure)
    """

    if isinstance(annuli, torch.Tensor):
        annuli = annuli.numpy()

    if not labels:
        labels = ["" for i in range(len(colors))]

    for this_annulus, this_color, label in zip(annuli, colors, labels):
        # Passing transposed annulus, as 'add_annulus' expects shape (3, n_points)
        add_annulus(
            ax=ax,
            points=this_annulus.T,
            color=this_color,
            add_fitted_plane=True,
            label=label,
        )
        # Plotting first point on annulus:
        ax.scatter(
            this_annulus[0, 0], this_annulus[0, 1], this_annulus[0, 2], color=this_color
        )

    ax.view_init(azim=-127, elev=-160)

    ax.set_xticks([0, 1 / 2, 1])
    ax.set_yticks([0, 1 / 2, 1])
    ax.set_zticks([0, 1 / 2, 1])

    if set_limits:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    return ax


def get_sample_difference(predictions, labels):
    """Return dict with key differences."""
    preds_center = predictions.mean(-2)
    labels_center = labels.mean(-2)

    center_distances = preds_center - labels_center

    # Get angle to first point:
    pred_center_to_ao = predictions[:, 0, :] - preds_center
    labels_center_to_ao = labels[:, 0, :] - labels_center

    dot_prod = torch.einsum(
        "...ij, ...ij->...i", pred_center_to_ao, labels_center_to_ao
    )
    normalize = pred_center_to_ao.norm(dim=-1) * labels_center_to_ao.norm(dim=-1)
    angle = torch.acos(dot_prod / normalize)
    angle_degrees = angle * 180 / torch.Tensor([math.pi])  # torch.pi
    metrics = {"center_distance": center_distances, "angle_degrees": angle_degrees}

    print(angle_degrees, " (degrees between)")
    return metrics


def create_3d_annuli_plots(
    annuli: torch.Tensor,
    labels: None | list[str] = None,
):
    fig = plt.figure()

    # Step 3: Add two 3D projection axes
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")  # First subplot
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")  # Second subplot

    if isinstance(annuli, list):
        annuli = torch.cat(annuli, dim=0)

    assert annuli.ndim == 3

    if labels:
        assert annuli.shape[0] == len(labels)
    else:
        labels = []

    colors = ["red", "green", "black", "orange"]

    create_annulus_plot(annuli, colors=colors, ax=ax1)
    create_annulus_plot(annuli, colors=colors, ax=ax2, labels=labels)

    # dist_n = norm_dist(labels_, predictions_.detach())
    # dist_c2c = curve_to_curve(labels_, predictions_.detach())
    # plt.suptitle(f"{dist_n} c2c: {dist_c2c * 90}")
    if labels:
        plt.legend()
    plt.show()
