import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def add_plane(ax, plane, par=None, plane_color="orange", plot_wire=False):
    if par is None:
        par = np.linspace(0, 1, 20)

    g_dir1, g_dir2 = np.meshgrid(par, par)
    origin = plane["origin"]

    dir1 = plane["dir_x"]
    dir2 = plane["dir_y"]

    xx = origin[0] + dir1[0] * g_dir1 + dir2[0] * g_dir2
    yy = origin[1] + dir1[1] * g_dir1 + dir2[1] * g_dir2
    zz = origin[2] + dir1[2] * g_dir1 + dir2[2] * g_dir2

    if plot_wire:
        ax.plot_wireframe(xx, yy, zz, alpha=0.4, color=plane_color)
    else:
        ax.plot_surface(xx, yy, zz, alpha=0.4, color=plane_color)


def get_plane_dict_for_plot(base_matrix, points, plane_center=None):
    """Return 'plane' dict with origin and directions from basis and points.

    Note that for plotting, the magnitude of the directions are used,
    which is why this function requires the points.

    Keyword arguments:
        base_matrix: 3x3 matrix with basis in column vectors
        points: array with points in 3 dimensions, shape (3, n_points)
    Returns:
        plane: dictionary with keys 'origin', 'dir_x', 'dir_y'

    """
    assert points.shape[-2] == 3
    M_inv = np.linalg.inv(base_matrix)

    if plane_center is None:
        plane_center = points.mean(axis=1)

    # Project annulus to new basis:
    data_rel_center = points - plane_center[:, None]
    annulus_svd_basis = np.matmul(M_inv, data_rel_center)

    svd_basis_min = annulus_svd_basis.min(axis=1)
    svd_basis_max = annulus_svd_basis.max(axis=1)
    svd_basis_data_span = svd_basis_max - svd_basis_min

    # Plot plane:
    plane = {
        "origin": plane_center
        + svd_basis_min[0] * base_matrix[:, 0]
        + svd_basis_min[1] * base_matrix[:, 1],
        "dir_x": base_matrix[:, 0] * svd_basis_data_span[0],
        "dir_y": base_matrix[:, 1] * svd_basis_data_span[1],
    }

    return plane


def add_line(ax, line, direction_name=None, color_override=None):
    r = np.linspace(0, 1, 10)

    x = line["origin"][0] + r * line["direction"][0]
    y = line["origin"][1] + r * line["direction"][1]
    z = line["origin"][2] + r * line["direction"][2]

    if color_override is not None:
        color = color_override
    else:
        color = None
        # ax.plot(x, y, z, label=direction_name)

    ax.plot(x, y, z, label=direction_name, color=color)
