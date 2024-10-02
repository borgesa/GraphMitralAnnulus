import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max

from mvq_graph_model.utils.torch_utils import dcn


def peak_local_min(array, *args, **kwargs):
    return peak_local_max(-array, *args, **kwargs)


def plot_hm_dist_error(
    volume, label, prediction, landmarks=2, sample_idx=0, slice_idx=None, pred_ch=0
):
    """Plots error between 'label' and prediction' overlaid on volume (in slice).

    Shows single slice of volume.

    Assumes all input has dimensionality: (B, feat, x, y, z).
    Accepts tensors on GPU (detach->cpu->numpy).

    Args:
        volume: Volume tensor
        label: Distance map, same spatial shape as volume
        prediction: Model output, same spatial shape as volume
        sample_idx: Sample to pick from the batch dimension
        pred_ch: Channel to pick from prediction output
    """
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    vol = dcn(volume[sample_idx, 0, slice_idx]).T[::-1, :]
    lab = dcn(label[sample_idx, 0, slice_idx]).T[::-1, :]
    pred = dcn(prediction[sample_idx, pred_ch, slice_idx]).T[::-1, :]

    # Get label and 'prediction candidate':
    points_lab = peak_local_min(lab, min_distance=20, num_peaks=landmarks)
    points_pred = peak_local_min(pred, min_distance=20, num_peaks=landmarks)

    scatter_s = 30
    # Contour levels to plot:
    c_levels = np.arange(0, 1, 0.01)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3)

    # us, focused US,

    ax_im = fig.add_subplot(gs[0, 0])
    ax_lab = fig.add_subplot(gs[1, 0])
    ax_diff = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[1, 1])

    ax_ch1 = fig.add_subplot(gs[0, 2])
    ax_ch2 = fig.add_subplot(gs[1, 2])

    ax_im.axes.xaxis.set_ticks([])
    ax_im.axes.yaxis.set_ticks([])
    ax_lab.axes.xaxis.set_ticks([])
    ax_lab.axes.yaxis.set_ticks([])
    ax_diff.axes.xaxis.set_ticks([])
    ax_diff.axes.yaxis.set_ticks([])
    ax_pred.axes.xaxis.set_ticks([])
    ax_pred.axes.yaxis.set_ticks([])

    ax_ch1.axes.xaxis.set_ticks([])
    ax_ch1.axes.yaxis.set_ticks([])
    ax_ch2.axes.xaxis.set_ticks([])
    ax_ch2.axes.yaxis.set_ticks([])

    ax_im.imshow(vol, origin="lower", cmap="gray")
    ax_im.set_title("Slice from US image")

    ax_lab.set_title("Labels (contours)")
    ax_lab.imshow(vol, origin="lower", cmap="gray")
    ax_lab.contour(lab, colors="cyan", linewidths=0.5, levels=c_levels)
    ax_lab.scatter(
        points_lab[:, 1], points_lab[:, 0], c="cyan", s=scatter_s, marker="+"
    )

    ax_diff.set_title("Difference (label - prediction)")
    ax_diff.imshow(vol, origin="lower", cmap="gray")
    im_diff = ax_diff.imshow(
        lab - pred,
        origin="lower",
        cmap="seismic",
        vmin=-3 / 128,
        vmax=3 / 128,
        alpha=0.7,
    )
    ax_diff.scatter(
        points_lab[:, 1], points_lab[:, 0], c="cyan", s=scatter_s, marker="+"
    )
    ax_diff.scatter(
        points_pred[:, 1], points_pred[:, 0], c="orange", s=scatter_s, marker="x"
    )
    fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

    ax_pred.set_title("Prediction (contours)")
    ax_pred.imshow(vol, origin="lower", cmap="gray")
    ax_pred.contour(pred, colors="orange", linewidths=0.5, levels=c_levels)
    ax_pred.scatter(
        points_lab[:, 1], points_lab[:, 0], c="cyan", s=scatter_s, marker="+"
    )
    ax_pred.scatter(
        points_pred[:, 1], points_pred[:, 0], c="orange", s=scatter_s, marker="x"
    )
    plt.tight_layout()

    return fig
