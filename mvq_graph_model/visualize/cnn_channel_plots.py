import matplotlib.pyplot as plt


def plot_channels(volume, tensor, batch_idx, x_slice=None, y_slice=None, z_slice=None):
    """
    Plots each channel of a tensor for a given batch index across three different slices.
    volume: A tensor with shape (B, channels, x, y, z), representing the raw data.
    tensor: A tensor with shape (B, channels, x, y, z), representing the processed data.
    batch_idx: The index of the batch to visualize.
    z_slice, x_slice, y_slice: The specific slices along the z, x, and y axes to visualize.
                               Defaults to the middle slice for each dimension.
    """
    # Ensure the batch index is within bounds
    if batch_idx >= tensor.shape[0]:
        print("Batch index out of bounds.")
        return

    # Default to the middle slice if not specified

    if x_slice is None:
        x_slice = tensor.shape[2] // 2
    if y_slice is None:
        y_slice = tensor.shape[3] // 2
    if z_slice is None:
        z_slice = tensor.shape[4] // 2
    # Number of channels
    channels = tensor.shape[1]

    # Creating subplots with 3 columns for each slice dimension and rows for each channel plus raw data
    fig, axs = plt.subplots(nrows=channels + 1, ncols=3, figsize=(15, channels * 5 + 5))

    # Plot raw data slices
    raw_x = volume[batch_idx, 0, x_slice, :, :].detach().numpy()
    raw_y = volume[batch_idx, 0, :, y_slice, :].detach().numpy()
    raw_z = volume[batch_idx, 0, :, :, z_slice].detach().numpy()

    axs[0, 0].imshow(raw_z, cmap="gray")
    axs[0, 1].imshow(raw_x, cmap="gray")
    axs[0, 2].imshow(raw_y, cmap="gray")
    axs[0, 0].set_title("Raw Data (Z slice)")
    axs[0, 1].set_title("Raw Data (X slice)")
    axs[0, 2].set_title("Raw Data (Y slice)")

    for channel_i in range(channels):
        i = channel_i + 1
        # Z slice
        channel_data_z = tensor[batch_idx, channel_i, :, :, z_slice].detach().numpy()
        axs[i, 0].imshow(channel_data_z, cmap="gray")
        axs[i, 0].set_title(f"Channel {channel_i} (Z slice)")
        axs[i, 0].axis("off")

        # X slice
        channel_data_x = tensor[batch_idx, channel_i, :, x_slice, :].detach().numpy()
        axs[i, 1].imshow(channel_data_x, cmap="gray")
        axs[i, 1].set_title(f"Channel {channel_i} (X slice)")
        axs[i, 1].axis("off")

        # Y slice
        channel_data_y = tensor[batch_idx, channel_i, y_slice, :, :].detach().numpy()
        axs[i, 2].imshow(channel_data_y, cmap="gray")
        axs[i, 2].set_title(f"Channel {channel_i} (Y slice)")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
