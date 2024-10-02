import os
from pathlib import Path

import numpy as np
from lightning.pytorch import Callback, Trainer


class SaveLatestCheckpointCallback(Callback):
    """Saves latest checkpoint (with epoch name).

    Intended in addition to storing n-best checkpoints.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        # Construct the base save directory path
        save_dir = (
            Path(trainer.logger.save_dir)
            / trainer.logger.name
            / f"version_{trainer.logger.version}"
            / "checkpoints"
        )
        current_epoch = trainer.current_epoch
        filepath = save_dir / f"latest-checkpoint-epoch-{current_epoch}.ckpt"
        trainer.save_checkpoint(filepath, weights_only=False)

        # After successful save, delete the previous epoch's checkpoint
        previous_epoch = current_epoch - 1
        previous_filepath = save_dir / f"latest-checkpoint-epoch-{previous_epoch}.ckpt"
        if previous_filepath.exists() and filepath.exists():
            os.remove(previous_filepath)


class RandomValBatchCallback(Callback):
    """Callback selecting batches to visualize from, in dataloader.
    Called on start of validation epoch, updating which batches to plot for.
    """

    def __init__(self, num_vis_per_epoch: int):
        self.num_vis_per_epoch = num_vis_per_epoch

    def on_validation_epoch_start(self, trainer: Trainer, pl_module):

        num_batches = len(trainer.datamodule.val_dataloader())  # type: ignore
        random_batch_indices = np.random.choice(
            num_batches, self.num_vis_per_epoch, replace=False
        ).tolist()
        # Store the indices in the model
        pl_module.random_batch_indices = random_batch_indices
