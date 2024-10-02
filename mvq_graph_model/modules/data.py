#!/usr/bin/env python3
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from mvq_graph_model.data.dataset import MVQDataset


class MVQDataModule(L.LightningDataModule):
    """Lightning data module."""

    def __init__(
        self,
        data_dir: Path | str,
        batch_size: int,
        train_epoch_size: int,
    ):
        """
        Args:
            data_dir (Path): The directory where the data is stored.
            batch_size (int): The size of the batches for the DataLoader.
            train_epoch_size (int): The size of a training epoch.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_epoch_size = train_epoch_size

    def setup(self, stage: str):
        """
        Set up the data for training, validation, and testing.

        Args:
            stage (str): The stage of the model (training, validation, testing).
        """

        if stage == "fit":
            self.data_train = MVQDataset(data_root=self.data_dir / "train")
            self.data_val = MVQDataset(data_root=self.data_dir / "val")
        elif stage == "test":
            self.data_test = MVQDataset(data_root=self.data_dir / "test_plain")

    def _create_dataloader(self, dataset, batch_size, sampler=None) -> DataLoader:
        """
        Create a DataLoader for a given dataset.

        Args:
            dataset: The dataset for which to create the DataLoader.
            sampler: The sampler to use for the DataLoader. Defaults to None.

        Return:
            DataLoader: The DataLoader for the given dataset.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=16,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Create a DataLoader for the training data, using weighted sampler."""
        # Weighted sampler to balance dicoms/patients:
        train_weights = self.data_train.weights
        weighted_sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=self.train_epoch_size, replacement=False
        )
        return self._create_dataloader(
            self.data_train, batch_size=self.batch_size, sampler=weighted_sampler
        )

    def val_dataloader(self) -> DataLoader:
        """Create a DataLoader for the validation data."""
        # Random sampler, mainly for randomizing what example is shown:
        return self._create_dataloader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test data."""
        return self._create_dataloader(self.data_test, batch_size=1)
