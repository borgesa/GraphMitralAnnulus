#!/usr/bin/env python3
import os
from collections import Counter
from pathlib import Path

from joblib import Memory
from loguru import logger
from torch.utils.data import Dataset

from mvq_graph_model.data.data_loader import VolumeHandler
from mvq_graph_model.data.utils import load_json

cachedir = os.getenv("MVQ_CACHE_DIR")


def load_volume(sample_path):
    sample = VolumeHandler(sample_path=sample_path, verify_hash=False)
    return sample.get_sample()


if cachedir is None:  # disable cache if no directory is set
    logger.warning("CACHING DIRECTORY HAS NOT BEEN SET!!!")
    load_cached_volume = load_volume

else:
    logger.info(f"Cache directory is set to: {cachedir}")
    # Initialize joblib.Memory with the cache directory
    memory = Memory(cachedir, compress="lz4", verbose=0)
    # Decorate the function for caching
    load_cached_volume = memory.cache(load_volume)


def compute_sample_weights(patient_ids: list[str]) -> list[float]:
    """
    Compute sample weights based on the frequency of each patient_id in the dataset.
    Args:
        patient_ids (list[str]): List of patient IDs.
    Return:
        weights (list[float]): List of computed weights for each sample.
    """
    patient_id_counter = Counter(patient_ids)
    num_patient_ids = len(patient_id_counter)
    weights = [1 / (patient_id_counter[id_] * num_patient_ids) for id_ in patient_ids]
    return weights


def get_dataset_meta_and_weights(
    data_root: Path, dataset_pattern: str
) -> tuple[list[Path], list[float]]:
    """
    Returns list with paths to all dataset samples and list with sample weights for each sample.
    The sample weights are computed to enable normalized sampling according to 'patient_id'.
    Args:
        data_root (Path): Root directory for the dataset.
        dataset_pattern (str): String used by 'glob' to find samples in 'data_root'.
    Return:
        paths (list[Path]): List of paths to all dataset samples.
        weights (list[float]): List of sample weights for each sample.
    """
    config_files = Path(data_root).glob(dataset_pattern)

    paths = []
    patient_ids = []
    for config_file_path in config_files:
        content = load_json(config_file_path)

        paths.append(config_file_path)
        patient_ids.append(content["dicom"]["patient_id"])

    weights = compute_sample_weights(patient_ids)

    return paths, weights


class MVQDataset(Dataset):
    """
    PyTorch Dataset class for MVQDataset.
    Args:
        data_root (Path): Root directory for the dataset.
        dataset_pattern (str): String used by 'glob' to find samples in 'data_root'.
    """

    def __init__(
        self,
        data_root: Path,
        dataset_pattern: str = "ID_*/*/geometry_*/frame_*/info_*.json",
    ):
        self.data_root = data_root
        self.sample_paths, self.weights = get_dataset_meta_and_weights(
            data_root=data_root, dataset_pattern=dataset_pattern
        )
        logger.debug(
            f"Instantiated MVQDataset with {self.__len__()} samples (root: '{data_root}')."
        )

    def __len__(self):
        """
        Get the number of samples in the dataset.
        Return:
            length (int): The number of samples in the dataset.
        """
        return len(self.sample_paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Args:
            idx (int): The index of the sample to fetch.
        Return:
            sample (VolumeHandler): The fetched sample.
        """
        sample_path = self.sample_paths[idx]

        return load_cached_volume(sample_path)
