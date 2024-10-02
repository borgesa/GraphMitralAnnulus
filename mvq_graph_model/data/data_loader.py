#!/usr/bin/env python3
"""Contains 'VolumeHandler' class that handles samples generated in this repository."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from loguru import logger

from mvq_graph_model.data.basemodel import SampleMeta
from mvq_graph_model.data.utils import check_hash, load_file
from mvq_graph_model.utils.distance_map import distance_map


def _get_json_path(sample_path: str | Path) -> Path:
    """Retrieves the path to a 'config' JSON file related to 'sample_path' in VolumeHandler.

    Args:
        sample_path: Path to directory with single 'info_*.json' file, or direct to json file.

    Returns:
        Path to config file.
    """
    sample_path = Path(sample_path)

    if not sample_path.exists():
        raise ValueError(f"The path {sample_path} does not exist.")

    if sample_path.is_dir():
        json_files = list(sample_path.glob("info_*.json"))
        if len(json_files) != 1:
            raise ValueError(
                f"Expected a single 'info_*.json' file, found {len(json_files)}."
            )
        json_file = json_files[0]
    elif sample_path.is_file() and sample_path.suffix == ".json":
        json_file = sample_path
    else:
        raise ValueError(
            "The path must be a directory containing 'info_*.json' or a direct path to a JSON file."
        )

    return json_file


class VolumeHandler:
    """Handles samples generated in this repository. It allows to load data, verify hashes and get sample data."""

    def __init__(
        self,
        sample_path: Path | str,
        verify_hash: bool = False,
    ):
        """Initializes the VolumeHandler with the given sample path. Optionally, it can verify the hashes of the files.

        Args:
            sample_path: Path to the sample data.
            verify_hash: Whether to verify the hashes of the files.
        """
        json_path = _get_json_path(sample_path=sample_path)
        self.meta = SampleMeta.load_from_json(file_path=json_path)

        if verify_hash:
            self._verify_hashes()

    def _verify_hashes(self):
        """Verifies the hashes of the files in the meta data."""
        for field_name in self.meta.fields.keys():
            files_instance = self.meta.fields[field_name]
            expected_hash = files_instance.hash
            file_path = self.meta.get_path(file_name=field_name)
            check_hash(file_path=file_path, expected_hash=expected_hash)
        logger.debug("Successfully verified all hashes.")

    def load(self, f_type: str):
        """Returns loaded data of type 'f_type'.

        Args:
            f_type: The type of the file to load.

        Returns:
            The loaded data of the given type.
        """
        file_path = self.meta.get_path(f_type)
        return load_file(file_path=file_path)

    def get_sample(self, normalize: bool = True) -> dict[str, Any]:
        """Returns dictionary with sample data for the class instance.

        Args:
            normalize: Whether to normalize the volume data.

        Returns:
            A dictionary with the sample data.
        """

        def inverse_frequency_weighting(arr):
            # Find unique elements and their indices and counts
            unique, inverse, counts = np.unique(
                arr, return_inverse=True, return_counts=True
            )
            # Calculate frequency
            weights = (arr.size / counts).astype(np.float32)
            # Map frequencies back to the original array positions
            weights_array = weights[inverse]

            return (weights_array / weights_array.mean()).reshape(arr.shape)

        label_dict = self.load("labels")
        labels_normed = label_dict["sample_normed"]
        landmarks = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in labels_normed["landmarks"].items()
        }
        annulus = labels_normed["annulus"]
        annulus_tensor = torch.tensor(annulus, dtype=torch.float32)

        coord_system = self.load("coord_system")
        coord_system = {
            key: value.astype(np.float32) for key, value in coord_system.items()
        }

        # Volumetric data:
        volume = self.load("image")
        volume_tensor = rearrange(torch.from_numpy(volume), "... -> 1 ...")

        # Assuming same spatial dimension
        res = float(volume.shape[-1])
        dmap = distance_map(volume.shape, annulus)  # type: ignore
        dmap_tensor = rearrange(torch.from_numpy(dmap / res), "... -> 1 ...")

        dmap_weights = inverse_frequency_weighting(dmap.astype(np.int16))
        dmap_weights_tensor = rearrange(torch.from_numpy(dmap_weights), "... -> 1 ...")

        dmap_ao = distance_map(volume.shape, annulus[:1, :])  # type: ignore

        dmap_ao_tensor = rearrange(torch.from_numpy(dmap_ao / res), "... -> 1 ...")

        dmap_ao_weights = inverse_frequency_weighting(dmap_ao.astype(np.int16))
        dmap_ao_weights_tensor = rearrange(
            torch.from_numpy(dmap_ao_weights), "... -> 1 ..."
        )

        if normalize:
            volume_tensor = volume_tensor / 255.0

        result = {
            "volume": volume_tensor,
            "dmap": dmap_tensor,
            "dmap_weights": dmap_weights_tensor,
            "dmap_ao": dmap_ao_tensor,
            "dmap_ao_weights": dmap_ao_weights_tensor,
            "target": annulus_tensor,
            "landmarks": landmarks,
            "coord_system": coord_system,
            "dicom_name": self.meta.dicom.name,
            "patient": self.meta.dicom.patient_id,
            "volume_file": self.meta.files.image.file,
        }

        return result
