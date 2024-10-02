#!/usr/bin/env python3
from pathlib import Path

from pydantic import BaseModel

from mvq_graph_model.data.utils import load_json


class DicomInfo(BaseModel):
    """
    Meta data related to the DICOM file sample was exported from.

    Args:
        dataset (str): Name of the dataset.
        included_frames (list[int]): List of frames included.
        name (str): DICOM file name.
        patient_id (int): Patient identifier.
    """

    dataset: str
    included_frames: list[int]
    name: str
    patient_id: int


class FileDetail(BaseModel):
    """
    Represents details about a single file related to a sample.

    Args:
        file (str): File name (not full path).
        hash (str): Hash of the file.
        valid (bool): Indicates if the file is valid. None by default.
    """

    file: str
    hash: str
    valid: bool | None = None


class Files(BaseModel):
    """
    Files expected for a single sample.

    Args:
        coord_system (FileDetail): File details for coordinate system file.
        figure (FileDetail): File details for figure file.
        image (FileDetail): File details for image file.
        labels (FileDetail): File details for labels file.
    """

    coord_system: FileDetail
    figure: FileDetail
    image: FileDetail
    labels: FileDetail


class SampleMeta(BaseModel):
    """
    Represents metadata for a sample.

    Args:
        dicom (DicomInfo): DICOM information for the sample.
        files (Files): Files related to the sample.
        frame (int): Frame number of the sample.
        base_path (Path): Base path of the sample. Set in 'load_from_json'
    """

    dicom: DicomInfo
    files: Files
    frame: int
    base_path: Path | None = None

    @classmethod
    def load_from_json(cls, file_path: Path) -> "SampleMeta":
        """
        Loads sample metadata from a JSON file.

        Args:
            file_path (Path): Path of the JSON file.

        Return:
            SampleMeta: An instance of SampleMeta.
        """
        json_content = load_json(file_path)
        instance = cls.model_validate(json_content)
        instance.base_path = file_path.parent
        return instance

    def get_path(self, file_name: str) -> Path:
        """
        Returns the full path of the specified file.

        Args:
            file_name (str): Name of the file type.

        Return:
            Path: Full path of the file.
        """
        if hasattr(self.files, file_name):
            file_detail = getattr(self.files, file_name)
            return self.base_path / file_detail.file
        else:
            raise AttributeError(f"Field '{file_name}' does not exist in Files class.")

    def __iter__(self):
        """
        Returns an iterator over all items in 'files'.

        Return:
            Iterator: An iterator over all items in 'files'.
        """
        yield from self.files
