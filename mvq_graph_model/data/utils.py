#!/usr/bin/env python3
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger


class JsonLoadingError(Exception):
    """Exception raised when JSON loading fails."""

    pass


def get_md5_hash(file_path: Path) -> str:
    """Returns the md5 hash of a file.

    Args:
        file_path: Path to the file

    Return:
        str: md5 hash of the file
    """
    hasher = hashlib.md5()
    with file_path.open("rb") as handle:
        hasher.update(handle.read())
    md5_hash = hasher.hexdigest()[:8]

    return md5_hash


def check_hash(file_path: Path, expected_hash: str) -> None:
    """Compares computed hash with expected hash, raises ValueError if not equal."""
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")
    # Checking hash:
    computed_hash = get_md5_hash(file_path)
    if not computed_hash == expected_hash:
        raise ValueError(
            f"The md5 of {file_path} does not match ('{computed_hash}' versus expected '{expected_hash}'."
        )


def load_json(file_path: str | Path) -> dict:
    """Load json file."""
    try:
        with Path(file_path).open() as f:
            loaded_content = json.load(f)
    except FileNotFoundError as e:
        logger.debug(f"File not found (file name: '{file_path}').")
        raise JsonLoadingError from e
    except PermissionError as e:
        logger.debug(f"Permission denied for saving (file name: '{file_path}').")
        raise JsonLoadingError from e
    except json.JSONDecodeError as e:
        logger.debug(f"Invalid JSON format (file name: '{file_path}').")
        raise JsonLoadingError from e
    except ValueError as e:
        logger.debug(f"Empty file. (file name: '{file_path}').")
        raise JsonLoadingError from e
    except TypeError as e:
        logger.debug(f"Invalid input type (file name: '{file_path}').")
        raise JsonLoadingError from e
    return loaded_content


def load_file(file_path: Path) -> dict | np.ndarray:
    """Handles loading of json, pickle and .npy files.
    The function requires that the file extension is explicit (does not infer content).

    Args:
        file_path: Path to the file

    Return:
        dict: content of the file
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileExistsError(f"File not found: {file_path}")

    if file_path.suffix == ".json":
        content = load_json(file_path=file_path)
    elif file_path.suffix == ".pickle":
        content = pickle.loads(file_path.read_bytes())
    elif file_path.suffix == ".npy":
        content = np.load(file_path)
    else:
        raise ValueError(
            f"Invalid extension ({file_path.suffix}). "
            + "Valid options: 'json', 'pickle', 'npy'."
        )
    return content
