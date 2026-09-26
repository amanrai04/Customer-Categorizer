"""Serialisation, configuration file and IO helpers shared by the components."""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.exception import CustomerException
from src.logger import logging


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read a CSV file into a :class:`pandas.DataFrame`.

    Args:
        file_path: Location of the CSV file.

    Returns:
        The parsed dataframe.

    Raises:
        CustomerException: If the file cannot be read.
    """
    logging.info(f"Reading dataframe from {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as error:
        raise CustomerException(error, sys) from error


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Load a numpy array previously saved with :func:`save_numpy_array_data`."""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as error:
        raise CustomerException(error, sys) from error


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """Persist a numpy array to disk, creating parent directories as needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as error:
        raise CustomerException(error, sys) from error


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """Dump ``content`` to ``file_path`` as YAML.

    Args:
        file_path: Destination path.
        content: Any object ``yaml.dump`` can serialise.
        replace: Remove the file first when it already exists.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file_obj:
            yaml.dump(content, file_obj, sort_keys=False)
    except Exception as error:
        raise CustomerException(error, sys) from error


class MainUtils:
    """Thin, dependency-free helpers around config files and (de)serialisation.

    Model selection itself is *not* handled here: the search space in
    ``config/model.yaml`` is driven by ``neuro_mf``'s ``ModelFactory`` from
    :class:`~src.components.model_trainer.ModelTrainer`.
    """

    def read_yaml_file(self, filename: str) -> dict:
        """Parse a YAML file into a dictionary."""
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as error:
            raise CustomerException(error, sys) from error

    def read_schema_config_file(self) -> dict:
        """Expected columns and columns to drop from ``config/schema.yaml``."""
        try:
            return self.read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as error:
            raise CustomerException(error, sys) from error

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """Pickle ``obj`` to ``file_path``."""
        logging.info(f"Entered save_object of MainUtils for {file_path}")

        try:
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info(f"Exited save_object of MainUtils for {file_path}")
        except Exception as error:
            raise CustomerException(error, sys) from error

    @staticmethod
    def load_object(file_path: str) -> object:
        """Unpickle the object stored at ``file_path``."""
        logging.info(f"Entered load_object of MainUtils for {file_path}")

        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info(f"Exited load_object of MainUtils for {file_path}")
            return obj
        except Exception as error:
            raise CustomerException(error, sys) from error
