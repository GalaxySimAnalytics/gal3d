import logging
import os
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    from gal3d.optimization.result import ModelResult

logger = logging.getLogger("gal3d.optimization.util")


def truncate(num: float, n: int) -> float:
    """
    Truncate a float to n decimal places (without rounding).

    Parameters
    ----------
    num : float
        The number to truncate.
    n : int
        Number of decimal places to keep.

    Returns
    -------
    float
        The truncated number. If num is inf, -inf, or nan, returns num unchanged.
    """
    import numpy as np
    if not np.isfinite(num):
        return num
    factor = 10.0 ** n
    return float(int(num * factor) / factor)


def provide_save_keys(shape_name: str, error_name: str) -> tuple[list[str], list[str]]:
    save_keys = ["pos", "angle", "a", "eps_ab", "eps_bc", "parameter"]
    if shape_name == "Ellipsoid_S":
        save_keys = save_keys + ["sa", "sb", "sc", "parent_fun"]

    res_keys = ["fun"]

    return save_keys, res_keys


def save_dict_to_hdf5(file: h5py.File | h5py.Group | str, data_dict: dict | None = None) ->  h5py.File | h5py.Group | None:
    """Save a dictionary to an HDF5 file or group."""
    if data_dict is None:
        data_dict = {}
    if isinstance(file, str):
        if not os.path.exists(file):
            f = h5py.File(file, "w")
            logger.info("Create %s", file)
            f.close()
        with h5py.File(file, "r+") as f:
            for i in data_dict:
                f.create_dataset(i, data=data_dict[i])
    if isinstance(file, h5py.File) or isinstance(file, h5py.Group):
        for i in data_dict:
            file.create_dataset(i, data=data_dict[i])
        return file
    return None


def save_model_hdf5(
    model: "ModelResult",
    hdf5_file_name: str,
    shape_name: str,
    error_name: str,
    all_header: str = "/",
    other_info: dict | None = None,
) -> None:
    import os

    if other_info is None:
        other_info = {}
    if all_header[-1] == "/":
        save_header = all_header + shape_name + "/" + error_name + "/"
    else:
        save_header = all_header + "/" + shape_name + "/" + error_name + "/"

    save_keys, res_keys = provide_save_keys(shape_name, error_name)

    if not os.path.exists(hdf5_file_name):
        f = h5py.File(hdf5_file_name, "w")
        f.close()
    with h5py.File(hdf5_file_name, "r+") as f:
        for i in save_keys:
            f.create_dataset(save_header + i, data=model[i])
        for i in res_keys:
            f.create_dataset(save_header + i, data=getattr(model,i))
        if other_info:
            save_dict_to_hdf5(f[all_header], other_info)
