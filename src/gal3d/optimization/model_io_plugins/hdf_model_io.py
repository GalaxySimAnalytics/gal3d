
import logging
import os
from typing import Any, Literal

import h5py
import numpy as np

from gal3d.optimization.model_io import MetaDataDict, ModelIOBase
from gal3d.optimization.optimizer import OptimizeResult
from gal3d.optimization.parameter import Parameter, Parameters

logger = logging.getLogger("gal3d.optimization.model_io")

__all__ = ["HDF5ModelIO"]

def _set_hdf_value(group: h5py.Group, key: str, value: Any, compression: str | None = None) -> None:
    """
    Utility to set a value in an HDF5 group as either an attribute or dataset.
    """
    try:
        if value is None:
            group.attrs[f"{key}_is_none"] = True
        elif isinstance(value, (bool, int, float, str)):
            group.attrs[key] = value
        elif isinstance(value, (np.ndarray, list, tuple)) and np.asarray(value).shape:
            group.create_dataset(key, data=value, compression=compression)
        else:
            group.attrs[f"{key}_repr"] = repr(value)
    except Exception as e:
        logger.warning(
            "Failed to set key '%s' in HDF5 group '%s': %s. Value type: %s",
            key, group.name, e, type(value)
        )
        group.attrs[f"{key}_repr"] = repr(value)

def _load_hdf_value(group: h5py.Group, key: str) -> Any:
    """
    Utility to load a value from an HDF5 group, handling attributes and datasets.
    """
    try:
        # Handle None marker
        if f"{key}_is_none" in group.attrs:
            return None
        # Handle repr marker (fallback)
        if f"{key}_repr" in group.attrs:
            return group.attrs[f"{key}_repr"]
        # Try attribute
        if key in group.attrs:
            return group.attrs[key]
        # Try dataset
        if key in group:
            return group[key][()]
        # Not found
        return None
    except Exception as e:
        logger.warning(
            "Failed to load key '%s' from HDF5 group '%s': %s",
            key, group.name, e
        )
        return None

class HDF5ModelIO(ModelIOBase):
    """
    HDF5-based implementation of ModelIOBase for saving and loading model results.
    Handles metadata, parameters, and optimization results using HDF5 groups and datasets.
    """
    meta_group: str = "meta"
    parameter_group: str = "parameters"
    opt_group: str = "opt_info"


    @classmethod
    def _save(
        cls,
        data: dict[Literal["meta", "parameters", "opt_info"], dict[str, Any]],
        filename: str,
        overwrite: bool = False,
        group_path: str = "/",
        compression: str | None = "gzip",
        **kwargs: Any)-> None:
        """Save the data to an HDF5 file.

        Parameters
        ----------
        data : dict[str, Any]
            The data to save, extracted from the model.
        filename : str
            The name of the file to save the data to.
        overwrite : bool
            Whether to overwrite the file if it exists.
        group_path : str, optional
            Path within the HDF5 file where data should be stored, default is "/"
        compression : str, optional
            Compression type for datasets, default is "gzip"
        **kwargs : Any
            Additional keyword arguments to pass to the save function.
        """
        # Create or open the file
        file_mode = "a" if os.path.exists(filename) else "w"
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, mode=file_mode) as f:
            # Check if the group already exists
            if group_path in f:
                if overwrite:
                    logger.warning(
                        "Overwriting group '%s' in file '%s'.",
                        group_path, filename
                    )
                    del f[group_path]
                else:
                    raise FileExistsError(
                        f"Group '{group_path}' already exists in file '{filename}'. "
                        "Set overwrite=True to replace."
                    )

            # Create the group
            group = f.create_group(group_path)

            # Save metadata
            meta_group = group.create_group(cls.meta_group)
            cls._save_metadata(meta_group, data["meta"], compression=compression)

            # Save parameters
            params_group = group.create_group(cls.parameter_group)
            cls._save_parameters(params_group, data["parameters"], compression=compression)

            # Save optimization results
            opt_group = group.create_group(cls.opt_group)
            cls._save_opt_results(opt_group, data["opt_info"], compression)

    @classmethod
    def _save_metadata(cls, group: h5py.Group, meta: dict[str, Any], compression: str | None) -> None:
        """ Save metadata as group attributes or datasets. """
        for key, value in meta.items():
            _set_hdf_value(group, key, value, compression=compression)

    @classmethod
    def _save_parameters(cls, params_group: h5py.Group, params_data: dict[str, Any], compression: str | None) -> None:
        """ Save model parameters to the HDF5 group. """
        # Save parameter names to attrs
        params_group.attrs["param_names"] = params_data["param_names"]

        for param_key in params_data["param_names"]:
            params_group.create_dataset(param_key, data=params_data[param_key], compression=compression)
            params_group.create_dataset(f"{param_key}_lb", data=params_data[f"{param_key}_lb"], compression=compression)
            params_group.create_dataset(f"{param_key}_ub", data=params_data[f"{param_key}_ub"], compression=compression)
            params_group.create_dataset(f"{param_key}_err", data=params_data[f"{param_key}_err"], compression=compression)

         # Save info keys
        params_group.attrs["info_names"] = params_data["info_names"]
        for info_key in params_data["info_names"]:
            params_group.create_dataset(info_key, data=params_data[info_key], compression=compression)

    @classmethod
    def _save_opt_results(cls, opt_group: h5py.Group, opt_data: dict[str, Any], compression: str | None) -> None:
        """ Save optimization results to the HDF5 group. """
        for result_name, result_data in opt_data.items():
            opt_set = opt_group.create_group(result_name)
            for key, value in result_data.items():
                _set_hdf_value(opt_set, key, value, compression=compression)

    @classmethod
    def _load_metadata_from_file(cls,
        filename: str,
        group_path: str = "/",
        **kwargs: Any
    ) -> MetaDataDict:
        """ Load metadata from the HDF5 file. """
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. "
                    "Check the group path or file integrity."
                )
            group = f[group_path]
            meta_group = group[cls.meta_group]
            meta = MetaDataDict()
            # Load all keys from attrs and datasets
            for key in list(meta_group.attrs.keys()) + list(meta_group.keys()):
                meta[key] = _load_hdf_value(meta_group, key)
        return meta

    @classmethod
    def _load_parameters_from_file(cls, filename: str, group_path: str = "/", **kwargs: Any) -> list[Parameters]:
        """ Load model parameters from the HDF5 file. """
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. "
                    "Check the group path or file integrity."
                )
            group = f[group_path]
            params_group = group[cls.parameter_group]

            # Load parameter names from attrs
            param_names = list(params_group.attrs["param_names"])
            info_names = list(params_group.attrs["info_names"])

            # Extract parameter data as arrays
            param_values = {name: params_group[name][()] for name in param_names}
            param_lbs = {name: params_group[f"{name}_lb"][()] for name in param_names}
            param_ubs = {name: params_group[f"{name}_ub"][()] for name in param_names}
            param_errs = {name: params_group[f"{name}_err"][()] for name in param_names}
            info_values = {name: params_group[name][()] for name in info_names}

            # Create parameter sets
            param_sets = []
            n_sets = len(next(iter(param_values.values())))
            for i in range(n_sets):
                param_set = Parameters()
                # First create parameters with their bounds and errors
                for name in param_names:
                    value = float(param_values[name][i])
                    lb = float(param_lbs[name][i])
                    ub = float(param_ubs[name][i])
                    err = float(param_errs[name][i])
                    param_set[name] = Parameter(value, lb=lb, ub=ub, err=err)
                # Then add info data
                for key, values in info_values.items():
                    try:
                        if i < len(values):
                            value = values[i]
                            if isinstance(value, str) and value.lower() == "none":
                                continue
                            param_set.add_info(**{key: value})
                    except Exception as e:
                        logger.warning(
                            "Failed to add info key '%s' for parameter set %d in file '%s': %s",
                            key, i, filename, e
                        )
                param_sets.append(param_set)
        return param_sets

    @classmethod
    def _load_opt_from_file(cls, filename:str, group_path: str = "/", **kwargs: Any)-> list[OptimizeResult]:
        """ Load optimization results from the HDF5 file. """
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. "
                    "Check the group path or file integrity."
                )
            group = f[group_path]
            opt_group = group[cls.opt_group]

            opt_results = []
            for opt_result_name in sorted(opt_group.keys()):
                opt_data = opt_group[opt_result_name]
                result_dict: dict[str, Any] = {}
                # Load all keys from attrs and datasets
                for key in list(opt_data.attrs.keys()) + list(opt_data.keys()):
                    result_dict[key] = _load_hdf_value(opt_data, key)
                opt_results.append(OptimizeResult(**result_dict))
        return opt_results

    @classmethod
    def standardize_group_path(cls, group_path: str) -> str:
        # Standardize group path
        if not group_path.startswith("/"):
            group_path = "/" + group_path
        if not group_path.endswith("/"):
            group_path += "/"
        return group_path
