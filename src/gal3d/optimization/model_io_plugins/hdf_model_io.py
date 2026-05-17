"""
HDF5-based Model I/O plugin for saving and loading model optimization results.

"""

import logging
import os
import time
from typing import Any, Literal

import h5py
import numpy as np

from gal3d.optimization.model_io import MetaDataDict, ModelIOBase
from gal3d.optimization.optimizer import OptimizeResult
from gal3d.optimization.parameter import Parameter, Parameters
from gal3d.optimization.result import ModelResult
from gal3d.shape import Structure3D, StructureCore

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
        logger.warning("Failed to set key '%s' in HDF5 group '%s': %s. Value type: %s", key, group.name, e, type(value))
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
        logger.warning("Failed to load key '%s' from HDF5 group '%s': %s", key, group.name, e)
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
        compression: str | None = "lzf",
        **kwargs: Any,
    ) -> None:
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
            Compression type for datasets, default is ``"lzf"`` (fast read/write
            bundled with h5py). Use ``"gzip"`` for a better compression ratio.
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
                    logger.warning("Overwriting group '%s' in file '%s'.", group_path, filename)
                    del f[group_path]
                else:
                    raise FileExistsError(
                        f"Group '{group_path}' already exists in file '{filename}'. Set overwrite=True to replace."
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
        """Save metadata as group attributes or datasets."""
        for key, value in meta.items():
            _set_hdf_value(group, key, value, compression=compression)

    @classmethod
    def _save_parameters(cls, params_group: h5py.Group, params_data: dict[str, Any], compression: str | None) -> None:
        """Save model parameters to the HDF5 group."""
        # Save parameter names to attrs
        params_group.attrs["param_names"] = params_data["param_names"]

        for param_key in params_data["param_names"]:
            params_group.create_dataset(param_key, data=params_data[param_key], compression=compression)
            params_group.create_dataset(f"{param_key}_lb", data=params_data[f"{param_key}_lb"], compression=compression)
            params_group.create_dataset(f"{param_key}_ub", data=params_data[f"{param_key}_ub"], compression=compression)
            params_group.create_dataset(
                f"{param_key}_err", data=params_data[f"{param_key}_err"], compression=compression
            )

        # Save info keys
        params_group.attrs["info_names"] = params_data["info_names"]
        for info_key in params_data["info_names"]:
            params_group.create_dataset(info_key, data=params_data[info_key], compression=compression)

    @classmethod
    def _save_opt_results(cls, opt_group: h5py.Group, opt_data: dict[str, Any], compression: str | None) -> None:
        """Save optimization results to the HDF5 group."""
        for result_name, result_data in opt_data.items():
            opt_set = opt_group.create_group(result_name)
            for key, value in result_data.items():
                _set_hdf_value(opt_set, key, value, compression=compression)

    @classmethod
    def _load_metadata_from_file(
        cls, filename: str, keys: list[str] | None = None, group_path: str = "/", **kwargs: Any
    ) -> MetaDataDict:
        """Load metadata from the HDF5 file."""
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. Check the group path or file integrity."
                )
            group = f[group_path]
            meta_group = group[cls.meta_group]
            meta = MetaDataDict()
            # Load all keys from attrs and datasets
            load_keys = list(meta_group.attrs.keys()) + list(meta_group.keys()) if keys is None else keys
            for key in load_keys:
                meta[key] = _load_hdf_value(meta_group, key)
        return meta

    @classmethod
    def _load_parameters_from_file(cls, filename: str, group_path: str = "/", **kwargs: Any) -> list[Parameters]:
        """Load model parameters from the HDF5 file."""
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. Check the group path or file integrity."
                )
            return cls._load_params_from_group(f[group_path][cls.parameter_group])

    @classmethod
    def _load_opt_from_file(cls, filename: str, group_path: str = "/", **kwargs: Any) -> list[OptimizeResult]:
        """Load optimization results from the HDF5 file."""
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. Check the group path or file integrity."
                )
            return cls._load_opt_from_group(f[group_path][cls.opt_group])

    # ------------------------------------------------------------------
    # Internal group-based helpers (no file I/O; reused by load() override)
    # ------------------------------------------------------------------

    @classmethod
    def _load_params_from_group(cls, params_group: h5py.Group) -> list[Parameters]:
        """Build a Parameters list from an already-open HDF5 group."""
        param_names: list[str] = list(params_group.attrs["param_names"])
        info_names: list[str] = list(params_group.attrs["info_names"])

        # Read every column in one shot — single HDF5 read per dataset
        param_values = {name: params_group[name][()] for name in param_names}
        param_lbs = {name: params_group[f"{name}_lb"][()] for name in param_names}
        param_ubs = {name: params_group[f"{name}_ub"][()] for name in param_names}
        param_errs = {name: params_group[f"{name}_err"][()] for name in param_names}
        info_values = {name: params_group[name][()] for name in info_names}

        n_sets = len(next(iter(param_values.values()))) if param_values else 0
        param_sets: list[Parameters] = []
        for i in range(n_sets):
            param_set = Parameters()
            for name in param_names:
                param_set[name] = Parameter(
                    float(param_values[name][i]),
                    lb=float(param_lbs[name][i]),
                    ub=float(param_ubs[name][i]),
                    err=float(param_errs[name][i]),
                )
            for key, values in info_values.items():
                try:
                    if i < len(values):
                        value = values[i]
                        if isinstance(value, str) and value.lower() == "none":
                            continue
                        param_set.add_info(**{key: value})
                except Exception as e:
                    logger.warning("Failed to add info key '%s' for parameter set %d: %s", key, i, e)
            param_sets.append(param_set)
        return param_sets

    @classmethod
    def _load_opt_from_group(cls, opt_group: h5py.Group) -> list[OptimizeResult]:
        """Build an OptimizeResult list from an already-open HDF5 group."""
        opt_results: list[OptimizeResult] = []
        for opt_result_name in sorted(opt_group.keys()):
            opt_data = opt_group[opt_result_name]
            result_dict: dict[str, Any] = {}
            for key in list(opt_data.attrs.keys()) + list(opt_data.keys()):
                result_dict[key] = _load_hdf_value(opt_data, key)
            opt_results.append(OptimizeResult(**result_dict))
        return opt_results

    # ------------------------------------------------------------------
    # Single-open load() override
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, filename: str, structure: Structure3D | StructureCore | None = None, group_path: str = "/", **kwargs: Any
    ) -> ModelResult:
        """
        Load the model from an HDF5 file using a single file open.

        Parameters
        ----------
        filename : str
            The name of the file to load the model from.
        structure : Structure3D | StructureCore | None, optional
            Structure object to associate with the loaded model. If None, a
            structure is reconstructed from the metadata stored in the file.
        group_path : str, optional
            Path within the HDF5 file where data is stored, default is "/".
        **kwargs : Any
            Unused; kept for API compatibility.

        Returns
        -------
        ModelResult
            The loaded model result.
        """
        start_time = time.time()
        group_path = cls.standardize_group_path(group_path)

        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. Check the group path or file integrity."
                )
            group = f[group_path]

            # Reconstruct structure from stored metadata if not provided
            if structure is None:
                meta_group = group[cls.meta_group]
                meta_keys = ["coordinate_name", "geometry_name", "error_method_name", "error_func_name"]
                metadata = MetaDataDict({k: _load_hdf_value(meta_group, k) for k in meta_keys})
                if metadata.get("error_method_name") is None or metadata.get("error_func_name") is None:
                    structure = StructureCore(
                        coordinate=metadata["coordinate_name"], geometry=metadata["geometry_name"]
                    )
                else:
                    structure = Structure3D(
                        coordinate=metadata["coordinate_name"],
                        geometry=metadata["geometry_name"],
                        error_func=metadata["error_func_name"],
                        error_method=metadata["error_method_name"],
                    )

            param_sets = cls._load_params_from_group(group[cls.parameter_group])
            opt_results = cls._load_opt_from_group(group[cls.opt_group])

        # add_derived is cheap (no I/O) and must happen after file closes
        derived_funcs = structure.derived_param_funcs()
        for param_set in param_sets:
            param_set.add_derived(derived_funcs)

        result = ModelResult(structure=structure, optimize_result=opt_results[0], parameters=param_sets[0])
        result._param_sets = param_sets
        result._opt_results = opt_results

        elapsed = time.time() - start_time
        logger.debug("Model loaded from %s in %.2f seconds (n: %d)", filename, elapsed, len(param_sets))
        return result

    # ------------------------------------------------------------------
    # Selective column loader  (fast path — no Parameters construction)
    # ------------------------------------------------------------------

    @classmethod
    def _load_columns_from_file(
        cls, filename: str, param_keys: list[str] | None = None, group_path: str = "/", **kwargs: Any
    ) -> dict[str, Any]:
        """
        Load specific parameter columns as raw numpy arrays without constructing
        Parameters objects.

        Parameters
        ----------
        filename : str
            The name of the file to load from.
        param_keys : list[str] | None, optional
            Column names to load. If None, all parameter value columns and info
            columns are returned (bounds/errors excluded from the default set).
            To load bounds or errors, include the suffixed names explicitly,
            e.g. ``"a_lb"``, ``"a_err"``.
        group_path : str, optional
            Path within the HDF5 file, default is "/".
        **kwargs : Any
            Unused; kept for API compatibility.

        Returns
        -------
        dict[str, Any]
            Mapping of column name to numpy array of shape ``(n_models,)``.
        """
        group_path = cls.standardize_group_path(group_path)
        with h5py.File(filename, "r") as f:
            if group_path not in f:
                raise ValueError(
                    f"Group '{group_path}' not found in file '{filename}'. Check the group path or file integrity."
                )
            params_group = f[group_path][cls.parameter_group]
            all_param_names: list[str] = list(params_group.attrs["param_names"])
            all_info_names: list[str] = list(params_group.attrs["info_names"])

            keys_to_load: list[str] = all_param_names + all_info_names if param_keys is None else list(param_keys)

            result: dict[str, Any] = {}
            for key in keys_to_load:
                if key in params_group:
                    result[key] = params_group[key][()]
                else:
                    logger.warning("Column '%s' not found in parameters group of '%s'.", key, filename)
        return result

    @classmethod
    def standardize_group_path(cls, group_path: str) -> str:
        # Standardize group path
        if not group_path.startswith("/"):
            group_path = "/" + group_path
        if not group_path.endswith("/"):
            group_path += "/"
        return group_path
