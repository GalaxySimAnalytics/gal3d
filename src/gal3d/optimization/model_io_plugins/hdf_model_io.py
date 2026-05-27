"""
HDF5-based Model I/O plugin for saving and loading model optimization results.

"""

import ast
import json
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


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _encode_bool_column(values: list[Any]) -> dict[str, Any]:
    arr = np.full(len(values), np.nan, dtype=float)
    for i, value in enumerate(values):
        if value is not None:
            arr[i] = float(bool(value))
    return {"kind": "scalar_bool", "values": arr}


def _encode_numeric_column(values: list[Any]) -> dict[str, Any]:
    arr = np.full(len(values), np.nan, dtype=float)
    for i, value in enumerate(values):
        if value is not None:
            arr[i] = float(value)
    return {"kind": "scalar_float", "values": arr}


def _encode_string_column(values: list[Any]) -> dict[str, Any]:
    text = np.asarray(["" if value is None else value for value in values], dtype=h5py.string_dtype("utf-8"))
    mask = np.asarray([value is None for value in values], dtype=bool)
    return {"kind": "string", "values": text, "isnull": mask}


def _encode_array_column(values: list[Any]) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=float) for value in values if value is not None]
    shapes = [arr.shape for arr in arrays]

    if len(set(shapes)) == 1:
        stacked = np.full((len(values),) + shapes[0], np.nan, dtype=float)
        mask = np.asarray([value is None for value in values], dtype=bool)
        for i, value in enumerate(values):
            if value is not None:
                stacked[i] = np.asarray(value, dtype=float)
        return {"kind": "ndarray_stack", "values": stacked, "isnull": mask}

    flat_parts: list[np.ndarray] = []
    offsets = np.zeros(len(values), dtype=np.int64)
    lengths = np.zeros(len(values), dtype=np.int64)
    shape_rows: list[str] = []
    mask = np.asarray([value is None for value in values], dtype=bool)

    cursor = 0
    for i, value in enumerate(values):
        offsets[i] = cursor
        if value is None:
            lengths[i] = 0
            shape_rows.append(repr(()))
            continue

        arr = np.asarray(value, dtype=float)
        flat = arr.ravel()
        flat_parts.append(flat)
        lengths[i] = flat.size
        shape_rows.append(repr(arr.shape))
        cursor += flat.size

    flat_values = np.concatenate(flat_parts) if flat_parts else np.asarray([], dtype=float)
    return {
        "kind": "ndarray_ragged",
        "values": flat_values,
        "offsets": offsets,
        "lengths": lengths,
        "shapes": np.asarray(shape_rows, dtype=h5py.string_dtype("utf-8")),
        "isnull": mask,
    }


def _encode_json_column(values: list[Any]) -> dict[str, Any]:
    payload = np.asarray(
        ["" if value is None else json.dumps(value) for value in values], dtype=h5py.string_dtype("utf-8")
    )
    mask = np.asarray([value is None for value in values], dtype=bool)
    return {"kind": "json", "values": payload, "isnull": mask}


def _encode_repr_column(values: list[Any]) -> dict[str, Any]:
    payload = np.asarray(["" if value is None else repr(value) for value in values], dtype=h5py.string_dtype("utf-8"))
    mask = np.asarray([value is None for value in values], dtype=bool)
    return {"kind": "repr", "values": payload, "isnull": mask}


def _encode_opt_column(values: list[Any]) -> dict[str, Any]:
    non_null = [value for value in values if value is not None]
    encoded: dict[str, Any]

    if not non_null:
        encoded = _encode_numeric_column(values)
    elif all(isinstance(value, (bool, np.bool_)) for value in non_null):
        encoded = _encode_bool_column(values)
    elif all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) for value in non_null
    ):
        encoded = _encode_numeric_column(values)
    elif all(isinstance(value, str) for value in non_null):
        encoded = _encode_string_column(values)
    elif all(isinstance(value, (np.ndarray, list, tuple)) for value in non_null):
        encoded = _encode_array_column(values)
    elif all(_is_jsonable(value) for value in non_null):
        encoded = _encode_json_column(values)
    else:
        encoded = _encode_repr_column(values)

    return encoded


def _decode_scalar_float(field_group: h5py.Group) -> list[Any]:
    values = field_group["values"][()]
    return [np.nan if np.isnan(value) else float(value) for value in values]


def _decode_scalar_bool(field_group: h5py.Group) -> list[Any]:
    values = field_group["values"][()]
    return [None if np.isnan(value) else bool(int(value)) for value in values]


def _decode_string(field_group: h5py.Group) -> list[Any]:
    values = field_group["values"].asstr()[()]
    mask = field_group["isnull"][()]
    return [None if mask[i] else values[i] for i in range(len(values))]


def _decode_ndarray_stack(field_group: h5py.Group) -> list[Any]:
    values = field_group["values"][()]
    mask = field_group["isnull"][()]
    return [None if mask[i] else values[i] for i in range(len(values))]


def _decode_ndarray_ragged(field_group: h5py.Group) -> list[Any]:
    flat = field_group["values"][()]
    offsets = field_group["offsets"][()]
    lengths = field_group["lengths"][()]
    shapes = field_group["shapes"].asstr()[()]
    mask = field_group["isnull"][()]
    out: list[Any] = []

    for i in range(len(offsets)):
        if mask[i]:
            out.append(None)
            continue
        start = int(offsets[i])
        stop = start + int(lengths[i])
        shape = ast.literal_eval(shapes[i])
        out.append(flat[start:stop].reshape(shape))

    return out


def _decode_json(field_group: h5py.Group) -> list[Any]:
    values = field_group["values"].asstr()[()]
    mask = field_group["isnull"][()]
    return [None if mask[i] else json.loads(values[i]) for i in range(len(values))]


def _decode_repr(field_group: h5py.Group) -> list[Any]:
    values = field_group["values"].asstr()[()]
    mask = field_group["isnull"][()]
    return [None if mask[i] else values[i] for i in range(len(values))]


_OPT_COLUMN_DECODERS: dict[str, Any] = {
    "scalar_float": _decode_scalar_float,
    "scalar_bool": _decode_scalar_bool,
    "string": _decode_string,
    "ndarray_stack": _decode_ndarray_stack,
    "ndarray_ragged": _decode_ndarray_ragged,
    "json": _decode_json,
    "repr": _decode_repr,
}


def _decode_opt_column(field_group: h5py.Group) -> list[Any]:
    kind = field_group.attrs["kind"]
    decoder = _OPT_COLUMN_DECODERS.get(kind)
    if decoder is None:
        raise ValueError(f"Unsupported opt_info kind: {kind}")
    return decoder(field_group)


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
        group_path = cls.standardize_group_path(group_path)
        is_root_group = group_path == "/"
        file_exists = os.path.exists(filename)

        # Overwriting the root group means replacing the whole file contents.
        file_mode = "a" if file_exists else "w"

        with h5py.File(filename, mode=file_mode) as f:
            if is_root_group:
                group = f
                existing_sections = [cls.meta_group, cls.parameter_group, cls.opt_group]
                has_existing_data = any(name in group for name in existing_sections)

                if has_existing_data:
                    if overwrite:
                        logger.warning("Overwriting root group in file '%s'.", filename)
                        for name in existing_sections:
                            if name in group:
                                del group[name]
                    else:
                        raise FileExistsError(
                            f"Root group '/' in file '{filename}' already contains model data. "
                            "Set overwrite=True to replace."
                        )
            else:
                # Check if the target subgroup already exists
                if group_path in f:
                    if overwrite:
                        logger.warning("Overwriting group '%s' in file '%s'.", group_path, filename)
                        del f[group_path]
                    else:
                        raise FileExistsError(
                            f"Group '{group_path}' already exists in file '{filename}'. Set overwrite=True to replace."
                        )

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
        result_names = [
            name
            for name in opt_data.get("result_names", [])
            if name in opt_data and any(value is not None for value in opt_data[name])
        ]
        opt_group.attrs["result_names"] = result_names
        opt_group.attrs["result_count"] = int(opt_data.get("result_count", 0))

        for key in result_names:
            field_group = opt_group.create_group(key)
            encoded = _encode_opt_column(opt_data[key])
            field_group.attrs["kind"] = encoded["kind"]
            for name, value in encoded.items():
                if name == "kind":
                    continue
                field_group.create_dataset(name, data=value, compression=compression)

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
        if "result_names" in opt_group.attrs:
            result_names = list(opt_group.attrs["result_names"])
            columns = {key: _decode_opt_column(opt_group[key]) for key in result_names if key in opt_group}
            n_results = int(
                opt_group.attrs.get("result_count", max((len(values) for values in columns.values()), default=0))
            )

            results: list[OptimizeResult] = []
            for i in range(n_results):
                row: dict[str, Any] = {}
                for key, values in columns.items():
                    if i < len(values):
                        row[key] = values[i]
                results.append(OptimizeResult(**row))
            return results

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
