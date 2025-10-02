import logging
import os
import time
from typing import Any

import h5py
import numpy as np

from gal3d.shape import Structure3D, StructureCore

from .result import ModelResult

logger = logging.getLogger("gal3d.optimization.model_io")

class ModelIO:
    """
    Handle I/O operations for ModelResult objects.

    This class provides methods for saving and loading model results to/from
    HDF5 files, separating these concerns from the main ModelResult class.
    """

    @staticmethod
    def _prepare_metadata(model: ModelResult, metadata: dict[str, Any] | None) -> dict[str, Any]:
        # Initialize metadata
        meta = {
            "save_timestamp": time.time(),
            "coordinate_name": model._structure._coordinate_name,
            "geometry_name": model._structure._geometry_name,
        }
        if hasattr(model._structure, "_error_method_name"):
            meta["error_method_name"] = model._structure._error_method_name
        if hasattr(model._structure, "_error_func_name"):
            meta["error_func_name"] = model._structure._error_func_name
        meta["parameter_count"] = len(model._param_sets)

        # Add user metadata
        if metadata:
            meta.update(metadata)
        return meta

    @staticmethod
    def _save_metadata(group, meta, compression):
        for key, value in meta.items():
            if isinstance(value, int | float | str | bool):
                group.attrs[key] = value
            else:
                try:
                    if isinstance(value, (np.ndarray | list | tuple)) and len(value) > 0:
                        group.create_dataset(f"meta_{key}", data=value, compression=compression)
                    else:
                        group.attrs[f"meta_{key}"] = str(value)
                except (TypeError, ValueError):
                    group.attrs[f"meta_{key}"] = str(value)

    @staticmethod
    def _save_parameters(params_group, model, compression, info_keys):
        for i in model.keys():
            params_group.create_dataset(i, data=model[i], compression=compression)
            params_group.create_dataset(f"{i}_lb", data=model[f"{i}_lb"], compression=compression)
            params_group.create_dataset(f"{i}_ub", data=model[f"{i}_ub"], compression=compression)
            params_group.create_dataset(f"{i}_err", data=model[f"{i}_err"], compression=compression)

         # Save info keys for each parameter
        for info_key in info_keys:
            try:
                info_values = [param_set.get_info(info_key) for param_set in model._param_sets]
                if any(v is not None for v in info_values):
                    try:
                        info_array = np.array(info_values)
                        params_group.create_dataset(f"{info_key}", data=info_array, compression=compression)
                    except (ValueError, TypeError):
                        str_info = [str(v) if v is not None else "None" for v in info_values]
                        params_group.create_dataset(f"{info_key}", data=np.array(str_info), compression=compression)
            except Exception as e:
                logger.warning("Failed to save info key '%s' for parameter set: %s", info_key, e)

    @staticmethod
    def _save_opt_results(opt_group, model, compression, result_keys):
        for i, opt_result in enumerate(model._opt_results):
            opt_set = opt_group.create_group(f"result_{i}")
            for key in result_keys:
                if key not in opt_result:
                    continue
                value = opt_result[key]
                try:
                    if value is None:
                        opt_set.attrs[f"{key}_is_none"] = True
                    elif isinstance(value, (bool | int | float | str)):
                        opt_set.attrs[key] = value
                    else:
                        try:
                            array_value = np.asarray(value)
                            if array_value.shape:
                                opt_set.create_dataset(key, data=array_value, compression=compression)
                            else:
                                opt_set.create_dataset(key, data=array_value)
                        except Exception:
                            opt_set.attrs[f"{key}_repr"] = repr(value)
                except (TypeError, ValueError):
                    opt_set.attrs[f"{key}_repr"] = repr(value)

    @staticmethod
    def save_to_hdf5(
        model: ModelResult,
        filename: str,
        group_path: str = "/",
        metadata: dict[str, Any] | None = None,
        compression: str | None = "gzip",
        overwrite: bool = False,
        result_keys: tuple[str, ...] = ("cost", "success", "n_fun_evals", "n_iterations"),
        info_keys: tuple[str, ...] = ("parameter",)
    ) -> None:
        """
        Save a model result to an HDF5 file.

        Parameters
        ----------
        model : ModelResult
            The model result to save
        filename : str
            Path to the HDF5 file
        group_path : str, optional
            Path within the HDF5 file where data should be stored, default is "/"
        metadata : dict, optional
            Additional metadata to store with the model
        compression : str, optional
            Compression type for datasets, default is "gzip"
        overwrite : bool, optional
            Whether to overwrite existing data at the specified group path, default is False
        result_keys : tuple[str, ...], optional
            Keys from OptimizeResult to save, default is ("cost","success","n_fun_evals","n_iterations")
        info_keys : tuple[str, ...], optional
            Parameter info keys to save, default is ("parameter",)

        Raises
        ------
        IOError
            If file writing fails
        ValueError
            If the group path already exists and overwrite=False
        """
        try:
            start_time = time.time()

            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filename))
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Standardize group path
            if not group_path.startswith("/"):
                group_path = "/" + group_path
            if not group_path.endswith("/"):
                group_path += "/"

            meta = ModelIO._prepare_metadata(model, metadata)

            # Create or open the file
            file_mode = "a" if os.path.exists(filename) else "w"
            with h5py.File(filename, file_mode) as f:
                # Check if group exists and handle accordingly
                if group_path in f:
                    if overwrite:
                        del f[group_path]
                    else:
                        raise ValueError(f"Group {group_path} already exists in {filename}. Set overwrite=True to replace.")

                # Create group for this model
                group = f.create_group(group_path)

                # Save metadata directly as attributes
                ModelIO._save_metadata(group, meta, compression)


                # Save parameters
                params_group = group.create_group("parameters")
                ModelIO._save_parameters(params_group, model, compression, info_keys)

                # Save optimization results
                opt_group = group.create_group("optimization_results")
                ModelIO._save_opt_results(opt_group, model, compression, result_keys)

            elapsed = time.time() - start_time
            logger.info("Model saved to %s in %.2f seconds (parameters: %d)", filename, elapsed, len(model._param_sets))

        except OSError as e:
            logger.error("Failed to save model to %s: %s", filename, e)
            raise OSError(f"Failed to save model to {filename}: ") from e
        except Exception as e:
            logger.error("Unexpected error saving model: %s", e)
            raise Exception("Failed to save model") from e


    @staticmethod
    def _load_structure(group, structure):
        if structure is not None:
            return structure
        if ("error_func_name" in group.attrs) and ("error_method_name" in group.attrs):
            return Structure3D(
                group.attrs["coordinate_name"],
                group.attrs["geometry_name"],
                group.attrs["error_func_name"],
                group.attrs["error_method_name"],
            )
        else:
            return StructureCore(
                group.attrs["coordinate_name"],
                group.attrs["geometry_name"],
            )

    @staticmethod
    def _load_parameters(params_group):
        from gal3d.optimization.parameter import Parameter, Parameters

        all_keys = list(params_group.keys())

        # Identify all parameter names with complete data
        param_names = set()
        info_key_mapping = {}

        # Find candidate parameter names (those without suffixes)
        candidates = {key for key in all_keys if not any(key.endswith(suffix) for suffix in ["_lb", "_ub", "_err"])}

        # Check each candidate to see if it's a complete parameter
        for name in candidates:
            if (f"{name}_lb" in all_keys and
                f"{name}_ub" in all_keys and
                f"{name}_err" in all_keys):
                param_names.add(name)
            else:
                info_key_mapping[name] = params_group[name][()]

        # Extract parameter data as arrays
        param_values = {name: params_group[name][()] for name in param_names}
        param_lbs = {name: params_group[f"{name}_lb"][()] for name in param_names}
        param_ubs = {name: params_group[f"{name}_ub"][()] for name in param_names}
        param_errs = {name: params_group[f"{name}_err"][()] for name in param_names}

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
            for key, values in info_key_mapping.items():
                try:
                    if i < len(values):
                        value = values[i]
                        if isinstance(value, str) and value.lower() == "none":
                            continue
                        param_set.add_info(**{key: value})
                except Exception as e:
                    logger.warning("Failed to add info key '%s': %s", key, e)
            param_sets.append(param_set)
        return param_sets

    @staticmethod
    def _load_opt_results(opt_group):
        from gal3d.optimization.optimizer import OptimizeResult

        opt_results = []
        for opt_result_name in sorted(opt_group.keys()):
            opt_data = opt_group[opt_result_name]
            result_dict: dict[str, Any] = {}
            # Load attributes
            for key, value in opt_data.attrs.items():
                if key.endswith("_is_none"):
                    base_key = key.rsplit("_is_none", 1)[0]
                    result_dict[base_key] = None
                elif key.endswith("_repr"):
                    continue
                else:
                    result_dict[key] = value
            # Load datasets
            for key in opt_data.keys():
                result_dict[key] = opt_data[key][()]
            opt_results.append(OptimizeResult(**result_dict))
        return opt_results

    @staticmethod
    def load_from_hdf5(
        filename: str,
        group_path: str = "/",
        structure: Structure3D | StructureCore | None = None
    ) -> ModelResult:
        """
        Load a ModelResult from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file
        group_path : str, optional
            Path within the HDF5 file where data is stored, default is "/"
        structure : Structure3D, optional
            Structure object to associate with the loaded model. If None, a structure
            will be created from metadata in the file.

        Returns
        -------
        ModelResult
            The loaded model result

        Raises
        ------
        IOError
            If file reading fails
        KeyError
            If the group path or required data is not found
        ValueError
            If the model structure doesn't match the saved metadata
        """
        try:
            start_time = time.time()

            # Standardize group path
            if not group_path.startswith("/"):
                group_path = "/" + group_path
            if not group_path.endswith("/"):
                group_path += "/"

            # Verify structure compatibility
            with h5py.File(filename, "r") as f:
                if group_path not in f:
                    raise KeyError(f"Group {group_path} not found in {filename}")
                group = f[group_path]

                structure = ModelIO._load_structure(group, structure)
                param_sets = ModelIO._load_parameters(group["parameters"])
                opt_results = ModelIO._load_opt_results(group["optimization_results"])

                # Create and initialize the model result
                if param_sets and opt_results:
                    result = ModelResult(
                        structure=structure,
                        optimize_result=opt_results[0],
                        parameters=param_sets[0]
                    )

                    # Replace the lists with all loaded data
                    result._param_sets = param_sets
                    result._opt_results = opt_results

                    elapsed = time.time() - start_time
                    logger.info("Model loaded from %s in %.2f seconds (parameters: %d)", filename, elapsed, len(param_sets))
                    return result
                else:
                    raise ValueError("No parameters or optimization results found in the file")

        except OSError as e:
            logger.error("Failed to load model from %s: %s",filename, e)
            raise OSError(f"Failed to load model from {filename}: ") from e
        except Exception as e:
            logger.error("Unexpected error loading model: %s", e)
            raise RuntimeError("Failed to load model: ") from e
