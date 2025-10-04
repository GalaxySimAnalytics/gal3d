import logging
import os
import time
from abc import abstractmethod
from typing import Any, Literal

from gal3d.plugin import PluginBase, PluginManager
from gal3d.shape import Structure3D, StructureCore

from .optimizer import OptimizeResult
from .parameter import Parameters
from .result import ModelResult

logger = logging.getLogger("gal3d.optimization.model_io")

class MetaDataDict(dict[str, Any]):
    """
    Dictionary subclass for storing model metadata.

    Attributes
    ----------
    coordinate_name : str
        Name of the coordinate system.
    geometry_name : str
        Name of the geometry.
    error_method_name : str | None
        Name of the error method, if any.
    error_func_name : str | None
        Name of the error function, if any.
    model_count : int
        Number of models stored.
    """
    coordinate_name: str
    geometry_name: str
    error_method_name: str | None
    error_func_name: str | None
    model_count: int

class ModelIOBase(PluginBase):
    """
    Abstract base class for model I/O operations.

    Subclasses must implement methods for saving and loading model data.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Registers the subclass as a model I/O plugin.
        """
        super().__init_subclass__(**kwargs)
        ModelIO.register(cls)

    @classmethod
    def save(cls,
        model: ModelResult,
        filename: str,
        info_keys: tuple[str, ...] = ("parameter",),
        result_keys: tuple[str, ...] = ("cost", "success", "n_fun_evals", "n_iterations"),
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        model : ModelResult
            The model to save.
        filename : str
            The name of the file to save the model to.
        info_keys : tuple[str, ...], optional
            Keys of additional info to save from the model.
        result_keys : tuple[str, ...], optional
            Keys of optimization results to save.
        metadata : dict[str, Any] | None, optional
            Additional metadata to include in the saved file.
        overwrite : bool, optional
            Whether to overwrite the file if it exists.
        **kwargs : Any
            Additional keyword arguments for the save function.
        """
        start_time = time.time()
        cls.check_file_path(filename=filename)
        data = cls.extract_data_from_model(model, info_keys, result_keys)
        if metadata is not None:
            data["meta"].update(metadata)
        cls._save(data, filename, overwrite=overwrite, **kwargs)
        elapsed = time.time() - start_time
        logger.info(
            "Model saved to %s in %.2f seconds (n: %d)",
            filename, elapsed, data["meta"]["model_count"]
        )

    @classmethod
    @abstractmethod
    def _save(cls,
        data: dict[Literal["meta", "parameters", "opt_info"], dict[str, Any]],
        filename: str,
        overwrite: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Abstract method to save data to a file.

        Parameters
        ----------
        data : dict
            The data to save.
        filename : str
            The name of the file to save the data to.
        overwrite : bool, optional
            Whether to overwrite the file if it exists.
        **kwargs : Any
            Additional keyword arguments.
        """


    @classmethod
    def load(cls,
        filename: str,
        structure: Structure3D | StructureCore | None = None,
        **kwargs: Any
    ) -> ModelResult:
        """
        Load the model from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the model from.
        structure : Structure3D | StructureCore | None, optional
            Structure object to associate with the loaded model. If None, a structure
            will be created from metadata in the file.
        **kwargs : Any
            Additional keyword arguments for the load function.

        Returns
        -------
        ModelResult
            The loaded model result.
        """
        start_time = time.time()

        if structure is None:
            metadata = cls._load_metadata_from_file(filename,**kwargs)
            if metadata["error_method_name"] is None or metadata["error_func_name"] is None:
                structure = StructureCore(
                    coordinate=metadata["coordinate_name"],
                    geometry=metadata["geometry_name"]
                )
            else:
                structure = Structure3D(
                    coordinate=metadata["coordinate_name"],
                    geometry=metadata["geometry_name"],
                    error_method=metadata["error_method_name"],
                    error_func=metadata["error_func_name"]
                )

        param_sets = cls._load_parameters_from_file(filename,**kwargs)
        opt_results = cls._load_opt_from_file(filename,**kwargs)
        result = ModelResult(
            structure = structure,
            optimize_result=opt_results[0],
            parameters=param_sets[0]
        )
        # Replace the lists with all loaded data
        result._param_sets = param_sets
        result._opt_results = opt_results
        elapsed = time.time() - start_time
        logger.info(
            "Model loaded from %s in %.2f seconds (n: %d)",
            filename, elapsed, len(param_sets)
        )
        return result

    @classmethod
    @abstractmethod
    def _load_metadata_from_file(
        cls, filename: str, **kwargs: Any
    ) -> MetaDataDict:
        """
        Abstract method to load metadata from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load metadata from.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        MetaDataDict
            The loaded metadata.
        """

    @classmethod
    @abstractmethod
    def _load_parameters_from_file(cls, filename: str, **kwargs: Any) -> list[Parameters]:
        """
        Abstract method to load parameters from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load parameters from.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[Parameters]
            The loaded parameter sets.
        """

    @classmethod
    @abstractmethod
    def _load_opt_from_file(cls, filename: str, **kwargs: Any) -> list[OptimizeResult]:
        """
        Abstract method to load optimization results from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load optimization results from.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[OptimizeResult]
            The loaded optimization results.
        """

    @classmethod
    def check_file_path(cls, filename: str) -> str:
        """
        Check and create the directory path for the file if it does not exist.

        Parameters
        ----------
        filename : str
            The file path to check.

        Returns
        -------
        str
            The directory path.
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("Created directory: %s", directory)
        return directory

    @classmethod
    def _extract_metadata_from_model(cls, model: ModelResult) -> MetaDataDict:
        """
        Extract metadata from the model.

        Parameters
        ----------
        model : ModelResult
            The model to extract metadata from.

        Returns
        -------
        MetaDataDict
            The extracted metadata.
        """
        meta_data: MetaDataDict = MetaDataDict(
            coordinate_name=model._structure._coordinate_name,
            geometry_name=model._structure._geometry_name,
            error_method_name=getattr(model._structure, "_error_method_name", None),
            error_func_name=getattr(model._structure, "_error_func_name", None)
        )
        meta_data["model_count"] = len(model._param_sets)
        return meta_data

    @classmethod
    def _extract_parameters_from_model(
        cls,
        model: ModelResult,
        info_keys: tuple[str, ...] = ("parameter",)
    ) -> dict[str, Any]:
        """
        Extract parameters from the model.

        Parameters
        ----------
        model : ModelResult
            The model to extract parameters from.
        info_keys : tuple[str, ...], optional
            Keys of additional info to extract.

        Returns
        -------
        dict[str, Any]
            The extracted parameters and info.
        """
        param_data = {}
        param_names = list(model.keys())
        info_names = []
        for i in param_names:
            param_data[i] = model[i]
            param_data[f"{i}_lb"] = model[f"{i}_lb"]
            param_data[f"{i}_ub"] = model[f"{i}_ub"]
            param_data[f"{i}_err"] = model[f"{i}_err"]

        for i in info_keys:
            try:
                info = [param_set.get_info(i) for param_set in model._param_sets]
                param_data[i] = info
                info_names.append(i)
            except KeyError:
                logger.warning("Info key '%s' not found in full parameter sets.", i)

        param_data["param_names"] = param_names
        param_data["info_names"] = info_names
        return param_data

    @classmethod
    def _extract_opt_from_model(
        cls,
        model: ModelResult,
        result_keys: tuple[str, ...] = ("cost", "success", "n_fun_evals", "n_iterations")
    ) -> dict[str, Any]:
        """
        Extract optimization results from the model.

        Parameters
        ----------
        model : ModelResult
            The model to extract optimization results from.
        result_keys : tuple[str, ...], optional
            Keys of optimization results to extract.

        Returns
        -------
        dict[str, Any]
            The extracted optimization results.
        """
        result_data = {}
        for i, opt_result in enumerate(model._opt_results):
            this_res = {}
            for key in result_keys:
                if key not in opt_result:
                    logger.debug("Result key '%s' not found in optimization result %d.", key, i)
                    continue
                this_res[key] = opt_result[key]
            result_data[f"result_{i}"] = this_res
        return result_data

    @classmethod
    def extract_data_from_model(
        cls,
        model: ModelResult,
        info_keys: tuple[str, ...] = ("parameter",),
        result_keys: tuple[str, ...] = ("cost", "success", "n_fun_evals", "n_iterations")
    ) -> dict[Literal["meta", "parameters", "opt_info"], dict[str, Any] | MetaDataDict]:
        """
        Extract all relevant data from a ModelResult for saving or management.

        Parameters
        ----------
        model : ModelResult
            The model from which to extract data.
        info_keys : tuple[str, ...], optional
            Keys of additional info to extract.
        result_keys : tuple[str, ...], optional
            Keys of optimization results to extract.

        Returns
        -------
        dict
            Dictionary containing metadata, parameters, and optimization info.
        """
        # Combine all extracted data
        combined_data: dict[Literal["meta", "parameters", "opt_info"], dict[str, Any]] = {
            "meta": cls._extract_metadata_from_model(model),
            "parameters": cls._extract_parameters_from_model(model, info_keys),
            "opt_info": cls._extract_opt_from_model(model, result_keys)
        }
        return combined_data

class ModelIO(PluginManager[ModelIOBase]):
    """Factory class for accessing registered model I/O plugins."""
    _plugins = {}
    _plugin_module = "gal3d.optimization.model_io_plugins"
    _base_class = ModelIOBase
