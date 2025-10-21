import abc
from .optimizer import OptimizeResult as OptimizeResult
from .parameter import Parameters as Parameters
from .result import ModelResult as ModelResult
from _typeshed import Incomplete
from abc import abstractmethod
from gal3d.plugin import PluginBase as PluginBase, PluginManager as PluginManager
from gal3d.shape import Structure3D as Structure3D, StructureCore as StructureCore
from typing import Any, Literal
from typing import Literal, overload
from gal3d.optimization.model_io_plugins.hdf_model_io import HDF5ModelIO

logger: Incomplete

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

class ModelIOBase(PluginBase, metaclass=abc.ABCMeta):
    """
    Abstract base class for model I/O operations.

    This class defines the interface for saving and loading model data.
    Subclasses must implement the following abstract methods to provide
    concrete I/O functionality (e.g., for different file formats):

    Methods to implement
    --------------------
    - _save(data, filename, overwrite=False, **kwargs):
        Save the extracted model data to a file.

    - _load_metadata_from_file(filename, **kwargs):
        Load only the metadata from a file.

    - _load_parameters_from_file(filename, **kwargs):
        Load all parameter sets from a file.

    - _load_opt_from_file(filename, **kwargs):
        Load all optimization results from a file.

    Notes
    -----
    Subclasses should ensure compatibility with the ModelResult structure
    and handle all required keys for saving and loading.

    The base class also provides utility methods for extracting data from
    ModelResult objects and for checking file paths.
    """
    def __init_subclass__(cls, **kwargs) -> None:
        """
        Registers the subclass as a model I/O plugin.
        """
    @classmethod
    def save(cls, model: ModelResult, filename: str, info_keys: tuple[str, ...] = ('parameter',), result_keys: tuple[str, ...] = ('cost', 'success', 'n_fun_evals', 'n_iterations'), metadata: dict[str, Any] | None = None, overwrite: bool = False, **kwargs: Any) -> None:
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
    @classmethod
    @abstractmethod
    def _save(cls, data: dict[Literal['meta', 'parameters', 'opt_info'], dict[str, Any]], filename: str, overwrite: bool = False, **kwargs: Any) -> None:
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
    def load(cls, filename: str, structure: Structure3D | StructureCore | None = None, **kwargs: Any) -> ModelResult:
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
    @classmethod
    @abstractmethod
    def _load_metadata_from_file(cls, filename: str, **kwargs: Any) -> MetaDataDict:
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
    @classmethod
    def _extract_parameters_from_model(cls, model: ModelResult, info_keys: tuple[str, ...] = ('parameter',)) -> dict[str, Any]:
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
    @classmethod
    def _extract_opt_from_model(cls, model: ModelResult, result_keys: tuple[str, ...] = ('cost', 'success', 'n_fun_evals', 'n_iterations')) -> dict[str, Any]:
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
    @classmethod
    def extract_data_from_model(cls, model: ModelResult, info_keys: tuple[str, ...] = ('parameter',), result_keys: tuple[str, ...] = ('cost', 'success', 'n_fun_evals', 'n_iterations')) -> dict[Literal['meta', 'parameters', 'opt_info'], dict[str, Any] | MetaDataDict]:
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

class ModelIO(PluginManager[ModelIOBase]):
    """Factory class for accessing registered model I/O plugins."""
    _plugins: Incomplete
    _plugin_module: str
    _base_class = ModelIOBase

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["HDF5ModelIO"]) -> type[HDF5ModelIO]: ...
