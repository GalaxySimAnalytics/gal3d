"""
Model result classes for optimization algorithms.

"""
import copy
import logging
from typing import TYPE_CHECKING, Any, Union, cast, overload

import numpy as np

from gal3d.shape import Structure3D, StructureCore

from .optimizer import OptimizeResult

if TYPE_CHECKING:
    from .parameter import Parameters

logger = logging.getLogger("gal3d.optimization.result")


class ModelResult:
    """
    A class to store and manage the results of a 3D galaxy morphology fitting process.

    This class stores the structure model, optimization results, and fitted parameters,
    supporting operations to access, combine, and manipulate fitting results.

    Parameters
    ----------
    structure : Structure3D
        An instance of the `Structure3D` class that defines the 3D structure model.
    optimize_result : OptimizeResult
        The result of the optimization process, containing fit details and metrics.
    parameters : Parameters
        An instance of the `Parameters` class containing the fitted parameters.
    **kwargs : dict
        Additional keyword arguments that may be passed to the class.

    Attributes
    ----------
    """

    def __init__(
        self,
        structure: Structure3D | StructureCore,
        optimize_result: OptimizeResult,
        parameters: "Parameters",
    ):
        """
        Initialize the ModelResult with structure, optimization results, and parameters.

        Parameters
        ----------
        structure : Structure3D or StructureCore
            The structure model used for fitting.
        optimize_result : OptimizeResult
            The result of the optimization process.
        parameters : Parameters
            The fitted parameters.
        """
        self._structure = structure
        self._opt_results = [optimize_result]
        self._param_sets = [parameters]

        # Cache OptimizeResult fields for __getattr__ and __dir__
        self._optimize_result_attrs = set(optimize_result.keys())

    @property
    def cost(self) -> np.ndarray:
        """Get cost values from all optimization results."""
        return np.array([r.cost for r in self._opt_results])

    def keys(self):
        """
        Get parameter keys from the first parameter set.
        """
        return self._param_sets[0].keys()

    def available_keys(self):
        """
        Get all available keys of parameters.
        """
        return self._param_sets[0].available_keys()

    @property
    def structure(self) -> Union["Structure3D", "StructureCore"]:
        """Get the 3D structure model"""
        return self._structure

    def estimate_errors(self, param_name: list[str] | str | None=None) -> dict[str, Any]:
        """
        Get the estimated errors for the model.
        """
        from gal3d.model_workflow.error_workflow import ErrorWorkflow
        if isinstance(param_name, str):
            param_name = [param_name]
        return ErrorWorkflow.estimate_error(self, param_name=param_name)

    def __call__(
        self,
        pos: np.ndarray | list | tuple,
        *,
        item: int = 0,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Evaluate the 3D structure model at given positions using specified parameter set.

        Parameters
        ----------
        pos : array-like
            Positions at which to evaluate the model.
        item : int, optional
            Index of parameter set to use (default: 0).
        **kwargs : dict
            Additional keyword arguments for the model.

        Returns
        -------
        array-like
            Model values at the given positions.
        """
        return self[item](pos, **kwargs)

    def get(self, key: str, index: int | None = None, default: Any = None) -> list[Any] | Any:
        """
        Mimic dict.get: Return the values for the given key from all parameter sets.

        Parameters
        ----------
        key : str
            The key to look up in the parameter sets.
        index : int, optional
            The index of the parameter set to return (default: None, return all).
        default : Any, optional
            Default value to return if key is not found.

        Returns
        -------
        Any
            The values for the given key from all parameter sets, or default if not found.
        """
        try:
            if index is not None:
                return self._param_sets[index][key]
            return [params[key] for params in self._param_sets]
        except KeyError:
            return default

    # Type hint overloads for __getitem__
    @overload
    def __getitem__(self, k: int) -> Structure3D | StructureCore: ...

    @overload
    def __getitem__(self, k: str) -> np.ndarray: ...

    @overload
    def __getitem__(self, k: slice) -> "ModelResult": ...

    @overload
    def __getitem__(self, k: np.ndarray) -> "ModelResult": ...

    def __getitem__(self, k: int | str | slice | np.ndarray) -> Union["StructureCore", "Structure3D", np.ndarray, "ModelResult"]:
        """
        Retrieves either specific parameters, a specific parameter set, or a slice of parameter sets.

        Parameters
        ----------
        k : str, int, slice, or numpy.ndarray
            - If `k` is a string: returns the corresponding parameter value from all parameter sets.
            - If `k` is an integer: returns the `StructureCore` or `Structure3D` instance initialized with the k-th parameter set.
            - If `k` is a slice: returns a new `ModelResult` with the specified slice of parameter sets.
            - If `k` is a numpy.ndarray of integers: returns a new `ModelResult` with parameter sets at those indices.
            - If `k` is a numpy.ndarray of booleans: returns a new `ModelResult` with parameter sets where mask is True.

        Returns
        -------
        numpy.ndarray, StructureCore, Structure3D, or ModelResult
            The requested parameter values, initialized structure model, or sliced ModelResult.

        Raises
        ------
        KeyError
            If `k` is not a valid string key, integer index, slice, or numpy array.
        IndexError
            If any index in the numpy array is out of bounds.
        ValueError
            If boolean array length doesn't match number of parameter sets.
        """
        if isinstance(k, str):
            try:
                return cast("np.ndarray", np.array([params[k] for params in self._param_sets]))
            except KeyError:
                pass

            try :
                if k.endswith(("_lb", "_ub", "_err")):
                    name, atr =k.rsplit("_",1)
                    return cast("np.ndarray", np.array([getattr(params[name], atr) for params in self._param_sets]))
            except KeyError:
                pass

            raise KeyError(f"Key '{k}' not found in parameter sets")

        elif isinstance(k, int):
            # Return the Structure3D initialized with parameters at index k
            if k < 0 or k >= len(self._param_sets):
                raise IndexError(f"Index {k} out of bounds (0-{len(self._param_sets)-1})")

            return self._structure.clone_with_parameters(
                **self._param_sets[k].structure_parameters
            )

        elif isinstance(k, slice):
            # Create a new ModelResult with sliced parameters and results
            sliced_result = copy.copy(self)
            sliced_result._param_sets = self._param_sets[k]
            sliced_result._opt_results = self._opt_results[k]
            return sliced_result

        elif isinstance(k, np.ndarray):
            # Handle numpy array indexing
            if k.dtype == bool:
                # Boolean mask array
                if len(k) != len(self._param_sets):
                    raise ValueError(
                        f"Boolean index array length ({len(k)}) doesn't match "
                        f"number of parameter sets ({len(self._param_sets)})"
                    )

                # Create a new ModelResult with masked parameters and results
                masked_result = copy.copy(self)
                masked_result._param_sets = [p for p, mask in zip(self._param_sets, k, strict=False) if mask]
                masked_result._opt_results = [r for r, mask in zip(self._opt_results, k, strict=False) if mask]
                return masked_result

            elif np.issubdtype(k.dtype, np.integer):
                # Integer index array

                # Check if all indices are in bounds
                if np.any((k < 0) | (k >= len(self._param_sets))):
                    out_of_bounds = k[(k < 0) | (k >= len(self._param_sets))]
                    raise IndexError(
                        f"Indices {out_of_bounds} out of bounds (0-{len(self._param_sets)-1})"
                    )

                # Create a new ModelResult with indexed parameters and results
                indexed_result = copy.copy(self)
                indexed_result._param_sets = [self._param_sets[i] for i in k]
                indexed_result._opt_results = [self._opt_results[i] for i in k]
                return indexed_result

            else:
                raise TypeError(
                    f"Numpy array index must be integers or booleans, got {k.dtype}"
                )
        else:
            raise KeyError(f"Key must be a string, integer, slice, or numpy array, got {type(k).__name__}")

    def _ipython_key_completions_(self) -> list[str]:
        return list(self._param_sets[0]._ipython_key_completions_())

    def __getattr__(self, name: str) -> list[Any] | np.ndarray:
        """
        Access attributes from all OptimizeResult instances.

        This provides access to optimization result attributes like 'fun', 'success',
        'message', etc. from all results as a list or array.

        Parameters
        ----------
        name : str
            Name of the OptimizeResult attribute to access

        Returns
        -------
        numpy.ndarray or list
            Array of values for numeric attributes, list for others

        Raises
        ------
        AttributeError
            If attribute doesn't exist in OptimizeResult
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if not hasattr(self, "_optimize_result_attrs"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in self._optimize_result_attrs:
            # Get attribute from all optimization results
            values = [getattr(res, name) for res in self._opt_results]

            # Convert to numpy array if all values are numeric
            if all(isinstance(v, (int, float, bool, np.number)) or v is None for v in values):
                # Replace None with np.nan for numeric arrays
                clean_values = [np.nan if v is None else v for v in values]
                return np.array(clean_values)
            return values

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __copy__(self):
        """
        Create a shallow copy of this ModelResult object.
        """
        new_obj = ModelResult(
            structure=self._structure,
            optimize_result=self._opt_results[0],
            parameters=self._param_sets[0]
        )
        new_obj._opt_results = self._opt_results.copy()
        new_obj._param_sets = self._param_sets.copy()
        new_obj._optimize_result_attrs = self._optimize_result_attrs.copy()
        return new_obj

    # Define __dir__ to show available attributes including OptimizeResult attributes
    def __dir__(self) -> list[str]:
        """List all attributes including those from OptimizeResult."""
        return sorted(set(super().__dir__()) | self._optimize_result_attrs)

    def __repr__(self) -> str:
        """
        Returns a string representation of the ModelResult object.

        Returns
        -------
        str
            A formatted string describing the ModelResult object.
        """
        coor = self._structure._coordinate_name
        shape = self._structure._geometry_name
        error = getattr(self._structure, "_error_method_name", "N/A")
        lin1 = (
            "<ModelResult| num="
            + str(len(self._param_sets))
            + " | "
            + coor
            + " | "
            + shape
            + " | "
            + error
            + " |>"
        )
        lin2 = "Parameters: " + str(list(self.keys()))
        lenmax = max(len(lin1), len(lin2))
        lins = [lin1, lin2]
        result = "".join([i.center(lenmax, " ") + "\n" for i in lins])

        return result

    def __add__(self, other: "ModelResult") -> "ModelResult":
        """
        Combines two ModelResult objects if they share the same structure.

        Parameters
        ----------
        other : ModelResult
            Another ModelResult object to combine with.

        Returns
        -------
        ModelResult
            The combined ModelResult object.

        Raises
        ------
        ValueError
            If the structures of the two ModelResult objects are different.
        TypeError
            If `other` is not an instance of ModelResult.
        """
        # Handle empty results
        if isinstance(other, EmptyModelResult):
            return self
        if isinstance(self, EmptyModelResult):
            return other

        if not isinstance(other, ModelResult):
            raise TypeError(f"{other} is not a ModelResult type")

        if not self._structure.is_equal(other._structure):
            raise ValueError(f"{other} has a different structure")

        # Create a new copy of self
        combined = self.__copy__()
        # Append the other's parameters and results
        combined._param_sets.extend(other._param_sets)
        combined._opt_results.extend(other._opt_results)
        return combined

    def __len__(self) -> int:
        """
        Returns the number of parameter sets stored in the ModelResult object.

        Returns
        -------
        int
            The number of parameter sets.
        """
        return len(self._param_sets)

    def save_to_file(
        self,
        filename: str,
        handler: str = "HDF5ModelIO",
        info_keys: tuple[str, ...] = ("parameter",),
        result_keys: tuple[str, ...] = ("cost","success","n_fun_evals","n_iterations"),
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Save the model result to an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file
        metadata : dict, optional
            Additional metadata to store with the model
        compression : str, optional
            Compression type for datasets, default is "gzip"
        overwrite : bool, optional
            Whether to overwrite existing data at the specified group path, default is False

        Raises
        ------
        IOError
            If file writing fails
        ValueError
            If the group path already exists and overwrite=False
        """
        from .model_io import ModelIO
        load = ModelIO.get_plugin(handler)
        load.save(self, filename,info_keys=info_keys,result_keys=result_keys, metadata=metadata,overwrite=overwrite, **kwargs)

    @classmethod
    def load_from_file(
        cls,
        filename: str,
        handler: str = "HDF5ModelIO",
        structure: Structure3D | StructureCore | None = None,
        **kwargs: Any,
    ) -> "ModelResult":
        """
        Load a ModelResult from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file
        structure : Structure3D or StructureCore
            Structure object to associate with the loaded model

        Returns
        -------
        ModelResult
            The loaded model result
        """
        from .model_io import ModelIO
        load = ModelIO.get_plugin(handler)
        return load.load(filename, structure,**kwargs)

load_model = ModelResult.load_from_file

class EmptyModelResult(ModelResult):
    def __init__(self):
        self._opt_results = []
        self._param_sets = []
        self._optimize_result_attrs = set()

    def __call__(self, *args, **kwargs):
        raise ValueError("EmptyModelResult: No results available.")

    def __getitem__(self, k):
        raise ValueError("EmptyModelResult: No results available.")

    def __repr__(self):
        return "<EmptyModelResult| No parameter sets |>"

    def __bool__(self):
        return False
