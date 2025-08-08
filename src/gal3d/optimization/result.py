import copy
import logging
import time
from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, overload, Self,cast

import numpy as np

from ..shape import Structure3D
from .optimizer import OptimizeResult
from .parameter import Parameters
from .util import save_model_hdf5

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
        structure: Structure3D, 
        optimize_result: OptimizeResult,
        parameters: Parameters, 
    ):
        """
        Initialize the ModelResult with structure, optimization results, and parameters.
        
        Parameters
        ----------
        structure : Structure3D
            The 3D structure model used for fitting.
        optimize_result : OptimizeResult
            The result of the optimization process.
        parameters : Parameters
            The fitted parameters.
        """
        self._structure = structure
        self._opt_results = [optimize_result]
        self._param_sets = [parameters]
        
        # Cache OptimizeResult fields for __getattr__ and __dir__
        self._optimize_result_attrs = {
            field.name for field in fields(OptimizeResult)
        }
        
        # Include properties from OptimizeResult
        for name in ['x', 'x0', 'nfev', 'nit', 'njev', 'nhev']:
            self._optimize_result_attrs.add(name)
            
    # OptimizeResult property accessors with proper type hints
    @property
    def params(self) -> np.ndarray:
        """Get parameter arrays from all optimization results."""
        return np.array([r.params for r in self._opt_results])
    
    @property
    def fun(self) -> np.ndarray:
        """Get objective function values from all optimization results."""
        return np.array([r.fun for r in self._opt_results])
    
    @property
    def start_params(self) -> np.ndarray:
        """Get starting parameter arrays from all optimization results."""
        return np.array([r.start_params for r in self._opt_results])
    
    @property
    def start_fun(self) -> np.ndarray:
        """Get starting objective function values from all optimization results."""
        return np.array([r.start_fun for r in self._opt_results])
    
    @property
    def algorithm(self) -> List[str | None]:
        """Get algorithm names from all optimization results."""
        return [r.algorithm for r in self._opt_results]
    
    @property
    def success(self) -> np.ndarray:
        """Get success flags from all optimization results."""
        values = [r.success if r.success is not None else np.nan for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def message(self) -> List[str]:
        """Get messages from all optimization results."""
        return [r.message if r.message is not None else "" for r in self._opt_results]
    
    @property
    def status(self) -> np.ndarray:
        """Get status codes from all optimization results."""
        values = [r.status if r.status is not None else np.nan for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def n_fun_evals(self) -> np.ndarray:
        """Get number of function evaluations from all optimization results."""
        values = [r.n_fun_evals if r.n_fun_evals is not None else np.nan for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def n_jac_evals(self) -> np.ndarray:
        """Get number of Jacobian evaluations from all optimization results."""
        values = [r.n_jac_evals if r.n_jac_evals is not None else np.nan for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def n_hess_evals(self) -> np.ndarray:
        """Get number of Hessian evaluations from all optimization results."""
        values = [r.n_hess_evals if r.n_hess_evals is not None else np.nan for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def n_iterations(self) -> np.ndarray:
        """Get number of iterations from all optimization results."""
        values = [r.n_iterations if r.n_iterations is not None else np.nan for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def jac(self) -> List[Optional[np.ndarray]]:
        """Get Jacobian arrays from all optimization results."""
        return [r.jac for r in self._opt_results]
    
    @property
    def hess(self) -> List[Optional[np.ndarray]]:
        """Get Hessian arrays from all optimization results."""
        return [r.hess for r in self._opt_results]
    
    @property
    def hess_inv(self) -> List[Optional[np.ndarray]]:
        """Get inverse Hessian arrays from all optimization results."""
        return [r.hess_inv for r in self._opt_results]
    
    @property
    def max_constraint_violation(self) -> np.ndarray:
        """Get maximum constraint violations from all optimization results."""
        values = [r.max_constraint_violation if r.max_constraint_violation is not None else np.nan 
                 for r in self._opt_results]
        return np.array(values, dtype=float)
    
    @property
    def history(self) -> List[Any]:
        """Get optimization histories from all optimization results."""
        return [r.history for r in self._opt_results]
    
    @property
    def algorithm_output(self) -> List[Optional[Dict[str, Any]]]:
        """Get algorithm outputs from all optimization results."""
        return [r.algorithm_output for r in self._opt_results]
    
    @property
    def multistart_info(self) -> List[Optional[Dict[str, Any]]]:
        """Get multistart information from all optimization results."""
        return [r.multistart_info for r in self._opt_results]
    
    # Property aliases for scipy.optimize compatibility
    @property
    def x(self) -> np.ndarray:
        """Alias for params (scipy.optimize compatibility)."""
        return self.params
    
    @property
    def x0(self) -> np.ndarray:
        """Alias for start_params (scipy.optimize compatibility)."""
        return self.start_params
    
    @property
    def nfev(self) -> np.ndarray:
        """Alias for n_fun_evals (scipy.optimize compatibility)."""
        return self.n_fun_evals
    
    @property
    def nit(self) -> np.ndarray:
        """Alias for n_iterations (scipy.optimize compatibility)."""
        return self.n_iterations
    
    @property
    def njev(self) -> np.ndarray:
        """Alias for n_jac_evals (scipy.optimize compatibility)."""
        return self.n_jac_evals
    
    @property
    def nhev(self) -> np.ndarray:
        """Alias for n_hess_evals (scipy.optimize compatibility)."""
        return self.n_hess_evals

    def keys(self):
        """
        Get parameter keys from the first parameter set.

        """
        return self._param_sets[0].keys()

    def __call__(
        self, 
        pos: Union[np.ndarray, list, tuple], 
        *, 
        item: int = 0, 
        **kwargs
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
    
    # Type hint overloads for __getitem__
    @overload
    def __getitem__(self, k: int) -> Structure3D: ...
    
    @overload
    def __getitem__(self, k: str) -> np.ndarray: ...

    @overload
    def __getitem__(self, k: slice) -> 'ModelResult': ...

    def __getitem__(self, k: Union[int, str, slice]) -> Union[Structure3D, np.ndarray, 'ModelResult']:
        """
        Retrieves either specific parameters, a specific parameter set, or a slice of parameter sets.

        Parameters
        ----------
        k : str, int, or slice
            - If `k` is a string: returns the corresponding parameter value from all parameter sets.
            - If `k` is an integer: returns the `Structure3D` instance initialized with the k-th parameter set.
            - If `k` is a slice: returns a new `ModelResult` with the specified slice of parameter sets.

        Returns
        -------
        numpy.ndarray, Structure3D, or ModelResult
            The requested parameter values, initialized structure model, or sliced ModelResult.

        Raises
        ------
        KeyError
            If `k` is not a valid string key, integer index, or slice.
        """
        if isinstance(k, str):
            # Return parameter values for all sets
            return cast(np.ndarray, np.array([params[k] for params in self._param_sets]))
        
        elif isinstance(k, int):
            # Return the Structure3D initialized with parameters at index k
            if k < 0 or k >= len(self._param_sets):
                raise IndexError(f"Index {k} out of bounds (0-{len(self._param_sets)-1})")
                
            return self._structure.from_parameters(
                **self._param_sets[k].structure_parameters
            )
            
        elif isinstance(k, slice):
            # Create a new ModelResult with sliced parameters and results
            sliced_result = copy.copy(self)
            sliced_result._param_sets = self._param_sets[k]
            sliced_result._opt_results = self._opt_results[k]
            return sliced_result
        else:
            raise KeyError(f"Key must be a string, integer, or slice, got {type(k).__name__}")

    def __getattr__(self, name: str) -> Union[List[Any], np.ndarray]:
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
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if not hasattr(self, '_optimize_result_attrs'):
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
    def __dir__(self) -> List[str]:
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
        error = self._structure._error_method_name
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
        result = ''.join([i.center(lenmax, " ") + "\n" for i in lins])

        return result

    def __add__(self, other) -> Self:
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
        if not isinstance(other, ModelResult):
            raise TypeError(f"{other} is not a ModelResult type")
            
        if self._structure != other._structure:
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



def model_to_hdf5(
    model: ModelResult,
    hdf5_file_name: str,
    shape_name: str,
    error_name: str,
    all_header: str = '/',
    large_model_threshold: int = 1000,
    other_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model results to an HDF5 file.
    
    Parameters
    ----------
    model : ModelResult
        The fitted model result to save.
    hdf5_file_name : str
        Path to the output HDF5 file.
    shape_name : str
        Name identifier for the shape being saved.
    error_name : str
        Name identifier for the error method used.
    all_header : str, optional
        Group path within the HDF5 file, by default '/'.
    large_model_threshold : int, optional
        Threshold for parameter sets to trigger a warning (default: 1000).
    other_info : dict, optional
        Additional metadata to store with the model.

    Raises
    ------
    IOError
        If file cannot be written.
    ValueError
        If model or parameters are invalid.
    TimeoutError
        If operation takes too long.
    """
    try:
        start_time = time.time()
        max_time = 300  # 5 minutes timeout
        
        info_dict = {} if other_info is None else other_info.copy()
        
        # Record metadata
        info_dict.update({
            'save_timestamp': time.time(),
            'geometry_name': getattr(model._structure, '_geometry_name', 'unknown'),
            'coordinate_name': getattr(model._structure, '_coordinate_name', 'unknown'),
        })
            
        logger.info(f"Saving model to {hdf5_file_name} (shape={shape_name}, error={error_name})")
        
        # Warn for large models
        if len(model) > large_model_threshold:
            logger.warning(
                f"Large model with {len(model)} parameter sets exceeds threshold "
                f"({large_model_threshold}) and may take time to save"
            )
            
        save_model_hdf5(
            model, hdf5_file_name, shape_name, error_name, all_header, info_dict
        )
        
        elapsed = time.time() - start_time
        if elapsed > 5.0:  # Log timing if significant
            logger.info(f"Model saving completed in {elapsed:.1f} seconds")
            
        logger.info(f"Successfully saved model with {len(model)} parameter sets")
        
    except ImportError as e:
        logger.error(
            f"Failed to save model to HDF5: {e}. "
            f"Possible causes: insufficient disk space, file permission issues, "
            f"or invalid model structure.",
            exc_info=True
        )
        raise
    except TimeoutError:
        logger.error(f"Save operation timed out after {max_time} seconds")
        raise
    except Exception as e:
        logger.error(f"Failed to save model to HDF5: {e}", exc_info=True)
        raise
