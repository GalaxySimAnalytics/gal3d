import copy
import logging
from dataclasses import is_dataclass

import numpy as np

from ..shape import Structure3D
from .parameter import Parameters
from .util import save_model_hdf5


logger = logging.getLogger("gal3d.optimization.result")


class OptimizeResult:
    """
    A class to store and manage the results of optimization processes.

    This class is designed to handle the results of optimization procedures,
    allowing for easy access, combination, and representation of the results.

    Parameters
    ----------
    optimize_result : dataclass
        The result of an optimization process, expected to be a dataclass.

    Attributes
    ----------
    _results : list
        A list containing the optimization results.

    Methods
    -------
    keys()
        Returns the keys of the fields in the first optimization result.
    __len__()
        Returns the number of optimization results stored.
    __contains__(key)
        Checks if a key is present in the fields of the first optimization result.
    __repr__()
        Returns a string representation of the OptimizeResult object.
    __getitem__(key)
        Retrieves the values associated with a specific key across all results.
    __add__(other)
        Combines the results of two OptimizeResult objects if they are compatible.

    Raises
    ------
    TypeError
        If the input `optimize_result` is not a dataclass.
    ValueError
        If attempting to combine with an incompatible object.
    KeyError
        If attempting to access a non-existent key.
    """

    def __init__(self, optimize_result):
        if not is_dataclass(optimize_result):
            raise (f'{optimize_result} is not a dataclass')

        self._results = [optimize_result]

    def keys(self):
        """
        Returns the keys of the fields in the first optimization result.

        Returns
        -------
        list
            A list of keys representing the fields in the first optimization result.
        """
        return self._results[0].__dataclass_fields__.keys()

    def __len__(self):
        """
        Returns the number of optimization results stored.

        Returns
        -------
        int
            The number of optimization results stored in the object.
        """
        return len(self._results)

    def __contains__(self, key):
        """
        Checks if a key is present in the fields of the first optimization result.

        Parameters
        ----------
        key : str
            The key to check for in the fields of the first optimization result.

        Returns
        -------
        bool
            True if the key is present, False otherwise.
        """
        return key in self._results[0].__dataclass_fields__

    def __repr__(self):
        """
        Returns a string representation of the OptimizeResult object.

        Returns
        -------
        str
            A string representation of the OptimizeResult object.
        """
        return f"<|OptimizeResult| {len(self._results)} |>"

    def __getitem__(self, key):
        """
        Retrieves the values associated with a specific key across all results.

        Parameters
        ----------
        key : str
            The key to retrieve values for.

        Returns
        -------
        list
            A list of values associated with the key across all results.

        Raises
        ------
        KeyError
            If the key is not present in the fields of the first optimization result.
        """
        if key in self:
            return [getattr(i, key) for i in self._results]

        raise KeyError(key)

    def __add__(self, other):
        """
        Combines the results of two OptimizeResult objects if they are compatible.

        Parameters
        ----------
        other : OptimizeResult or dataclass
            Another OptimizeResult object or a dataclass to combine with.

        Returns
        -------
        OptimizeResult
            The combined OptimizeResult object.

        Raises
        ------
        ValueError
            If the objects are not compatible for combination.
        TypeError
            If the input is not a dataclass or OptimizeResult object.
        """
        if isinstance(other, OptimizeResult):
            if self.keys() <= other.keys():
                self._results = self._results + other._results
                return self

            raise ValueError(f'{other} is not the same dataclass')

        if not is_dataclass(other):
            raise TypeError(f'{other} is not a dataclass')

        if self.keys() <= other.__dataclass_fields__.keys():
            self._results.append(other)
            return self
        raise ValueError(f'{other} is not the same dataclass')


class ModelResult:
    """
    A class to store and manage the results of a 3D galaxy morphology fitting process.

    Parameters
    ----------
    structure : Structure3D
        An instance of the `Structure3D` class that defines the 3D structure model.
    optimize_result : dict or OptimizeResult
        The result of the optimization process, typically returned by a fitting algorithm.
    parameters : Parameters
        An instance of the `Parameters` class containing the fitted parameters.
    **kwargs : dict
        Additional keyword arguments that may be passed to the class.

    Attributes
    ----------
    _structure : Structure3D
        The 3D structure model used for fitting.
    res : OptimizeResult
        The optimization results, wrapped in an `OptimizeResult` instance.
    _parameters : list of Parameters
        A list containing the fitted parameters. Initially, it contains a single `Parameters` instance.

    Methods
    -------
    keys()
        Returns the keys of the parameters stored in the first `Parameters` instance.
    __call__(pos, item=0, **kwargs)
        Evaluates the 3D structure model at the given position using the specified set of parameters.
    __getitem__(k)
        Retrieves either a specific parameter or a specific set of parameters.
    __repr__()
        Returns a string representation of the `Result` object.
    __add__(other)
        Combines two `Result` objects if they share the same structure.
    __len__()
        Returns the number of parameter sets stored in the `Result` object.
    """

    def __init__(
        self, structure: Structure3D, optimize_result, parameters: Parameters, **kwargs
    ):
        """
        Initializes the `Result` class with the given structure, optimization result, and parameters.

        Parameters
        ----------
        structure : Structure3D
            The 3D structure model used for fitting.
        optimize_result : dict or OptimizeResult
            The result of the optimization process.
        parameters : Parameters
            The fitted parameters.
        **kwargs : dict
            Additional keyword arguments.
        """
        self._structure = structure
        self.res = OptimizeResult(optimize_result)

        self._parameters = [parameters]

    def keys(self):
        """
        Returns the keys of the parameters stored in the first `Parameters` instance.

        Returns
        -------
        list
            A list of parameter keys.
        """
        return self._parameters[0].keys()

    def __call__(self, pos, *, item: int = 0, **kwargs):
        """
        Evaluates the 3D structure model at the given position using the specified set of parameters.

        Parameters
        ----------
        pos : array-like
            The position at which to evaluate the model.
        item : int, optional
            The index of the parameter set to use (default is 0).
        **kwargs : dict
            Additional keyword arguments passed to the model evaluation.

        Returns
        -------
        array-like
            The evaluated model values at the given position.
        """
        return self[item](pos, **kwargs)

    def __getitem__(self, k):
        """
        Retrieves either a specific parameter or a specific set of parameters.

        Parameters
        ----------
        k : str or int
            If `k` is a string, returns the corresponding parameter value from all parameter sets.
            If `k` is an integer, returns the `Structure3D` instance initialized with the k-th parameter set.

        Returns
        -------
        numpy.ndarray or Structure3D
            The requested parameter values or the initialized structure model.

        Raises
        ------
        KeyError
            If `k` is neither a valid string key nor an integer index.
        """
        if isinstance(k, str):
            return np.array([i[k] for i in self._parameters])

        if isinstance(k, int):
            return self._structure.from_parameters(
                **self._parameters[k].structure_parameters
            )
        raise KeyError(f"{k} is not a valid key")

    def __repr__(self):
        """
        Returns a string representation of the `Result` object.

        Returns
        -------
        str
            A formatted string describing the `Result` object.
        """
        coor = self._structure._coordinate_name
        shape = self._structure._geometry_name
        error = self._structure._error_method_name
        lin1 = (
            "<Resullt| num="
            + str(len(self._parameters))
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

    def __add__(self, other):
        """
        Combines two `Result` objects if they share the same structure.

        Parameters
        ----------
        other : Result
            Another `Result` object to combine with.

        Returns
        -------
        Result
            The combined `Result` object.

        Raises
        ------
        ValueError
            If the structures of the two `Result` objects are different.
        TypeError
            If `other` is not an instance of `Result`.
        """
        if isinstance(other, ModelResult):
            if self._structure == other._structure:
                self.res = self.res + other.res
                self._parameters = self._parameters + other._parameters
                return self
            raise ValueError(f"{other} have a different structure")
        raise TypeError(f"{other} is not a Result type")

    def __len__(self):
        """
        Returns the number of parameter sets stored in the `Result` object.

        Returns
        -------
        int
            The number of parameter sets.
        """
        return len(self._parameters)


def model_to_hdf5(
    model: ModelResult,
    hdf5_file_name: str,
    shape_name: str,
    error_name: str,
    all_header='/',
    other_info=dict(),
):

    save_model_hdf5(
        model, hdf5_file_name, shape_name, error_name, all_header, other_info
    )
