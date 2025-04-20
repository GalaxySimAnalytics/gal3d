import os
import logging
from abc import ABC, abstractmethod
from typing import List

from ..optimization.parameter import Parameters

logger = logging.getLogger("gal3d.shape.with_parameter")

required_attrs = ['PN', 'UB', 'LB']
required_type = [tuple, dict, dict]


class WithParameter(ABC):
    """
    Abstract base class for parameterized geometry or model components.

    Subclasses must define the following class attributes:
    - `PN` : tuple
        Names of parameters.
    - `UB` : dict
        Upper bounds of parameters.
    - `LB` : dict
        Lower bounds of parameters.
    """

    def __init_subclass__(cls, **kwargs) -> bool:
        """
        Validates that the subclass defines required attributes.

        Returns
        -------
        bool
            True if all required attributes are correctly defined, False otherwise.
        """
        for key, typ in zip(required_attrs, required_type):
            if not hasattr(cls, key):
                logger.warning(
                    f"{cls.__name__} does not define the {key} with type of {typ}"
                )
                return False
            if not isinstance(getattr(cls, key), typ):
                logger.warning(
                    f"{cls.__name__} does define the {key} with type of {type(getattr(cls,key))}, need {typ}"
                )
                return False
        return True

    @staticmethod
    @abstractmethod
    def init_parameters(**kwargs) -> Parameters:
        """
        Initialize and return a Parameters instance using the provided keyword arguments.

        Returns
        -------
        Parameters
            A new Parameters object initialized with derived values.
        """

        pass

    @staticmethod
    @abstractmethod
    def get_parameters() -> Parameters:
        """
        Return a default Parameters instance.

        Returns
        -------
        Parameters
            The default Parameters object for this component.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        str
            String showing the class name and parameter values.
        """

        param_repr = repr(self.parameters)

        return f"<{self.__class__.__name__}|: " + param_repr[10:] + "|>"

    def __getitem__(self, item):
        """
        Get the value of a specific parameter.

        Parameters
        ----------
        item : str
            The name of the parameter.

        Returns
        -------
        float
            The value of the parameter.

        Raises
        ------
        KeyError
            If the parameter name is invalid.
        """
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')

    def keys(self):
        """
        Return a list of parameter names.

        Returns
        -------
        list of str
            List of keys in the Parameters object.
        """

        return list(self.parameters.keys())

    def __contains__(self, item):
        """
        Check if a parameter exists.

        Parameters
        ----------
        item : str
            The name of the parameter to check.

        Returns
        -------
        bool
            True if the parameter exists, False otherwise.
        """

        return item in self.parameters
