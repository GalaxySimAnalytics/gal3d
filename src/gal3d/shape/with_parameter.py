import os
import logging
from abc import ABC, abstractmethod
from typing import List

from ..optimization.parameter import Parameters

logger = logging.getLogger("gal3d.shape.with_parameter")

required_attrs = ['PN', 'UB', 'LB']
required_type = [tuple, dict, dict]


class WithParameter(ABC):

    def __init_subclass__(cls, **kwargs) -> bool:
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
        """Initializes and returns the parameters with derived values."""

        pass

    @staticmethod
    @abstractmethod
    def get_parameters() -> Parameters:
        """Returns a default set of parameters."""
        pass

    def __repr__(self) -> str:

        param_repr = repr(self.parameters)

        return f"<{self.__class__.__name__}|: " + param_repr[10:] + "|>"

    def __getitem__(self, item):
        """
        Returns the value of the specified parameter.

        Parameters
        ----------
        item : str
            The name of the parameter.

        Returns
        -------
        float
            The value of the specified parameter.

        Raises
        ------
        KeyError
            If the specified parameter is not valid.
        """
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')

    def keys(self):

        return list(self.parameters.keys())

    def __contains__(self, item):

        return item in self.parameters
