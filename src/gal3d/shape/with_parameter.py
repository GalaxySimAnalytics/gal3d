import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from ..optimization.parameter import Parameters

logger = logging.getLogger("gal3d.shape.with_parameter")

required_attrs = ['PN', 'UB', 'LB']
required_type = [tuple, dict, dict]

T = TypeVar('T', bound='WithParameter')

class WithParameter(ABC):
    """
    Abstract base class for objects that contain parameters with bounds.
    
    This class defines the interface for objects that need to maintain and manipulate
    parameters with lower and upper bounds. It ensures consistent parameter handling
    across different shape and coordinate implementations.
    
    Subclasses must define:
    - PN: Parameter names tuple
    - UB: Upper bounds dictionary
    - LB: Lower bounds dictionary
    
    Attributes
    ----------
    PN : ClassVar[Tuple[str, ...]]
        Names of all parameters (class attribute)
    UB : ClassVar[Dict[str, float]]
        Upper bounds for parameters (class attribute)
    LB : ClassVar[Dict[str, float]]
        Lower bounds for parameters (class attribute)
    """

    PN: ClassVar[Tuple[str, ...]]
    UB: ClassVar[Dict[str, float]]
    LB: ClassVar[Dict[str, float]]
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate that concrete subclasses properly define required parameter attributes.
        
        This method only validates concrete classes (non-abstract classes) to ensure
        they define all required parameter attributes correctly.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent __init_subclass__.
        """
        # Call parent __init_subclass__
        super().__init_subclass__(**kwargs)
        
        # Skip validation for abstract classes
        if inspect.isabstract(cls):
            return
            
        # For concrete implementations, verify required attributes
        for attr, typ in zip(required_attrs, required_type):
            if not hasattr(cls, attr):
                logger.error(
                    f"{cls.__name__} missing required attribute '{attr}', "
                    f"please add '{attr}' to the class definition."
                )
                return False
            
            attr_value = getattr(cls, attr)
            if not isinstance(attr_value, typ):
                logger.error(
                    f"{cls.__name__} has attribute '{attr}' with incorrect type. "
                    f"Expected {typ}, got {type(attr_value)}."
                )
                return False
                
        # Verify that all parameter names in PN have corresponding entries in UB and LB
        missing_in_ub = set(cls.PN) - set(cls.UB.keys())
        missing_in_lb = set(cls.PN) - set(cls.LB.keys())
        
        if missing_in_ub:
            logger.error(
                f"{cls.__name__} missing upper bounds for parameters: {missing_in_ub}"
            )
            return False
            
        if missing_in_lb:
            logger.error(
                f"{cls.__name__} missing lower bounds for parameters: {missing_in_lb}"
            )
            return False
            
        return True
    
    @classmethod
    def get_parameters(cls) -> Parameters:
        """
        Create a Parameters object with the default values and bounds.
        
        Returns
        -------
        Parameters
            A new Parameters object initialized with default values
            and the bounds specified by UB and LB.
            
        Notes
        -----
        This method creates default parameters with values at the midpoint
        between the lower and upper bounds.
        """
        # Initialize parameters with midpoint values between bounds
        param_dict = {}
        for name in cls.PN:
            lower = cls.LB[name]
            upper = cls.UB[name]
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                # Use midpoint if both bounds are numeric
                param_dict[name] = (lower + upper) / 2
            else:
                # Use lower bound as default if not numeric
                param_dict[name] = lower
                
        parameters = Parameters(**param_dict)
        
        # Set bounds
        parameters.set_lb(**cls.LB)
        parameters.set_ub(**cls.UB)
        
        return parameters
    
    @classmethod
    def init_parameters(cls, **kwargs: Any) -> Parameters:
        """
        Initialize parameters with custom values while respecting bounds.
        
        Parameters
        ----------
        **kwargs
            Parameter values to override defaults.
            
        Returns
        -------
        Parameters
            A Parameters object with the specified values, 
            constrained by the defined bounds.
            
        Notes
        -----
        This method is useful for creating parameters with specific initial values,
        while ensuring they stay within their defined bounds.
        """
        parameters = cls.get_parameters()
        
        # Update with provided values
        for name, value in kwargs.items():
            if name in cls.PN:
                parameters[name] = max(min(value, cls.UB[name]), cls.LB[name])
                
        return parameters
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluate the parameterized object.
        
        This method must be implemented by subclasses to provide functionality
        specific to the object being parameterized.
        
        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
            
        Returns
        -------
        Any
            Subclass-specific output.
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
        
    def __getitem__(self, key: str) -> Any:
        """
        Access parameter values using dictionary-style indexing.
        
        Parameters
        ----------
        key : str
            Parameter name to access.
            
        Returns
        -------
        Any
            The value of the requested parameter.
            
        Raises
        ------
        KeyError
            If parameter does not exist or cannot be accessed.
        """
        if hasattr(self, 'parameters') and self.parameters is not None:
            return self.parameters[key]
        raise KeyError(f"Parameter '{key}' does not exist or cannot be accessed in {self.__class__.__name__}")
    
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
