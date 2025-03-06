
import copy
import logging

import numpy as np
from scipy import optimize
#from optimagic import Bounds

from .util import truncate


logger = logging.getLogger("gal3d.fitter.parameter")


__all__ = ['Parameters']


class Parameter(float):
    """
    A class representing a parameter with lower and upper bounds.

    This class extends the built-in `float` type to include lower (`lb`) and upper (`ub`) bounds.
    It is used to define parameters that can be fitted within a specified range.

    Attributes
    ----------
    _lb : float
        The lower bound of the parameter.
    _ub : float
        The upper bound of the parameter.

    Methods
    -------
    __new__(cls, value, **kwargs)
        Creates a new instance of the Parameter class.
    assign_value(value)
        Assigns a new value to the parameter while preserving the bounds.
    lb
        Property getter for the lower bound.
    ub
        Property getter for the upper bound.
    lb.setter
        Property setter for the lower bound.
    ub.setter
        Property setter for the upper bound.
    """
    
    __slots__ = ["_lb","_ub"]
    def __new__(cls,value,**kwargs):
        """
        Creates a new instance of the Parameter class.

        Parameters
        ----------
        value : float or Parameter
            The initial value of the parameter. If `value` is an instance of `Parameter`,
            the new instance will inherit the bounds from `value` unless overridden by `kwargs`.
        **kwargs : dict
            Additional keyword arguments:
            - lb : float, optional
                The lower bound of the parameter. Default is `-np.inf`.
            - ub : float, optional
                The upper bound of the parameter. Default is `np.inf`.

        Returns
        -------
        Parameter
            A new instance of the Parameter class.
        """
        instance = float.__new__(cls,value)
        
        if isinstance(value,Parameter):
            object.__setattr__(instance,'_lb',kwargs.get('lb',value.lb))
            object.__setattr__(instance,'_ub',kwargs.get('ub',value.ub))
            return instance
        object.__setattr__(instance,'_lb',kwargs.get('lb',-np.inf))
        object.__setattr__(instance,'_ub',kwargs.get('ub',np.inf))
        return instance
    
    def assign_value(self,value):
        """
        Assigns a new value to the parameter while preserving the bounds.

        Parameters
        ----------
        value : float
            The new value to assign to the parameter.

        Returns
        -------
        Parameter
            A new instance of the Parameter class with the updated value and the same bounds.
        """
        inst = self.__class__.__new__(self.__class__,value,lb=self.lb,ub=self.ub)
        return inst
        
    @property
    def lb(self):
        """
        Get the lower bound of the parameter.

        Returns
        -------
        float
            The lower bound of the parameter.
        """
        return self._lb
    
    @property
    def ub(self):
        """
        Get the upper bound of the parameter.

        Returns
        -------
        float
            The upper bound of the parameter.
        """
        return self._ub
    
    @ub.setter
    def ub(self,value):
        """
        Set the upper bound of the parameter.

        Parameters
        ----------
        value : float
            The new upper bound of the parameter.
        """
        self._ub = value
        
    @lb.setter
    def lb(self,value):
        """
        Set the lower bound of the parameter.

        Parameters
        ----------
        value : float
            The new lower bound of the parameter.
        """
        self._lb = value



class Parameters():
    """
    A class representing a collection of parameters with bounds and derived values.

    This class is used to manage a set of parameters, each of which is an instance of the `Parameter` class.
    It also supports derived parameters and additional information associated with the parameters.

    Attributes
    ----------
    __parameters : dict
        A dictionary of `Parameter` instances.
    _derived : dict
        A dictionary of derived parameters, where keys are function names and values are functions.
    _info : dict
        A dictionary of additional information associated with the parameters.

    Methods
    -------
    __init__(**kwargs)
        Initializes the Parameters class with a set of parameters.
    new(*args, **kwargs)
        Creates a new instance of Parameters with updated values and bounds.
    truncate_dict(n=3)
        Truncates the parameter values to `n` decimal places.
    available_keys()
        Returns a set of all available keys, including parameter keys, derived keys, and info keys.
    keys()
        Returns the keys of the parameters.
    derived(f)
        Adds a derived parameter function to the collection.
    __len__()
        Returns the number of parameters.
    __contains__(key)
        Checks if a key is in the parameters.
    __repr__()
        Returns a string representation of the Parameters instance.
    __setitem__(key, value)
        Sets the value of a parameter.
    __getitem__(k)
        Gets the value of a parameter or derived parameter.
    __copy__()
        Creates a shallow copy of the Parameters instance.
    update(other)
        Updates the parameters with values from another Parameters instance or dictionary.
    set_value(*args, **kwargs)
        Sets the values of multiple parameters.
    set_ub(**kwargs)
        Sets the upper bounds of multiple parameters.
    set_lb(**kwargs)
        Sets the lower bounds of multiple parameters.
    lb
        Property getter for the lower bounds of all parameters.
    ub
        Property getter for the upper bounds of all parameters.
    __add__(other)
        Merges two Parameters instances or a Parameters instance with a dictionary.
    scipy_bounds
        Property getter for the bounds in a format suitable for scipy.optimize.
    add_info(**kwargs)
        Adds additional information to the Parameters instance.
    """
    def __init__(self,**kwargs):
        """
        Initializes the Parameters class with a set of parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their initial values.
        """
        self.__parameters = dict(**{key: Parameter(value) for key, value in kwargs.items()})    
        self._derived = {}
        self._info = {}
        
    def new(self,*args,**kwargs):
        """
        Creates a new instance of Parameters with updated values and bounds.

        Parameters
        ----------
        *args : list
            A list of values to set for the parameters, in the order of the keys.
        **kwargs : dict
            A dictionary of parameter names and their new values.

        Returns
        -------
        Parameters
            A new instance of Parameters with updated values and bounds.
        """
        new = Parameters(**self)

        new._derived = self._derived.copy()
        new._info = self._info.copy()
        new.set_lb(**self.lb)
        new.set_ub(**self.ub)
        if args:
            params = dict(zip(self.keys(),*args))
            new.set_value(**params)
        if kwargs:
            new.set_value(**kwargs)
        
        return new
    
    def truncate_dict(self,n=3):
        """
        Truncates the parameter values to `n` decimal places.

        Parameters
        ----------
        n : int, optional
            The number of decimal places to truncate to. Default is 3.

        Returns
        -------
        dict
            A dictionary of parameter names and their truncated values.
        """
        return {i:truncate(self.__parameters[i],n) for i in self.__parameters}
        
        
    def available_keys(self):
        """
        Returns a set of all available keys, including parameter keys, derived keys, and info keys.

        Returns
        -------
        set
            A set of all available keys.
        """
        return self.keys() | self._derived.keys() | self._info.keys()
    
    def keys(self):
        """
        Returns the keys of the parameters.

        Returns
        -------
        dict_keys
            The keys of the parameters.
        """
        return self.__parameters.keys()
    
    def derived(self, f):
        """
        Adds a derived parameter function to the collection.

        Parameters
        ----------
        f : function
            A function that computes a derived parameter.
        """
        self._derived[f.__name__] = f
        
        
    def __len__(self):
        """
        Returns the number of parameters.

        Returns
        -------
        int
            The number of parameters.
        """
        return len(self.data)
    def __contains__(self, key):
        """
        Checks if a key is in the parameters.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        bool
            True if the key is in the parameters, False otherwise.
        """
        return key in self.__parameters
    
    def __repr__(self):
        """
        Returns a string representation of the Parameters instance.

        Returns
        -------
        str
            A string representation of the Parameters instance.
        """
        dict_repr = repr(self.__parameters)
        
        return "Parameters( " +dict_repr[1:-1]+ " )"

    def __setitem__(self, key, value):
        """
        Sets the value of a parameter.

        Parameters
        ----------
        key : str
            The name of the parameter.
        value : float or Parameter
            The new value of the parameter.

        Raises
        ------
        KeyError
            If the key is not in the parameters.
        """
        if key in self.__parameters:
            if isinstance(value,Parameter):
                self.__parameters[key] = value
                return 
            else:
                self.__parameters[key] = self.__parameters[key].assign_value(value)
                return 
        raise KeyError(f"Only Parameter: {list(self.keys())} can be set values")
    
    def __getitem__(self, k):
        """
        Gets the value of a parameter or derived parameter.

        Parameters
        ----------
        k : str
            The name of the parameter or derived parameter.

        Returns
        -------
        float or any
            The value of the parameter or derived parameter.

        Raises
        ------
        KeyError
            If the key is not found in the parameters, derived parameters, or info.
        """
        if k in self.__parameters:
            return self.__parameters[k]
        elif k in self._derived:
            return self._derived[k](self)
        elif k in self._info:
            return self._info[k]
        else:
            raise KeyError(k)
        
    def __copy__(self):
        """
        Creates a shallow copy of the Parameters instance.

        Returns
        -------
        Parameters
            A shallow copy of the Parameters instance.
        """
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        return inst
    
    def update(self,other):
        """
        Updates the parameters with values from another Parameters instance or dictionary.

        Parameters
        ----------
        other : Parameters or dict
            Another Parameters instance or a dictionary of parameter names and values.

        Returns
        -------
        None
        """
        if isinstance(other,Parameters):
            return self.__parameters.update(other)
        else:
            for i in other:
                if i in self.__parameters:
                    self.__parameters[i] = self.__parameters[i].assign_value(other[i])
                else:
                    self.__parameters[i] = Parameter(other[i])
            return 
    
    def set_value(self,*args,**kwargs):
        """
        Sets the values of multiple parameters.

        Parameters
        ----------
        *args : list
            A list of values to set for the parameters, in the order of the keys.
        **kwargs : dict
            A dictionary of parameter names and their new values.

        Returns
        -------
        Parameters
            The updated Parameters instance.
        """
        if args:
            params = dict(zip(self.__parameters.keys(),*args))
            for i in params:
                if i not in self.__parameters:
                    print(f"{i} is not a parameter name")
                else:
                    self.__parameters[i] = self.__parameters[i].assign_value(params[i])
        
        for i in kwargs:
            if i not in self.__parameters:
                print(f"{i} is not a parameter name")
            else:
                self.__parameters[i] = self.__parameters[i].assign_value(kwargs[i])
                
        return self
    
    def set_ub(self,**kwargs):
        """
        Sets the upper bounds of multiple parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new upper bounds.

        Returns
        -------
        Parameters
            The updated Parameters instance.
        """
        for i in kwargs:
            if i not in self.__parameters:
                print(f"{i} is not a parameter name")
            else:
                self.__parameters[i].ub = kwargs[i]
        return self
    def set_lb(self,**kwargs):
        """
        Sets the lower bounds of multiple parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new lower bounds.

        Returns
        -------
        Parameters
            The updated Parameters instance.
        """
        for i in kwargs:
            if i not in self.__parameters:
                print(f"{i} is not a parameter name")
            else:
                self.__parameters[i].lb = kwargs[i]
        return self
    @property
    def lb(self):
        """
        Get the lower bounds of all parameters.

        Returns
        -------
        dict
            A dictionary of parameter names and their lower bounds.
        """
        return {i:j.lb for i,j in self.__parameters.items()}

    @property
    def ub(self):
        """
        Get the upper bounds of all parameters.

        Returns
        -------
        dict
            A dictionary of parameter names and their upper bounds.
        """
        return {i:j.ub for i,j in self.__parameters.items()}
    
    def __add__(self,other):
        """
        Merges two Parameters instances or a Parameters instance with a dictionary.

        Parameters
        ----------
        other : Parameters or dict
            Another Parameters instance or a dictionary of parameter names and values.

        Returns
        -------
        Parameters
            The merged Parameters instance.

        Raises
        ------
        TypeError
            If `other` is not a dictionary or Parameters instance.
        """
        if isinstance(other,Parameters):
            logger.info(f"merge {self} and {other}")
            
            h1 = copy.copy(self)
            h2 = copy.copy(other)
            h1.update(h2)
            h1._derived.update(h2._derived)
            h1._info.update(h2._info)
            return h1
        if isinstance(other,dict):
            logger.info(f"merge {self} and {other}")
            
            h1 = copy.copy(self)
            h1.update(other)
            return h1
        logger.error(f"{other} is not a dict")
        raise TypeError("Must be dict")
    
    @property
    def scipy_bounds(self):
        """
        Get the bounds in a format suitable for scipy.optimize.

        Returns
        -------
        scipy.optimize.Bounds
            A Bounds object containing the lower and upper bounds of all parameters.
        """
        return optimize.Bounds(lb=[i.lb for i in self.__parameters.values()],ub=[i.ub for i in self.__parameters.values()])
    
    
    def add_info(self,**kwargs):
        """
        Adds additional information to the Parameters instance.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of additional information to add to the Parameters instance.
        """
        self._info.update(kwargs)