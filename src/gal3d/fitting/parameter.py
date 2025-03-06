
import copy
import logging

import numpy as np
from scipy import optimize
#from optimagic import Bounds

from .util import truncate


logger = logging.getLogger("gal3d.fitter.parameter")


__all__ = ['Parameters']


class Parameter(float):
    
    __slots__ = ["_lb","_ub"]
    def __new__(cls,value,**kwargs):
        instance = float.__new__(cls,value)
        
        if isinstance(value,Parameter):
            object.__setattr__(instance,'_lb',kwargs.get('lb',value.lb))
            object.__setattr__(instance,'_ub',kwargs.get('ub',value.ub))
            return instance
        object.__setattr__(instance,'_lb',kwargs.get('lb',-np.inf))
        object.__setattr__(instance,'_ub',kwargs.get('ub',np.inf))
        return instance
    
    def assign_value(self,value):
        inst = self.__class__.__new__(self.__class__,value,lb=self.lb,ub=self.ub)
        return inst
        
    @property
    def lb(self):
        return self._lb
    
    @property
    def ub(self):
        return self._ub
    
    @ub.setter
    def ub(self,value):
        self._ub = value
        
    @lb.setter
    def lb(self,value):
        self._lb = value



class Parameters():
    
    def __init__(self,**kwargs):
        
        self.__parameters = dict(**{key: Parameter(value) for key, value in kwargs.items()})    
        self._derived = {}
        self._info = {}
        
    def new(self,*args,**kwargs):
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
        
        return {i:truncate(self.__parameters[i],n) for i in self.__parameters}
        
        
    def available_keys(self):
        return self.keys() | self._derived.keys() | self._info.keys()
    
    def keys(self):
        return self.__parameters.keys()
    
    def derived(self, f):
        self._derived[f.__name__] = f
        
        
    def __len__(self):
        return len(self.data)
    def __contains__(self, key):
        return key in self.__parameters
    
    def __repr__(self):
        dict_repr = repr(self.__parameters)
        
        return "Parameters( " +dict_repr[1:-1]+ " )"

    def __setitem__(self, key, value):
        if key in self.__parameters:
            if isinstance(value,Parameter):
                self.__parameters[key] = value
                return 
            else:
                self.__parameters[key] = self.__parameters[key].assign_value(value)
                return 
        raise KeyError(f"Only Parameter: {list(self.keys())} can be set values")
    
    def __getitem__(self, k):
        if k in self.__parameters:
            return self.__parameters[k]
        elif k in self._derived:
            return self._derived[k](self)
        elif k in self._info:
            return self._info[k]
        else:
            raise KeyError(k)
        
    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        return inst
    
    def update(self,other):
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
        for i in kwargs:
            if i not in self.__parameters:
                print(f"{i} is not a parameter name")
            else:
                self.__parameters[i].ub = kwargs[i]
        return self
    def set_lb(self,**kwargs):
        for i in kwargs:
            if i not in self.__parameters:
                print(f"{i} is not a parameter name")
            else:
                self.__parameters[i].lb = kwargs[i]
        return self
    @property
    def lb(self):
        return {i:j.lb for i,j in self.__parameters.items()}

    @property
    def ub(self):
        return {i:j.ub for i,j in self.__parameters.items()}
    
    def __add__(self,other):
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
        return optimize.Bounds(lb=[i.lb for i in self.__parameters.values()],ub=[i.ub for i in self.__parameters.values()])
    
    
    def add_info(self,**kwargs):
        self._info.update(kwargs)