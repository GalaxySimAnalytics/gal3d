
import copy
import logging
from dataclasses import is_dataclass

import numpy as np

from ..structure.structure_main import Structure_3D,Structure_3D_fitter
from .parameter import Parameters



logger = logging.getLogger("gal3d.fitting.result")






class OptimizeResults:
    def __init__(self,optimize_result):
        if not is_dataclass(optimize_result):
            raise(f'{optimize_result} is not a dataclass')
        
        self._results = [optimize_result]
        

    def keys(self):
        return self._results[0].__dataclass_fields__.keys()
    
    
    def __len__(self):
        return len(self._results)
    
    
    def __contains__(self, key):
        return key in self._results[0].__dataclass_fields__
    
    def __repr__(self):
        return f"<|OptimizeResults| {len(self._results)} |>"

    def __getitem__(self,key):
        if key in self:
            return [getattr(i,key) for i in self._results]
        
        raise KeyError(key)

    def __add__(self,other):
        
        if isinstance(other,OptimizeResults):
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



class Result:
    def __init__(self,structure: Structure_3D_fitter,optimize_result,parameters:Parameters,**kwargs):
        
        self._structure = structure
        self.res = OptimizeResults(optimize_result)
        
        self._parameters = [parameters]

    
    def keys(self):
        return self._parameters[0].keys()
    
    def __call__(self,pos,*,item: int =0,**kwargs):
        return self[item](pos,**kwargs)
    
    def __getitem__(self, k):
        if isinstance(k,str):
            return np.array([i[k] for i in self._parameters])
        
        if isinstance(k,int):
            return self._structure.from_parameters(**self._parameters[k])
        raise KeyError(f"{k} is not a valid key")
    
        
    def __repr__(self):
        coor = self._structure._coordinate_name
        shape = self._structure._shape_name
        error = self._structure._error_name
        lin1 = "<Resullt| num="+str(len(self._parameters))+" | "+coor+" | "+shape+" | "+error+" |>"
        lin2 = "Parameters: "+str(list(self.keys()))
        lenmax = max(len(lin1),len(lin2))
        lins = [lin1,lin2]
        result = ''.join([i.center(lenmax," ")+ "\n" for i in lins])
        
        return result[:-2]
    
    
    def __add__(self,other):
        if isinstance(other,Result):
            if self._structure == other._structure:
                self.res = self.res + other.res
                self._parameters = self._parameters + other._parameters
                return self
            raise ValueError(f"{other} have a different structure")
        raise TypeError(f"{other} is not a Result type")
                
    
    def __len__(self):
        return len(self._parameters)
