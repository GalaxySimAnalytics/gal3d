import typing
import inspect
from collections.abc import Callable
from functools import cached_property



class MySignature(inspect.Signature):
    
    @cached_property
    def params(self) -> dict:
        return {i: (self.parameters[i].kind,self.parameters[i].default) for i in self.parameters}
    
    @cached_property
    def kwargs(self) -> bool:
        for i in self.params:
            if self.params[i][0] == 4:
                return True
        return False
    
    @cached_property
    def args(self) -> bool:
        for i in self.params:
            if self.params[i][0] == 2:
                return True
        return False
    
    def get_params(self, positional: int = 0,keyword: int = 0, empty: int = 0) -> dict:
        '''Give the required paramter name of a function
        
        Parameters:
            positional: int, optional
                0: anything; 1: parameters can be positional; 2: only positional. Default 0.
            keyword: int, optional
                0: anything; 1: parameter can be keyword; 2: only keyword. Default 0.
            empty: int, optional
                0: anything; 1: parameter default values is empty; 2: parameter default values is not empty. Default 0.             
        
        Return:
            params: dict
                key is the param name, value is its default value in the function.
        
        Examples:
            get the empty parameters:
                args: get_params(positional=2,keyword=0,empty=1,)
                kwargs: get_params(positional=0,keyword=1,empty=1,)
            or:
                args: get_params(positional=1,keyword=0,empty=1,)
                kwargs: get_params(positional=0,keyword=2,empty=1,)
        Notes:
            POSITIONAL_ONLY 0
            POSITIONAL_OR_KEYWORD 1
            VAR_POSITIONAL 2
            KEYWORD_ONLY 3
            VAR_KEYWORD 4
        '''
        avaiable = [0,1,3]
        
        levels = [0,1,2]
        
        if positional not in levels:
            raise ValueError(f"'positional' = {positional}, this is not a valid value")
        if keyword not in levels:
            raise ValueError(f"'keyword' = {keyword}, this is not a valid value")
        if empty not in levels:
            raise ValueError(f"'positional' = {empty}, this is not a valid value")
        
        
        if positional == 1:
            avaiable = list(filter(lambda x: x<2, avaiable))
        if positional == 2:
            avaiable = list(filter(lambda x: x==0, avaiable))
        if keyword ==1 :
            avaiable = list(filter(lambda x: (x==1 or x==3), avaiable))
        if keyword ==2:
            avaiable = list(filter(lambda x: (x==3),avaiable))
        
        if empty == 0:
            emptysel = lambda x: True
        else:
            emptysel = (lambda x: x==inspect.Parameter.empty) if empty==1 else (lambda x: x!=inspect.Parameter.empty)
        
        params = {}
        for i in self.params:
            if self.params[i][0] in avaiable:
                if emptysel(self.params[i][1]):
                    params[i] = self.params[i][1]
                
        return params
    
    

def update_dict_value(origin: dict, other: dict, **kwargs)->dict:
    
    ret = origin.copy()
    same_key = ret.keys() & other.keys()
    for i in same_key:
        ret[i] = other[i]
    same_key = ret.keys() & kwargs.keys()
    for i in same_key:
        ret[i] = kwargs[i]
    return ret
    
func_optional_key = lambda x: MySignature.from_callable(x).get_params(positional=0,keyword=1,empty=2,)
func_required_key = lambda x: MySignature.from_callable(x).get_params(positional=0,keyword=0,empty=1,)