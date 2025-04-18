import functools
import json
import time

import numpy as np


__all__=['timer','classproperty','lru_cache']

def timer(logger):
    '''
    Decorator to measure the execution time of a function.

    Parameters
    ----------
    logger: 
    
    fun : function
        The function to be wrapped.

    Returns
    -------
    wrapper : function
    '''
    logger = logger
    def _timer(fun):
        name = fun.__name__
        @functools.wraps(fun)
        def wrapper(*args,**kwargs):
            s =  time.time()
            result = fun(*args,**kwargs)
            e =  time.time()
            logger.info(name.replace('_',' ').capitalize()+': '+f"{(e-s):.5f} sec")
            return result
        
        return wrapper
    
    return _timer

class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)
    
    
HashDict = type('HashDict', (dict,), {'__hash__': lambda self: hash(json_hash(self))})
HashList = type('HashList', (list,), {'__hash__': lambda self: hash(json_hash(self))})
HashSet = type('HashSet', (set,), {'__hash__': lambda self: hash(json_hash(self))})

def json_hash(o, *args, **kwargs):
    return json.dumps(o, *args, **dict({"sort_keys": True, "default": repr}, **kwargs))


@functools.singledispatch
def to_hash_object(obj):
    return obj

@to_hash_object.register
def _(obj : dict):
    return HashDict(obj)

@to_hash_object.register
def _(obj : list):
    return HashList(obj)

@to_hash_object.register
def _(obj : set):
    return HashSet(obj)

@to_hash_object.register
def _(obj : np.ndarray):
    return HashList(obj.tolist())



# https://luoruiqing.github.io/blog/2021/12/09/Python%E7%BC%93%E5%AD%98%E5%87%BD%E6%95%B0/
def lru_cache(*lru_args, **lru_kwargs):
    def wrapper_cache(func):
        func = functools.lru_cache(*lru_args, **lru_kwargs)(func)

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*map(to_hash_object, args), **{k: to_hash_object(v) for k, v in kwargs.items()})

        wrapped_func.cache_info = func.cache_info
        wrapped_func.cache_clear = func.cache_clear
        return wrapped_func
    return wrapper_cache