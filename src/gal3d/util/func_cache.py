import functools
import json

import numpy as np

HashDict = type('HashDict', (dict,), {'__hash__': lambda self: hash(json_hash(self))})
HashList = type('HashList', (list,), {'__hash__': lambda self: hash(json_hash(self))})
HashSet = type('HashSet', (set,), {'__hash__': lambda self: hash(json_hash(self))})

def json_hash(o, *args, **kwargs):
    return json.dumps(o, *args, **dict({"sort_keys": True, "default": repr}, **kwargs))


def to_hash_object(obj, ):
    if isinstance(obj, dict):
        return HashDict(obj)
    elif isinstance(obj, list):
        return HashList(obj)
    elif isinstance(obj, set):
        return HashSet(obj)
    elif isinstance(obj,np.ndarray):
        return HashList(list(obj))
    return obj

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