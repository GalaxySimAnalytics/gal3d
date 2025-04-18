
from typing import Callable


class MinimizeFunc:

    minimize_fn = {}
    
    @staticmethod
    def fn_registry(fn: str | Callable) -> Callable:
        if callable(fn):
            MinimizeFunc.minimize_fn[fn.__name__] = fn
            return fn

        fn_name = fn
        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                MinimizeFunc.minimize_fn[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")
        
        return decorator
    
    
from .fns import *







