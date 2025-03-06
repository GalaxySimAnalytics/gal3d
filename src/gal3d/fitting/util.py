

from functools import wraps




def nlopt_wrap(function):
    wraps(function)
    def wrapper(*args, **kwargs):
        result = function(params =args[0])
        return result
    return wrapper

def truncate(num,n):
    return float(int(num*(10**n))/10**n)