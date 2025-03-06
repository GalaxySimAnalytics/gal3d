

from functools import partial


import numpy as np
from scipy import optimize
import optimagic as om
from optimagic.optimizers.nlopt_optimizers import _process_nlopt_results
from optimagic.optimizers.scipy_optimizers import process_scipy_result
import nlopt

from .util import nlopt_wrap

'''
fun like this (params, kwargs), params is the tuple of parameters, kwargs is any thing depend on how you use this in the fun.

'''


class Optimizer:
    """
    A class to handle optimization algorithms for fitting 3D galaxy morphologies.

    This class provides an interface to various optimization algorithms from different libraries
    such as `scipy`, `optimagic`, and `nlopt`. It allows users to select an algorithm and configure
    its options for minimizing a given objective function.

    Attributes
    ----------
    _algos : dict
        A dictionary containing the available optimization algorithms from different libraries.
        The keys are `'scipy'`, `'optimagic'`, and `'nlopt'`, and the values are lists of algorithm names.

    algo_name : str
        The name of the selected optimization algorithm.

    algo_options : dict
        A dictionary of options specific to the selected optimization algorithm.

    Methods
    -------
    __init__(algorithm, algo_options)
        Initializes the Optimizer with the specified algorithm and options.

    check_algorithm(algorithm)
        Checks if the specified algorithm is valid and available.

    fitting(fun, x0, bounds, args)
        Performs the optimization using the selected algorithm.

    available_algorithm
        Returns a copy of the available algorithms.

    """
    
    # `__algos` contains keys `'scipy'`, `'optimagic'`, and `'nlopt'`, each
        # associated with a list of optimization algorithm names.
    _algos = {}
    _algos['scipy']=['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B','TNC','COBYLA','COBYQA'
                        'SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']
    _algos['optimagic'] = om.algos.AvailableNames
    _algos['nlopt'] = list(filter(lambda x: x[:3] in ['GN_','GD_','LN_','LD_'],dir(nlopt)))
    def __init__(self,algorithm,algo_options):
        """
        Initializes the Optimizer with the specified algorithm and options.

        Parameters
        ----------
        algorithm : str
            The name of the optimization algorithm to use.

        algo_options : dict
            A dictionary of options specific to the selected optimization algorithm.

        Raises
        ------
        ValueError
            If the specified algorithm is not valid or available.
        """
        
        self.check_algorithm(algorithm)
        
        self.algo_name = algorithm
        self.algo_options = algo_options
        
        
        
        
    def check_algorithm(self,algorithm):
        """
        Checks if the specified algorithm is valid and available.

        Parameters
        ----------
        algorithm : str
            The name of the optimization algorithm to check.

        Raises
        ------
        ValueError
            If the specified algorithm is not valid or available.
        """
        if algorithm in self._algos['optimagic']:
            return 
        if algorithm in self._algos['scipy']:
            return 
        if algorithm in self._algos['nlopt']:
            return
        if algorithm in om.algos.AllNames:
            raise ValueError(f"{algorithm} is not a valid value, you should pip install related packages,\n" 
                             "see optimagic document for more details.")
        raise ValueError(f"{algorithm} is not a valid algorithm name.\n"
                         f"valid algorithm names:\n" 
                         f"scipy.optimize.minimize: {self._algos['scipy']} \n optimagic.minimize: {om.algos.AvailableNames}")
        
    def fitting(self,fun,x0,bounds,args,):
        """
        Performs the optimization using the selected algorithm.

        Parameters
        ----------
        fun : callable
            The objective function to minimize.

        x0 : list or array-like
            The initial guess of the parameters for the optimization.

        bounds : scipy.optimize.Bounds
            Lower and upper bounds on the parameters.

        args : tuple
            Additional arguments for the objective function.

        Returns
        -------
        result : object
            The result of the optimization, which may vary depending on the selected algorithm.

        Raises
        ------
        ValueError
            If the specified algorithm is not valid or available.
        """
        algorithm = self.algo_name 
        algo_options = self.algo_options
        
        if algorithm in self._algos['optimagic']:
            res = om.minimize(fun=fun,params=np.asarray(x0),algorithm=algorithm,bounds=bounds,
                              fun_kwargs={'kwargs':args}, algo_options=algo_options,)
            return res
        
        if algorithm in self._algos['scipy']:
            res = optimize.minimize(fun=fun,x0=x0,args=args, method=algorithm,bounds=bounds,options=algo_options)
            return process_scipy_result(res)
        
        if algorithm in self._algos['nlopt']:
            is_global = (algorithm[0]=='G')
            opt  = nlopt.opt(getattr(nlopt,algorithm),len(x0))
            warp_fun = nlopt_wrap(partial(fun,kwargs=args))
            opt.set_min_objective(warp_fun)
            opt.set_lower_bounds(bounds.lb)
            opt.set_upper_bounds(bounds.ub)
 
            opt.set_ftol_abs(algo_options.get('ftol_abs',0))
           # opt.set_ftol_rel(algo_options.get('ftol_res',0))
            opt.set_xtol_rel(algo_options.get('xtol_rel',1e-5))
            opt.set_xtol_abs(algo_options.get('xtol_abs',0))
            opt.set_maxeval(algo_options.get('maxeval',len(x0)*1000))

            xopt = opt.optimize(x0)
            
            return _process_nlopt_results(opt, xopt, is_global)

        if algorithm in om.algos.AllNames:
            raise ValueError(f"{algorithm} is not a valid value, you should pip install related packages,\n" 
                             "see optimagic document for more details.")
        raise ValueError(f"{algorithm} is not a valid algorithm name.\n"
                         f"valid algorithm names:\n" 
                         f"scipy.optimize.minimize: {self._algos['scipy']} \n optimagic.minimize: {om.algos.AvailableNames}")
        
    @property
    def available_algorithm(self):
        """
        Returns a copy of the available algorithms.

        Returns
        -------
        dict
            A dictionary containing the available optimization algorithms from different libraries.
        """
        return self._algos.copy()