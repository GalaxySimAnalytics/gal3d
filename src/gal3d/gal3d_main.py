

from collections.abc import Iterable
import time


import numpy as np
from tqdm import tqdm

from .preprocessing.processor import Particles
from .structure.structure_main import Structure_3D,Structure_3D_fitter
from .fitting.optimizer import Optimizer
from .fitting.result import Result





class Galaxy3d(Particles):
    """
    The main class to fit a 3D galaxy model using particle data.

    This class provides functionality to fit an equal density surface to a 3D galaxy model
    using particle positions and weights. It supports various shape functions, coordinate systems,
    and error functions for optimization.

    Parameters
    ----------
    pos : array-like, shape (N, 3)
        The positions of N particles in 3D Cartesian coordinates (x, y, z).
    weight : array-like, shape (N,)
        The property of N particles, such as mass or density.
    parameter_mode : str, optional, default='Density'
        Determines how to calculate the target parameter. Options: {'Density', 'Mean'}.
    num_near : int, optional, default=32
        The number of nearest neighbors used in KDTree to estimate the target parameter.
    kdtree_options : dict, optional, default={}
        Additional options for building and querying the KDTree.
    verbose : bool, optional, default=True
        Controls verbosity of the class.

    Returns
    -------
    Galaxy3d
        An instance of the Galaxy3d class.
    """
    
    def __init__(self,pos,weight,parameter_mode: str = 'Density', num_near: int = 32, kdtree_options=dict(),verbose=True):
        super().__init__(pos=pos,weight=weight,parameter_mode=parameter_mode,num_near=num_near,verbose=verbose,**kdtree_options)
        
            
    def set_structure_options(self,coordinate_class='Coordinate3D',shape_class='Ellipsoid',error_func='isodensity_sums_fdev_rscale',**kwargs):
        """
        Set the structure options for model fitting.

        Parameters
        ----------
        coordinate_class : str, optional, default='Coordinate3D'
            The coordinate system used for model fitting. Options: {'Coordinate3D'}.
        shape_class : str, optional, default='Ellipsoid'
            The shape function used for model fitting. Options: {'Ellipsoid', 'Ellipsoid_S'}.
        error_func : str, optional, default='isodensity_sums_fdev_rscale'
            The error function used for optimization. Determines the minimize function for fitting.

        Returns
        -------
        Galaxy3d
            The instance with updated structure options.
        """
        
        self._structure = Structure_3D_fitter(coordinate_class=coordinate_class,shape_class=shape_class,error_func=error_func,**kwargs)
        
        self.params = self._structure.parameters

        if error_func[:10] =='isodensity':
            self.data_generator = self.__isodensity_data
            return self
        if error_func[:5] == 'shell':
            self.data_generator = self.__shell_data  
            return self
        if error_func[:4] == 'grid':
            self.data_generator = self.__grid_data
            return self
        raise ValueError(f"{error_func} is not identified, something wrong !!")

    
    def set_optimizer_options(self,algorithm,algo_options = dict()):
        """
        Set the optimizer and its options for model fitting.

        Parameters
        ----------
        algorithm : str
            The optimization algorithm name. See available algorithms for options.
        algo_options : dict, optional, default={}
            Additional options for the optimizer.

        Returns
        -------
        Galaxy3d
            The instance with updated optimizer options.
        """
        
        self._optimizer = Optimizer(algorithm=algorithm,algo_options=algo_options)
        return self
    
    def get_structure(self, r: float | Iterable = np.geomspace(1,10,200), **kwargs):
        """
        Fit the galaxy structure at a given radius or a range of radii.

        Parameters
        ----------
        r : float or Iterable, optional, default=np.geomspace(1, 10, 200)
            The maximum radius or a range of radii for fitting the structure.
        **kwargs : dict
            Additional fitting options:
            - var_a : float, optional
                The variation of the semi-major axis `a`.
            - data_level : int, optional, default=0
                The level of data to use for fitting.

        Returns
        -------
        Result
            The fitting result for the given radius or radii.
        """
        get_one_structure = self._get_one_ell_structure
        
        if not isinstance(r,Iterable):
            
            return get_one_structure(r,**kwargs)
        
        else:
            resall = []
            for i in tqdm(r):
                try:
                    if resall:
                        res_value = {i:resall[-1][i][0] for i in resall[-1].keys()}
                        kwargs['init_parameters'] = res_value
                    res = get_one_structure(i,**kwargs)
                    resall.append(res)
                except Exception as e:
                    print(f'Exception: skip fitting at radius {i:.2f} : {e}')
            if len(resall)>0:
                return sum(resall[1:],resall[0])
            else:
                return resall
    
    
    
    def _get_one_structure(self,r ,coordinate_class='Coordinate3D',
                           shape_class='Ellipsoid',
                           error_func='isodensity_sums_fdev_rscale',
                           init_parameters=dict(),upper_bounds=dict(),lower_bounds=dict(),**kwargs):
        """
        Fit a single structure at a given radius.

        Parameters
        ----------
        r : float
            The radius for fitting the structure.
        coordinate_class : str, optional, default='Coordinate3D'
            The coordinate system for fitting.
        shape_class : str, optional, default='Ellipsoid'
            The shape function for fitting.
        error_func : str, optional, default='isodensity_sums_fdev_rscale'
            The error function for optimization.
        init_parameters : dict, optional, default={}
            Initial parameters for fitting.
        upper_bounds : dict, optional, default={}
            Upper bounds for fitting parameters.
        lower_bounds : dict, optional, default={}
            Lower bounds for fitting parameters.
        **kwargs : dict
            Additional fitting options.

        Returns
        -------
        Result
            The fitting result for the given radius.
        """

        structure = Structure_3D_fitter(coordinate_class=coordinate_class,shape_class=shape_class,error_func=error_func)
        
        params = structure.parameters
        data_generator = None
        if error_func[:10] =='isodensity':
            data_generator = self.__isodensity_data
        if error_func[:5] == 'shell':
            data_generator = self.__shell_data  
        if error_func[:4] == 'grid':
            data_generator = self.__grid_data
        
        if data_generator is None:
            raise ValueError(f"{error_func} is not identified, something wrong !!")
        
        fun = structure.error
        fitdata = data_generator(r,**kwargs)
        
        parameters_set = params.new()
        
        if 'info' in fitdata:
            parameters_set.add_info(**fitdata['info'])
            del fitdata['info']
            
        for i in (upper_bounds.keys() & parameters_set.keys()):
            parameters_set[i].ub = upper_bounds[i]
        
        for i in (lower_bounds.keys() & parameters_set.keys()):
            parameters_set[i].lb = lower_bounds[i]
        
        if init_parameters:
            new_value = {i: init_parameters[i] for i in (init_parameters.keys()& parameters_set.keys())}
            parameters_set.set_value(**new_value)
        else:
            parameters_set = parameters_set.set_value(**structure.estimate_init_parameter(fitdata))


        bounds = parameters_set.scipy_bounds  
        
        x0_dict =  parameters_set.truncate_dict(n=4)
        each = self._optimizer.fitting(fun,list(x0_dict.values()),bounds,args=fitdata)
        parameters_set =parameters_set.set_value(each.x)

        res = Result(self._structure,each,parameters_set)
        
        return res
    
    
    def _get_one_ell_structure(self,a,**kwargs):
        """
        Fit an ellipsoid or generalized ellipsoid (Ellipsoid_S) structure.

        Parameters
        ----------
        a : float
            The semi-major axis for fitting.
        **kwargs : dict
            Additional fitting options:
            - var_a : float, optional
                The variation of the semi-major axis `a`.
            - init_parameters : dict, optional
                Initial parameters for fitting.
            - upper_bounds : dict, optional
                Upper bounds for fitting parameters.
            - lower_bounds : dict, optional
                Lower bounds for fitting parameters.

        Returns
        -------
        Result
            The fitting result for the given semi-major axis.
        """
        
        var_a = kwargs.get('var_a',0)
        var_a = min(var_a,0.99)
        var_a = max(var_a,0)
        
        init_parameters = kwargs.get('init_parameters',dict())
        upper_bounds=kwargs.get('upper_bounds',dict())
        lower_bounds=kwargs.get('lower_bounds',dict())
        fitonce = kwargs.get('fitonce',False)
        
        fitdata = self.data_generator(a,**kwargs)
        
        if (self._structure._shape_name == 'Ellipsoid') or (self._structure._shape_name == 'Ellipsoid_S' and fitonce):
            
            parameters_set = self.params.new()
            fun = parameters_set.decorate_func_contraints(self._structure.error)
            
            if 'info' in fitdata:
                parameters_set.add_info(**fitdata['info'])
                del fitdata['info']
                
            parameters_set.set_lb(a=(a*(1-var_a)))
            parameters_set.set_ub(a=(a*(1+var_a)))

            bounds = parameters_set.scipy_bounds  
                
            if init_parameters:
                parameters_set = parameters_set.set_value(**init_parameters)
            else:
                parameters_set = parameters_set.set_value(**self._structure.estimate_init_parameter(fitdata))
            
            parameters_set.set_value(a=a)
            
            x0_dict =  parameters_set.truncate_dict(n=4)

            each = self._optimizer.fitting(fun,list(x0_dict.values()),bounds,args=fitdata)
            parameters_set =parameters_set.set_value(each.x)

            res = Result(self._structure,each,parameters_set)
            
            # we need first fitting Ellipsoid, then fit sa,sb,sc and a for Elllipsoid_S
        if self._structure._shape_name == 'Ellipsoid_S':
            
            parameters_set = self.params.new()
            parameters_set.set_lb(a=(a*(1-var_a)))
            parameters_set.set_ub(a=(a*(1+var_a)))
            
            if init_parameters:
                parameters_set = parameters_set.set_value(**init_parameters)
            else:
                parameters_set = parameters_set.set_value(**self._structure.estimate_init_parameter(fitdata))
            
            parameters_set.set_value(a=a)
            
            
            
            parent_res = self._get_one_structure(a,coordinate_class=self._structure._coordinate_name,
                                                   shape_class='Ellipsoid',
                                                   error_func=self._structure._error_name,
                                                   init_parameters=dict(**parameters_set),
                                                   upper_bounds={i: parameters_set[i].ub for i in parameters_set.keys()},
                                                   lower_bounds={i: parameters_set[i].lb for i in parameters_set.keys()})
            
            res_value = {i:parent_res[i][0] for i in parent_res.keys()}
            
            parameters_set.set_value(**res_value)
            
            
            x0_dict =  parameters_set.truncate_dict(n=4)
            x0_dict.update(res_value)
            
            del res_value['a']
            parameters_set.set_lb(**res_value)
            parameters_set.set_ub(**res_value)
            
            bounds = parameters_set.scipy_bounds 
            
            parameters_set.add_info(parent_fun = parent_res.res['fun'][0])
            
            fun = parameters_set.decorate_func_contraints(self._structure.error)

            if 'info' in fitdata:
                parameters_set.add_info(**fitdata['info'])
                del fitdata['info']
            
            each = self._optimizer.fitting(fun,list(x0_dict.values()),bounds,args=fitdata)
            parameters_set =parameters_set.set_value(each.x)

            res = Result(self._structure,each,parameters_set)
            
        
        return res
    
        
    def __isodensity_data(self,r,**kwargs):
        """
        Generate isodensity data for fitting.

        Parameters
        ----------
        a : float
            The radius for generating isodensity data.
        **kwargs : dict
            Additional options:
            - data_level : int, optional, default=0
                The level of data to use for fitting.

        Returns
        -------
        dict
            A dictionary containing the isodensity data.
        """
        
        
        data_level = kwargs.get('data_level',0) # 0: [(0,0)], 1: (0,(1,0,-1)), 2: ((1,0,-1),0), 3: (1,0,-1),(1,0,-1)
        Level = {0:[(0,0)],1:[(0,1),(0,-1)],2:[(1,0),(-1,0)],
                 3:[(0,1),(0,-1),(1,0),(1,-1)],
                 4:[(0,1),(0,-1),(1,0),(1,-1),(1,1),(1,-1),(-1,1),(-1,-1)]}
        
                
        fitdata = self.field.generate(r,level= (0,0))
        fitdata['pos'] = fitdata['pos'][~np.isnan(fitdata['r'])]
        fitdata['r'] = fitdata['r'][~np.isnan(fitdata['r'])]
           # fitdata['r'] = fitdata['r']/np.sqrt(np.sum(fitdata['r']**2)/len(fitdata['r']))   #  normalization as this used for calculate error
           # fitdata['info'] = {'parameter': fitdata['parameter']}
        
        if data_level !=0:
            le = Level[data_level]
            for i in le:
                other = self.field.generate(r,level= i)
                other['pos'] = other['pos'][~np.isnan(other['r'])]
                other['r'] = other['r'][~np.isnan(other['r'])]
                
                fitdata['pos'] = np.concatenate((fitdata['pos'],other['pos']))
                fitdata['r'] = np.concatenate((fitdata['r'],other['r']))
    
        fitdata['r'] = fitdata['r']/np.sqrt(np.sum(fitdata['r']**2)/len(fitdata['r']))   #  normalization as this used for calculate error
        fitdata['info'] = {'parameter': fitdata['parameter']}
        return fitdata
    
    def __shell_data(self,a):
        """
        Generate shell data for fitting.

        Parameters
        ----------
        a : float
            The radius for generating shell data.

        Returns
        -------
        dict
            A dictionary containing the shell data.
        """
        
        
        fitdata={'pos': self.pos, 'pa': self.parameter}
        return fitdata
    
    
    def __grid_data(self,a,logscale = True):
        """
        Generate grid data for fitting.

        Parameters
        ----------
        a : float
            The radius for generating grid data.
        logscale : bool, optional, default=True
            Whether to use logarithmic scaling for the parameter.

        Returns
        -------
        dict
            A dictionary containing the grid data.
        """

        fitdata={'pos': self.grid.grid_pos,'volumn': self.grid.grid_volumn} # using log10 pa ?? #TODO
        fitdata['parameter'] = np.log10(self.grid.grid_denpa) if logscale else self.grid.grid_denpa

        return fitdata
    
    @property
    def available_shape(self):
        """
        Get the available shape functions.

        Returns
        -------
        list
            A list of available shape functions.
        """
        return list(Structure_3D._shape_3d_fn.keys())
    
    @property
    def available_coordinate(self):
        """
        Get the available coordinate systems.

        Returns
        -------
        list
            A list of available coordinate systems.
        """
        return list(Structure_3D._coordinate_fn.keys())
    
    @property
    def available_error(self):
        """
        Get the available error functions.

        Returns
        -------
        list
            A list of available error functions.
        """
        return list(Structure_3D_fitter._error_fcall_fn.keys())

    @property
    def available_algorithm(self):
        """
        Get the available optimization algorithms.

        Returns
        -------
        dict
            A dict of available optimization algorithms.
        """
        return Optimizer._algos.copy()

    def __repr__(self):
        """
        Get a string representation of the Galaxy3d instance.

        Returns
        -------
        str
            A formatted string representation of the instance.
        """
        lin1 = "< Galaxy3d | " +  "N_particles = "+str(len(self.pos)) + " | "+self.pa_mode+ " >"

        if hasattr(self,"_structure"):
            lin2 = "| Coordinate: " + self._structure._coordinate_name+ " |"
            lin3 = "| Shape: " + self._structure._shape_name+ " |"
            lin4 = "| Error: "+self._structure._error_name + " |"
        else:
            lin2 = "| Coordinate: not set |"
            lin3 = "| Shape: not set |"
            lin4 = "| Error: not set |"
            

        if hasattr(self,"_optimizer"):
            lin5 = "| Optimizer: "+ self._optimizer.algo_name+ " |"
        else:
            lin5 = "| Optimizer: not set |"
            
        lins = [lin1,lin2,lin3,lin4,lin5]
        lenmax = len(max(lins,key=len))+4
        result = ''.join([i.center(lenmax,"*")+ "\n" for i in lins])
        
        return result[:-2]