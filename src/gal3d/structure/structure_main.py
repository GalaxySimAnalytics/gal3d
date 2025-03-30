
import logging
import copy

import numpy as np
import optimagic as om
from scipy import optimize,spatial

from .util import ellipsoid_fit
from ..optimization.parameter import Parameters
from ..util.func_signature import func_required_key

from ..preprocess.spherical_field.util import fibonacci_sampling, vector_length3d

__all__ = ['Structure_3D','Structure_3D_fitter']

logger = logging.getLogger('gal3d.structure.structure_main')
class Structure_3D:
    '''
    A class to represent a 3D structure, combining coordinate and shape information.

    Attributes
    ----------
    _shape_3d_fn : dict
        A dictionary mapping shape function names to their corresponding functions.
    _coordinate_fn : dict
        A dictionary mapping coordinate function names to their corresponding functions.
    _coordinate_name : str
        The name of the coordinate function used.
    _shape_name : str
        The name of the shape function used.
    _coordinate : function
        The coordinate function instance.
    _shape : function
        The shape function instance.
    _coordinate_quick_params : list
        List of quick parameters for the coordinate function.
    _shape_quick_params : list
        List of quick parameters for the shape function.
    __coor_pa_num : int
        Number of coordinate parameters.
    parameters : dict
        Dictionary of parameters for both coordinate and shape functions.
    d_a : float
        Distance parameter (default is 0).
    constraints : None or dict
        Constraints for the parameters (default is None).

    Methods
    -------
    __init__(coordinate_class='Coordinate3D', shape_class='Ellipsoid', **kwargs)
        Initializes the Structure_3D object.
    init_parameters(*args, **kwargs)
        Initializes the parameters for the coordinate and shape functions.
    set_parameters(*args, **kwargs)
        Sets the parameters for the structure.
    from_parameters(*args, **kwargs)
        Creates a new Structure_3D instance with the given parameters.
    __repr__()
        Returns a string representation of the Structure_3D object.
    __eq__(other)
        Checks equality with another Structure_3D object.
    __call__(pos, **kwargs)
        Evaluates the structure at the given position.
    quick_call(*args, pos, **kwargs)
        Quick evaluation of the structure at the given position.
    quick_call_d(*args, pos, **kwargs)
        Quick evaluation of the derivative of the structure at the given position.
    quick_call_dist(*args, pos, **kwargs)
        Quick evaluation of the distance function of the structure at the given position.
    check_boundary(params_list, mode='periodic')
        Adjusts parameters to stay within specified bounds.
    generate_points(random_np=1024)
        Generates random points on the surface of the structure.
    shape_func(fn)
        Decorator to register a shape function.
    coordinate_func(fn)
        Decorator to register a coordinate function.
    '''

    
    
    _shape_3d_fn = {}
    _coordinate_fn = {}    

    def __init__(self,coordinate_class='Coordinate3D',shape_class='Ellipsoid',**kwargs):
        '''
        Initializes the Structure_3D object.

        Parameters
        ----------
        coordinate_class : str, optional
            The name of the coordinate function to use (default is 'Coordinate3D').
        shape_class : str, optional
            The name of the shape function to use (default is 'Ellipsoid').
        **kwargs : dict
            Additional keyword arguments, including 'd_a' for distance parameter.

        Raises
        ------
        ValueError
            If the coordinate_class or shape_class is not valid.
        '''
        
        if coordinate_class not in Structure_3D._coordinate_fn:
            raise ValueError('not a valid coordinate_class')
        
        if shape_class not in Structure_3D._shape_3d_fn:
            raise ValueError('not a valid shape_class')
        
        
        self._coordinate_name = coordinate_class
        self._shape_name = shape_class
        
        self._coordinate = self._coordinate_fn[coordinate_class]
        self._shape = self._shape_3d_fn[shape_class]
        
        self._coordinate_quick_params = list(func_required_key(self._coordinate.quick_call).keys())[:-1]
        self._shape_quick_params = list(func_required_key(self._shape.quick_call).keys())[:-1]
        self.__coor_pa_num = len(self._coordinate_quick_params)
        
        self.parameters = self._coordinate.get_parameters() + self._shape.get_parameters()
        
        self.d_a = kwargs.get('d_a',0)
        self.constraints = None
        
    def init_parameters(self,*args,**kwargs):
        """
        Initializes the parameters for the coordinate and shape functions.

        Parameters
        ----------
        *args : tuple
            Positional arguments to initialize parameters.
        **kwargs : dict
            Keyword arguments to initialize parameters.

        Returns
        -------
        Parameters
            An Intance of Parameters.
        """
        if args:
            params = dict(zip(self.parameters.keys(),*args))
            return self._coordinate.init_parameters(**params)+self._shape.init_parameters(**params)
        
        return self._coordinate.init_parameters(**kwargs)+self._shape.init_parameters(**kwargs)
    
    def set_parameters(self,*args,**kwargs):
        '''
        Sets the parameters for the structure.

        Parameters
        ----------
        *args : tuple
            Positional arguments to set parameters.
        **kwargs : dict
            Keyword arguments to set parameters.
        '''
        self.parameters = self.parameters + self.init_parameters(*args,**kwargs)
        
        
    def from_parameters(self,*args,**kwargs):
        '''
        Creates a new Structure_3D instance with the given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments to initialize parameters.
        **kwargs : dict
            Keyword arguments to initialize parameters.

        Returns
        -------
        Structure_3D
            A new Structure_3D instance with the given parameters.
        '''
        ret = Structure_3D(coordinate_class=self._coordinate_name,shape_class=self._shape_name)
        ret.set_parameters(*args,**kwargs)
        return ret

    
    def __repr__(self):
        '''
        Returns a string representation of the Structure_3D object.

        Returns
        -------
        str
            String representation of the object.
        '''
        coor_repr = repr(self._coordinate(**self.parameters))
        shape_repr = repr(self._shape(**self.parameters))
        lin1 = "<Structure_3D|: "+'\n'
        lin2 = "   "+ coor_repr + '\n'
        lin3 = "   "+ shape_repr
        return lin1 +lin2 + lin3
        

    def __eq__(self, other):
        '''
        Checks equality with another Structure_3D object.

        Parameters
        ----------
        other : Structure_3D
            Another Structure_3D object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        '''
        if isinstance(other,Structure_3D):
            if self._coordinate_name == other._coordinate_name:
                if self._shape_name == other._shape_name:
                    return True
        return False
        
    def __call__(self, pos,**kwargs):
        '''
        Evaluates the structure at the given position.

        Parameters
        ----------
        pos : array_like
            The position at which to evaluate the structure.
        **kwargs : dict
            Additional keyword arguments for the evaluation.

        Returns
        -------
        array_like
            The evaluated structure at the given position.
        '''
        pos = np.asarray(pos)
        if kwargs:
            coord_pa = self._coordinate.init_parameters(**kwargs)
            shape_pa = self._shape.init_parameters(**kwargs)
            return self._shape(**shape_pa)(self._coordinate(**coord_pa)(pos))
        return self._shape(**self.parameters)(self._coordinate(**self.parameters)(pos))

    
    
    def quick_call(self, *args,pos,**kwargs):
        '''
        Quick evaluation of the structure at the given position.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the quick evaluation.
        pos : array_like
            The position at which to evaluate the structure.
        **kwargs : dict
            Additional keyword arguments for the evaluation.

        Returns
        -------
        array_like
            The quick evaluated structure at the given position.

        Raises
        ------
        KeyError
            If parameters are not provided.
        '''
        pos = np.asarray(pos)
        if args:
            return self._shape.quick_call(*args[self.__coor_pa_num:],
                                          pos = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos=pos))

        coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
        shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
        return self._shape.quick_call(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos = pos))
        
    
    
    def quick_call_d(self, *args,pos,**kwargs):
        '''
        Quick evaluation of the distance fraction of the structure at the given position.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the quick evaluation.
        pos : array_like
            The position at which to evaluate the distance fraction.
        **kwargs : dict
            Additional keyword arguments for the evaluation.

        Returns
        -------
        array_like
            The quick evaluated distance fraction at the given position.

        Raises
        ------
        KeyError
            If parameters are not provided.
        '''
        pos = np.asarray(pos)
        if args:
            return self._shape.quick_call_d(*args[self.__coor_pa_num:],
                                          pos = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_d(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos=pos))
        coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
        shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
        return self._shape.quick_call_d(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos = pos))
        
    
    def quick_call_dist(self, *args,pos,**kwargs):
        '''
        Quick evaluation of the distance function of the structure at the given position.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the quick evaluation.
        pos : array_like
            The position at which to evaluate the distance function.
        **kwargs : dict
            Additional keyword arguments for the evaluation.

        Returns
        -------
        array_like
            The quick evaluated distance function at the given position.

        Raises
        ------
        KeyError
            If parameters are not provided.
        '''
        pos = np.asarray(pos)
        if args:
            return self._shape.quick_call_raydistance(*args[self.__coor_pa_num:],
                                          pos = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_raydistance(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos=pos))
        coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
        shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
        return self._shape.quick_call_raydistance(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos = pos))
        
    def quick_call_intersect(self,*args,pos1,pos2,**kwargs):
        
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        if args:
            return self._shape.quick_call_lineintersect(
                *args[self.__coor_pa_num:],
                pos1 = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos1),
                pos2 = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos2))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_lineintersect(
                **shape_pa,
                pos1 = self._coordinate.quick_call(**coord_pa,pos=pos1),
                pos2 = self._coordinate.quick_call(**coord_pa,pos=pos2),)
        coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
        shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
        return self._shape.quick_call_lineintersect(
            **shape_pa,
            pos1 = self._coordinate.quick_call(**coord_pa,pos = pos1),
            pos2 = self._coordinate.quick_call(**coord_pa,pos = pos2))
        
    def check_boundary(self,params_list,mode='periodic'):
        '''
        Adjusts parameters to stay within specified bounds.

        Parameters
        ----------
        params_list : list
            List of parameters to check.
        mode : str, optional
            The mode to use for boundary checking ('cut' or 'periodic', default is 'periodic').

        Raises
        ------
        ValueError
            If the mode is not valid.
        '''
        if mode == 'cut':
            for i in params_list:
                lb = self.parameters[i].lb
                ub = self.parameters[i].ub
                self.parameters[i] = np.clip(self.parameters[i],lb,ub)
            return 
        if mode == 'periodic':
            for i in params_list:
                lb = self.parameters[i].lb
                ub = self.parameters[i].ub
                self.parameters[i] = (self.parameters[i]-lb)%(ub-lb)+lb
            return 
        raise ValueError(f"not a valid mode, {mode}, only 'cut', 'periodic',")
    
    
    
    def generate_points(self,random_np: int = 1024):
        '''
        Generates random points on the surface of the structure.

        Parameters
        ----------
        random_np : int, optional
            Number of random points to generate (default is 1024).

        Returns
        -------
        array_like
            Generated points on the surface of the structure.
        '''
        cpos,spos = fibonacci_sampling(random_np)
        
        coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
        shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
        
        points, _ = self._shape(**shape_pa).ray_point(cpos)
        
        return self._coordinate(**coord_pa).inverse(points)
        
        
    
    @staticmethod
    def shape_func(fn):
        '''
        Decorator to register a shape function.

        Parameters
        ----------
        fn : function
            The shape function to register.

        Returns
        -------
        function
            The registered shape function.
        '''
        Structure_3D._shape_3d_fn[fn.__name__] = fn
        return fn
    
    @staticmethod
    def coordinate_func(fn):
        '''
        Decorator to register a coordinate function.

        Parameters
        ----------
        fn : function
            The coordinate function to register.

        Returns
        -------
        function
            The registered coordinate function.
        '''
        Structure_3D._coordinate_fn[fn.__name__] = fn
        return fn
    

class Structure_3D_fitter(Structure_3D):
    '''
    A 3D structures with an error function.

    Attributes
    ----------
    _error_fcall_fn : dict
        A dictionary mapping error function names to their corresponding functions.
    _error_name : str
        The name of the error function used.
    _error : function
        The error function instance.
    _error_params : list
        List of parameters for the error function.

    Methods
    -------
    __init__(coordinate_class='Coordinate3D', shape_class='Ellipsoid', error_func='isodensity_sums_fdev_rscale', **kwargs)
        Initializes the Structure_3D_fitter object.
    _set_error(error_func)
        Sets the error function based on the provided name.
    from_parameters(*args, **kwargs)
        Creates a new Structure_3D_fitter instance with the given parameters.
    __eq__(other)
        Checks equality with another Structure_3D_fitter object.
    _error_call_from_isodensity(params, kwargs)
        Evaluates the error function using isodensity.
    _error_call_from_shell(params, kwargs)
        Evaluates the error function using shell method.
    _error_call_from_grid(params, kwargs)
        Evaluates the error function using grid method.
    _error_dist_from_isodensity(params, kwargs)
        Evaluates the error function using distance from isodensity.
    _error_d_from_isodensity(params, kwargs)
        Evaluates the error function using derivative from isodensity.
    fit(algorithm="scipy_neldermead", **kwargs)
        Fits the structure using the specified algorithm.
    estimate_error(test_parameter, random_np=1024, **kwargs)
        Estimates the error for the given parameters.
    estimate_init_parameter(fitdata)
        Estimates initial parameters for fitting.
    __repr__()
        Returns a string representation of the Structure_3D_fitter object.
    minimize_func_fcall(fn)
        Decorator to register an error function.
    '''
    _error_fcall_fn = {}
    def __init__(self,coordinate_class='Coordinate3D',shape_class='Ellipsoid',error_func='isodensity_sums_fdev_rscale',**kwargs):
        """
        Initialize the Structure_3D_fitter instance.

        Parameters
        ----------
        coordinate_class : str, optional
            The class name for the coordinate system to be used. Default is 'Coordinate3D'.
        shape_class : str, optional
            The class name for the shape model to be used. Default is 'Ellipsoid'.
        error_func : str, optional
            The name of the error function to be used for fitting. Default is 'isodensity_sums_fdev_rscale'.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the parent class.
        """
        super().__init__(coordinate_class=coordinate_class,shape_class=shape_class)
        
        
        
        self._error_name = error_func
        self._error = self._error_fcall_fn[error_func]

        self._error_params = list(func_required_key(self._error).keys())
        self._error_params = list(filter(lambda x: ('f_call' not in x) and ('d_call' not in x), self._error_params))  
        
        self._set_error(error_func)

    
    def _set_error(self,error_func):
        """
        Set the error function based on the provided error function name.

        Parameters
        ----------
        error_func : str
            The name of the error function to be used.

        Raises
        ------
        ValueError
            If the provided error function name is not recognized.
        """
        if error_func[:5] == 'shell':
            self.error = self._error_call_from_shell
            return 
        if error_func[:4] == 'grid':
            self.error = self._error_call_from_grid
            return
        if error_func[:10] =='isodensity':
            if 'dist' in error_func:
                self.error = self._error_dist_from_isodensity
                return
            if 'ddev' in error_func:
                self.error = self._error_d_from_isodensity
                return
            if 'fdev' in error_func:
                self.error = self._error_call_from_isodensity
                return
        raise ValueError(f"{error_func} is not identified")
    
    
    def from_parameters(self,*args,**kwargs):
        """
        Create a new instance of Structure_3D_fitter with the same configuration as the current instance.

        Parameters
        ----------
        *args : list
            Positional arguments to be passed to the new instance.
        **kwargs : dict
            Keyword arguments to be passed to the new instance.

        Returns
        -------
        Structure_3D_fitter
            A new instance of Structure_3D_fitter with the same configuration.
        """
        ret = Structure_3D_fitter(coordinate_class=self._coordinate_name,shape_class=self._shape_name,error_func=self._error_name)
        ret.set_parameters(*args,**kwargs)
        
        return ret
    
    def __eq__(self, other):
        """
        Check if two Structure_3D_fitter instances are equal.

        Parameters
        ----------
        other : Structure_3D_fitter
            The other instance to compare with.

        Returns
        -------
        bool
            True if the instances are equal, False otherwise.
        """
        if isinstance(other,Structure_3D_fitter):
            if self._coordinate_name == other._coordinate_name:
                if self._shape_name == other._shape_name:
                    if self._error_name == other._error_name:
                        return True
        return False
    
    def _error_call_from_isodensity(self, params,kwargs):
        """
        Calculate the error using the isodensity error function.

        Parameters
        ----------
        params : list or dict
            The parameters to be used in the error calculation.
        kwargs : dict
            Additional keyword arguments required by the error function.

        Returns
        -------
        float
            The calculated error value.
        """
        f_call = self.quick_call(*params,pos=kwargs['pos'])
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(f_call=f_call,**error_pa)
    
    def _error_call_from_shell(self, params, kwargs):
        """
        Calculate the error using the shell error function.

        Parameters
        ----------
        params : list or dict
            The parameters to be used in the error calculation.
        kwargs : dict
            Additional keyword arguments required by the error function.

        Returns
        -------
        float
            The calculated error value.
        """
        if not isinstance(params,dict):
            params = dict(zip(self.parameters.keys(),params))
        params1 = params.copy()
        params1['a'] = 0.98*params1['a']
        params2 = params.copy()
        params2['a'] = 1.02*params2['a']
        f_call1 = self.quick_call(pos=kwargs['pos'],**params1)
        f_call2 = self.quick_call(pos=kwargs['pos'],**params2)
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(f_call1=f_call1,f_call2=f_call2,**error_pa)
    
    def _error_call_from_grid(self, params, kwargs):
        """
        Calculate the error using the grid error function.

        Parameters
        ----------
        params : list or dict
            The parameters to be used in the error calculation.
        kwargs : dict
            Additional keyword arguments required by the error function.

        Returns
        -------
        float
            The calculated error value.
        """
        if not isinstance(params,dict):
            params = dict(zip(self.parameters.keys(),params))
        params1 = params.copy()
        params1['a'] = 0.9*params1['a']
        params2 = params.copy()
        params2['a'] = 1.1*params2['a']
        f_call1 = self.quick_call(pos=kwargs['pos'],**params1)
        f_call2 = self.quick_call(pos=kwargs['pos'],**params2)
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(f_call1=f_call1,f_call2=f_call2,**error_pa)
        
    def _error_dist_from_isodensity(self, params, kwargs):
        """
        Calculate the error using the isodensity distance error function.

        Parameters
        ----------
        params : list or dict
            The parameters to be used in the error calculation.
        kwargs : dict
            Additional keyword arguments required by the error function.

        Returns
        -------
        float
            The calculated error value.
        """
        d_call = self.quick_call_dist(*params,pos=kwargs['pos'])
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(d_call=d_call,**error_pa)
    
    def _error_d_from_isodensity(self, params, kwargs):
        """
        Calculate the error using the isodensity distance fraction error function.

        Parameters
        ----------
        params : list or dict
            The parameters to be used in the error calculation.
        kwargs : dict
            Additional keyword arguments required by the error function.

        Returns
        -------
        float
            The calculated error value.
        """
        d_call = self.quick_call_d(*params,pos=kwargs['pos'])
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(d_call=d_call,**error_pa)
    
    def fit(self, algorithm="scipy_neldermead", **kwargs):
        """
        Fit the structure using the specified algorithm.

        Parameters
        ----------
        algorithm : str, optional
            The optimization algorithm to be used. Default is 'scipy_neldermead'.
        **kwargs : dict
            Additional keyword arguments to be passed to the optimizer.

        Returns
        -------
        object
            The result of the optimization process.
        """
        other_kwargs = dict(**kwargs)
        fun_kwargs = {i: other_kwargs.pop(i) for i in self._error_params}
        if 'pos' not in fun_kwargs:
            fun_kwargs['pos'] = other_kwargs.pop('pos')
        return om.minimize(
            fun = self.error_call_from_isodensity,
            params = dict(**self.parameters),
            algorithm=algorithm,
            fun_kwargs={'kwargs':fun_kwargs},
            constraints = self.constraints,
            bounds = self.parameters.scipy_bounds,**other_kwargs)
        
    def estimate_error(self,test_parameter , random_np = 1024,**kwargs):
        """
        Estimate the error for a given set of parameters.

        Parameters
        ----------
        test_parameter : dict
            The parameters to be tested.
        random_np : int, optional
            The number of random points to generate for the estimation. Default is 1024.
        **kwargs : dict
            Additional keyword arguments to be passed to the error function.

        Returns
        -------
        float
            The estimated error value.
        """
        points = kwargs.get('pos',self.generate_points(random_np))
        kwargs={'pos': points}
        if 'r' in self._error_params:
            r = vector_length3d(points)
            kwargs['r'] = r/np.sqrt(np.sum(r**2)/len(r))
        params = {i: test_parameter[i] for i in self.parameters}
        return self.error(list(params.values()),kwargs)

    
    
    def estimate_init_parameter(self,fitdata):
        """
        Estimate initial parameters for fitting based on the provided data.

        Parameters
        ----------
        fitdata : dict
            The data to be used for estimating initial parameters.

        Returns
        -------
        dict
            The estimated initial parameters.
        """
        r25 = np.nanpercentile(fitdata['r'],25)
        r75 = np.nanpercentile(fitdata['r'],75)
        va = np.mean(np.abs(fitdata['pos'][fitdata['r']>r75]),axis=0)    #use all r>r75 points to estimate a vector 
        vc = np.mean(np.abs(fitdata['pos'][fitdata['r']<r25]),axis=0)    #use all r<r25 points to estimate c vector
        
        ini_parameter = self.parameters.new()
        
        Matrixxyz = spatial.transform.Rotation.align_vectors(np.eye(3), np.array([va,[0,0,0],vc]), weights=[1,0,1])
        ini_Mxyz = Matrixxyz[0].as_euler('zyx')
        
        ini_parameter['ang1'] =(ini_Mxyz[0] + np.pi) % (2*np.pi)-np.pi 
        ini_parameter['ang2'] = (ini_Mxyz[1] + np.pi/2) % (np.pi)-np.pi/2
        ini_parameter['ang3'] = (ini_Mxyz[2] + np.pi) % (2*np.pi)-np.pi 
        if ('eps_ab' in ini_parameter) and ('eps_bc' in ini_parameter) :
            least = ellipsoid_fit(fitdata['pos'])
            cba = np.abs(np.sort(least[2]))
            ini_parameter['eps_ab'] = 1-cba[1]/cba[2]
            ini_parameter['eps_bc'] = 1-cba[0]/cba[1]
        return ini_parameter
    
    
    def __repr__(self):
        """
        Return a string representation of the Structure_3D_fitter instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        coor_repr = repr(self._coordinate(**self.parameters))
        shape_repr = repr(self._shape(**self.parameters))
        error = self._error_name
        
        lin1 = "<Structure_3D|:"+'\n'
        lin2 = "   "+ coor_repr + '\n'
        lin3 = "   "+ shape_repr + '\n'
        lin4 = "   "+ "<Error| "+ error + " |>"

        return lin1 + lin2 + lin3 + lin4
    
    @staticmethod
    def minimize_func_fcall(fn):
        """
        Decorator to register an error function for use in the fitting process.

        Parameters
        ----------
        fn : callable
            The error function to be registered.

        Returns
        -------
        callable
            The registered error function.
        """
        Structure_3D_fitter._error_fcall_fn[fn.__name__] = fn
        return fn
    
    
    



    