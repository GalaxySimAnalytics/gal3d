
from functools import wraps
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from .estimate.local_estimate import Local_est
from .estimate.global_property import Global_prop
from .spherical_field.spherical_vector import Sphere_vector
from .spherical_field.field import Field



def ray_method(fun):
    '''
    Decorator to constrain a method to be used only after `build_ray_vector()` has been called.

    Parameters
    ----------
    fun : function
        The function to be wrapped.

    Returns
    -------
    wrapper : function
        The wrapped function that checks if `rays_index` and `rays` attributes exist.

    Raises
    ------
    AttributeError
        If `rays_index` or `rays` attributes are not found, indicating that `build_ray_vector()` must be called first.
    '''
    
    @wraps(fun)
    def wrapper(*args,**kwargs):
        self = args[0]
        if (not hasattr(self,'rays_index')) or (not hasattr(self,'rays')):
            raise AttributeError("This is a ray method that needs build_ray_vector() first")
        result = fun(*args,**kwargs)
        return result
    
    return wrapper
    
def field_method(fun):
    '''
    Decorator to constrain a method to be used only after `build_field_boundary()` has been called.

    Parameters
    ----------
    fun : function
        The function to be wrapped.

    Returns
    -------
    wrapper : function
        The wrapped function that checks if `field` attribute exists.

    Raises
    ------
    AttributeError
        If `field` attribute is not found, indicating that `build_field_boundary()` must be called first.
    '''
    @wraps(fun)
    def wrapper(*args,**kwargs):
        self = args[0]
        if (not hasattr(self,'field')):
            raise AttributeError("This is a field method that needs build_field_boundary() first")
        result = fun(*args,**kwargs)
        return result
    
    return wrapper

def timer(fun):
    '''
    Decorator to measure the execution time of a function.

    Parameters
    ----------
    fun : function
        The function to be wrapped.

    Returns
    -------
    wrapper : function
        The wrapped function that prints the execution time if `verbose` is True.
    '''
    name = fun.__name__
    @wraps(fun)
    def wrapper(*args,**kwargs):
        self = args[0]
        if (hasattr(self,'_verbose')) and self._verbose:
            s =  time.time()
            print(name.replace('_',' ').capitalize()+': ',end='')
        result = fun(*args,**kwargs)
        
        if (hasattr(self,'_verbose')) and self._verbose:
            e =  time.time()
            print(f"{(e-s):.5f} sec")
        return result
    
    return wrapper



class Particles(Local_est,Global_prop):
    def __init__(self,pos,weight,parameter_mode: str = 'Density', num_near: int = 32,verbose = True,**kwargs):
        '''
        Initialize the Particles class with particle positions, weights, and other parameters.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            The positions of N particles in 3D Cartesian coordinates (x, y, z).
        weight : array_like, shape (N,)
            The properties of N particles, such as mass.
        parameter_mode : str, optional
            {'Density', 'Mean'}, determines how to calculate the target parameter. Default is 'Density'.
        num_near : int, optional
            The number of nearest neighbors used in KDTree to estimate the target parameter. Default is 32.
        verbose : bool, optional
            If True, prints progress and timing information. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments passed to `Local_est` and `Global_prop`.

        Returns
        -------
        self : Particles
            The initialized Particles object.
        '''

        
        self._verbose = verbose
        
        if verbose:
            s= time.time()
            print("Init global property: ",end='')
        
        Global_prop.__init__(self,pos,weight)
        
        if verbose:
            e = time.time()
            print(f"{(e-s):.5f} seconds")
            print("Init local estimator: ",end='')
            
        Local_est.__init__(self,pos,weight,parameter_mode=parameter_mode,num_near=num_near,**kwargs)
        

        if verbose:
            f = time.time()
            print(f"{(f-e):.5f} seconds")
    
    
    
    
    @timer
    def build_ray_vector(self,N_ray=1024, method = 'fibonacci'):
        '''
        Build ray vectors uniformly distributed on a sphere.

        Parameters
        ----------
        N_ray : int, optional
            The number of rays to generate. Default is 1024.
        method : str, optional
            {'fibonacci', 'muller'}, the method used to generate unit ray vectors. Default is 'fibonacci'.

        Returns
        -------
        self : Particles
            The Particles object with updated ray vectors and indices.
        '''
            
        self.rays = Sphere_vector(N_ray,method)
        self.rays_index = self.rays.assign_points(self.pos)
        self.rays_points_num = np.bincount(self.rays_index)
        if self._verbose:
            print(f"Ray {np.argmin(self.rays_points_num)} has the minimum particle count of {np.min(self.rays_points_num)}.  "
                  ,end='')
        if np.min(self.rays_points_num)<3:
            print(f"Ray {np.argmin(self.rays_points_num)} has {np.min(self.rays_points_num)} particles. ")
            print(f"It should be > 2, so please make the ray num smaller.  ")

        ind = [[] for _ in range(N_ray)]
        for i,j in enumerate(self.rays_index):
            ind[j].append(i)
        self.points_index = [np.array(i) for i in ind]
        
        return self
    
    @timer
    @ray_method
    def build_field_boundary(self,inner=0.5, outer=95, inner_mode='dist', outer_mode='pct'):
        '''
        Determine the inner and outer boundaries of the field for each ray.

        Parameters
        ----------
        inner : float, optional
            The value used to determine the inner boundary of the field. Default is 0.5.
        outer : float, optional
            The value used to determine the outer boundary of the field. Default is 95.
        inner_mode : str, optional
            {'dist', 'pct', 'value'}, the mode used to determine the inner radius. Default is 'dist'.
        outer_mode : str, optional
            {'dist', 'pct', 'value'}, the mode used to determine the outer radius. Default is 'pct'.

        Returns
        -------
        self : Particles
            The Particles object with updated field boundaries.
        '''
        
        self.field = Field(self,inner=inner,outer=outer,inner_mode=inner_mode,outer_mode=outer_mode)

        return self
    
    @timer
    @field_method
    def build_field_sample(self,num_p:int = 500, step_mode: str = 'log'):
        '''
        Build sample points along each ray for field interpolation.

        Parameters
        ----------
        num_p : int, optional
            The number of sample points on each ray. Default is 500.
        step_mode : str, optional
            {'log', 'lin'}, the scale used for sampling points. Default is 'log'.

        Returns
        -------
        self : Particles
            The Particles object with updated field samples.
        '''
        self.field.build_sample(self,num_p=num_p, step_mode=step_mode)
        
        return self
    
    @timer
    @field_method
    def build_field_interpolator(self,interpolator_method = 'LU', f_de=True,interpolator_kwargs=dict(),**kwargs):
        '''
        Interpolate the field along each ray.

        Parameters
        ----------
        interpolator_method : str, optional
            {'SG', 'LU'}, the method used for interpolation. Default is 'LU'.
        f_de : bool, optional
            If True, enables additional interpolation options. Default is True.
        interpolator_kwargs : dict, optional
            Additional keyword arguments passed to the interpolator. Default is an empty dict.

        Returns
        -------
        self : Particles
            The Particles object with updated field interpolator.
        '''
        
        
        self.field.build_interpolate(interpolator_method=interpolator_method,f_de=f_de,
                                     interpolator_kwargs=interpolator_kwargs,**kwargs)

        return self
    
    @timer
    @field_method
    def build_field_isoprofile(self,method='pair',from_rays_func = False,res_b=0.2,res_c=0.1,**kwargs):
        '''
        Determine the isodensity profile for the field.

        Parameters
        ----------
        method : str, optional
            The method used to determine the isoprofile. Default is 'pair'.
        from_rays_func : bool, optional
            If True, retrieves points' parameters from the ray interpolator. Default is False.
        res_b : float, optional
            The target structure resolution at xy within a unit sphere. Default is 0.2.
        res_c : float, optional
            The target structure resolution at z within a unit sphere. Default is 0.1.
        **kwargs : dict, optional
            Additional keyword arguments passed to the isoprofile builder.

        Returns
        -------
        self : Particles
            The Particles object with updated isodensity profile.
        '''
        
        
        
        self.field.build_isoprofile(self,method = method,from_rays_func=from_rays_func,res_b=res_b,res_c=res_c,**kwargs)
        return self
        
        
    @ray_method
    def pos_ray_n(self, n:int) -> np.ndarray:
        '''
        Retrieve the positions of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The coordinates of the points closest to the nth ray.
        '''
        return self.pos[self.points_index[n]]
    
    @ray_method
    def r_ray_n(self, n:int) -> np.ndarray:
        '''
        Retrieve the radii of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The radii of the points closest to the nth ray.
        '''
        return self.r[self.points_index[n]]
    
    @ray_method
    def weight_ray_n(self,n:int)->np.ndarray:
        '''
        Retrieve the weights of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The weights of the points closest to the nth ray.
        '''
        return self.weight[self.points_index[n]]
    
    @ray_method
    def parameter_ray_n(self,n:int)->np.ndarray:
        '''
        Retrieve the parameters of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        np.ndarray
            The parameters of the points closest to the nth ray.
        '''
        return self.parameter[self.points_index[n]]
    
    @ray_method
    def gradient_ray_n(self,n:int)->tuple:
        '''
        Retrieve the gradients of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        tuple
            The gradients of the points closest to the nth ray.
        '''
        return ((self.gradient[0][0][self.points_index[n]],self.gradient[0][1][self.points_index[n]]),
                (self.gradient[1][0][self.points_index[n]],self.gradient[1][1][self.points_index[n]]))
        
    @ray_method
    def fparameter_ray_n(self,n:int)->PchipInterpolator:
        '''
        Retrieve the interpolator for the parameters of particles closest to the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        PchipInterpolator
            The interpolator for the parameters of the points closest to the nth ray.
        '''
        return self.fparameter_r[n]
        
    @field_method
    @ray_method
    def inner_ray_n(self, n:int) -> float:
        '''
        Retrieve the inner boundary of the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        float
            The inner boundary of the nth ray.
        '''
        return self.field.inner_r[n]
    
    @field_method
    @ray_method
    def outer_ray_n(self, n:int) -> float:
        '''
        Retrieve the outer boundary of the nth ray.

        Parameters
        ----------
        n : int
            The index of the ray (0 to num_ray-1).

        Returns
        -------
        float
            The outer boundary of the nth ray.
        '''
        return self.field.outer_r[n]
    
    
    
