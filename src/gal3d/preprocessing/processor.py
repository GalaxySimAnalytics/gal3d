
from functools import wraps
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from .estimate.local_estimate import Local_est
from .estimate.global_property import Global_prop
from .spherical_field.spherical_vector import Sphere_vector
from .spherical_field.field import Field



def ray_method(fun):
    ''' constrain the method, after build ray'''
    
    @wraps(fun)
    def wrapper(*args,**kwargs):
        self = args[0]
        if (not hasattr(self,'rays_index')) or (not hasattr(self,'rays')):
            raise AttributeError("This is a ray method that needs build_ray_vector() first")
        result = fun(*args,**kwargs)
        return result
    
    return wrapper
    
def field_method(fun):
    ''' constrain the method, after build field'''
    @wraps(fun)
    def wrapper(*args,**kwargs):
        self = args[0]
        if (not hasattr(self,'field')):
            raise AttributeError("This is a field method that needs build_field_boundary() first")
        result = fun(*args,**kwargs)
        return result
    
    return wrapper

def timer(fun):
    ''' measure cost time'''
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
        Parameter:
            pos: array (N,3),
                the position of N particles, in 3D cartis.. (x,y,z) coordinate,
            weight: array (N,)
                the property of N particles, such as mass ...
            parameter_mode: str, 
                {'Density','Mean'}, determine how to calcualte the target parameter.
            num_near: int,
                used in KDtree, determine how much num of nerighbors to estimate the target parameter.
            kdtree_options: dict,
                the options used to build KDtree and KDtree.query.
            verbose: bool,
                verbose control
        
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
        build ray vectors uniformly on the sphere
        
        Parameter:
            N_ray: int,
                the number of ray.
            method: str,
                {'fibonacci', 'muller'}, method to generate unit ray vector
        Return
            self
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
        Determine field inner and outer radius in each ray.
        
        Parameters
            inner: float,
                used to determine the inner boundary of the field
            
            outer: float,
                used to determine the outer boundary of the field
            
            inner_mode: str,
                {'dist','pct','value'}, mode to determine inner radius
            
            outer_mode: str,
                {'dist','pct','value'}, mode to determine outer radius
                
        Return 
            self
        '''
        
        self.field = Field(self,inner=inner,outer=outer,inner_mode=inner_mode,outer_mode=outer_mode)

        return self
    
    @timer
    @field_method
    def build_field_sample(self,num_p:int = 500, step_mode: str = 'log'):
        '''
        
        
        Parameters
            num_p: int,
                the sample points on each ray
            step_mode: str,
                {'log','lin'}, logscale or linear scale
        
        Return self
        '''
        self.field.build_sample(self,num_p=num_p, step_mode=step_mode)
        
        return self
    
    @timer
    @field_method
    def build_field_interpolator(self,interpolator_method = 'LU', f_de=True,interpolator_kwargs=dict(),**kwargs):
        '''
        interpolate each ray
        
        Parameter:
            interpolator_method: str,
                {'SG','LU'},method to interpolate each ray.
            interpolator_kwargs: dict,
                options used in interpolate
        
        Return
            self
        
        '''
        
        
        self.field.build_interpolate(interpolator_method=interpolator_method,f_de=f_de,
                                     interpolator_kwargs=interpolator_kwargs,**kwargs)

        return self
    
    @timer
    @field_method
    def build_field_isoprofile(self,method='pair',from_rays_func = False,res_b=0.2,res_c=0.1,**kwargs):
        '''
        determine how the isodensity profile look like
        
        Parameter:
            method: str,
                method to determine the isoprofile
            from_rays_func: bool,
                get the points' parameters, from KDtree or ray ingterpolator.
            res_b: float, 0< <1
                the target structure resolution at xy, within a unit sphere
            res_c: float, 0< <1
                the target structure resolution at z, within a unit sphere
            
        kwargs:

        Return 
            self
        
        
        
        '''
        
        
        
        self.field.build_isoprofile(self,method = method,from_rays_func=from_rays_func,res_b=res_b,res_c=res_c,**kwargs)
        return self
        
        
    @ray_method
    def pos_ray_n(self, n:int) -> np.ndarray:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the coordinates of the points closest to the nth ray
        '''
        return self.pos[self.points_index[n]]
    
    @ray_method
    def r_ray_n(self, n:int) -> np.ndarray:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the radius of the points closest to the nth ray
        '''
        return self.r[self.points_index[n]]
    
    @ray_method
    def weight_ray_n(self,n:int)->np.ndarray:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the weight of the points closest to the nth ray
        '''
        return self.weight[self.points_index[n]]
    
    @ray_method
    def parameter_ray_n(self,n:int)->np.ndarray:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the parameter of the points closest to the nth ray
        '''
        return self.parameter[self.points_index[n]]
    
    @ray_method
    def gradient_ray_n(self,n:int)->tuple:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the gradient of the points closest to the nth ray
        '''
        return ((self.gradient[0][0][self.points_index[n]],self.gradient[0][1][self.points_index[n]]),
                (self.gradient[1][0][self.points_index[n]],self.gradient[1][1][self.points_index[n]]))
        
    @ray_method
    def fparameter_ray_n(self,n:int)->PchipInterpolator:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the PchipInterpolator between the points r and parameter closest to the nth ray
        '''
        return self.fparameter_r[n]
        
    @field_method
    @ray_method
    def inner_ray_n(self, n:int) -> float:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the inner boundary of the nth ray
        '''
        return self.field.inner_r[n]
    
    @field_method
    @ray_method
    def outer_ray_n(self, n:int) -> float:
        '''
        Parameters:
            n: the index of the ray, 0 ~ num_ray-1
        Return:
            the outer boundary of the nth ray
        '''
        return self.field.outer_r[n]
    
    
    
