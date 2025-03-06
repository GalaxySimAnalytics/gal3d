
import logging
from collections.abc import Iterable

import numpy as np


from .util import iso_profile_by_moi,iso_profile_by_pair
from ..estimate.local_estimate import Local_est
from ..ray.ray_profile import Ray


logger = logging.getLogger("gal3d.preprocessing.spherical_field.field")

class Field:
    def __init__(self,Base,inner=0.5, outer=95, inner_mode='dist', outer_mode='pct'):
        '''
        Initialize the Field class for calculating parameters at various distances in a 3D galaxy model.

        Parameters
        ----------
        Base : Galaxy3D
            An instance of the Galaxy3D class used to compute parameters at specific positions.
        inner : float, optional
            The inner boundary value. Default is 0.5.
        outer : float, optional
            The outer boundary value. Default is 95.
        inner_mode : str, optional
            The mode for calculating the inner boundary. Options are 'dist', 'pct', or 'value'. Default is 'dist'.
        outer_mode : str, optional
            The mode for calculating the outer boundary. Options are 'dist', 'pct', or 'value'. Default is 'pct'.
        '''
        self.__bound_method = {'dist': self.__bound_dist,
                               'pct': self.__bound_pct,
                               'value': self.__bound_value}
        
        
        self.inner_r = self.__bound_method[inner_mode](inner, Base,mode='min')
        self.outer_r = self.__bound_method[outer_mode](outer, Base,mode='max')
        self.rays_vect = Base.rays.pos
        self.check_boundary()
        
    
    def check_boundary(self):
        '''
        Check if the outer boundaries are greater than the inner boundaries.

        Raises
        ------
        ValueError
            If any outer boundary is not greater than the corresponding inner boundary.
        '''
        if not all(self.outer_r > self.inner_r):
            logger.error('The outer boundaries need to be greater than the inner boundaries')
            raise ValueError('The outer boundaries need to be greater than the inner boundaries')
        return 
    
    
    
    def build_sample(self,base: Local_est, num_p:int = 500, step_mode: str = 'log',):
        '''
        Build a sample of points along the rays for parameter calculation.

        Parameters
        ----------
        base : Local_est
            An instance of the Local_est class used to compute parameters.
        num_p : int, optional
            The number of points to sample along each ray. Default is 500.
        step_mode : str, optional
            The mode for spacing the points. Options are 'lin' for linear spacing or 'log' for logarithmic spacing. Default is 'log'.

        Raises
        ------
        ValueError
            If the base is not an instance of Local_est.
        '''
        if not isinstance(base,Local_est):
            raise ValueError(f"base shoud be Local_est")
        
        self.__point_method = {'lin': self.__point_lin,
                               'log': self.__point_log}
        
        self.points_r = self.__point_method[step_mode](num_p)
        self.points_pos = np.einsum('ij,ik->ijk', self.points_r, base.rays.pos)
        
        
        points_que = self.points_pos.reshape(self.points_pos.shape[0]*self.points_pos.shape[1], 3)
        self.points_parameter = base.get_parameter(points_que).reshape(self.points_r.shape)
    
    
    def build_interpolate(self,interpolator_method = 'LU',f_de=True,interpolator_kwargs=dict(),**kwargs):
        '''
        Build interpolators for the sampled points.

        Parameters
        ----------
        interpolator_method : str, optional
            The method used for interpolation. Default is 'LU'.
        f_de : bool, optional
            Whether to use density estimation. Default is True.
        interpolator_kwargs : dict, optional
            Additional keyword arguments for the interpolator. Default is an empty dictionary.
        **kwargs : dict
            Additional keyword arguments.
        '''
        self.rays_func = [Ray(self.points_r[i],self.points_parameter[i],f_de=f_de,
                              interpolator_method=interpolator_method,interpolator_kwargs=interpolator_kwargs,**kwargs) 
                          for i in range(len(self.points_parameter))]
        
        
    
    def build_isoprofile(self,Base,method: str = 'pair', from_rays_func = False,res_b=0.2,res_c=0.1,**kwargs):
        '''
        Build isoprofiles for the galaxy model.

        Parameters
        ----------
        Base : Galaxy3D
            An instance of the Galaxy3D class used to compute parameters.
        method : str, optional
            The method used for building isoprofiles. Options are 'moi' or 'pair'. Default is 'pair'.
        from_rays_func : bool, optional
            Whether to use the ray functions for building isoprofiles. Default is False.
        res_b : float, optional
            Resolution parameter for the isoprofile. Default is 0.2.
        res_c : float, optional
            Resolution parameter for the isoprofile. Default is 0.1.
        **kwargs : dict
            Additional keyword arguments.
        '''
    
        self.set_isosphere(Base=Base,from_rays_func=from_rays_func,**kwargs)
        
        Method = {'moi':iso_profile_by_moi,'pair':iso_profile_by_pair}
        
        self.iso_pro_parameter = Method[method](self.rays_vect,self.iso_parameters,res_b,res_c) 
      #  iso_profile_by_moi(self.rays_vect,self.iso_parameters,res_b,res_c)   
        interpolator_method = kwargs.get('interpolator_method','LU')
        
               
        self.iso_pro_func = Ray(self.iso_pro_r,self.iso_pro_parameter,interpolator_method=interpolator_method)

        
    def set_isosphere(self,Base,from_rays_func = False,**kwargs):
        '''
        Set the isosphere for the galaxy model.

        Parameters
        ----------
        Base : Galaxy3D
            An instance of the Galaxy3D class used to compute parameters.
        from_rays_func : bool, optional
            Whether to use the ray functions for setting the isosphere. Default is False.
        **kwargs : dict
            Additional keyword arguments.
        '''
        num_p = kwargs.get('num_p',self.points_r.shape[1])
        
        if from_rays_func:
            self.iso_pro_r = np.geomspace(np.max(self.inner_r),
                                      np.min(self.outer_r),num_p)
            self.iso_points = np.einsum('ij,k->ikj',self.rays_vect,self.iso_pro_r)
            self.iso_parameters = np.array([self.rays_func[i](self.iso_pro_r) for i in range(len(self.rays_func))])
            
        else:
            self.iso_pro_r = np.geomspace(np.percentile(self.inner_r,50),
                                      np.percentile(self.outer_r,50),num_p)
            self.iso_points = np.einsum('ij,k->ikj',self.rays_vect,self.iso_pro_r)
        
            self.iso_parameters = Base.get_parameter(self.iso_points.reshape(self.iso_points.shape[0]*self.iso_points.shape[1],3)).reshape(
                (self.iso_points.shape[0],self.iso_points.shape[1]))
            
            
            
            
    def generate(self, r: Iterable | float, level: tuple = (0,0))->dict:
        '''
        Generate the equivalent surface for a given radius.

        Parameters
        ----------
        r : Iterable | float
            The radius or radii for which to generate the equivalent surface.
        level : tuple, optional
            The level of the equivalent surface. Default is (0, 0).

        Returns
        -------
        dict
            A dictionary containing the positions, parameters, and radii of the equivalent surface.
        '''
        
        ftarget = self.query_iso_f(r, which =level[0])   
        
        rtarget = self.query_rays_r(ftarget,which=level[1])  # shape (len(ftarget),num_rays)
        
        if isinstance(r,Iterable):
            target_pos = np.einsum('ji,ik->jik', rtarget, self.rays_vect) # (len(ftarget),num_rays) * (num_rays , 3)

        else:

            target_pos = np.einsum('i,ik->ik',rtarget,self.rays_vect)
            
        Eq_surface = {}
        Eq_surface['pos'] = target_pos
        Eq_surface['parameter'] = ftarget
        Eq_surface['r'] = rtarget
        return Eq_surface
    
    
    def generate_by_f(self, f: Iterable | float, level: tuple = (0,0))->dict:
        '''
        Generate the equivalent surface for a given parameter value.

        Parameters
        ----------
        f : Iterable | float
            The parameter value or values for which to generate the equivalent surface.
        level : tuple, optional
            The level of the equivalent surface. Default is (0, 0).

        Returns
        -------
        dict
            A dictionary containing the positions, radii, and isoradii of the equivalent surface.
        '''
        rtarget = self.query_rays_r(f,which=level[1]) 
        
        if isinstance(f,Iterable):
            target_pos = np.einsum('ji,ik->jik', rtarget, self.rays_vect)
        else:   
            target_pos = np.einsum('i,ik->ik',rtarget,self.rays_vect)
            
        iso_r = self.query_iso_r(f, which=level[0])
        
        Eq_surface = {}
        Eq_surface['pos'] = target_pos
        Eq_surface['r'] = rtarget
        Eq_surface['iso_r'] = iso_r
        return Eq_surface
              
        
    def query_rays_f(self,r, which=0):
        '''
        Query the parameter value at a given radius along the rays.

        Parameters
        ----------
        r : float
            The radius at which to query the parameter value.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        np.ndarray
            The parameter values at the given radius.
        '''
        if which > 0:
            return np.array([i.upper(r,inv = False) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(r,inv = False) for i in self.rays_func])
        
        return np.array([i(r,inv = False) for i in self.rays_func]).T
    
    
    def query_rays_r(self,f, which = 0):
        '''
        Query the radius for a given parameter value along the rays.

        Parameters
        ----------
        f : float
            The parameter value for which to query the radius.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        np.ndarray
            The radii corresponding to the given parameter value.
        '''
        if which > 0:
            return np.array([i.upper(f,inv = True) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(f,inv = True) for i in self.rays_func])
        
        return np.array([i(f, inv = True) for i in self.rays_func]).T
    
    
    def query_iso_f(self,r,which=0):
        '''
        Query the parameter value at a given radius for the isoprofile.

        Parameters
        ----------
        r : float
            The radius at which to query the parameter value.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        float
            The parameter value at the given radius.
        '''
        if which > 0:
            return self.iso_pro_func.upper(r,inv=False)
        if which < 0:
            return self.iso_pro_func.lower(r,inv=False)
        
        return self.iso_pro_func(r,inv=False)
    
    
    def query_iso_r(self,f,which=0):
        '''
        Query the radius for a given parameter value for the isoprofile.

        Parameters
        ----------
        f : float
            The parameter value for which to query the radius.
        which : int, optional
            The level of the query. Default is 0.

        Returns
        -------
        float
            The radius corresponding to the given parameter value.
        '''
        if which > 0:
            return self.iso_pro_func.upper(f,inv=True)
        if which < 0:
            return self.iso_pro_func.lower(f,inv=True)
        
        return self.iso_pro_func(f,inv=True)
    
    
    
    def __bound_dist(self, value, Base,**kwargs):
        '''
        Calculate the boundary based on a fixed distance.

        Parameters
        ----------
        value : float
            The distance value.
        Base : Galaxy3D
            An instance of the Galaxy3D class.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The boundary values.
        '''
        return value*np.ones(Base.rays.num)
    def __bound_pct(self, value, Base,**kwargs):
        '''
        Calculate the boundary based on a percentile.

        Parameters
        ----------
        value : float
            The percentile value.
        Base : Galaxy3D
            An instance of the Galaxy3D class.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The boundary values.
        '''
        return np.array([np.percentile(Base.r_ray_n(i), value) for i in range(Base.rays.num)])
    
    def __bound_value(self,value,Base,mode='max',**kwargs):
        '''
        Calculate the boundary based on a parameter value.

        Parameters
        ----------
        value : float
            The parameter value.
        Base : Galaxy3D
            An instance of the Galaxy3D class.
        mode : str, optional
            The mode for calculating the boundary. Options are 'max' or 'min'. Default is 'max'.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The boundary values.

        Raises
        ------
        ValueError
            If the mode is not 'max' or 'min'.
        '''
        if mode =='max':
    
            return np.array([np.max(Base.r_ray_n(i)[Base.parameter_ray_n(i)>value]) for i in range(Base.rays.num)])
        if mode =='min':
            return np.array([np.min(Base.r_ray_n(i)[Base.parameter_ray_n(i)<value]) for i in range(Base.rays.num)])
        raise ValueError(f"{mode} is not a valid value. Only 'max' and 'min' are valid.")
    
    
    def __point_lin(self,num_p):
        '''
        Generate linearly spaced points between the inner and outer boundaries.

        Parameters
        ----------
        num_p : int
            The number of points to generate.

        Returns
        -------
        np.ndarray
            The generated points.
        '''
        return np.linspace(self.inner_r, self.outer_r, num_p).T
    def __point_log(self,num_p):
        '''
        Generate logarithmically spaced points between the inner and outer boundaries.

        Parameters
        ----------
        num_p : int
            The number of points to generate.

        Returns
        -------
        np.ndarray
            The generated points.
        '''
        return np.geomspace(self.inner_r, self.outer_r,num_p).T 