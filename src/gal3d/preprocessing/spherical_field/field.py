
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
        Parameters
            inner: float,
            
            outer: float,
            
            inner_mode: str,
            
            outer_mode: str,
        '''
        self.__bound_method = {'dist': self.__bound_dist,
                               'pct': self.__bound_pct,
                               'value': self.__bound_value}
        
        
        self.inner_r = self.__bound_method[inner_mode](inner, Base,mode='min')
        self.outer_r = self.__bound_method[outer_mode](outer, Base,mode='max')
        self.rays_vect = Base.rays.pos
        self.check_boundary()
        
    
    def check_boundary(self):
        if not all(self.outer_r > self.inner_r):
            logger.error('The outer boundaries need to be greater than the inner boundaries')
            raise ValueError('The outer boundaries need to be greater than the inner boundaries')
        return 
    
    
    
    def build_sample(self,base: Local_est, num_p:int = 500, step_mode: str = 'log',):
        if not isinstance(base,Local_est):
            raise ValueError(f"base shoud be Local_est")
        
        self.__point_method = {'lin': self.__point_lin,
                               'log': self.__point_log}
        
        self.points_r = self.__point_method[step_mode](num_p)
        self.points_pos = np.einsum('ij,ik->ijk', self.points_r, base.rays.pos)
        
        
        points_que = self.points_pos.reshape(self.points_pos.shape[0]*self.points_pos.shape[1], 3)
        self.points_parameter = base.get_parameter(points_que).reshape(self.points_r.shape)
    
    
    def build_interpolate(self,interpolator_method = 'LU',f_de=True,interpolator_kwargs=dict(),**kwargs):
        self.rays_func = [Ray(self.points_r[i],self.points_parameter[i],f_de=f_de,
                              interpolator_method=interpolator_method,interpolator_kwargs=interpolator_kwargs,**kwargs) 
                          for i in range(len(self.points_parameter))]
        
        
    
    def build_isoprofile(self,Base,method: str = 'pair', from_rays_func = False,res_b=0.2,res_c=0.1,**kwargs):
    
        self.set_isosphere(Base=Base,from_rays_func=from_rays_func,**kwargs)
        
        Method = {'moi':iso_profile_by_moi,'pair':iso_profile_by_pair}
        
        self.iso_pro_parameter = Method[method](self.rays_vect,self.iso_parameters,res_b,res_c) 
      #  iso_profile_by_moi(self.rays_vect,self.iso_parameters,res_b,res_c)   
        interpolator_method = kwargs.get('interpolator_method','LU')
        
               
        self.iso_pro_func = Ray(self.iso_pro_r,self.iso_pro_parameter,interpolator_method=interpolator_method)

        
    def set_isosphere(self,Base,from_rays_func = False,**kwargs):
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
        
        if which > 0:
            return np.array([i.upper(r,inv = False) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(r,inv = False) for i in self.rays_func])
        
        return np.array([i(r,inv = False) for i in self.rays_func]).T
    
    
    def query_rays_r(self,f, which = 0):
        if which > 0:
            return np.array([i.upper(f,inv = True) for i in self.rays_func])
        if which < 0:
            return np.array([i.lower(f,inv = True) for i in self.rays_func])
        
        return np.array([i(f, inv = True) for i in self.rays_func]).T
    
    
    def query_iso_f(self,r,which=0):
        if which > 0:
            return self.iso_pro_func.upper(r,inv=False)
        if which < 0:
            return self.iso_pro_func.lower(r,inv=False)
        
        return self.iso_pro_func(r,inv=False)
    
    
    def query_iso_r(self,f,which=0):
        if which > 0:
            return self.iso_pro_func.upper(f,inv=True)
        if which < 0:
            return self.iso_pro_func.lower(f,inv=True)
        
        return self.iso_pro_func(f,inv=True)
    
    
    
    def __bound_dist(self, value, Base,**kwargs):
        return value*np.ones(Base.rays.num)
    def __bound_pct(self, value, Base,**kwargs):
        return np.array([np.percentile(Base.r_ray_n(i), value) for i in range(Base.rays.num)])
    
    def __bound_value(self,value,Base,mode='max',**kwargs):
        if mode =='max':
    
            return np.array([np.max(Base.r_ray_n(i)[Base.parameter_ray_n(i)>value]) for i in range(Base.rays.num)])
        if mode =='min':
            return np.array([np.min(Base.r_ray_n(i)[Base.parameter_ray_n(i)<value]) for i in range(Base.rays.num)])
        raise ValueError(f"{mode} is not a valid value. Only 'max' and 'min' are valid.")
    
    
    def __point_lin(self,num_p):
        return np.linspace(self.inner_r, self.outer_r, num_p).T
    def __point_log(self,num_p):
        return np.geomspace(self.inner_r, self.outer_r,num_p).T 