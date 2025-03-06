

from functools import cached_property
import logging
import os

import numpy as np
from scipy.spatial import KDTree

from ...util.array_operate import vector_length3d,Matmul,unit_vector3d
from ...util.func_signature import MySignature, func_optional_key, update_dict_value



logger = logging.getLogger('gal3d.preprocessing.local_estimate')


__all__ =['Local_kde']

class Local_est():
    '''Estimate the parameter value at any position by kd-tree'''
    def __init__(self, pos, weight, parameter_mode: str = 'Density', num_near: int = 32,**kwargs):
        '''
        Parameters:
            pos: ndarray, shape(n,3)
                the coordinates (x,y,z) of the n data points
            weight: array, shape(n,)
                the property of the n points
            parameter_mode: str, optional
                if 'Density', default, estimation of the density, 
                for example: input pa is mass, then return density 
                if 'Mean':, return the average value
            num_near: int, default 64,
                nearest points of this number are used to estimate the target parameter
        Attributes:
            pos: equal to input pos
            weight: equal to input weight
            r: the radius of input points
            tree: scipy.spatial.KDTree
            pa_mode: equal to parameter_mode
        Methods:
            get_pa(target_pos[x,y,z]) 
                estimate the parameter for the target position
            get_gradient(target_pos[x,y,z]) 
                estimate the get_gradient for the target position
        '''
        self.__generate_kd_options(num_near,**kwargs)
        pos = self._shape_check(pos)
        r = vector_length3d(pos)
        ind = np.argsort(r)
        
        self.pos = pos[ind]
        self.weight = weight[ind]
        self.r = r[ind]
        
        self.tree = KDTree(self.pos,**self._tree_build_options)
        self.pa_mode = parameter_mode
        

        
    @cached_property
    def parameter(self):
        return self.get_parameter(self.pos)
    
    @cached_property
    def gradient(self):
        return self.get_gradient(self.pos)
    
    def _shape_check(self,pos):
        if len(np.shape(pos))!=2:
            logger.info(f"pos is 1d array with shape={np.shape(pos)}, so we reshape it")
            pos = np.array(pos).reshape(-1,3)
        if np.shape(pos)[1]==3:
            return pos
        if np.shape(pos)[0]==3:
            logger.info(f"pos have the shape= {np.shape(pos)}, so we transpose it")
            return np.array(pos).T
        logger.info(f"pos have the shape={np.shape(pos)}, target shape: (n,3), so we reshape this")
        return np.array(pos).reshape(-1,3)
        
    
    def get_parameter(self, target_pos, **kwargs):
        '''
        Parameters:
            target_pos: ndarray, shape(n,3)
            the target positions (x,y,z) where the parameter values are to be estimated
        Return:
            results: array shape (n,)
                the estimated parameter values
        '''
        target_pos = self._shape_check(target_pos)
        query_options = update_dict_value(self._tree_query_options,kwargs)
        
        n_d, n_index = self.tree.query(target_pos, **query_options)
        
        return self._cal_pa(n_d,n_index)
        
    def get_gradient(self,target_pos, **kwargs):
        target_pos = self._shape_check(target_pos)
        query_options = update_dict_value(self._tree_query_options,kwargs)
        
        n_d, n_index = self.tree.query(target_pos,**query_options)
                
        target_pa = self._cal_pa(n_d,n_index)
        
        parameter = self.parameter[n_index]
        
        up_grad_ind = np.argmax(parameter,axis=1, keepdims=True)
        lo_grad_ind = np.argmin(parameter,axis=1, keepdims=True)
        
        up_grad_index = np.take_along_axis(n_index,up_grad_ind,axis=1)[:,0]
        lo_grad_index = np.take_along_axis(n_index,lo_grad_ind,axis=1)[:,0]
        
        up_dist = np.clip(np.take_along_axis(n_d,up_grad_ind,axis=1)[:,0],a_min=1e-8,a_max=None)
        lo_dist = np.clip(np.take_along_axis(n_d,lo_grad_ind,axis=1)[:,0],a_min=1e-8,a_max=None)
        
        up_pa = np.take_along_axis(parameter,up_grad_ind,axis=1)[:,0] - target_pa
        lo_pa = np.take_along_axis(parameter,lo_grad_ind,axis=1)[:,0] - target_pa
        
        up_vect = self.pos[up_grad_index] - target_pos
        lo_vect = self.pos[lo_grad_index] - target_pos
        
        up_gradient = up_pa/up_dist
        lo_gradient = lo_pa/lo_dist
        
        return ((up_gradient,unit_vector3d(up_vect)),(lo_gradient,unit_vector3d(lo_vect)))
    
    def _cal_pa(self,n_d,n_index):
        n_d_max = n_d[:,-1]
        if self.pa_mode == 'Density':
            n_pain = np.sum(self.weight[n_index], axis=1)
            fit_pa = n_pain/(4/3*np.pi*np.power(n_d_max, 3))
        elif self.pa_mode == 'Mean':
            fit_pa = np.mean(self.weight[n_index], axis=1)
        else:
            logger.error("KeyError: No such method, use parameter_mode = Density")
            n_pain = np.sum(self.weight[n_index], axis=1)
            fit_pa = n_pain/(4/3*np.pi*np.power(n_d_max, 3))
        return fit_pa
    
    
    def __generate_kd_options(self,num_near,**kwargs):
        
        self._tree_build_options = func_optional_key(KDTree)
        self._tree_query_options = func_optional_key(KDTree.query)
        
        self._tree_build_options['leafsize'] = num_near
        self._tree_query_options['workers'] = os.cpu_count()
        self._tree_query_options['k'] = num_near
        
        logger.info(f"cpu nums: {self._tree_query_options['workers']}")
        
        
        self._tree_build_options = update_dict_value(self._tree_build_options,kwargs)
        
        self._tree_query_options = update_dict_value(self._tree_query_options,kwargs)