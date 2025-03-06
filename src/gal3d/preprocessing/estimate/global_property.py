

import logging
from functools import cached_property

import numpy as np

from .util import shrink_sphere_center as ssc
from .util import center_of_mass,centroid,moment_of_inertia,abc_vect


logger = logging.getLogger('gal3d.preprocessing.estimate.global_property')





class Global_prop:
    
    __slots__ = ["pos","weight"]
    def __init__(self,pos,weight):
        self.pos = pos
        self.weight = weight


    @cached_property
    def ssc_center(self):
        return self.shrink_sphere_center(self.pos,self.weight)
    
    @cached_property
    def weight_center(self):
        return center_of_mass(self.pos,self.weight)
    
    @cached_property
    def shape_center(self):
        return centroid(self.pos)
    
    @cached_property
    def moi(self):
        return moment_of_inertia(self.pos,self.weight)
    
    @cached_property
    def abc(self):
        return abc_vect(self.pos,self.weight)
    
    @staticmethod
    def shrink_sphere_center(pos, weight, shrink_factor=0.7, begin_r = None, min_points=100, itermax = 100):
        if begin_r is None:
            begin_r = (np.max(pos[:,0]) - np.min(pos[:,0]))/2

        logger.info(f"Using a begin_r= {begin_r:.2f}")
        
        
        cen, final_r, v_r, iternum = ssc(np.array(pos),np.array(weight),min_points,0,shrink_factor,begin_r,itermax)
        
        logger.info(f"Iteration num= {iternum}")
        
        if iternum > itermax:
            logger.error(f"shrink_sphere_center failed to converge after {iternum} iterations")
        
        logger.info(f"After iteration, final_r= {final_r:.2f}")
        
        return cen
    
    @staticmethod
    def moment_of_inertia(pos,weight):
        return moment_of_inertia(pos,weight)
    
    @staticmethod
    def abc_vector(pos,weight):
        return abc_vect(pos,weight)