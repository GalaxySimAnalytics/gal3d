
import logging


import numpy as np
from scipy.spatial import SphericalVoronoi

from .util import *


logger = logging.getLogger('gal3d.preprocessing.spherical_field.spherical_vector')




__all__ = ["Sphere_vector"]

class Sphere_vector():
    '''The coordinates of N points uniformly distributed on the unit sphere'''
    
    def __init__(self,N_sample=512, method = 'fibonacci'):
        '''
        Parameters:
            N_sample: int, default 512,
                the numble of points 
            method: str, optional, 'muller' or 'fibonacci', default 'fibonacci'
                the method to generate points on the sphere
                
        Attributes:
            num: int, 
                equal to N_sample,the numble of points 
            pos: ndarray, shape (n,3)
                coordinates of each point (x,y,z)
            sph: ndarray, shape (n,3)
                spherical coordinates of each point (r,phi,theta)
            voronoi: SphericalVoronoi
                Voronoi diagrams on the surface of a sphere. 
                see scipy.spatial.SphericalVoronoi
            area: SphericalVoronoi.calculate_areas
                the areas of the Voronoi regions
            uniformity: float
                the ratio of variance to mean of areas
        '''
        
        METHOD = {'fibonacci': self.fibonacci_sampling,
                'muller': self.muller_sampling,}
            #TODO other method
        self.num = N_sample
        self.pos,self.sph = METHOD[method](self.num)
        self.voronoi = SphericalVoronoi(self.pos)
        self.voronoi.sort_vertices_of_regions()
        self.area = SphericalVoronoi.calculate_areas(self.voronoi)
        self.uniformity = np.std(self.area) / np.mean(self.area)
        logger.info(f"{self.num} points on the sphere by {method} method have the uniformity of {self.uniformity:.2e}")
        
        
    def assign_points(self,pos):
        pos_uni = unit_vector3d(pos)
        
        batchsize = 200000      # prevent memory overflow
        n_split = len(pos_uni) // batchsize
        n_split = max(n_split,1)
        
        logger.info(f"Splitting pos into {n_split} parts, prevent memory overflow")
        
        target_pos_split = np.array_split(pos_uni, n_split, axis=0)
        
        return np.concatenate([np.argmax(Matmul(i,self.pos.T), axis=1) for i in target_pos_split], axis=0)
        
    @staticmethod
    def fibonacci_sampling(Num_sampling: int = 256):
        '''
        Parameters: 
            Num_sampling: int, default 256,
                the numble of points 
        
        Return: [x,y,z],[r,phi,theta]
        '''
        logger.info(f"Sampling {Num_sampling} random points on the sphere by fibonacci method")
        return fibonacci_sampling(Num_sampling)
    
    @staticmethod
    def muller_sampling(Num_sampling=256):
        '''
        Parameters: 
            Num_sampling: int, default 256,
                the numble of points 
        
        Return: [x,y,z],[r,phi,theta]
        '''
        logger.info(f"Sampling {Num_sampling} random points on the sphere by muller method")
        
        u, v, w = np.random.randn(3, Num_sampling)
       # unit_vector3d(np.array([u,v,w]).T)
       # norm = vector_length3d(np.array([u,v,w]))
       # x, y, z = (u,v,w) / norm
        sampling_pos = unit_vector3d(np.array([u,v,w]).T)
        sampling_sphere_coor = trans_to_Spherical_coordinates(sampling_pos)
        
        return sampling_pos, sampling_sphere_coor