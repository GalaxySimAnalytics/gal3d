
import numpy as np
from tqdm import tqdm

from ..preprocessing.spherical_field.spherical_vector import Sphere_vector


class VisualModel:
    def __init__(self,inner_model,outer_model,N_ray:int = 6000,num_p :int = 200):
        ray_model = Sphere_vector(N_ray)
        total_outer_r = 1.1*(1-outer_model.quick_call_dist(pos =ray_model.pos))
        total_inner_r = 0.9*(1-inner_model.quick_call_dist(pos =ray_model.pos))
        
        points_r = np.geomspace(total_inner_r ,total_outer_r,num_p).T 
        
        inner_r = np.array([np.convolve(points_r[i],[0.5,0.5],mode='same',) for i in range(len(points_r))])
        inner_r[:,0] = 0
        outer_r = np.roll(inner_r,-1,axis=1)
        outer_r[:,-1] = (points_r[:,-1]*3 - points_r[:,-2])/2

        volumn = np.einsum('ij,i->ij',(outer_r**3-inner_r**3),ray_model.area/3)
        volumn = volumn.flatten()


        posall = np.einsum('ij,ik->ijk', points_r, ray_model.pos).reshape(-1,3)
        
        self.pos = posall
        self.volumn = volumn
        
    def from_model(self,model):
        density = np.zeros(len(self.volumn))
        indices = np.arange(len(self.volumn))

        for i in tqdm(range(len(model))):
            sel = model[i](self.pos[indices])<=1
            density[indices[sel]] =model['parameter'][i]
            
            indices = indices[~sel]

        weight = self.volumn*density
        
        self.density = density
        self.weight = weight