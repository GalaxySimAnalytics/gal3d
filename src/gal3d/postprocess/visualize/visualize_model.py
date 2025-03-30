
import abc
from functools import wraps
import logging

import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate

from .hist2d import hist_2d
from ...preprocess.spherical_field.spherical_vector import Sphere_vector
from ...util.array_operate import Rotate

logger = logging.getLogger('gal3d.postprocess.visualize.visualize_model')

class AbstractBaseVisualize(abc.ABC):
    
    def __init__(self):
        self._image_cache = {}

    def ImageCache(func):
        @wraps(func)
        def wrapper(self, x_range,y_range,nbins, z_range, rotation,**kwargs):
            recod = (x_range[0],x_range[1],y_range[0],y_range[1],nbins,z_range[0],z_range[1],rotation.tobytes())
            if recod in self._image_cache:
                return self._image_cache[recod]
            else:
                self._image_cache[recod] = func(self, x_range,y_range,nbins, z_range, rotation,**kwargs)
            return self._image_cache[recod]
        return wrapper
    
    @ImageCache
    @abc.abstractmethod
    def image(self, x_range,y_range, nbins: int = 100, z_range=(-20,20), rotation=np.eye(3)):
        pass


    def image_xz(self, x_range,y_range,nbins: int = 100, z_range=(-20,20), ):
        return self.image(x_range,y_range,nbins, z_range, rotation = np.array([[1.,0,0],[0,0,1.],[0,1.,0.]]).T)
    
    def image_yz(self, x_range,y_range,nbins: int = 100, z_range=(-20,20), ):
        return self.image(x_range,y_range,nbins, z_range, rotation = np.array([[0,1.,0.],[0,0,1.],[1.,0,0.]]).T)
        
class VisualModel_SphGrid(AbstractBaseVisualize):
    def __init__(self,inner_model,outer_model, model ,N_ray:int = 6000,num_p :int = 200,**kwargs):
        super().__init__()
        ray_model = Sphere_vector(N_ray)
        total_outer_r = 1.1*(1-outer_model.quick_call_dist(pos =ray_model.pos))
        total_inner_r = 0.9*(1-inner_model.quick_call_dist(pos =ray_model.pos))
        
        points_r = np.geomspace(total_inner_r ,total_outer_r,num_p).T 
        
        inner_r = np.array([np.convolve(points_r[i],[0.5,0.5], mode='same',) for i in range(len(points_r))])
        inner_r[:,0] = 0
        outer_r = np.roll(inner_r,-1,axis=1)
        outer_r[:,-1] = (points_r[:,-1]*3 - points_r[:,-2])/2

        volumn = np.einsum('ij,i->ij',(outer_r**3-inner_r**3),ray_model.area/3)
        volumn = volumn.flatten()

        posall = np.einsum('ij,ik->ijk', points_r, ray_model.pos).reshape(-1,3)
        
        self.pos = posall
        self.volumn = volumn
        
        density = np.zeros(len(self.volumn))
        indices = np.arange(len(self.volumn))

        for i in tqdm(range(len(model))):
            sel = model[i](self.pos[indices])<=1
            density[indices[sel]] =model['parameter'][i]
            
            indices = indices[~sel]

        weight = self.volumn*density
        
        self.density = density
        self.weight = weight
        
    @AbstractBaseVisualize.ImageCache
    def image(self,x_range,y_range,nbins: int = 100, z_range=(-20,20),  rotation=np.eye(3)):
        
        
        new_pos = Rotate(self.pos,rotation)
        sel = (new_pos[:,2] > z_range[0]) & (new_pos[:,2] < z_range[1])
        model_image,xs,ys=hist_2d(new_pos[:,0][sel],new_pos[:,1][sel], weights=self.weight[sel],
                   x_range=x_range,y_range=y_range,density=True,nbins=nbins)
        
        
        return model_image,xs,ys
        
class VisualModel_IntegrateLine(AbstractBaseVisualize):
    def __init__(self, model, model_cric = None,**kwargs):
        
        super().__init__()
        sel = kwargs.get("sel",None)
        
        self.model = model
        self.model_cric = model_cric
        self.model_sel = np.arange(len(self.model['parameter']))
        
        if (self.model_cric is not None) or (sel is not None):
            if sel is None:
                sel = (np.array(self.model.res['fun'])<self.model_cric)
            else:
                if self.model_cric is not None:
                    sel = (sel&(np.array(self.model.res['fun'])<self.model_cric))
            
            self.model_sel = self.model_sel[sel]

            
    @AbstractBaseVisualize.ImageCache
    def image(self,x_range,y_range,nbins: int = 100, z_range=(-20,20),  rotation=np.eye(3)):
        
        
        deproject_array = np.ones((nbins,nbins),dtype=np.float64)
        cen_indices = np.array(deproject_array.shape)/2 - 0.5
        indices = np.transpose(np.nonzero(deproject_array))
        pos = np.zeros((nbins*nbins,2),dtype=np.float64)
        
        xs = np.linspace(x_range[0],x_range[1],nbins+1)
        ys = np.linspace(y_range[0],y_range[1],nbins+1)
        xs = .5 * (xs[:-1] + xs[1:])
        ys = .5 * (ys[:-1] + ys[1:])
        
        pos[:,0] = (indices-cen_indices)[:,0]*(x_range[1]-x_range[0])/nbins
        pos[:,1] = (indices-cen_indices)[:,1]*(y_range[1]-y_range[0])/nbins
        
        pos1 = np.zeros((len(pos),3))
        pos2 = np.zeros((len(pos),3))

        pos1[:,0] = pos[:,0]
        pos1[:,1] = pos[:,1]
        pos1[:,2] = z_range[1]

        pos2[:,0] = pos[:,0]
        pos2[:,1] = pos[:,1]
        pos2[:,2] = z_range[0]
        
        pos1 = Rotate(pos1,rotation)
        pos2 = Rotate(pos2,rotation)
        
        deproject_array = np.zeros((nbins,nbins),dtype=np.float64)
        project_profile = [list([list(),list()]) for _ in range(len(pos1))]
        model_sel = self.model_sel
        
        alll = self.model[int(model_sel[-1])].quick_call_intersect(pos1=pos1,pos2=pos2)
        ind = np.arange(len(pos1))
        ind_in = ind[(alll[:,0]>0.)]
        ind_total = ind_in.copy()

        para = self.model['parameter']
        for i in tqdm(model_sel[::-1]):
            
            sec = self.model[int(i)].quick_call_intersect(pos1=pos1[ind_in],pos2=pos2[ind_in])
            tar = ind_in[(sec[:,0]>0.)]
            sec = sec[(sec[:,0]>0.)]

            for j in range(len(tar)):
                project_profile[tar[j]][0].append(sec[j][0])
                project_profile[tar[j]][0].append(sec[j][1])
                project_profile[tar[j]][1].append(para[i])
                project_profile[tar[j]][1].append(para[i])
            ind_in = tar
        
        for i in tqdm(ind_total):
            x = np.array(project_profile[i][0])
            y = np.array(project_profile[i][1])
            xsort = np.argsort(x)
            inte =  integrate.trapezoid(y[xsort],x[xsort])
            deproject_array[tuple(indices[i])] = inte
        
        
        
        return deproject_array.T,xs,ys
    