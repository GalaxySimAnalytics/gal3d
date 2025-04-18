

import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate


from ..model_projector import ModelProjectorBase
from ...field.spherical_field.spherical_vector import SphVector
from ...util.array_operate import Rotate
from ..hist2d import hist_2d


class ProjectorLineIntegration(ModelProjectorBase):
    def __init__(self, model, model_cric = None,cache_len = 100,**kwargs):
        
        super().__init__(cache_len=cache_len)
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

            
    def _image(self,x_range,y_range,nbins: int = 100, z_range=(-20,20),  rotation=np.eye(3)):
        
        
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