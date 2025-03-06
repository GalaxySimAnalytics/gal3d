
from .util import *


__all__ = ['Grid']

class Grid:
    def __init__(self, pos, parameter, maxdepth: int = 14, splitpart: int = 128,**kwargs):
        
        self.set_bound(pos,**kwargs)
        
        sele = np.sum(pos<=self.bound_upper,axis=1) + np.sum(pos>=self.bound_lower,axis=1)
        
        self.base_pos = pos[sele==6]
        self.base_pa = parameter[sele==6]
        
        self.maxdepth = maxdepth
        self.splitpart = splitpart
        
        
        self.make_grid(method = kwargs.get('method','make_grid_by_num'))
        
    def set_bound(self,pos,**kwargs):
        bound_lower = np.min(pos,axis=0,keepdims=True)
        bound_upper = np.max(pos,axis=0,keepdims=True)
        posmin = ['xmin','ymin','zmin']
        posmax = ['xmax','ymax','zmax']
        for i,j in enumerate(posmin):
            if j in kwargs:
                bound_lower[0][i] = float(kwargs[j]) #  min
                
        for i,j in enumerate(posmax):
            if j in kwargs:
                bound_upper[0][i] = float(kwargs[j]) #  max
        self.bound_lower = bound_lower
        self.bound_upper = bound_upper
        
        
    def make_grid(self,method = 'make_grid_by_num'):
        splid_method = {'make_grid_by_num': make_grid_by_num,'make_grid_by_diff': make_grid_by_diff}
        
        
        lower_pos,upper_pos,Depth,Nums,Indice = splid_method[method](self.base_pos,self.maxdepth,self.splitpart,self.bound_lower,self.bound_upper)
        volumn,masses,density = cal_volumn_density(lower_pos,upper_pos,self.base_pa,Indice)
        grid_pos = (lower_pos+upper_pos)/2
        
        self.base_indice = Indice
        self.grid_pos_l = lower_pos[Nums>0]
        self.grid_pos_u = upper_pos[Nums>0]
        self.grid_depth = Depth[Nums>0]
        self.grid_volumn = volumn[Nums>0]
        self.grid_sumpa = masses[Nums>0]
        self.grid_denpa = density[Nums>0]
        self.grid_pos = grid_pos[Nums>0]
        print(f"Remove {len(Nums[Nums==0])} void grids")
        self.grid_nums = Nums[Nums>0]
        
    def provide_griddata(self) ->dict:
        retdict = {}
        retdict['pos'] = self.grid_pos
        retdict['density'] = self.grid_denpa
        retdict['volumn'] = self.grid_volumn
        
        return retdict
        
        
        