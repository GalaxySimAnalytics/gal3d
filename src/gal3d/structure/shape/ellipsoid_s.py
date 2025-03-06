import copy

import numpy as np

if __name__=='__main__':
    from util_fcall import *
    from util_distance import *
else:
    from .util_fcall import *
    from .util_distance import *
    
    
from ..structure_main import Structure_3D, Parameters


__all__ =["Ellipsoid_S"]



@Structure_3D.shape_func
class Ellipsoid_S:
    
    PA = ('a','eps_ab','eps_bc','sa','sb','sc')   ## not use set !!!##
    LB = {'a':0.1,'eps_ab':0.01,'eps_bc':0.01,'sa':0.2,'sb':0.2,'sc':0.2}
    UB = {'a':np.inf,'eps_ab':0.99,'eps_bc':0.99,'sa':2,'sb':2,'sc':2}
    def __init__(self,**kwargs):
        '''
        a,b,c
        sa,sb,sc
        '''
        self.parameters = self.init_parameters(**kwargs)
        
    @staticmethod
    def init_parameters(**kwargs):
        param = Parameters(**kwargs)
        param._derived['eps_ab'] = lambda d: 1.-d['b']/d['a']
        param._derived['eps_bc'] = lambda d: 1.-d['c']/d['b']
        param._derived['eps_ac'] = lambda d: 1.-d['c']/d['a']
        param._derived['b'] = lambda d: d['a']*(1-d['eps_ab']) if 'eps_ab' in d else d['c']/(1-d['eps_bc'])
        param._derived['c'] = lambda d: d['b']*(1-d['eps_bc']) if 'eps_bc' in d else d['a']*(1-d['eps_ac'])
        param._derived['a'] = lambda d: d['b']/(1-d['eps_ab']) if 'eps_ab' in d else d['c']/(1-d['eps_ac'])
        
        parameters = Parameters(**{i:param[i] for i in Ellipsoid_S.PA})
        parameters._derived.update(param._derived)
        parameters.set_lb(**Ellipsoid_S.LB)
        parameters.set_ub(**Ellipsoid_S.UB)
        return parameters
    
    @staticmethod
    def get_parameters():
        return Ellipsoid_S.init_parameters(a=3.,eps_ab=0.2,eps_bc=0.5,sa=1.,sb=1.,sc=1.)
    
    
    def __repr__(self):
        param_repr = repr(self.parameters)

        return "<Ellipsoid_S|: "+ param_repr[10:] + "|>"

            
    def __getitem__(self, item):
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')
        
    def __call__(self, pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid(self['a'],self['b'],self['c'],self['sa'],self['sb'],self['sc'], pos)
    
    def jacobian(self,pos) -> tuple:
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid_jacobian(self['a'],self['b'],self['c'],self['sa'],self['sb'],self['sc'], pos)
    
    @staticmethod
    def quick_call(a,eps_ab,eps_bc,sa,sb,sc,pos):
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return f_shaped_ellipsoid(a,b,c,sa,sb,sc,pos)
    
    @staticmethod
    def quick_call_d(a,eps_ab,eps_bc,sa,sb,sc,pos):
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return d_shaped_ellipsoid(a,b,c,sa,sb,sc,pos,100)
    
    @staticmethod
    def quick_call_raydistance(a,eps_ab,eps_bc,sa,sb,sc,pos):
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return DistanceRayPointsEllipsoid_S(float(a),b,c,sa,sb,sc,pos,100)[1]
    
    @staticmethod
    def f_shaped_ellipsoid(a,b,c,sa,sb,sc,pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid(float(a),float(b),float(c),float(sa),float(sb),float(sc),pos)
    
    def ray_point(self,pos):
        '''
        distance between points and raypoint on the shaped ellipsoid
        '''
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return DistanceRayPointsEllipsoid_S(self['a'],self['b'],self['c'],self['sa'],self['sb'],self['sc'], pos, 100)
    
    @staticmethod
    def DistanceRayPointsEllipsoid_S(a,b,c,sa,sb,sc,pos):
        return DistanceRayPointsEllipsoid_S(float(a),float(b),float(c),float(sa),float(sb),float(sc),pos,100)
    
    @staticmethod
    def f_shaped_ellipsoid_jacobian(a,b,c,sa,sb,sc,pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid_jacobian(float(a),float(b),float(c),float(sa),float(sb),float(sc),pos)