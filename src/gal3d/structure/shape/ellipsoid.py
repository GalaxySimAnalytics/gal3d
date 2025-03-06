
import copy

import numpy as np

if __name__=='__main__':
    from util_fcall import *
    from util_distance import *
else:
    from .util_fcall import *
    from .util_distance import *
from ..structure_main import Structure_3D, Parameters


__all__ =["Ellipsoid"]


@Structure_3D.shape_func
class Ellipsoid:
    
    
    PA = ('a','eps_ab','eps_bc')        ### not use set !!!
    LB = {'a':0.1,'eps_ab':0.01,'eps_bc':0.01}
    UB = {'a':np.inf,'eps_ab':0.99,'eps_bc':0.99}
    def __init__(self,*args,**kwargs):
        '''
        args: default: a , b, c
        a>b>c
        eps_ab = 1 - b/a : 0~1
        eps_bc = 1 = c/b : 0~1
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
        
        parameters = Parameters(**{i:param[i] for i in Ellipsoid.PA})
        parameters._derived.update(param._derived)
        parameters.set_lb(**Ellipsoid.LB)
        parameters.set_ub(**Ellipsoid.UB)
        return parameters
    
    @staticmethod
    def get_parameters():
        return Ellipsoid.init_parameters(a=3.,eps_ab=0.2,eps_bc=0.5)
    
        
    def __call__(self, pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_ellipsoid(self['a'],self['b'],self['c'], pos)
    
    def __getitem__(self, item):
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')
    
    def __repr__(self):

        param_repr = repr(self.parameters)

        return "<Ellipsoid|: "+ param_repr[10:] + "|>"
    
    def closed_point(self,pos,maxIterations:int = 100) -> tuple:
        '''
        Distance between points and the ellipsoid
        '''
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        x0,x1,x2,d = DistancePointsEllipsoid(self['a'],self['b'],self['c'],pos[:,0],pos[:,1],pos[:,2],maxIterations)
        
        return np.array([x0,x1,x2]).T,d
    
    def ray_point(self,pos):
        '''
        distance between points and raypoint on the ellipsoid
        '''
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return DistanceRayPointsEllipsoid(self['a'],self['b'],self['c'], pos)
    
    def jacobian(self,pos) -> tuple:
        return f_ellipsoid_jacobian(self['a'],self['b'],self['c'], pos)
    
    @staticmethod
    def quick_call(a,eps_ab,eps_bc,pos):
        b = a*(1.-eps_ab)
        c = b*(1.-eps_bc)
        return f_ellipsoid(float(a),b,c,pos)
    
    @staticmethod
    def quick_call_d(a,eps_ab,eps_bc,pos):
        b = a*(1.-eps_ab)
        c = b*(1.-eps_bc)
        return d_ellipsoid(float(a),b,c,pos)
    
    @staticmethod
    def quick_call_raydistance(a,eps_ab,eps_bc,pos):
        b = a*(1.-eps_ab)
        c = b*(1.-eps_bc)
        return DistanceRayPointsEllipsoid(float(a),b,c,pos)[1]
    
    @staticmethod
    def f_ellipsoid(a,b,c,pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_ellipsoid(float(a),float(b),float(c), pos)
    
    
    @staticmethod
    def DistancePointsEllipsoid(a,b,c,pos, maxIterations: int = 100):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        x0,x1,x2,d = DistancePointsEllipsoid(float(a),float(b),float(c),pos[:,0],pos[:,1],pos[:,2],maxIterations)
        
        return np.array([x0,x1,x2]).T,d
    
    @staticmethod
    def DistanceRayPointsEllipsoid(a,b,c,pos):
        return DistanceRayPointsEllipsoid(float(a),float(b),float(c),np.float64(pos))
    
    @staticmethod
    def f_ellipsoid_jacobian(a,b,c,pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_ellipsoid_jacobian(float(a),float(b),float(c), pos)