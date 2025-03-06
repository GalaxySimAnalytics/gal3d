

import numpy as np
import optimagic as om
from scipy import optimize,spatial
import copy

from .util import ellipsoid_fit
from ..fitting.parameter import Parameters
from ..util.func_signature import func_required_key

from ..preprocessing.spherical_field.util import fibonacci_sampling, vector_length3d

__all__ = ['Structure_3D','Structure_3D_fitter']


class Structure_3D:
    
    _shape_3d_fn = {}
    _coordinate_fn = {}    

    def __init__(self,coordinate_class='Coordinate3D',shape_class='Ellipsoid',**kwargs):
        
        if coordinate_class not in Structure_3D._coordinate_fn:
            raise ValueError('not a valid coordinate_class')
        
        if shape_class not in Structure_3D._shape_3d_fn:
            raise ValueError('not a valid shape_class')
        
        
        self._coordinate_name = coordinate_class
        self._shape_name = shape_class
        
        self._coordinate = self._coordinate_fn[coordinate_class]
        self._shape = self._shape_3d_fn[shape_class]
        
        self._coordinate_quick_params = list(func_required_key(self._coordinate.quick_call).keys())[:-1]
        self._shape_quick_params = list(func_required_key(self._shape.quick_call).keys())[:-1]
        self.__coor_pa_num = len(self._coordinate_quick_params)
        
        self.parameters = self._coordinate.get_parameters() + self._shape.get_parameters()
        
        self.d_a = kwargs.get('d_a',0)
        self.constraints = None
        
    def init_parameters(self,*args,**kwargs):
        if args:
            params = dict(zip(self.parameters.keys(),*args))
            return self._coordinate.init_parameters(**params)+self._shape.init_parameters(**params)
        
        return self._coordinate.init_parameters(**kwargs)+self._shape.init_parameters(**kwargs)
    
    def set_parameters(self,*args,**kwargs):
        self.parameters = self.parameters + self.init_parameters(*args,**kwargs)
        
        
    def from_parameters(self,*args,**kwargs):
        ret = Structure_3D(coordinate_class=self._coordinate_name,shape_class=self._shape_name)
        ret.set_parameters(*args,**kwargs)
        return ret

    
    def __repr__(self):

        coor_repr = repr(self._coordinate(**self.parameters))
        shape_repr = repr(self._shape(**self.parameters))
        lin1 = "<Structure_3D|: "+'\n'
        lin2 = "   "+ coor_repr + '\n'
        lin3 = "   "+ shape_repr
        return lin1 +lin2 + lin3
        

    def __eq__(self, other):
        if isinstance(other,Structure_3D):
            if self._coordinate_name == other._coordinate_name:
                if self._shape_name == other._shape_name:
                    return True
        return False
        
    def __call__(self, pos,**kwargs):
        pos = np.asarray(pos)
        if kwargs:
            coord_pa = self._coordinate.init_parameters(**kwargs)
            shape_pa = self._shape.init_parameters(**kwargs)
            return self._shape(**shape_pa)(self._coordinate(**coord_pa)(pos))
        return self._shape(**self.parameters)(self._coordinate(**self.parameters)(pos))

    
    
    def quick_call(self, *args,pos,**kwargs):
        pos = np.asarray(pos)
        if args:
            return self._shape.quick_call(*args[self.__coor_pa_num:],
                                          pos = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos=pos))
        if self.parameters:
            coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
            shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos = pos))
        
        raise KeyError("Need a parameters' tuple or dict, or set parameters")
    
    
    def quick_call_d(self, *args,pos,**kwargs):
        pos = np.asarray(pos)
        if args:
            return self._shape.quick_call_d(*args[self.__coor_pa_num:],
                                          pos = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_d(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos=pos))
        if self.parameters:
            coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
            shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_d(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos = pos))
        
        raise KeyError("Need a parameters' tuple or dict, or set parameters")
    
    def quick_call_dist(self, *args,pos,**kwargs):
        pos = np.asarray(pos)
        if args:
            return self._shape.quick_call_raydistance(*args[self.__coor_pa_num:],
                                          pos = self._coordinate.quick_call(*args[:self.__coor_pa_num],pos=pos))
        if kwargs:
            try:
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            except:
                coord_parameters = self._coordinate.init_parameters(**kwargs)
                shape_parameters = self._shape.init_parameters(**kwargs)
                coord_pa = {i:coord_parameters[i] for i in self._coordinate_quick_params}
                shape_pa = {i:shape_parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_raydistance(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos=pos))
        if self.parameters:
            coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
            shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
            return self._shape.quick_call_raydistance(**shape_pa,pos = self._coordinate.quick_call(**coord_pa,pos = pos))
        
        raise KeyError("Need a parameters' tuple or dict, or set parameters")
    
        
    def check_boundary(self,params_list,mode='periodic'):
        if mode == 'cut':
            for i in params_list:
                lb = self.parameters[i].lb
                ub = self.parameters[i].ub
                self.parameters[i] = np.clip(self.parameters[i],lb,ub)
            return 
        if mode == 'periodic':
            for i in params_list:
                lb = self.parameters[i].lb
                ub = self.parameters[i].ub
                self.parameters[i] = (self.parameters[i]-lb)%(ub-lb)+lb
            return 
        raise ValueError(f"not a valid mode, {mode}, only 'cut', 'periodic',")
    
    
    
    def generate_points(self,random_np: int = 1024):
        cpos,spos = fibonacci_sampling(random_np)
        
        coord_pa = {i:self.parameters[i] for i in self._coordinate_quick_params}
        shape_pa = {i:self.parameters[i] for i in self._shape_quick_params}
        
        points, _ = self._shape(**shape_pa).ray_point(cpos)
        
        return self._coordinate(**coord_pa).inverse(points)
        
        
        
    
    @staticmethod
    def shape_func(fn):
        Structure_3D._shape_3d_fn[fn.__name__] = fn
        return fn
    
    @staticmethod
    def coordinate_func(fn):
        Structure_3D._coordinate_fn[fn.__name__] = fn
        return fn
    

class Structure_3D_fitter(Structure_3D):
    
    _error_fcall_fn = {}
    def __init__(self,coordinate_class='Coordinate3D',shape_class='Ellipsoid',error_func='isodensity_sums_fdev_rscale',**kwargs):
        super().__init__(coordinate_class=coordinate_class,shape_class=shape_class)
        
        
        
        self._error_name = error_func
        self._error = self._error_fcall_fn[error_func]

        self._error_params = list(func_required_key(self._error).keys())
        self._error_params = list(filter(lambda x: ('f_call' not in x) and ('d_call' not in x), self._error_params))  
        
        self._set_error(error_func)

    
    def _set_error(self,error_func):
        if error_func[:5] == 'shell':
            self.error = self._error_call_from_shell
            return 
        if error_func[:4] == 'grid':
            self.error = self._error_call_from_grid
            return
        if error_func[:10] =='isodensity':
            if 'dist' in error_func:
                self.error = self._error_dist_from_isodensity
                return
            if 'ddev' in error_func:
                self.error = self._error_d_from_isodensity
                return
            if 'fdev' in error_func:
                self.error = self._error_call_from_isodensity
                return
        raise ValueError(f"{error_func} is not identified")
    
    
    def from_parameters(self,*args,**kwargs):
        ret = Structure_3D_fitter(coordinate_class=self._coordinate_name,shape_class=self._shape_name,error_func=self._error_name)
        ret.set_parameters(*args,**kwargs)
        
        return ret
    
    def __eq__(self, other):
        if isinstance(other,Structure_3D_fitter):
            if self._coordinate_name == other._coordinate_name:
                if self._shape_name == other._shape_name:
                    if self._error_name == other._error_name:
                        return True
        return False
    
    def _error_call_from_isodensity(self, params,kwargs):
        f_call = self.quick_call(*params,pos=kwargs['pos'])
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(f_call=f_call,**error_pa)
    
    def _error_call_from_shell(self, params, kwargs):
        if not isinstance(params,dict):
            params = dict(zip(self.parameters.keys(),params))
        params1 = params.copy()
        params1['a'] = 0.98*params1['a']
        params2 = params.copy()
        params2['a'] = 1.02*params2['a']
        f_call1 = self.quick_call(pos=kwargs['pos'],**params1)
        f_call2 = self.quick_call(pos=kwargs['pos'],**params2)
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(f_call1=f_call1,f_call2=f_call2,**error_pa)
    
    def _error_call_from_grid(self, params, kwargs):
        if not isinstance(params,dict):
            params = dict(zip(self.parameters.keys(),params))
        params1 = params.copy()
        params1['a'] = 0.9*params1['a']
        params2 = params.copy()
        params2['a'] = 1.1*params2['a']
        f_call1 = self.quick_call(pos=kwargs['pos'],**params1)
        f_call2 = self.quick_call(pos=kwargs['pos'],**params2)
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(f_call1=f_call1,f_call2=f_call2,**error_pa)
        
    def _error_dist_from_isodensity(self, params, kwargs):
        d_call = self.quick_call_dist(*params,pos=kwargs['pos'])
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(d_call=d_call,**error_pa)
    
    def _error_d_from_isodensity(self, params, kwargs):
        d_call = self.quick_call_d(*params,pos=kwargs['pos'])
        error_pa = {i: kwargs[i] for i in self._error_params}
        return self._error(d_call=d_call,**error_pa)
    
    def fit(self, algorithm="scipy_neldermead", **kwargs):
        other_kwargs = dict(**kwargs)
        fun_kwargs = {i: other_kwargs.pop(i) for i in self._error_params}
        if 'pos' not in fun_kwargs:
            fun_kwargs['pos'] = other_kwargs.pop('pos')
        return om.minimize(
            fun = self.error_call_from_isodensity,
            params = dict(**self.parameters),
            algorithm=algorithm,
            fun_kwargs={'kwargs':fun_kwargs},
            constraints = self.constraints,
            bounds = self.parameters.scipy_bounds,**other_kwargs)
        
    def estimate_error(self,test_parameter , random_np = 1024,**kwargs):
        points = kwargs.get('pos',self.generate_points(random_np))
        kwargs={'pos': points}
        if 'r' in self._error_params:
            r = vector_length3d(points)
            kwargs['r'] = r/np.sqrt(np.sum(r**2)/len(r))
        params = {i: test_parameter[i] for i in self.parameters}
        return self.error(list(params.values()),kwargs)

    
    
    def estimate_init_parameter(self,fitdata):
        r25 = np.nanpercentile(fitdata['r'],25)
        r75 = np.nanpercentile(fitdata['r'],75)
        va = np.mean(np.abs(fitdata['pos'][fitdata['r']>r75]),axis=0)    #use all r>r75 points to estimate a vector 
        vc = np.mean(np.abs(fitdata['pos'][fitdata['r']<r25]),axis=0)    #use all r<r25 points to estimate c vector
        
        ini_parameter = self.parameters.new()
        
        Matrixxyz = spatial.transform.Rotation.align_vectors(np.eye(3), np.array([va,[0,0,0],vc]), weights=[1,0,1])
        ini_Mxyz = Matrixxyz[0].as_euler('zyx')
        
        ini_parameter['ang1'] =(ini_Mxyz[0] + np.pi) % (2*np.pi)-np.pi 
        ini_parameter['ang2'] = (ini_Mxyz[1] + np.pi/2) % (np.pi)-np.pi/2
        ini_parameter['ang3'] = (ini_Mxyz[2] + np.pi) % (2*np.pi)-np.pi 
        if ('eps_ab' in ini_parameter) and ('eps_bc' in ini_parameter) :
            least = ellipsoid_fit(fitdata['pos'])
            cba = np.abs(np.sort(least[2]))
            ini_parameter['eps_ab'] = 1-cba[1]/cba[2]
            ini_parameter['eps_bc'] = 1-cba[0]/cba[1]
        return ini_parameter
    
    
    def __repr__(self):

        coor_repr = repr(self._coordinate(**self.parameters))
        shape_repr = repr(self._shape(**self.parameters))
        error = self._error_name
        
        lin1 = "<Structure_3D|:"+'\n'
        lin2 = "   "+ coor_repr + '\n'
        lin3 = "   "+ shape_repr + '\n'
        lin4 = "   "+ "<Error| "+ error + " |>"

        return lin1 + lin2 + lin3 + lin4
    
    @staticmethod
    def minimize_func_fcall(fn):
        Structure_3D_fitter._error_fcall_fn[fn.__name__] = fn
        return fn
    
    
    



    