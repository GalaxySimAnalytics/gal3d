

from collections.abc import Iterable
import time


import numpy as np
from tqdm import tqdm

from .preprocessing.processor import Particles
from .structure.structure_main import Structure_3D,Structure_3D_fitter
from .fitting.optimizer import Optimizer
from .fitting.result import Result





class Galaxy3d(Particles):
    
    def __init__(self,pos,weight,parameter_mode: str = 'Density', num_near: int = 32, kdtree_options=dict(),verbose=True):
        '''
        The main class to fit Galaxy 3D model.
        
        
        
        Parameter:
            pos: array (N,3),
                the position of N particles, in 3D cartis.. (x,y,z) coordinate,
            weight: array (N,)
                the property of N particles, such as mass ...
            parameter_mode: str, 
                {'Density','Mean'}, determine how to calcualte the target parameter.
            num_near: int,
                used in KDtree, determine how much num of nerighbors to estimate the target parameter.
            kdtree_options: dict,
                the options used to build KDtree and KDtree.query.
            verbose: bool,
                verbose control
        
        
        '''
        super().__init__(pos=pos,weight=weight,parameter_mode=parameter_mode,num_near=num_near,verbose=verbose,**kdtree_options)
        
            
    def set_structure_options(self,coordinate_class='Coordinate3D',shape_class='Ellipsoid',error_func='isodensity_sums_fdev_rscale',**kwargs):
        '''
        set the structure options, used for fit.
        
        Parameter:
            coordinate_class: str,
                {'Coordinate3D'},the coordinate system used for model fitting.
            shape_class: str,
                {'Ellipsoid','Ellipsoid_S'}, the structure shape for model fitting.
            error_func: str,
                use available error, determine the minimize function used in model fitting.
        
        Return:
            Galaxy3d
        
        '''
        
        self._structure = Structure_3D_fitter(coordinate_class=coordinate_class,shape_class=shape_class,error_func=error_func,**kwargs)
        
        self.params = self._structure.parameters

        if error_func[:10] =='isodensity':
            self.data_generator = self.__isodensity_data
            return self
        if error_func[:5] == 'shell':
            self.data_generator = self.__shell_data  
            return self
        if error_func[:4] == 'grid':
            self.data_generator = self.__grid_data
            return self
        raise ValueError(f"{error_func} is not identified, something wrong !!")

    
    def set_optimizer_options(self,algorithm,algo_options = dict()):
        '''
        set the optimizer and its options for fitting model.
        
        Parameter:
            algorithm: str,
                the algorithm name, used for optimize, see available algo.
            algo_options: dict,
                set the optimizer options
        
        Return:
            Galaxy3d
        
        '''
        
        
        
        
        
        self._optimizer = Optimizer(algorithm=algorithm,algo_options=algo_options)
        return self
    
    def get_structure(self, r: float | Iterable = np.geomspace(1,10,200), **kwargs):
        '''
        Parameter
            r: float | Iterable,
                the maximun radius of the structure
        
        kwargs:
            var_a, the variation of a,
            
            data_level = 0
        
        Return:
            Result
        
        '''
        get_one_structure = self._get_one_ell_structure
        
        if not isinstance(r,Iterable):
            
            return get_one_structure(r,**kwargs)
        
        else:
            resall = []
            for i in tqdm(r):
                try:
                    if resall:
                        res_value = {i:resall[-1][i][0] for i in resall[-1].keys()}
                        kwargs['init_parameters'] = res_value
                    res = get_one_structure(i,**kwargs)
                    resall.append(res)
                except Exception as e:
                    print(f'Exception: skip fitting at radius {i:.2f} : {e}')
            if len(resall)>0:
                return sum(resall[1:],resall[0])
            else:
                return resall
    
    
    
    def _get_one_structure(self,r ,coordinate_class='Coordinate3D',
                           shape_class='Ellipsoid',
                           error_func='isodensity_sums_fdev_rscale',
                           init_parameters=dict(),upper_bounds=dict(),lower_bounds=dict(),**kwargs):
        structure = Structure_3D_fitter(coordinate_class=coordinate_class,shape_class=shape_class,error_func=error_func)
        
        params = structure.parameters
        data_generator = None
        if error_func[:10] =='isodensity':
            data_generator = self.__isodensity_data
        if error_func[:5] == 'shell':
            data_generator = self.__shell_data  
        if error_func[:4] == 'grid':
            data_generator = self.__grid_data
        
        if data_generator is None:
            raise ValueError(f"{error_func} is not identified, something wrong !!")
        
        fun = structure.error
        fitdata = data_generator(r,**kwargs)
        
        parameters_set = params.new()
        
        if 'info' in fitdata:
            parameters_set.add_info(**fitdata['info'])
            del fitdata['info']
            
        for i in (upper_bounds.keys() & parameters_set.keys()):
            parameters_set[i].ub = upper_bounds[i]
        
        for i in (lower_bounds.keys() & parameters_set.keys()):
            parameters_set[i].lb = lower_bounds[i]
        
        if init_parameters:
            new_value = {i: init_parameters[i] for i in (init_parameters.keys()& parameters_set.keys())}
            parameters_set.set_value(**new_value)
        else:
            parameters_set = parameters_set.set_value(**structure.estimate_init_parameter(fitdata))


        bounds = parameters_set.scipy_bounds  
        
        x0_dict =  parameters_set.truncate_dict(n=4)
        each = self._optimizer.fitting(fun,list(x0_dict.values()),bounds,args=fitdata)
        parameters_set =parameters_set.set_value(each.x)

        res = Result(self._structure,each,parameters_set)
        
        return res
    
    
    def _get_one_ell_structure(self,a,**kwargs):
        '''
        fit Ellipsoid and Ellipsoid_S shape
        '''
        
        var_a = kwargs.get('var_a',0)
        var_a = min(var_a,0.99)
        var_a = max(var_a,0)
        
        init_parameters = kwargs.get('init_parameters',dict())
        upper_bounds=kwargs.get('upper_bounds',dict())
        lower_bounds=kwargs.get('lower_bounds',dict())
        
        fitdata = self.data_generator(a,**kwargs)
        
        if self._structure._shape_name == 'Ellipsoid':
            
            fun = self._structure.error
            parameters_set = self.params.new()
            
            if 'info' in fitdata:
                parameters_set.add_info(**fitdata['info'])
                del fitdata['info']
                
            parameters_set.set_lb(a=(a*(1-var_a)))
            parameters_set.set_ub(a=(a*(1+var_a)))

            bounds = parameters_set.scipy_bounds  
                
            if init_parameters:
                parameters_set = parameters_set.set_value(**init_parameters)
            else:
                parameters_set = parameters_set.set_value(**self._structure.estimate_init_parameter(fitdata))
            
            parameters_set.set_value(a=a)
            
            x0_dict =  parameters_set.truncate_dict(n=4)

            each = self._optimizer.fitting(fun,list(x0_dict.values()),bounds,args=fitdata)
            parameters_set =parameters_set.set_value(each.x)

            res = Result(self._structure,each,parameters_set)
            
            # we need first fitting Ellipsoid, then fit sa,sb,sc and a for Elllipsoid_S
        if self._structure._shape_name == 'Ellipsoid_S':
            
            parameters_set = self.params.new()
            parameters_set.set_lb(a=(a*(1-var_a)))
            parameters_set.set_ub(a=(a*(1+var_a)))
            
            if init_parameters:
                parameters_set = parameters_set.set_value(**init_parameters)
            else:
                parameters_set = parameters_set.set_value(**self._structure.estimate_init_parameter(fitdata))
            
            parameters_set.set_value(a=a)
            
            
            
            parent_res = self._get_one_structure(a,coordinate_class=self._structure._coordinate_name,
                                                   shape_class='Ellipsoid',
                                                   error_func=self._structure._error_name,
                                                   init_parameters=dict(**parameters_set),
                                                   upper_bounds={i: parameters_set[i].ub for i in parameters_set.keys()},
                                                   lower_bounds={i: parameters_set[i].lb for i in parameters_set.keys()})
            
            res_value = {i:parent_res[i][0] for i in parent_res.keys()}
            
            parameters_set.set_value(**res_value)
            
            
            x0_dict =  parameters_set.truncate_dict(n=4)
            x0_dict.update(res_value)
            
            del res_value['a']
            parameters_set.set_lb(**res_value)
            parameters_set.set_ub(**res_value)
            
            bounds = parameters_set.scipy_bounds 
            
            parameters_set.add_info(parent_fun = parent_res.res['fun'][0])
            
            fun = self._structure.error

            if 'info' in fitdata:
                parameters_set.add_info(**fitdata['info'])
                del fitdata['info']
            
            each = self._optimizer.fitting(fun,list(x0_dict.values()),bounds,args=fitdata)
            parameters_set =parameters_set.set_value(each.x)

            res = Result(self._structure,each,parameters_set)
            
        
        return res
    
        
    def __isodensity_data(self,a,**kwargs):
        '''
        Providing isodensity data when fitting isodensity.
        '''
        
        
        data_level = kwargs.get('data_level',0) # 0: [(0,0)], 1: (0,(1,0,-1)), 2: ((1,0,-1),0), 3: (1,0,-1),(1,0,-1)
        Level = {0:[(0,0)],1:[(0,1),(0,-1)],2:[(1,0),(-1,0)],
                 3:[(0,1),(0,-1),(1,0),(1,-1)],
                 4:[(0,1),(0,-1),(1,0),(1,-1),(1,1),(1,-1),(-1,1),(-1,-1)]}
        
                
        fitdata = self.field.generate(a,level= (0,0))
        fitdata['pos'] = fitdata['pos'][~np.isnan(fitdata['r'])]
        fitdata['r'] = fitdata['r'][~np.isnan(fitdata['r'])]
           # fitdata['r'] = fitdata['r']/np.sqrt(np.sum(fitdata['r']**2)/len(fitdata['r']))   #  normalization as this used for calculate error
           # fitdata['info'] = {'parameter': fitdata['parameter']}
        
        if data_level !=0:
            le = Level[data_level]
            for i in le:
                other = self.field.generate(a,level= i)
                other['pos'] = other['pos'][~np.isnan(other['r'])]
                other['r'] = other['r'][~np.isnan(other['r'])]
                
                fitdata['pos'] = np.concatenate((fitdata['pos'],other['pos']))
                fitdata['r'] = np.concatenate((fitdata['r'],other['r']))
    
        fitdata['r'] = fitdata['r']/np.sqrt(np.sum(fitdata['r']**2)/len(fitdata['r']))   #  normalization as this used for calculate error
        fitdata['info'] = {'parameter': fitdata['parameter']}
        return fitdata
    
    def __shell_data(self,a):
        '''
        Providing shell data when fitting by original particles.
        '''
        
        
        fitdata={'pos': self.pos, 'pa': self.parameter}
        return fitdata
    
    
    def __grid_data(self,a,logscale = True):
        '''
        Providing grid data when fitting by grid.
        '''

        fitdata={'pos': self.grid.grid_pos,'volumn': self.grid.grid_volumn} # using log10 pa ?? #TODO
        fitdata['parameter'] = np.log10(self.grid.grid_denpa) if logscale else self.grid.grid_denpa

        return fitdata
    
    @property
    def available_shape(self):
        return list(Structure_3D._shape_3d_fn.keys())
    
    @property
    def available_coordinate(self):
        return list(Structure_3D._coordinate_fn.keys())
    
    @property
    def available_error(self):
        return list(Structure_3D_fitter._error_fcall_fn.keys())

    @property
    def available_algorithm(self):
        return Optimizer._algos.copy()

    def __repr__(self):
        lin1 = "< Galaxy3d | " +  "N_particles = "+str(len(self.pos)) + " | "+self.pa_mode+ " >"

        if hasattr(self,"_structure"):
            lin2 = "| Coordinate: " + self._structure._coordinate_name+ " |"
            lin3 = "| Shape: " + self._structure._shape_name+ " |"
            lin4 = "| Error: "+self._structure._error_name + " |"
        else:
            lin2 = "| Coordinate: not set |"
            lin3 = "| Shape: not set |"
            lin4 = "| Error: not set |"
            

        if hasattr(self,"_optimizer"):
            lin5 = "| Optimizer: "+ self._optimizer.algo_name+ " |"
        else:
            lin5 = "| Optimizer: not set |"
            
        lins = [lin1,lin2,lin3,lin4,lin5]
        lenmax = len(max(lins,key=len))+4
        result = ''.join([i.center(lenmax,"*")+ "\n" for i in lins])
        
        return result[:-2]