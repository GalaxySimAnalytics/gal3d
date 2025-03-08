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
    """
    A class representing an ellipsoid with shape indices, used for fitting 3D galaxy morphologies.

    Attributes
    ----------
    PA : tuple
        A tuple of parameter names: ('a', 'eps_ab', 'eps_bc', 'sa', 'sb', 'sc').
    LB : dict
        A dictionary of lower bounds for the parameters: {'a': 0.1, 'eps_ab': 0.01, 'eps_bc': 0.01, 'sa': 0.2, 'sb': 0.2, 'sc': 0.2}.
    UB : dict
        A dictionary of upper bounds for the parameters: {'a': np.inf, 'eps_ab': 0.99, 'eps_bc': 0.99, 'sa': 2, 'sb': 2, 'sc': 2}.

    Methods
    -------
    __init__(**kwargs)
        Initializes the Ellipsoid_S instance with given parameters.
    init_parameters(**kwargs)
        Initializes and returns the parameters with derived values.
    get_parameters()
        Returns a default set of parameters for the ellipsoid.
    __repr__()
        Returns a string representation of the ellipsoid instance.
    __getitem__(item)
        Returns the value of the specified parameter.
    __call__(pos)
        Evaluates the ellipsoid function at the given positions.
    jacobian(pos)
        Computes the Jacobian of the ellipsoid function at the given positions.
    quick_call(a, eps_ab, eps_bc, sa, sb, sc, pos)
        Quickly evaluates the ellipsoid function with given parameters and positions.
    quick_call_d(a, eps_ab, eps_bc, sa, sb, sc, pos)
        Quickly evaluates the derivative of the ellipsoid function with given parameters and positions.
    quick_call_raydistance(a, eps_ab, eps_bc, sa, sb, sc, pos)
        Quickly computes the distance between points and ray points on the ellipsoid.
    f_shaped_ellipsoid(a, b, c, sa, sb, sc, pos)
        Evaluates the ellipsoid function with given parameters and positions.
    ray_point(pos)
        Computes the distance between points and ray points on the ellipsoid.
    DistanceRayPointsEllipsoid_S(a, b, c, sa, sb, sc, pos)
        Computes the distance between points and ray points on the ellipsoid.
    f_shaped_ellipsoid_jacobian(a, b, c, sa, sb, sc, pos)
        Computes the Jacobian of the ellipsoid function with given parameters and positions.
    """
    
    PA = ('a','eps_ab','eps_bc','sa','sb','sc')   ## not use set !!!##
    LB = {'a':0.1,'eps_ab':0.01,'eps_bc':0.01,'sa':0.2,'sb':0.2,'sc':0.2}
    UB = {'a':np.inf,'eps_ab':0.99,'eps_bc':0.99,'sa':2,'sb':2,'sc':2}
    def __init__(self,**kwargs):
        """
        Initializes the Ellipsoid_S instance with given parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameters to initialize the ellipsoid.
        """
        self.parameters = self.init_parameters(**kwargs)
        
    @staticmethod
    def init_parameters(**kwargs):
        """
        Initializes and returns the parameters with derived values.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameters to initialize the ellipsoid.

        Returns
        -------
        Parameters
            An instance of Parameters with initialized and derived values.
        """
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
        """
        Returns a default set of parameters for the ellipsoid.

        Returns
        -------
        Parameters
            An instance of Parameters with default values.
        """
        return Ellipsoid_S.init_parameters(a=3.,eps_ab=0.2,eps_bc=0.5,sa=1.,sb=1.,sc=1.)
    
    
    def __repr__(self):
        """
        Returns a string representation of the ellipsoid instance.

        Returns
        -------
        str
            A string representation of the ellipsoid instance.
        """
        param_repr = repr(self.parameters)

        return "<Ellipsoid_S|: "+ param_repr[10:] + "|>"

            
    def __getitem__(self, item):
        """
        Returns the value of the specified parameter.

        Parameters
        ----------
        item : str
            The name of the parameter.

        Returns
        -------
        float
            The value of the specified parameter.

        Raises
        ------
        KeyError
            If the specified parameter is not valid.
        """
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')
        
    def __call__(self, pos):
        """
        Evaluates the ellipsoid function at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        array_like
            The evaluated values of the ellipsoid function at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid(self['a'],self['b'],self['c'],self['sa'],self['sb'],self['sc'], pos)
    
    def jacobian(self,pos) -> tuple:
        """
        Computes the Jacobian of the ellipsoid function at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid_jacobian(self['a'],self['b'],self['c'],self['sa'],self['sb'],self['sc'], pos)
    
    @staticmethod
    def quick_call(a,eps_ab,eps_bc,sa,sb,sc,pos):
        """
        Quickly evaluates the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis length.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        sa : float
            The scale factor for the a axis.
        sb : float
            The scale factor for the b axis.
        sc : float
            The scale factor for the c axis.
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        array_like
            The evaluated values of the ellipsoid function at the given positions.
        """
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return f_shaped_ellipsoid(a,b,c,sa,sb,sc,pos)
    
    @staticmethod
    def quick_call_d(a,eps_ab,eps_bc,sa,sb,sc,pos):
        """
        Quickly evaluates the distance of the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis length.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        sa : float
            The scale factor for the a axis.
        sb : float
            The scale factor for the b axis.
        sc : float
            The scale factor for the c axis.
        pos : array_like
            An array of positions where the derivative is evaluated.

        Returns
        -------
        array_like
            The evaluated distance values at the given positions.
        """
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return d_shaped_ellipsoid(a,b,c,sa,sb,sc,pos,300)
    
    @staticmethod
    def quick_call_raydistance(a,eps_ab,eps_bc,sa,sb,sc,pos):
        """
        Quickly computes the distance between points and ray points on the ellipsoid.

        Parameters
        ----------
        a : float
            The semi-major axis length.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        sa : float
            The scale factor for the a axis.
        sb : float
            The scale factor for the b axis.
        sc : float
            The scale factor for the c axis.
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        array_like
            The computed distances between points and ray points on the ellipsoid.
        """
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return DistanceRayPointsEllipsoid_S(float(a),b,c,sa,sb,sc,pos,300)[1]
    
    @staticmethod
    def f_shaped_ellipsoid(a,b,c,sa,sb,sc,pos):
        """
        Evaluates the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis length.
        b : float
            The semi-intermediate axis length.
        c : float
            The semi-minor axis length.
        sa : float
            The scale factor for the a axis.
        sb : float
            The scale factor for the b axis.
        sc : float
            The scale factor for the c axis.
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        array_like
            The evaluated values of the ellipsoid function at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid(float(a),float(b),float(c),float(sa),float(sb),float(sc),pos)
    
    def ray_point(self,pos):
        """
        Computes the distance between points and ray points on the ellipsoid.

        Parameters
        ----------
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        array_like
            The computed distances between points and ray points on the ellipsoid.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return DistanceRayPointsEllipsoid_S(self['a'],self['b'],self['c'],self['sa'],self['sb'],self['sc'], pos, 300)
    
    @staticmethod
    def DistanceRayPointsEllipsoid_S(a,b,c,sa,sb,sc,pos):
        """
        Computes the distance between points and ray points on the ellipsoid.

        Parameters
        ----------
        a : float
            The semi-major axis length.
        b : float
            The semi-intermediate axis length.
        c : float
            The semi-minor axis length.
        sa : float
            The scale factor for the a axis.
        sb : float
            The scale factor for the b axis.
        sc : float
            The scale factor for the c axis.
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        array_like
            The computed distances between points and ray points on the ellipsoid.
        """
        return DistanceRayPointsEllipsoid_S(float(a),float(b),float(c),float(sa),float(sb),float(sc),pos,300)
    
    @staticmethod
    def f_shaped_ellipsoid_jacobian(a,b,c,sa,sb,sc,pos):
        """
        Computes the Jacobian of the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis length.
        b : float
            The semi-intermediate axis length.
        c : float
            The semi-minor axis length.
        sa : float
            The scale factor for the a axis.
        sb : float
            The scale factor for the b axis.
        sc : float
            The scale factor for the c axis.
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid_jacobian(float(a),float(b),float(c),float(sa),float(sb),float(sc),pos)