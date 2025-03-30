
import copy
import logging

import numpy as np


from .util_fcall import *
from .util_distance import *
from ..structure_main import Structure_3D, Parameters


__all__ =["Ellipsoid"]

logger = logging.getLogger('gal3d.structure.shape.ellipsoid')

@Structure_3D.shape_func
class Ellipsoid:
    """
    A class representing a standard ellipsoid in 3D space.

    Attributes
    ----------
    PA : tuple of str
        Tuple containing the parameter names: 'a', 'eps_ab', 'eps_bc'.
    LB : dict
        Lower bounds for the parameters: 'a', 'eps_ab', 'eps_bc'.
    UB : dict
        Upper bounds for the parameters: 'a', 'eps_ab', 'eps_bc'.

    Methods
    -------
    __init__(*args, **kwargs)
        Initialize the ellipsoid with given parameters.
    init_parameters(**kwargs)
        Initialize and return the parameters of the ellipsoid.
    get_parameters()
        Return a default set of parameters for the ellipsoid.
    __call__(pos)
        Evaluate the ellipsoid function at given positions.
    __getitem__(item)
        Get the value of a parameter by name.
    __repr__()
        Return a string representation of the ellipsoid.
    closed_point(pos, maxIterations=100)
        Compute the closest point on the ellipsoid to the given positions and the distance.
    ray_point(pos)
        Compute the distance between points and the ray point on the ellipsoid.
    jacobian(pos)
        Compute the Jacobian of the ellipsoid function at given positions.
    quick_call(a, eps_ab, eps_bc, pos)
        Quickly evaluate the ellipsoid function with given parameters and positions.
    quick_call_d(a, eps_ab, eps_bc, pos)
        Quickly evaluate the derivative of the ellipsoid function with given parameters and positions.
    quick_call_raydistance(a, eps_ab, eps_bc, pos)
        Quickly compute the distance between points and the ray point on the ellipsoid.
    f_ellipsoid(a, b, c, pos)
        Evaluate the ellipsoid function with given parameters and positions.
    DistancePointsEllipsoid(a, b, c, pos, maxIterations=100)
        Compute the closest point on the ellipsoid to the given positions and the distance.
    DistanceRayPointsEllipsoid(a, b, c, pos)
        Compute the distance between points and the ray point on the ellipsoid.
    f_ellipsoid_jacobian(a, b, c, pos)
        Compute the Jacobian of the ellipsoid function with given parameters and positions.
    """

    
    PA = ('a','eps_ab','eps_bc')        ### not use set !!!
    LB = {'a':0.1,'eps_ab':0.01,'eps_bc':0.01}
    UB = {'a':np.inf,'eps_ab':0.99,'eps_bc':0.99}
    def __init__(self,*args,**kwargs):
        """
        Initialize the ellipsoid with given parameters.

        Parameters
        ----------
        *args : tuple
            Default parameters: a, b, c.
        **kwargs : dict
            Additional parameters to initialize the ellipsoid.

        Notes
        -----
        The parameters a, b, c must satisfy a > b > c.
        The derived parameters eps_ab and eps_bc are defined as:
        eps_ab = 1 - b/a : 0~1
        eps_bc = 1 - c/b : 0~1
        """
        self.parameters = self.init_parameters(**kwargs)
        
    @staticmethod
    def init_parameters(**kwargs):
        """
        Initialize and return the parameters of the ellipsoid.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to initialize the ellipsoid.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing the ellipsoid parameters.
        """
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
        """
        Return a default set of parameters for the ellipsoid.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing default ellipsoid parameters.
        """
        return Ellipsoid.init_parameters(a=3.,eps_ab=0.2,eps_bc=0.5)
    
        
    def __call__(self, pos):
        """
        Evaluate the ellipsoid function at given positions.

        Parameters
        ----------
        pos : array_like
            Positions at which to evaluate the ellipsoid function.

        Returns
        -------
        float or ndarray
            The value of the ellipsoid function at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_ellipsoid(self['a'],self['b'],self['c'], pos)
    
    def __getitem__(self, item):
        """
        Get the value of a parameter by name.

        Parameters
        ----------
        item : str
            The name of the parameter to retrieve.

        Returns
        -------
        float
            The value of the parameter.

        Raises
        ------
        KeyError
            If the parameter name is not valid.
        """
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')
    
    def __repr__(self):
        """
        Return a string representation of the ellipsoid.

        Returns
        -------
        str
            A string representation of the ellipsoid.
        """
        param_repr = repr(self.parameters)

        return "<Ellipsoid|: "+ param_repr[10:] + "|>"
    
    def closed_point(self,pos,maxIterations:int = 100) -> tuple:
        """
        Compute the closest point on the ellipsoid to the given positions and the distance.

        Parameters
        ----------
        pos : array_like
            Positions for which to find the closest point on the ellipsoid.
        maxIterations : int, optional
            Maximum number of iterations for the optimization algorithm (default is 100).

        Returns
        -------
        tuple
            A tuple containing the closest points on the ellipsoid and the distances.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        x0,x1,x2,d = DistancePointsEllipsoid(self['a'],self['b'],self['c'],pos[:,0],pos[:,1],pos[:,2],maxIterations)
        
        return np.array([x0,x1,x2]).T,d
    
    def ray_point(self,pos):
        """
        Compute the distance between points and the ray point on the ellipsoid.

        Parameters
        ----------
        pos : array_like
            Positions for which to compute the distance.

        Returns
        -------
        float or ndarray
            The distance between the points and the ray point on the ellipsoid.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return DistanceRayPointsEllipsoid(self['a'],self['b'],self['c'], pos)
    
    def jacobian(self,pos) -> tuple:
        """
        Compute the Jacobian of the ellipsoid function at given positions.

        Parameters
        ----------
        pos : array_like
            Positions at which to compute the Jacobian.

        Returns
        -------
        tuple
            The Jacobian matrix of the ellipsoid function at the given positions.
        """
        return f_ellipsoid_jacobian(self['a'],self['b'],self['c'], pos)
    
    @staticmethod
    def quick_call(a,eps_ab,eps_bc,pos):
        """
        Quickly evaluate the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        pos : array_like
            Positions at which to evaluate the ellipsoid function.

        Returns
        -------
        float or ndarray
            The value of the ellipsoid function at the given positions.
        """
        b = a*(1.-eps_ab)
        c = b*(1.-eps_bc)
        return f_ellipsoid(float(a),b,c,pos)
    
    @staticmethod
    def quick_call_d(a,eps_ab,eps_bc,pos):
        """
        Quickly evaluate the distance fraction of the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        pos : array_like
            Positions at which to evaluate the distance fraction.

        Returns
        -------
        float or ndarray
            The distance fraction of the ellipsoid function at the given positions.
        """
        b = a*(1.-eps_ab)
        c = b*(1.-eps_bc)
        return d_ellipsoid(float(a),b,c,pos)
    
    @staticmethod
    def quick_call_raydistance(a,eps_ab,eps_bc,pos):
        """
        Quickly compute the distance between points and the ray point on the ellipsoid.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        pos : array_like
            Positions for which to compute the distance.

        Returns
        -------
        float or ndarray
            The distance between the points and the ray point on the ellipsoid.
        """
        b = a*(1.-eps_ab)
        c = b*(1.-eps_bc)
        return DistanceRayPointsEllipsoid(float(a),b,c,pos)[1]
    
    @staticmethod
    def quick_call_lineintersect(a,eps_ab,eps_bc,pos1,pos2):
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return IntersectLinesEllipsoid(float(a),float(b),float(c),pos1,pos2)
    
    @staticmethod
    def f_ellipsoid(a,b,c,pos):
        """
        Evaluate the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-intermediate axis of the ellipsoid.
        c : float
            The semi-minor axis of the ellipsoid.
        pos : array_like
            Positions at which to evaluate the ellipsoid function.

        Returns
        -------
        float or ndarray
            The value of the ellipsoid function at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_ellipsoid(float(a),float(b),float(c), pos)
    
    def line_points(self,pos1,pos2):
        if (len(np.shape(pos1))==2) and (np.shape(pos1)[1] == 3):
            pos1 = np.float64(pos1)
        if len(np.shape(pos1))==1:
            pos1 = np.float64([pos1])
        if (len(np.shape(pos2))==2) and (np.shape(pos2)[1] == 3):
            pos2 = np.float64(pos2)
        if len(np.shape(pos2))==1:
            pos2 = np.float64([pos2])
        
        return IntersectLinesEllipsoid(self['a'],self['b'],self['c'],pos1,pos2)
    
    @staticmethod
    def DistancePointsEllipsoid(a,b,c,pos, maxIterations: int = 100):
        """
        Compute the closest point on the ellipsoid to the given positions and the distance.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-intermediate axis of the ellipsoid.
        c : float
            The semi-minor axis of the ellipsoid.
        pos : array_like
            Positions for which to find the closest point on the ellipsoid.
        maxIterations : int, optional
            Maximum number of iterations for the optimization algorithm (default is 100).

        Returns
        -------
        tuple
            A tuple containing the closest points on the ellipsoid and the distances.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        x0,x1,x2,d = DistancePointsEllipsoid(float(a),float(b),float(c),pos[:,0],pos[:,1],pos[:,2],maxIterations)
        
        return np.array([x0,x1,x2]).T,d
    
    @staticmethod
    def DistanceRayPointsEllipsoid(a,b,c,pos):
        """
        Compute the distance between points and the ray point on the ellipsoid.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-intermediate axis of the ellipsoid.
        c : float
            The semi-minor axis of the ellipsoid.
        pos : array_like
            Positions for which to compute the distance.

        Returns
        -------
        float or ndarray
            The distance between the points and the ray point on the ellipsoid.
        """
        return DistanceRayPointsEllipsoid(float(a),float(b),float(c),np.float64(pos))
    
    @staticmethod
    def f_ellipsoid_jacobian(a,b,c,pos):
        """
        Compute the Jacobian of the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-intermediate axis of the ellipsoid.
        c : float
            The semi-minor axis of the ellipsoid.
        pos : array_like
            Positions at which to compute the Jacobian.

        Returns
        -------
        tuple
            The Jacobian matrix of the ellipsoid function at the given positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return f_ellipsoid_jacobian(float(a),float(b),float(c), pos)