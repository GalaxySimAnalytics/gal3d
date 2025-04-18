
import numpy as np
from numpy.typing import ArrayLike

from ._ellipsoid_util import *
from ..geometry import GeometryBase,classproperty,Parameters

__all__ = ['Ellipsoid']

class Ellipsoid(GeometryBase):
    PN = ('a','eps_ab','eps_bc')        ### not use set !!!
    LB = {'a':0.1,'eps_ab':0.01,'eps_bc':0.01}
    UB = {'a':np.inf,'eps_ab':0.99,'eps_bc':0.99}
    
    MaxIterationClosed = 100
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
        
        parameters = Parameters(**{i:param[i] for i in Ellipsoid.PN})
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
    
    
    def ray_intersect(self,pos):
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
        return IntersectRaysEllipsoid(self['a'],self['b'],self['c'], pos)
    
    def line_intersect(self, pos1, pos2):
        
        if (len(np.shape(pos1))==2) and (np.shape(pos1)[1] == 3):
            pos1 = np.float64(pos1)
        if len(np.shape(pos1))==1:
            pos1 = np.float64([pos1])
        if (len(np.shape(pos2))==2) and (np.shape(pos2)[1] == 3):
            pos2 = np.float64(pos2)
        if len(np.shape(pos2))==1:
            pos2 = np.float64([pos2])
        
        return IntersectLinesEllipsoid(self['a'],self['b'],self['c'],pos1,pos2)
    
    
    def f_ray_d(self, pos):
        
        return f_ray_ellipsoid(self['a'],self['b'],self['c'], pos)
    
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
    def quick_f_ray_d(a,eps_ab,eps_bc,pos):
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
        return f_ray_ellipsoid(float(a),b,c,pos)
    
    @staticmethod
    def quick_ray_dist(a,eps_ab,eps_bc,pos):
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
        return IntersectRaysEllipsoid(float(a),b,c,pos)[1]
    
    
    @staticmethod
    def quick_line_intersect(a,eps_ab,eps_bc,pos1,pos2):
        b = a*(1-eps_ab)
        c = b*(1-eps_bc)
        return IntersectLinesEllipsoid(float(a),float(b),float(c),pos1,pos2)
    
    @staticmethod
    def quick_jacobian(a,b,c,pos):
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
    
    
    def closed_point(self,pos) -> tuple:
        """
        Compute the closest point on the ellipsoid to the given positions and the distance.

        Parameters
        ----------
        pos : array_like
            Positions for which to find the closest point on the ellipsoid.
        Returns
        -------
        tuple
            A tuple containing the closest points on the ellipsoid and the distances.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        x0,x1,x2,d = DistancePointsEllipsoid(self['a'],self['b'],self['c'],pos[:,0],pos[:,1],pos[:,2],Ellipsoid.MaxIterationClosed)
        
        return np.array([x0,x1,x2]).T,d
        