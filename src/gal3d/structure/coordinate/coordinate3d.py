
import copy
import logging

import numpy as np
from scipy.spatial.transform import Rotation



from ...util.array_operate import Rotate,Shift
from ..structure_main import Structure_3D, Parameters


__all__ = ["Rotation3D","Coordinate3D"]

logger = logging.getLogger('gal3d.structure.coordinate.coordinate3d')
class Rotation3D(Rotation):
    """
    A class to handle 3D rotations using Euler angles.

    This class extends the `Rotation` class to provide additional functionality
    for 3D rotations, including the computation of Jacobians for Euler angles.

    Attributes
    ----------
    None
    """
    def jacobian_euler(self, pos, seq: str = 'zyx'):
        """
        Compute the Jacobian of the rotated position with respect to the Euler angles.

        Given a set of positions and a sequence of Euler angles, this function computes
        the Jacobian matrix that describes how the rotated positions change with respect
        to changes in the Euler angles.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be rotated.
        seq : str, optional
            The sequence of Euler angles to use for the rotation. Default is 'zyx'.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple of three Nx3x3 arrays representing the Jacobian matrices for the
            rotated positions with respect to each Euler angle.
        """
        d_theta1,d_theta2,d_theta3 = self.d_euler(seq)
        
        return (Rotate(pos,d_theta1),Rotate(pos,d_theta2),Rotate(pos,d_theta3))
        
        
    def d_euler(self,seq: str):
        """
        Compute the derivative matrices for the Euler angles.

        This function calculates the derivative matrices for the given sequence of Euler angles.
        These matrices are used to compute the Jacobian of the rotation.

        Parameters
        ----------
        seq : str
            The sequence of Euler angles to use for the rotation.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple of three 3x3 arrays representing the derivative matrices for each Euler angle.
        """
        angle={}
        angle[seq[0]],angle[seq[1]],angle[seq[2]] = self.as_euler(seq)
        
        C1, S1 = np.cos(angle['z']), np.sin(angle['z'])
        C2, S2 = np.cos(angle['y']), np.sin(angle['y'])
        C3, S3 = np.cos(angle['x']), np.sin(angle['x'])
        
        angle['R_z'] = np.array([
            [C1, -S1, 0],
            [S1, C1, 0],
            [0, 0, 1]  ])
        angle['d_R_z'] = np.array([
            [-S1, -C1, 0],
            [C1, -S1, 0],
            [0, 0, 1]  ])
        angle['R_y'] = np.array([
            [C2, 0, S2],
            [0, 1, 0],
            [-S2, 0, C2]])
        angle['d_R_y'] = np.array([
            [-S2, 0, C2],
            [0, 1, 0],
            [-C2, 0, -S2]])
        angle['R_x'] = np.array([
            [1, 0, 0],
            [0, C3, -S3],
            [0, S3, C3],])
        angle['d_R_x'] = np.array([
            [1, 0, 0],
            [0, -S3, -C3],
            [0, C3, -S3],])
        
        return (np.dot(np.dot(angle[f'd_R_{seq[0]}'],angle[f'R_{seq[1]}']),angle[f'R_{seq[2]}']),
                np.dot(np.dot(angle[f'R_{seq[0]}'],angle[f'd_R_{seq[1]}']),angle[f'R_{seq[2]}']),
                np.dot(np.dot(angle[f'R_{seq[0]}'],angle[f'R_{seq[1]}']),angle[f'd_R_{seq[2]}']))

@Structure_3D.coordinate_func
class Coordinate3D:
    """
    A class to handle 3D coordinate transformations, including translation and rotation.

    This class provides methods to transform coordinates using a combination of translation
    and rotation, and to compute the Jacobian of the transformation.

    Attributes
    ----------
    PA : tuple
        A tuple of parameter names: ('x', 'y', 'z', 'ang1', 'ang2', 'ang3').
    LB : dict
        A dictionary of lower bounds for the parameters.
    UB : dict
        A dictionary of upper bounds for the parameters.
    Euler_seq : str
        The default sequence of Euler angles for rotation.
    """
    

    PA = ('x','y','z','ang1','ang2','ang3')    ##!!!! not use set !!!!
    LB = {'x':-0.2,'y':-0.2,'z':-0.2, 'ang1':-np.pi,'ang2':-np.pi/2,'ang3':-np.pi}
    UB = {'x':0.2, 'y':0.2, 'z':0.2, 'ang1':np.pi, 'ang2':np.pi/2, 'ang3':np.pi}
    Euler_seq = 'zyx'
    def __init__(self,x,y,z,ang1,ang2,ang3,**kwargs):
        """
        Initialize the Coordinate3D object with translation and rotation parameters.

        Parameters
        ----------
        x, y, z : float
            The center position coordinates.
        ang1, ang2, ang3 : float
            The Euler angles for rotation, in units of pi.
        **kwargs : dict
            Additional keyword arguments, including:
            - seq : str, optional
                The sequence of Euler angles for rotation. Default is 'zyx'.
        """
        self.parameters = self.init_parameters(x=x,y=y,z=z,ang1=ang1,ang2=ang2,ang3=ang3)
        self._seq = kwargs.get('seq',Coordinate3D.Euler_seq)
        self._rotation = Rotation3D.from_euler(seq=self._seq, angles= [ang1,ang2,ang3])
    
    @staticmethod
    def init_parameters(**kwargs):
        """
        Initialize the parameters for the coordinate transformation.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter values.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing the initialized parameters.
        """
        
        param = Parameters(**kwargs)
        param._derived['pos'] = lambda d: np.array([d['x'],d['y'],d['z']])
        param._derived['angle'] = lambda d: np.array([d['ang1'],d['ang2'],d['ang3']])
        
        parameters = Parameters(**{i:param[i] for i in Coordinate3D.PA})
        parameters._derived.update(param._derived)
        parameters.set_lb(**Coordinate3D.LB)
        parameters.set_ub(**Coordinate3D.UB)

        return parameters
    
    @staticmethod
    def get_parameters():
        """
        Get the default parameters for the coordinate transformation.

        Returns
        -------
        Parameters
            An instance of the Parameters class with default values.
        """
        return Coordinate3D.init_parameters(x=0.,y=0.,z=0.,ang1=0.,ang2=0.,ang3=0.)
    
        
    def jacobian(self,pos):
        """
        Compute the Jacobian of the transformed positions with respect to the translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple of six arrays representing the Jacobian matrices for the transformed positions
            with respect to the translation and rotation parameters.
        """
        pos1 = Shift(pos,-self['pos'])
        N_p = len(pos)
        d_ang1,d_ang2,d_ang3 = self._rotation.jacobian_euler(pos=pos1,seq=self._seq)
        rotmatrix = self._rotation.as_matrix()
        
        d_Px = np.array([-np.ones(N_p),
                         np.zeros(N_p),
                         np.zeros(N_p),])
        d_Py = np.array([np.zeros(N_p),
                         -np.ones(N_p),
                         np.zeros(N_p)])
        d_Pz = np.array([np.zeros(N_p),
                         np.zeros(N_p),
                         -np.ones(N_p)])
        
        return (d_Px.T,d_Py.T,d_Pz.T,d_ang1,d_ang2,d_ang3)
    
    def __call__(self, pos):
        """
        Transform the given positions using the current translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        numpy.ndarray
            The transformed positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return Shift(Rotate(pos.copy(),self._rotation.as_matrix()),self['pos'])
    
    
    
    def inverse(self,pos):
        """
        Inverse transform the given positions using the current translation and rotation parameters.

        Parameters
        ----------
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be inverse transformed.

        Returns
        -------
        numpy.ndarray
            The inverse transformed positions.
        """
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return Rotate(Shift(pos.copy(),-self['pos']),self._rotation.as_matrix().T)
    
    @staticmethod
    def quick_call(x,y,z,ang1,ang2,ang3,pos):
        """
        Quickly transform the given positions using the specified translation and rotation parameters.

        Parameters
        ----------
        x, y, z : float
            The center position coordinates.
        ang1, ang2, ang3 : float
            The Euler angles for rotation, in units of pi.
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be transformed.

        Returns
        -------
        numpy.ndarray
            The transformed positions.
        """
        pc = np.float64([x,y,z])
        matrix = Rotation3D.from_euler(seq=Coordinate3D.Euler_seq, angles=[ang1,ang2,ang3]).as_matrix()
        return Shift(Rotate(pos.copy(),matrix),pc)
    
    @staticmethod
    def quick_call_inverse(x,y,z,ang1,ang2,ang3,pos):
        """
        Quickly inverse transform the given positions using the specified translation and rotation parameters.

        Parameters
        ----------
        x, y, z : float
            The center position coordinates.
        ang1, ang2, ang3 : float
            The Euler angles for rotation, in units of pi.
        pos : numpy.ndarray
            An Nx3 array representing the positions [x, y, z] to be inverse transformed.

        Returns
        -------
        numpy.ndarray
            The inverse transformed positions.
        """
        pc = np.float64([x,y,z])
        matrix = Rotation3D.from_euler(seq=Coordinate3D.Euler_seq, angles=[ang1,ang2,ang3]).as_matrix()
        return Rotate(Shift(pos.copy(),-pc),matrix.T)
    
    
    def __getitem__(self, item):
        """
        Get the value of a parameter.

        Parameters
        ----------
        item : str
            The name of the parameter.

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

    def __setitem__(self, key, value):
        """
        Set the value of a parameter.

        Parameters
        ----------
        key : str
            The name of the parameter.
        value : float
            The value to set.

        Raises
        ------
        KeyError
            Cannot change a parameter
        """
        raise KeyError('Cannot change a parameter')

    def __delitem__(self, key):
        """
        Delete a parameter.

        Parameters
        ----------
        key : str
            The name of the parameter.

        Raises
        ------
        KeyError
            Cannot delete a parameter
        """
        raise KeyError('Cannot delete a parameter')

    def __repr__(self):
        """
        Return a string representation of the Coordinate3D object.

        Returns
        -------
        str
            A string representation of the object.
        """

        param_repr = repr(self.parameters)

        return "<Coordinate3D|: "+ param_repr[10:] + "|>"
    
    def keys(self):
        """
        Return the keys of the coordinate transformation parameters.

        Returns
        -------
        list
            A list of parameter names.
        """
        return list(self.parameters.keys())

    def __contains__(self, item):
        """
        Check if a parameter is in the coordinate transformation parameters.

        Parameters
        ----------
        item : str
            The name of the parameter.

        Returns
        -------
        bool
            True if the parameter is in the parameters, False otherwise.
        """
        return item in self.parameters