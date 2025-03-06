
import copy

import numpy as np
from scipy.spatial.transform import Rotation



from ...util.array_operate import Rotate,Shift
from ..structure_main import Structure_3D, Parameters


__all__ = ["Rotation3D","Coordinate3D"]


class Rotation3D(Rotation):
        
    def jacobian_euler(self, pos, seq: str = 'zyx'):
        '''
        pos1 = (R @ pos0.T).T
        seq = 'zyx'
        calculate dpos2/d_R
        --------
        input: pos, Nx3 array [x,y,z]
        output: (Nx3,Nx3,Nx3 array) 
        dx_dgamma, dy_dgamma, dz_dgamma, dx_dbeta, dy_dbeta, dz_dbeta, d_x, d_alpha ... 
        '''
        d_theta1,d_theta2,d_theta3 = self.d_euler(seq)
        
        return (Rotate(pos,d_theta1),Rotate(pos,d_theta2),Rotate(pos,d_theta3))
        
        
    def d_euler(self,seq: str):
        '''
        return (3x3,3x3,3x3 array)
        '''
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
    '''Coordinate translation and rotation'''
    

    PA = ('x','y','z','ang1','ang2','ang3')    ##!!!! not use set !!!!
    LB = {'x':-0.2,'y':-0.2,'z':-0.2, 'ang1':-np.pi,'ang2':-np.pi/2,'ang3':-np.pi}
    UB = {'x':0.2, 'y':0.2, 'z':0.2, 'ang1':np.pi, 'ang2':np.pi/2, 'ang3':np.pi}
    Euler_seq = 'zyx'
    def __init__(self,x,y,z,ang1,ang2,ang3,**kwargs):
        '''
        Coordinate translation and rotation
        ------------
        Parameters:
        px,py,pz, the center position,
        ang1,ang2,ang3, the Euler angles, with unit of pi
        ------------
        kwargs:
        seq, default, 'zyx', 
        ---------------
        Pos0, original position
        Pos1, = Rot @ Pos0, rotation, Rot is the rotation matrix, determined by ang1,ang2,ang3
        Pos2, = Pos1 - Pc, translation, Pc represent [px, py, pz], final position,
        '''
        self.parameters = self.init_parameters(x=x,y=y,z=z,ang1=ang1,ang2=ang2,ang3=ang3)
        self._seq = kwargs.get('seq',Coordinate3D.Euler_seq)
        self._rotation = Rotation3D.from_euler(seq=self._seq, angles= [ang1,ang2,ang3])
    
    @staticmethod
    def init_parameters(**kwargs):
        
        
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
        return Coordinate3D.init_parameters(x=0.,y=0.,z=0.,ang1=0.,ang2=0.,ang3=0.)
    
        
    def jacobian(self,pos):
        '''
        return d_Pos2/d_Pc d_Pos2/d_angs
        '''
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
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return Shift(Rotate(pos.copy(),self._rotation.as_matrix()),self['pos'])
    
    
    
    def inverse(self,pos):
        if (len(np.shape(pos))==2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos))==1:
            pos = np.float64([pos])
        return Rotate(Shift(pos.copy(),-self['pos']),self._rotation.as_matrix().T)
    
    @staticmethod
    def quick_call(x,y,z,ang1,ang2,ang3,pos):
        pc = np.float64([x,y,z])
        matrix = Rotation3D.from_euler(seq=Coordinate3D.Euler_seq, angles=[ang1,ang2,ang3]).as_matrix()
        return Shift(Rotate(pos.copy(),matrix),pc)
    
    @staticmethod
    def quick_call_inverse(x,y,z,ang1,ang2,ang3,pos):
        pc = np.float64([x,y,z])
        matrix = Rotation3D.from_euler(seq=Coordinate3D.Euler_seq, angles=[ang1,ang2,ang3]).as_matrix()
        return Rotate(Shift(pos.copy(),-pc),matrix.T)
    
    
    def __getitem__(self, item):
        try:
            return self.parameters[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')

    def __setitem__(self, key, value):
        raise KeyError('Cannot change a parameter')

    def __delitem__(self, key):
        raise KeyError('Cannot delete a parameter')

    def __repr__(self):

        param_repr = repr(self.parameters)

        return "<Coordinate3D|: "+ param_repr[10:] + "|>"
    
    def keys(self):
        """Return the keys of the Coordinate transformation parameters"""
        return list(self.parameters.keys())

    def __contains__(self, item):
        return item in self.parameters