


from numba import (
    int32,
    deferred_type,
    optional,
    float64,
    boolean,
    int64,
    njit,
    jit,
    prange,
    types,
)

FCALPARA = True
import numpy as np
import math

from .util_distance import DistanceRayPointEllipsoid_S,DistanceRayPointEllipsoid

__all__=["f_ellipsoid","f_ellipsoid_jacobian","f_ellipsoids",
         "f_shaped_ellipsoid","f_shaped_ellipsoid_jacobian","f_shaped_ellipsoids",
         "d_ellipsoid","d_shaped_ellipsoid",]

@jit(float64[:](float64,float64,float64,float64[:,:]),nogil=True,parallel=FCALPARA,fastmath=True,cache=True)
def f_ellipsoid(a: float,b: float, c: float, pos):
    return (pos[:,0]*pos[:,0]/a/a+pos[:,1]*pos[:,1]/b/b+pos[:,2]*pos[:,2]/c/c)




@jit(float64[:](float64, float64, float64, float64[:,:]),
     nogil=True,parallel=FCALPARA,fastmath=True,cache=True)
def d_ellipsoid(a, b, c, pos):

    tarpos = np.zeros((len(pos),3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    r = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i,0],tarpos[i,1],tarpos[i,2],d[i], L[i] = DistanceRayPointEllipsoid(a,b,c,pos[i,0],pos[i,1],pos[i,2])
    for i in prange(len(pos)):
        r[i] = L[i]/d[i]
    return r

@jit(types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
     (float64,float64,float64,float64[:,:]),nogil=True,parallel=FCALPARA,fastmath=True,cache=True)
def f_ellipsoid_jacobian(a: float,b: float, c: float, pos):
    '''
    jacobian of ellipsoid, d/da,d/db,d/dc,d/dx,d/dy,d/dz
    '''
    dx = 2*pos[:,0]/a/a
    dy = 2*pos[:,1]/b/b
    dz = 2*pos[:,2]/c/c
    da = -dx*pos[:,0]/a
    db = -dy*pos[:,1]/b
    dc = -dz*pos[:,2]/c
    return (da,db,dc,dx,dy,dz)


@jit(float64[:,:](float64[:],float64[:],float64[:],float64[:,:]),nogil=True,parallel=FCALPARA,fastmath=True,cache=True )
def f_ellipsoids(a,b,c,pos):
    res = np.zeros((len(a),len(pos)))
    for i in prange(len(a)):
        res[i] = f_ellipsoid(a[i],b[i],c[i],pos)
    return res



@jit(float64[:](float64,float64,float64,float64,float64,float64,float64[:,:]),nogil=True,parallel=FCALPARA,fastmath=True,cache=True )
def f_shaped_ellipsoid(a,b,c,Sa,Sb,Sc,pos):
    h1 = pos[:,0]*pos[:,0]/a/a
    h2 = pos[:,1]*pos[:,1]/b/b
    h3 = pos[:,2]*pos[:,2]/c/c
    return (np.float_power(h1,Sa)+np.float_power(h2,Sb)+np.float_power(h3,Sc))


@jit(types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
     (float64,float64,float64,float64,float64,float64,float64[:,:]),nogil=True,parallel=FCALPARA,fastmath=True,cache=True )
def f_shaped_ellipsoid_jacobian(a,b,c,Sa,Sb,Sc,pos):
    cof0 = np.float_power(pos[:,0]*pos[:,0],Sa)
    cof1 = np.float_power(pos[:,1]*pos[:,1],Sb)
    cof2 = np.float_power(pos[:,2]*pos[:,2],Sc)
    dx = 2*Sa*cof0/pos[:,0]/a**(2*Sa)
    dy = 2*Sb*cof1/pos[:,1]/b**(2*Sb)
    dz = 2*Sc*cof2/pos[:,2]/c**(2*Sc)
    da = -2*Sa*cof0/a**(2*Sa+1)
    db = -2*Sb*cof1/b**(2*Sb+1)
    dc = -2*Sc*cof2/c**(2*Sc+1)
    dSa = 2*cof0*(np.log(np.abs(pos[:,0]))-np.log(a))/(a**2)**Sa
    dSb = 2*cof1*(np.log(np.abs(pos[:,1]))-np.log(b))/(b**2)**Sb
    dSc = 2*cof2*(np.log(np.abs(pos[:,2]))-np.log(c))/(c**2)**Sc
    return (da,db,dc,dSa,dSb,dSc,dx,dy,dz)
    
    
    
@jit(float64[:,:](float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:,:]),
     nogil=True,parallel=FCALPARA,fastmath=True,cache=True )
def f_shaped_ellipsoids(a,b,c,Sa,Sb,Sc,pos):
    res = np.zeros((len(a),len(pos)))
    for i in prange(len(a)):
        res[i] = f_shaped_ellipsoid(a[i],b[i],c[i],Sa[i],Sb[i],Sc[i],pos)
    return res


@jit(float64[:](float64, float64, float64,float64, float64, float64, float64[:,:],int32),
     nogil=True,parallel=FCALPARA,fastmath=True)
def d_shaped_ellipsoid(a, b, c, Sa, Sb, Sc, pos, maxIterations: int):
    tarpos = np.zeros((len(pos),3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    r = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i,0],tarpos[i,1],tarpos[i,2],d[i], L[i] = DistanceRayPointEllipsoid_S(
                                                a,b,c,Sa,Sb,Sc,
                                                pos[i,0],pos[i,1],pos[i,2],maxIterations)
    for i in prange(len(pos)):
        r[i] = L[i]/d[i]
    return r


if __name__ == "__main__":
    pass

