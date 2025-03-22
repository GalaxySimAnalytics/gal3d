

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

DISTANCEPARA = True
    
import numpy as np
import math

from ...util.array_operate import unit_vector3d, vector_length3d

__all__ = ['DistancePointsEllipsoid','DistancePointsEllipse','DistancePointEllipsoid','DistancePointEllipse',
           "DistanceRayPointsEllipsoid","DistanceRayPointsEllipsoid_S","IntersectLinesEllipsoid","IntersectLinesEllipsoid_S"]



@jit(float64(float64[:],float64[:]),fastmath=True,parallel=DISTANCEPARA,cache=True )
def Dot(v0,v1):
    dot = v0[0]*v1[0]
    for i in prange(1,len(v0)):
        dot += v0[i]*v1[i]
    return dot

@jit(float64(float64,float64),fastmath=True,cache=True )
def RobustLength2d(v0,v1):
    '''avoiding floating-point overflow that could occur normally when computing'''
    v0 = v0 if v0>0 else -v0
    v1 = v1 if v1>0 else -v1
    if v0>v1:
        return v0*math.sqrt(1+v1*v1/v0/v0)
    else:
        return v1*math.sqrt(1+v0*v0/v1/v1)
    
@jit(float64(float64,float64,float64),fastmath=True,cache=True )
def RobustLength3d(v0,v1,v2):
    '''avoiding floating-point overflow that could occur normally when computing'''
    v0 = v0 if v0>0 else -v0
    v1 = v1 if v1>0 else -v1
    v2 = v2 if v2>0 else -v2
    if (v0>v1) and (v0>v2):
        return v0*math.sqrt(1+v1*v1/v0/v0+v2*v2/v0/v0)
    elif (v2>v0) and (v2>v1):
        return v2*math.sqrt(1+v1*v1/v2/v2+v0*v0/v2/v2)
    else:
        return v1*math.sqrt(1+v0*v0/v1/v1+v2*v2/v1/v1)


@jit(float64(float64[:],boolean),cache=True )
def Length(v,robust: bool = False):
    if robust:
        maxAbsComp = abs(v[0])
        for i in range(1,len(v)):
            absComp = abs(v[i])
            if absComp> maxAbsComp:
                maxAbsComp = absComp
        if maxAbsComp > 0:
            scaled = v/maxAbsComp
            length = maxAbsComp * math.sqrt(Dot(scaled,scaled))
        else:
            length = 0.            
        return length 
    else:
        return math.sqrt(Dot(v,v))
    
@jit(float64(float64[:]),cache=True )
def RobustLength(v):
    '''avoiding floating-point overflow that could occur normally when computing'''
    return Length(v,robust=True)

    

@jit(float64(float64,float64,float64,float64,int64), fastmath=True,cache=True )
def GetRoot2d(r0: float, z0: float, z1: float, g: float, maxIterations: int) -> float: 
    '''The bisection algorithm to find the unique root'''
    n0 = r0*z0
    s0 = z1 -1
    s1 = 0. if g < 0 else RobustLength2d(n0,z1)-1.
    s = 0.
    i = 0
    while True:
        if i > maxIterations:
            print("maximum number of iterations exceeded!")
            break
        s = (s0 + s1) / 2
        if s == s0 or s == s1:
            break
        ratio0 = n0 / (s + r0)
        ratio1 = z1 / (s +  1)
        g = RobustLength2d(ratio0,ratio1) - 1.
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            break
        i += 1
    return s

@jit(float64(float64[:],float64[:],float64,int64),cache=True )
def GetRoot(r,z,g,maxIterations):
    n = np.zeros(len(z))
    for i in range(len(r)):
        n[i] = r[i]*z[i]
    n[-1] = z[-1]
    s0 = z[-1] -1
    s1 = 0. if g<0 else RobustLength(n) - 1.
    s = 0.
    j = 0
    while True:
        if j > maxIterations:
            print("maximum number of iterations exceeded!")
            break
        s = (s0+s1)/2
        if (s == s0) or (s == s1):
            break
        ratio = np.zeros(len(r))
        for i in range(len(r)):
            ratio[i] = n[i]/(s+r[i])
        ratio[-1] = z[-1]/(s+1)
        g = RobustLength(ratio) -1
        if g>0:
            s0 = s
        elif g<0:
            s1 = s
        else:
            break
        j += 1
    return s
    
@jit(float64(float64,float64,float64,float64,float64,float64,int64),fastmath=True,cache=True )
def GetRoot3d(r0,r1,z0,z1,z2,g,maxIterations):
    n0 = r0*z0
    n1 = r1*z1
    s0 = z2-1
    s1 = 0. if g<0 else RobustLength3d(n0,n1,z2)-1.
    s = 0.
    i = 0
    while True:
        if i > maxIterations:
            print("maximum number of iterations exceeded!")
            break
        s = (s0+s1)/2
        if (s == s0) or (s == s1):
            break
        ratio0 = n0/(s+r0)
        ratio1 = n1/(s+r1)
        ratio2 = z2/(s+1)
        g = RobustLength3d(ratio0,ratio1,ratio2) -1
        if g>0:
            s0 = s
        elif g<0:
            s1 = s
        else:
            break
        i += 1
    return s
        

@jit(types.Tuple((float64,float64,float64))(float64, float64, float64, float64, int64),fastmath=True,cache=True )
def _DistancePointEllipse(e0: float, e1: float, y0: float, y1: float, maxIterations: int = 100) -> float:
    '''
    e0>=e1>0
    y0>0
    y1>0
    '''
    if y1 > 0.:
        x1 =1.
        if y0 > 0.:
            z0 = y0 / e0
            z1 = y1 / e1
            g = z0*z0 + z1*z1 - 1.
            if abs(g) > 0.:
                r0 = e0*e0/e1/e1
                sbar = GetRoot2d(r0, z0, z1, g, maxIterations)
                x0 = r0 * y0 / (sbar + r0)
                x1 = y1 / (sbar + 1.)
                distance = RobustLength2d(x0-y0,x1-y1)
            else:
                x0 = y0
                x1 = y1
                distance = 0.
        else: # y0 == 0
            x0 = 0
            x1 = e1
            distance = abs(y1 - e1)
    else: # y1 == 0
        numer0 = e0 * y0
        denom0 = e0*e0 - e1*e1
        if numer0 < denom0:
            xde0 = numer0 / denom0
            x0 = e0 * xde0
            x1 = e1 * math.sqrt(1 - xde0*xde0)
            distance = RobustLength2d(x0-y0,x1)
        else:
            x0 = e0
            x1 = 0
            distance = abs(y0 - e0)
    return x0, x1, distance
            
@jit(types.Tuple((float64,float64,float64))(float64,float64,float64,float64,int64),cache=True )
def DistancePointEllipse(e0: float, e1: float, y0: float, y1: float, maxIterations: int = 100) -> float:
    if e0 < 0:
        e0 = -e0
    if e1 < 0:
        e1 = -e1
    c0 = (y0<0)
    c1 = (y1<0)
    if c0:
        y0 = -y0 
    if c1:
        y1 = -y1

    if e0 < e1:
        # Exchange roles of X and Y
        x1, x0, distance = _DistancePointEllipse(e1, e0, y1, y0, maxIterations)
    else:
        x0, x1, distance = _DistancePointEllipse(e0, e1, y0, y1, maxIterations)
    if c0:
        x0 = -x0
    if c1:
        x1 = -x1
    return x0, x1, distance

@jit(types.Tuple((float64[:],float64[:],float64[:]))(float64,float64,float64[:],float64[:],int64),nogil=True, parallel=DISTANCEPARA,cache=True )  
def DistancePointsEllipse(e0: float, e1: float, y0,y1,maxIterations):
    x0 = np.zeros(len(y0))
    x1 = np.zeros(len(y0))
    d = np.zeros(len(y0))
    for i in prange(len(y0)):
        x0i,x1i,di = DistancePointEllipse(e0,e1,y0[i],y1[i],maxIterations)
        x0[i] = x0i
        x1[i] = x1i
        d[i] = di
    return (x0,x1,d)


@jit(types.Tuple((float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64, int64),fastmath=True,cache=True )
def _DistancePointEllipsoid(e0, e1, e2, y0, y1, y2, maxIterations):
    '''
    e0>=e1>=e2>0
    y0,y1,y2>0
    '''
    if y2>0:
        if y1>0:
            if y0>0:
                z0 = y0/e0
                z1 = y1/e1
                z2 = y2/e2
                g = RobustLength3d(z0,z1,z2) - 1
                if g!= 0 :
                    r0 = e0*e0/e2/e2
                    r1 = e1*e1/e2/e2
                    sbar = GetRoot3d(r0,r1,z0,z1,z2,g, maxIterations)
                    x0 = r0*y0/(sbar+r0)
                    x1 = r1*y1/(sbar+r1)
                    x2 = y2/(sbar+1)
                    distance = RobustLength3d(x0-y0,x1-y1,x2-y2)
                else:
                    x0,x1,x2 = y0,y1,y2
                    distance = 0
            else: # y0==0
                x0=0
                x1,x2,distance = DistancePointEllipse(e1,e2,y1,y2, maxIterations)
        else: #y1==0
            x1 = 0.
            x0,x2,distance = DistancePointEllipse(e0,e2,y0,y2, maxIterations)
    else: # y2 ==0
        denom0 = e0*0-e2*e2
        denom1 = e1*e1-e2*e2
        numer0 = e0*y0
        numer1 = e1*y1
        computed = False
        if (numer0<denom0) and (numer1< denom1):
            xde0 = numer0/denom0
            xde1 = numer1/denom1
            xde0sqr = xde0*xde0
            xde1sqr = xde1*xde1
            discr = 1-xde0sqr -xde1sqr
            if discr >0:
                x0 = e0*xde0
                x1 = e1*xde1
                x2 = e2*math.sqrt(discr)
                distance = RobustLength3d(x0-y0,x1-y1,x2)
                computed = True
        if not computed:
            x2 =0
            x0,x1,distance = DistancePointEllipse(e0,e1,y0,y1, maxIterations)
    return x0,x1,x2,distance



@jit(types.Tuple((float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64, int64),cache=True)
def DistancePointEllipsoid(e0, e1, e2, y0, y1, y2, maxIterations):
    if e0 < 0:
        e0 = -e0
    if e1 < 0:
        e1 = -e1
    if e2 < 0:
        e2 = -e2
    c0 = (y0<0)
    c1 = (y1<0)
    c2 = (y2<0)
    if c0:
        y0 = -y0 
    if c1:
        y1 = -y1
    if c2:
        y2 = -y2 
    sor = sorted(zip([e0,e1,e2],[y0,y1,y2],[1,2,3]))
    
    x0,x1,x2,distance = _DistancePointEllipsoid(sor[0][0],sor[1][0],sor[2][0],
                                               sor[0][1],sor[1][1],sor[2][1],maxIterations)
    res = sorted(zip([sor[0][2],sor[1][2],sor[2][2]],[x0,x1,x2]))
    x0 = -res[0][1] if c0 else res[0][1]
    x1 = -res[1][1] if c1 else res[1][1]
    x2 = -res[2][1] if c2 else res[2][1]
    return x0,x1,x2,distance

@jit(types.Tuple((float64[:],float64[:],float64[:],float64[:]))
     (float64, float64, float64, float64[:], float64[:], float64[:], int64),nogil=True,parallel=DISTANCEPARA ,cache=True )
def DistancePointsEllipsoid(e0, e1, e2, y0, y1, y2, maxIterations):
    x0 = np.zeros(len(y0))
    x1 = np.zeros(len(y0))
    x2 = np.zeros(len(y0))
    d = np.zeros(len(y0))
    for i in prange(len(y0)):
        x0i,x1i,x2i, di = DistancePointEllipsoid(e0, e1, e2, y0[i], y1[i], y2[i], maxIterations)
        x0[i] = x0i
        x1[i] = x1i
        x2[i] = x2i
        d[i] = di
    return (x0,x1,x2,d)


@jit(types.Tuple((float64,float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64),fastmath=True,)
def DistanceRayPointEllipsoid(a, b, c, x, y, z):
    # a >= b >= c > 0, 
    L = RobustLength3d(x,y,z)   #x, y, z,
    xi = x/L
    yi = y/L
    zi = z/L
    
    d = math.sqrt(1/((xi/a)**2+(yi/b)**2+(zi/c)**2))
    return xi*d,yi*d,zi*d,d,L
    
@jit(types.Tuple((float64[:,:],float64[:]))(float64, float64, float64, float64[:,:]),
     nogil=True,parallel=DISTANCEPARA,fastmath=True,cache=True)
def DistanceRayPointsEllipsoid(a, b, c, pos):

    tarpos = np.zeros((len(pos),3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i,0],tarpos[i,1],tarpos[i,2],d[i], L[i] = DistanceRayPointEllipsoid(a,b,c,pos[i,0],pos[i,1],pos[i,2])
    return tarpos,L-d


@jit(float64(float64,float64,float64,float64,float64,float64,float64),fastmath=True,cache=True)
def _iter_f_DistanceRayPointEllipsoid_S(d,Sa,Sb,Sc,Ex,Ey,Ez):
    # g= f* 
    dd = d*d
    ExddSa = Ex*dd**Sa
    EyddSb = Ey*dd**Sb
    EzddSc = Ez*dd**Sc
    f = ExddSa + EyddSb + EzddSc - 1
    df = 2*(Sa*ExddSa+Sb*EyddSb+Sc*EzddSc)/d
    return -f/df

@jit(types.Tuple((float64,float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64,
    float64, float64, float64,int32),fastmath=True,cache=True)
def DistanceRayPointEllipsoid_S(a, b, c, Sa, Sb, Sc, x, y, z, maxIterations: int):
    # a >= b >= c > 0, 
    L = RobustLength3d(x,y,z)   #x, y, z,
    xi = x/L
    yi = y/L
    zi = z/L
    # dmin = c, dmax = a
    Ex = ((xi/a)**2)**Sa
    Ey = ((yi/b)**2)**Sb
    Ez = ((zi/c)**2)**Sc
    
    epsilon = 1e-9      # 1e-9 is ok ?
    i = 0
    d0 = (a+c)/2
    while True:
        if i > maxIterations:
            delta = _iter_f_DistanceRayPointEllipsoid_S(d0,Sa,Sb,Sc,Ex,Ey,Ez)
            print("maximum number of iterations exceeded!, next iter: ",delta)
            break
        d1 = d0 + _iter_f_DistanceRayPointEllipsoid_S(d0,Sa,Sb,Sc,Ex,Ey,Ez)
        if abs(d1-d0)<epsilon:
            break
        d0 = d1
        i = i + 1
    return d0*xi,d0*yi,d0*zi,d0,L

@jit(types.Tuple((float64[:,:],float64[:]))(float64, float64, float64,float64, float64, float64, float64[:,:],int32),
     nogil=True,parallel=DISTANCEPARA,fastmath=True,cache=True)
def DistanceRayPointsEllipsoid_S(a, b, c, Sa, Sb, Sc, pos, maxIterations):
    tarpos = np.zeros((len(pos),3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i,0],tarpos[i,1],tarpos[i,2],d[i], L[i] = DistanceRayPointEllipsoid_S(
                                                a,b,c,Sa,Sb,Sc,
                                                pos[i,0],pos[i,1],pos[i,2],maxIterations)
    return tarpos,L-d


@jit(float64[:](float64,float64,float64,float64[:],float64[:]),fastmath=True,)
def IntersectLineEllipsoid(a,b,c,pos1,vect):

    ts = -np.ones(2,dtype=np.float64)
    a2,b2,c2 = a**2,b**2,c**2
    ell11 = vect[0]**2/a2 + vect[1]**2/b2 + vect[2]**2/c2
    ell01 = vect[0]*pos1[0]/a2 + vect[1]*pos1[1]/b2 + vect[2]*pos1[2]/c2
    ell00 = pos1[0]**2/a2 + pos1[1]**2/b2 + pos1[2]**2/c2
    
    D = ell01**2 - ell11*(ell00-1)
    if D>0:
        ts[0] = (-ell01-math.sqrt(D))/ell11
        ts[1] = (-ell01+math.sqrt(D))/ell11
        return ts
    if D==0:
        ts[0] = (-ell01)/ell11
        ts[1] = ts[0]
        return ts
    
    return ts

@jit(float64[:,:](float64,float64,float64,float64[:,:],float64[:,:]),nogil=True,parallel=DISTANCEPARA,fastmath=True,cache=True)
def IntersectLinesEllipsoid(a,b,c,pos1,pos2):
    vects = unit_vector3d(pos2 - pos1)
    ts = np.ones((len(pos1),2),dtype=np.float64)
    
    for i in prange(len(ts)):
        ts[i] = IntersectLineEllipsoid(a,b,c,pos1[i],vects[i])
    
    return ts

@jit(types.Tuple((float64,float64))(float64[:],float64[:],float64,float64,float64,float64,float64,float64,float64,int32,float64,float64),fastmath=True,cache=True)
def _iter_IntersectLineEllipsoid_S(pos1,vect,t0,a,b,c,Sa,Sb,Sc,maxIteration,epsilon,delta_cut):
    i=0
    posi = pos1 + t0*vect
    while True:
        Ex,Ey,Ez = ((posi[0]/a)**2)**Sa,((posi[1]/b)**2)**Sb,((posi[2]/c)**2)**Sc
        f = Ex+Ey+Ez - 1
        
        if abs(f)<epsilon:       # this position is already at the target positon
            break
        
        df_x = Ex*Sa*vect[0]/posi[0] if posi[0]!=0 else 0.
        df_y = Ey*Sb*vect[1]/posi[1] if posi[1]!=0 else 0.
        df_z = Ez*Sc*vect[2]/posi[2] if posi[2]!=0 else 0.
        df = 4*(df_x+df_y+df_z)

        delta = -f/df
        delta = min(delta_cut,delta)        # avoid large update
        delta = max(-delta_cut,delta)
        t0 = t0+ delta
        posi = pos1 + t0*vect
        i = i+1
        
        if abs(delta)<epsilon:
            break
        if i >maxIteration:
            break
    Ex,Ey,Ez = ((posi[0]/a)**2)**Sa,((posi[1]/b)**2)**Sb,((posi[2]/c)**2)**Sc
    f = Ex+Ey+Ez - 1
    return t0,f

@jit(float64[:](float64,float64,float64,float64,float64,float64,float64[:],float64[:],float64,int32),fastmath=True,)
def IntersectLineEllipsoid_S(a,b,c,Sa,Sb,Sc,pos1,vect,tmax,maxIteration):
    
    delta_cut = abs(10*tmax/maxIteration)
    ts = -np.ones(2)
    epsilon = 1e-9                      # 1e-9 is ok ?
    
    # From both ends of the line segment  
    t0,f0 = _iter_IntersectLineEllipsoid_S(pos1,vect,epsilon,a,b,c,Sa,Sb,Sc,maxIteration,epsilon,delta_cut)
    t1,f1 = _iter_IntersectLineEllipsoid_S(pos1,vect,tmax,a,b,c,Sa,Sb,Sc,maxIteration,epsilon,delta_cut)
    
    if (abs(t0-t1)<=10*epsilon):                            # converge at the same positon, maybe one zero position, or no zero position
        if (abs(f0)>=10*epsilon) and (abs(f1)>=10*epsilon):
            return ts
        else:
            ts[0] = t0
            ts[1] = t1
            return ts
        
    if (abs(f0)<=10*epsilon) and (abs(f1)<=10*epsilon):     # converge at two positons, two zero positions
        ts[0] = t0
        ts[1] = t1
        return ts

    if (abs(f0)<=10*epsilon) or (abs(f1)<=10*epsilon):                 # one is not at the target position, will expore t0 -> t1
        ti_min = t0
        ti_max = t1
        delta_cut = abs(10*(t1-t0)/maxIteration)
        while True:
            if (abs(f0)>10*epsilon):
                t2,f2 = _iter_IntersectLineEllipsoid_S(pos1,vect,ti_min+(ti_max-ti_min)/3,a,b,c,Sa,Sb,Sc,maxIteration,epsilon,delta_cut)    
            else:
                t2,f2 = _iter_IntersectLineEllipsoid_S(pos1,vect,ti_min+2*(ti_max-ti_min)/3,a,b,c,Sa,Sb,Sc,maxIteration,epsilon,delta_cut)
                
            if (abs(t2-ti_min)>=10*epsilon) and (abs(t2-ti_max) >= 10*epsilon) and (abs(f2)<=10*epsilon):
                break
            
            if (abs(f0)>10*epsilon):
                ti_min = ti_min+(ti_max-ti_min)/3
            else:
                ti_max = ti_min+2*(ti_max-ti_min)/3
                
            if abs(ti_min-ti_max)<10*epsilon:
                break
            
        if (abs(f0)<=10*epsilon):
            ts[0] = t0
            ts[1] = t2
            return ts
        ts[0] = t2
        ts[1] = t1
        return ts
    
    return ts

@jit(float64[:,:](float64,float64,float64,float64,float64,float64,float64[:,:],float64[:,:],int32),nogil=True,parallel=DISTANCEPARA,fastmath=True,cache=True)
def IntersectLinesEllipsoid_S(a,b,c,Sa,Sb,Sc,pos1,pos2,maxIteration):
    vects = unit_vector3d(pos2 - pos1)
    tmax = vector_length3d(pos2 - pos1)
    
    ts = np.ones((len(pos1),2),dtype=np.float64)
    for i in prange(len(ts)):
        ts[i] = IntersectLineEllipsoid_S(a,b,c,Sa,Sb,Sc,pos1[i],vects[i],tmax[i],maxIteration)
    
    return ts

if __name__ == "__main__":
    pass



    