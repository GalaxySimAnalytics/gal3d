

import numpy as np
from scipy.special import sph_harm




def spherical_harmonics_in_real(phi,theta,m,l) -> float:
    '''
    calculate spherical harmonics in real
    '''
    
    if m<0:
        return (-1)**m*np.sqrt(2)*np.imag(sph_harm(m,l,phi,theta))
    elif m>0:
        return (-1)**m*np.sqrt(2)*np.real(sph_harm(m,l,phi,theta))
    else:
        return np.real(sph_harm(m,l,phi,theta))
                            

def spherical_harmonics_dec(theta,phi,density,lmax=4) -> dict:
    '''
    theta 0 ~ pi
    phi 0 ~ 2pi
    
    spherical harmonics decomposition
    '''
    
    coef={}
    for l in range(lmax+1):
        coef[l]=[]
        for m in np.linspace(l,-l,2*l+1):
            coef[l].append(np.sum(density*spherical_harmonics_in_real(phi,theta,m,l)*np.sin(theta)))
        coef[l] = np.array(coef[l])
    return coef