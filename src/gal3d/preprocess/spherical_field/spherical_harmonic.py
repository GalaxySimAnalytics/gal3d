

import numpy as np
from scipy.special import sph_harm




def spherical_harmonics_in_real(phi,theta,m,l) -> float:
    '''
    Calculate the real part of spherical harmonics for given angles and indices.

    Parameters
    ----------
    phi : float or np.ndarray
        Azimuthal angle (longitude) in radians, ranging from 0 to 2π.
    theta : float or np.ndarray
        Polar angle (colatitude) in radians, ranging from 0 to π.
    m : int
        Order of the spherical harmonic (integer, -l <= m <= l).
    l : int
        Degree of the spherical harmonic (integer, l >= 0).

    Returns
    -------
    float or np.ndarray
        The real part of the spherical harmonic Y_l^m(phi, theta).

    Notes
    -----
    The function computes the real part of the spherical harmonic using the relation:
    - For m < 0: Y_l^m = (-1)^m * sqrt(2) * imag(Y_l^{|m|})
    - For m > 0: Y_l^m = (-1)^m * sqrt(2) * real(Y_l^{|m|})
    - For m = 0: Y_l^0 = real(Y_l^0)
    '''
    
    if m<0:
        return (-1)**m*np.sqrt(2)*np.imag(sph_harm(m,l,phi,theta))
    elif m>0:
        return (-1)**m*np.sqrt(2)*np.real(sph_harm(m,l,phi,theta))
    else:
        return np.real(sph_harm(m,l,phi,theta))
                            

def spherical_harmonics_dec(theta,phi,density,lmax=4) -> dict:
    '''
    Perform spherical harmonics decomposition on a given density distribution.

    Parameters
    ----------
    theta : np.ndarray
        Polar angles (colatitude) in radians, ranging from 0 to π.
    phi : np.ndarray
        Azimuthal angles (longitude) in radians, ranging from 0 to 2π.
    density : np.ndarray
        Density distribution on the sphere, corresponding to the given theta and phi.
    lmax : int, optional
        Maximum degree of spherical harmonics to compute (default is 4).

    Returns
    -------
    dict
        A dictionary where the keys are the degrees l (from 0 to lmax), and the values
        are arrays of spherical harmonics coefficients for each order m (from -l to l).

    Notes
    -----
    The function computes the spherical harmonics coefficients by integrating the density
    distribution multiplied by the spherical harmonics over the sphere. The integration
    is approximated by a sum over the provided grid of theta and phi values.
    '''
    
    coef={}
    for l in range(lmax+1):
        coef[l]=[]
        for m in np.linspace(l,-l,2*l+1):
            coef[l].append(np.sum(density*spherical_harmonics_in_real(phi,theta,m,l)*np.sin(theta)))
        coef[l] = np.array(coef[l])
    return coef