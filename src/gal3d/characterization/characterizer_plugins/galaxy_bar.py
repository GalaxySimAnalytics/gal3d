import numpy as np
from ..characterizer import CharacterizerBase

__all__ = ['Bar']

class Bar(CharacterizerBase):
    def __init__(self, data,):
        '''
        Class for measuring galaxy bar parameters using ellipse/ellipsoid fitting results.

        Parameters
        ----------
        a : array_like
            Semi-major axis values (1D array)
        eps : array_like
            Ellipticity values (0-1, 1D array)
        pa : array_like
            Position angle values in degrees (0-180, 1D array)
        **kwargs : dict, optional
            Additional keyword arguments with arrays of same length as `a`

        Attributes
        ----------
        a : ndarray
            Stored semi-major axis array
        eps : ndarray
            Stored ellipticity array
        pa : ndarray
            Stored position angle array

        Notes
        -----
        - Automatically converts inputs to numpy arrays
        - Checks position angle range consistency (warns if <10 deg variation)
        '''
        super().__init__(data)
        
        self.a = self.data['a']
        self.eps = self.data['eps']
        self.pa = self.data['pa']

    def measure(
        self, eps_con=0.25, rangemin=0.2, startmax=3, angle_dev=10, decre=0.85
    ):
        '''
        Measure galaxy bar parameters using ellipticity profile analysis.

        Parameters
        ----------
        eps_con : float, optional
            Ellipticity threshold for bar detection (default: 0.25)
        rangemin : float, optional
            Minimum required length of bar region (default: 0.2)
        startmax : float, optional
            Maximum allowed starting radius (default: 3)
        angle_dev : float, optional
            Maximum allowed position angle deviation in degrees (default: 10)
        decre : float, optional
            Ellipticity decrease factor for bar length determination (default: 0.85)

        Returns
        -------
        result : dict
            Dictionary containing bar parameters with keys:
            - 'Rin' : float
                Start radius where eps >= eps_con
            - 'Rou' : float
                End radius where eps >= eps_con
            - 'epsmax' : float
                Maximum ellipticity in bar region
            - 'Rmax' : float
                Radius at maximum ellipticity
            - 'R' : float
                Bar length (where eps decreases to decre*epsmax)
            - 'eps' : float
                Ellipticity at R
            - 'Rpa' : float
                Radius where position angle deviates begins to > angle_dev
            - 'epspa' : float
                Ellipticity at Rpa
            - 'barflag' : int
                1 if bar detected, 0 otherwise

        Notes
        -----
        Bar detection criteria:
        1. Starting radius (Rin) < startmax
        2. Bar region length (Rou-Rin) > rangemin
        3. Peak ellipticity > eps_con
        4. Position angle stability within angle_dev
        '''

        R_start, R_end = self.select_region(epscon=eps_con)
        R_start, R_end = self.filter_region(
            R_start, R_end, startmax=startmax, rangemin=rangemin
        )
        Rstart = R_start[0] if R_end.size > 0 else 0
        Rend = R_end[0] if R_end.size > 0 else 0

        epsmax, Rmax, pamax = self.get_max_epsRpa(Rstart, Rend, Rcon=startmax)

        eps85, R85, feps_R = self.get_eps_decreR(Rmax, epsmax, decre=decre)
        epspa, Rpa = self.get_epspaR(Rmax, anglecon=angle_dev)

        barflag = 0
        if (Rstart > 0) & (epsmax > eps_con) & ((Rpa - Rmax) > rangemin):
            barflag = 1

        result = {
            'Rin': Rstart,
            'Rou': Rend,
            'epsmax': epsmax,
            'Rmax': Rmax,
            'eps': eps85,
            'R': R85,
            'epspa': epspa,
            'Rpa': Rpa,
            'barflag': barflag,
        }
        return result

    def select_region(
        self,
        epscon=0.25,
    ):
        '''
        Identify regions with ellipticity above threshold.

        Parameters
        ----------
        epscon : float, optional
            Ellipticity threshold (default: 0.25)

        Returns
        -------
        R_start : ndarray
            Starting radii of high-ellipticity regions
        R_end : ndarray
            Ending radii of high-ellipticity regions
        '''
        beginflag = True
        R_start = []
        R_end = []
        for i in range(len(self.eps)):
            if self.eps[i] >= epscon:
                if beginflag:
                    Rstart = self.a[i]
                    beginflag = False
            else:
                if not beginflag:
                    Rend = self.a[i - 1]
                    beginflag = True
                    R_start.append(Rstart)
                    R_end.append(Rend)
            if (i == len(self.eps) - 1) & (not beginflag):
                R_start.append(Rstart)
                R_end.append(self.a[i])
        R_start = np.array(R_start, dtype=np.float64)
        R_end = np.array(R_end, dtype=np.float64)
        return R_start, R_end

    def filter_region(self, R_start, R_end, startmax=3, rangemin=0.3):
        '''
        Filter regions based on spatial constraints.

        Parameters
        ----------
        R_start : array_like
            Region starting radii
        R_end : array_like
            Region ending radii
        startmax : float, optional
            Maximum allowed starting radius (default: 3)
        rangemin : float, optional
            Minimum required region length (default: 0.3)

        Returns
        -------
        R_start : ndarray
            Filtered starting radii
        R_end : ndarray
            Filtered ending radii
        '''
        selstart = R_start < startmax
        R_start = R_start[selstart]
        R_end = R_end[selstart]
        selrange = R_end - R_start > rangemin
        R_start = R_start[selrange]
        R_end = R_end[selrange]

        return R_start, R_end

    def get_max_epsRpa(self, Rstart, Rend, Rcon=3):
        '''
        Find maximum ellipticity and corresponding parameters in a region.

        Parameters
        ----------
        Rstart : float
            Region start radius
        Rend : float
            Region end radius
        Rcon : float, optional
            Fallback search radius if Rstart=Rend (default: 3)

        Returns
        -------
        epsmax : float
            Maximum ellipticity
        Rmax : float
            Radius at maximum ellipticity
        pamax : float
            Position angle at maximum ellipticity
        '''
        if Rstart != Rend:
            rangecut = (self.a >= Rstart) & (self.a <= Rend)
            epsmax = np.max(self.eps[rangecut])
            Rmax = self.a[rangecut][np.argmax(self.eps[rangecut])]
            pamax = self.pa[rangecut][np.argmax(self.eps[rangecut])]
        else:
            epsmax = np.max(self.eps[self.a < Rcon])
            Rmax = self.a[np.argmax(self.eps[self.a < Rcon])]
            pamax = self.pa[np.argmax(self.eps[self.a < Rcon])]
        return epsmax, Rmax, pamax

    def get_eps_decreR(self, Rmax, epsmax, decre=0.85):
        '''
        Determine bar length using ellipticity decrease criterion.

        Parameters
        ----------
        Rmax : float
            Radius of maximum ellipticity
        epsmax : float
            Maximum ellipticity value
        decre : float, optional
            Decrease factor (default: 0.85)

        Returns
        -------
        eps85 : float
            Ellipticity at bar length
        R85 : float
            Bar length radius
        feps_R : scipy.interpolate.PchipInterpolator
            Ellipticity interpolator
        '''
        import scipy

        feps_R = scipy.interpolate.PchipInterpolator(self.a, self.eps)
        eps85 = epsmax * decre
        rangecut = self.a >= epsmax
        R_85 = feps_R.solve(eps85, discontinuity=False, extrapolate=False)
        if R_85[R_85 > Rmax].size == 0:
            eps85 = self.eps[rangecut][-1]
            R85 = self.a[rangecut][-1]
        else:
            R85 = R_85[R_85 > Rmax][0]
        return eps85, R85, feps_R

    def get_epspaR(self, Rmax, anglecon=10):
        '''
        Find position angle deviation point.

        Parameters
        ----------
        Rmax : float
            Radius of maximum ellipticity
        anglecon : float, optional
            Position angle deviation threshold (default: 10 degrees)

        Returns
        -------
        epspa : float
            Ellipticity at deviation point
        Rpa : float
            Radius where angle deviation exceeds threshold
        '''
        findout = False
        selr = self.a >= Rmax
        if len(self.a[selr]) == 1:
            Rpa = self.a[selr][0]
            epspa = self.eps[selr][0]
            return epspa, Rpa
        for i in range(len(self.a[selr])):
            for j in range(i):
                Rpa = self.a[selr][i]
                epspa = self.eps[selr][i]
                if Bar.inter_angle(self.pa[selr][i], self.pa[selr][j]) > anglecon:
                    findout = True
                    break
            if findout:
                break
        return epspa, Rpa

    @staticmethod
    def distance_PBC_1d(a1, a2, b):
        '''
        Calculate periodic distance in 1D space.

        Parameters
        ----------
        a1 : float
            First value
        a2 : float
            Second value
        b : float
            Periodic boundary length

        Returns
        -------
        float
            Minimum distance considering periodicity
        '''
        d = abs(a2 - a1)
        d = d % b
        return min(d, b - d)

    @staticmethod
    def inter_angle(a1, a2):
        '''
        Calculate angular difference in 0-180 degree range.

        Parameters
        ----------
        a1 : float
            First angle in degrees
        a2 : float
            Second angle in degrees

        Returns
        -------
        float
            Minimum angular difference (0-90 degrees)
        '''
        return Bar.distance_PBC_1d(a1, a2, b=180)
