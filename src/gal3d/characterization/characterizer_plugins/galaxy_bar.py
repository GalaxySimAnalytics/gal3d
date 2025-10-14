"""
Module for measuring galaxy bar parameters using ellipse/ellipsoid fitting results.
"""
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from gal3d.characterization.characterizer import CharacterizerBase
from gal3d.optimization.result import ModelResult
from gal3d.shape.coordinate_plugins.euler_shift import EulerAngles

__all__ = ["Bar"]
class Bar(CharacterizerBase):
    def __init__(self, data: dict[str, np.ndarray] | ModelResult, angle_clip: float | None = 3.0):
        """
        Class for measuring galaxy bar parameters using ellipse/ellipsoid fitting results.

        Parameters
        ----------
        data: dict-like,
            Object supporting dict-like access (e.g., dict, pandas.DataFrame, custom mapping).
            Must provide keys:
                - 'a' : array_like
                    Semi-major axis values (1D array)
                - 'eps' or 'eps_ab' : array_like
                    Ellipticity values (0-1, 1D array)
                - 'pa' or 'angle' : array_like
                    Position angle values in radians (0-pi or 2pi, 1D array)
        angle_clip : float, optional
            Clip value for filtering outliers in the angle (default: 3.0)

        Attributes
        ----------
        a : ndarray
            Stored semi-major axis array
        eps : ndarray
            Stored ellipticity array
        pa : ndarray
            Stored position angle array
        """
        super().__init__(data)
        dex = np.argsort(data["a"])
        self.data={i: data[i][dex] for i in data.keys()}

        self.a = self.data["a"]

        if data.get("eps") is not None:
            self.data["eps"] = data["eps"][dex]
            self.eps = self.data["eps"]
        elif data.get("eps_ab") is not None:
            self.data["eps_ab"] = data["eps_ab"][dex]
            self.eps = self.data["eps_ab"]
        else:
            raise KeyError("Both 'eps' and 'eps_ab' are missing from the data dictionary.")
        if data.get("pa") is not None:
            self.data["pa"] = data["pa"][dex]
            self.pa = self.data["pa"]*180/np.pi         # radians to degrees
        elif data.get("angle") is not None:
            self.data["angle"] = data["angle"][dex]
            rota = EulerAngles.from_euler(seq="zyx",angles=self.data["angle"])
            self.pa = rota.magnitude()*180/np.pi
        else:
            raise KeyError("Both 'pa' and 'angle' are missing from the data dictionary.")

        assert len(self.a) == len(self.eps) == len(self.pa), "Inconsistent array lengths."
        assert len(self.a) > 0, "Empty array detected."
        assert len(self.a) > 5, "Insufficient data points."
        assert np.all(np.isfinite(self.a)), "Non-finite values detected in 'a'."
        assert np.all(np.isfinite(self.eps)), "Non-finite values detected in 'eps'."
        assert np.all(np.isfinite(self.pa)), "Non-finite values detected in 'pa'."

        if angle_clip is not None:
            self.filter_outlier_data_points(clip=angle_clip)

        self._f_eps_R = PchipInterpolator(self.a, self.eps, extrapolate=True)

    def measure(
        self,
        eps_cond: float = 0.25,
        range_min: float = 0.2,
        start_max: float = 3,
        angle_dev: float = 10,
        dec: float = 0.85,
        detail: bool = False,
        other_keys: list[str] | str | None = None
    ) -> dict[str, Any]:
        """
        Measure galaxy bar parameters using ellipticity profile analysis.

        Parameters
        ----------
        eps_cond : float, optional
            Ellipticity threshold for bar detection (default: 0.25)
        range_min : float, optional
            Minimum required length of bar region (default: 0.2)
        start_max : float, optional
            Maximum allowed starting radius (default: 3)
        angle_dev : float, optional
            Maximum allowed position angle deviation in degrees (default: 10)
        dec : float, optional
            Ellipticity decrease factor for bar length determination (default: 0.85)
        detail: bool, optional
            the return dict is more detailed. (default: False)
        other_keys:
            other property in data, need to measure at some correspond radii.

        Returns
        -------
        result : dict
            Dictionary containing bar parameters.

            flag : int
                1 if bar detected, 0 otherwise.

            eps_max : float
                Maximum ellipticity in bar region.

            R_max : float
                Radius at maximum ellipticity.

            R_bar : float
                Bar radius.

            R_in : float, optional
                Start radius where eps >= eps_cond (if detail=True).

            R_ou : float, optional
                End radius where eps >= eps_cond (if detail=True).

            R_dc : float, optional
                Where eps decreases to dec*eps_max (if detail=True).

            eps_dc : float, optional
                Ellipticity at R_dc (if detail=True).

            R_pa : float, optional
                Radius where position angle deviation begins to exceed angle_dev (if detail=True).

            eps_pa : float, optional
                Ellipticity at R_pa (if detail=True).

        Notes
        -----
        Bar detection criteria:

        1. Starting radius (Rin) < start_max
        2. Bar region length (Rou-Rin) > range_min
        3. Peak ellipticity > eps_cond
        4. Position angle stability within angle_dev
        """

        R_start, R_end = self.select_region(eps_cond=eps_cond)
        R_start, R_end = self.filter_region_length(
            R_start, R_end, start_max=start_max, range_min=range_min
        )
        if R_start.size == 0 or R_end.size == 0:
            result = {
                "flag": 0,
                "eps_max": np.max(self.eps),
                "R_max": self.a[np.argmax(self.eps)],
                "R_bar": 0.0,
            }

        R_start_val: float = float(R_start[0]) if R_end.size > 0 else 0.0
        R_end_val: float = float(R_end[0]) if R_end.size > 0 else 0.0

        eps_max, R_max, pa_max = self.get_max_epsRpa(R_start_val, R_end_val, R_cond=start_max)

        eps_dc, R_dc, f_eps_R = self.get_dec_epsRpa(R_max, eps_max, dec=dec)
        eps_pa, R_pa = self.get_dev_epsRpa(R_max, angle_cond=angle_dev)

        bar_flag = 0
        if (R_start_val > 0) and (eps_max > eps_cond) and ((R_pa - R_max) > range_min):
            bar_flag = 1
        result = {"flag": bar_flag, "eps_max":eps_max, "R_max": R_max, "R_bar": min(R_dc,R_pa)}

        if other_keys and not isinstance(other_keys, str | list | type(None)):
            raise TypeError("Parameter 'other_keys' must be a string, list, or None.")
        update: dict[str, float] = {
            "R_in": R_start_val,
            "R_ou": R_end_val,
            "eps_dc": eps_dc,
            "R_dc": R_dc,
            "eps_pa": eps_pa,
            "R_pa": R_pa,
        }

        if other_keys:
            if isinstance(other_keys,str):
                other_keys = [other_keys]
            for i in other_keys:
                f_r = PchipInterpolator(self.a, self.data[i],extrapolate=False)
                result[f"R_max_{i}"] = f_r(result["R_max"])
                result[f"R_bar_{i}"] = f_r(result["R_bar"])
                if detail:
                    update[f"R_in_{i}"] = f_r(update["R_in"])
                    update[f"R_ou_{i}"] = f_r(update["R_ou"])
                    update[f"R_dc_{i}"] = f_r(update["R_dc"])
                    update[f"R_pa_{i}"] = f_r(update["R_pa"])

        if not detail:
            return result

        result.update(update)
        return result

    def select_region(
        self,
        eps_cond: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify regions with ellipticity above threshold.

        Parameters
        ----------
        eps_cond : float, optional
            Ellipticity threshold (default: 0.25)

        Returns
        -------
        R_start : ndarray
            Starting radii of high-ellipticity regions
        R_end : ndarray
            Ending radii of high-ellipticity regions
        """
        begin_flag = True
        R_start: list[float] = []
        R_end: list[float] = []
        for i in range(len(self.eps)):
            if self.eps[i] >= eps_cond:
                if begin_flag:
                    R_st = self.a[i]
                    begin_flag = False
            elif not begin_flag:
                R_ed = self.a[i - 1]
                begin_flag = True
                R_start.append(R_st)
                R_end.append(R_ed)
            if (i == len(self.eps) - 1) and (not begin_flag):
                R_start.append(R_st)
                R_end.append(self.a[i])
        R_start_arr = np.array(R_start, dtype=np.float64)
        R_end_arr = np.array(R_end, dtype=np.float64)
        return R_start_arr, R_end_arr

    def filter_region_length(
        self, R_start: np.ndarray, R_end: np.ndarray, start_max: float = 3, range_min: float = 0.3
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter regions based on spatial constraints.

        Parameters
        ----------
        R_start : array_like
            Region starting radii
        R_end : array_like
            Region ending radii
        start_max : float, optional
            Maximum allowed starting radius (default: 3)
        range_min : float, optional
            Minimum required region length (default: 0.3)

        Returns
        -------
        R_start : ndarray
            Filtered starting radii
        R_end : ndarray
            Filtered ending radii
        """
        sel_start = R_start < start_max
        R_start = R_start[sel_start]
        R_end = R_end[sel_start]
        sel_range = R_end - R_start > range_min
        R_start = R_start[sel_range]
        R_end = R_end[sel_range]

        return R_start, R_end

    def filter_outlier_data_points(self, n: int | None = None, clip: float = 3.0) -> None:
        """
        Remove data points with large position angle jumps.

        This method calculates the mean and standard deviation of the position angle differences
        between each data point and its neighbors. Data points with mean deviation greater than
        `clip` times the standard deviation are filtered out.

        Parameters
        ----------
        n : int, optional
            Number of neighbors to consider on each side (default: len(self.a)//40, minimum 1).
        clip : float, optional
            Threshold multiplier for filtering outliers (default: 3.0).

        """
        n = max(int(len(self.a)/40),1) if n is None else n
        mean_dev = np.zeros(len(self.a))
        mean_std = np.zeros(len(self.a))
        for i in range(len(self.pa)):
            neighbors = []
            # left
            for k in range(1, n+1):
                if i - k >= 0:
                    neighbors.append(Bar.inter_angle(self.pa[i], self.pa[i-k]))
            # right
            for k in range(1, n+1):
                if i + k < len(self.pa):
                    neighbors.append(Bar.inter_angle(self.pa[i], self.pa[i+k]))
            # If one side is insufficient, supplement the other side
            while len(neighbors) < 2*n:
                # Preferentially supplement the left side
                if i - (n + len(neighbors) - n) >= 0:
                    neighbors.append(Bar.inter_angle(self.pa[i], self.pa[i - (n + len(neighbors) - n)]))
                # Then supplement the right side
                elif i + (n + len(neighbors) - n) < len(self.pa):
                    neighbors.append(Bar.inter_angle(self.pa[i], self.pa[i + (n + len(neighbors) - n)]))
                else:
                    break
            mean_dev[i] = np.mean(neighbors)
            mean_std[i] = np.std(neighbors)

        select = mean_dev <  clip * mean_std
        self.a = self.a[select]
        self.eps = self.eps[select]
        self.pa = self.pa[select]
        data_sel = {i:self.data[i][select] for i in self.data.keys()}
        self.data = data_sel
        self.pa_dev_mean = mean_dev
        self.pa_dev_std = mean_std

    def get_max_epsRpa(self, R_start: float, R_end: float, R_cond: float =3) -> tuple[float, float, float]:
        """
        Find maximum ellipticity and corresponding parameters in a region.

        Parameters
        ----------
        R_start : float
            Region start radius
        R_end : float
            Region end radius
        R_cond : float, optional
            Fallback search radius if Rstart=Rend (default: 3)

        Returns
        -------
        epsmax : float
            Maximum ellipticity
        Rmax : float
            Radius at maximum ellipticity
        pamax : float
            Position angle at maximum ellipticity
        """
        if R_start != R_end:
            range_cut = (self.a >= R_start) & (self.a <= R_end)
            if np.any(range_cut):
                eps_max = float(np.max(self.eps[range_cut]))
                R_max = float(self.a[range_cut][np.argmax(self.eps[range_cut])])
                pa_max = float(self.pa[range_cut][np.argmax(self.eps[range_cut])])
            else:
                eps_max, R_max, pa_max = 0.0, 0.0, 0.0
        else:
            mask = self.a < R_cond
            if np.any(mask):
                eps_max = float(np.max(self.eps[mask]))
                idx = np.argmax(self.eps[mask])
                R_max = float(self.a[mask][idx])
                pa_max = float(self.pa[mask][idx])
            else:
                eps_max, R_max, pa_max = 0.0, 0.0, 0.0
        return eps_max, R_max, pa_max

    def get_dec_epsRpa(self, R_max: float, eps_max: float, dec: float = 0.85) -> tuple[float, float, PchipInterpolator]:
        """
        Determine bar length using ellipticity decrease criterion.

        Parameters
        ----------
        R_max : float
            Radius of maximum ellipticity
        eps_max : float
            Maximum ellipticity value
        dec : float, optional
            Decrease factor (default: 0.85)

        Returns
        -------
        eps_dc : float
            Ellipticity at bar length
        R_dc : float
            Bar length radius
        f_eps_R : PchipInterpolator
            Ellipticity interpolator
        """

        f_eps_R = self._f_eps_R
        eps_dc = eps_max * dec
        range_cut = self.a >= eps_max
        R_dc = np.array(f_eps_R.solve(eps_dc, discontinuity=False, extrapolate=False))
        R_dc_val: float
        if R_dc[R_dc > R_max].size == 0:
            eps_dc = float(self.eps[range_cut][-1])
            R_dc_val = float(self.a[range_cut][-1])
        else:
            R_dc_val = float(R_dc[R_dc > R_max][0])
        return eps_dc, R_dc_val, f_eps_R

    def get_dev_epsRpa(self, R_max: float, angle_cond: float = 10) -> tuple[float, float]:
        """
        Find position angle deviation point.

        Parameters
        ----------
        R_max : float
            Radius of maximum ellipticity
        angle_cond : float, optional
            Position angle deviation threshold (default: 10 degrees)

        Returns
        -------
        eps_pa : float
            Ellipticity at deviation point
        R_pa : float
            Radius where angle deviation exceeds threshold
        """
        find_out = False
        sel_a = self.a >= R_max
        if len(self.a[sel_a]) == 1:
            R_pa = float(self.a[sel_a][0])
            eps_pa = float(self.eps[sel_a][0])
            return eps_pa, R_pa
        for i in range(len(self.a[sel_a])):
            for j in range(i):
                R_pa = float(self.a[sel_a][i])
                eps_pa = float(self.eps[sel_a][i])
                if Bar.inter_angle(self.pa[sel_a][i], self.pa[sel_a][j]) > angle_cond:
                    find_out = True
                    break
            if find_out:
                break
        return eps_pa, R_pa

    @staticmethod
    def distance_PBC_1d(a1: float, a2: float, b: float) -> float:
        """
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
        """
        d = abs(a2 - a1)
        d = d % b
        return min(d, b - d)

    @staticmethod
    def inter_angle(a1: float, a2: float) -> float:
        """
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
        """
        return Bar.distance_PBC_1d(a1, a2, b=180)
