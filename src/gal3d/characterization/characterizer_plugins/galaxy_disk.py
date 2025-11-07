"""
Module for measuring galaxy disk parameters using ellipse/ellipsoid fitting results.
"""

from typing import Any

import numpy as np

from gal3d.characterization.characterizer import CharacterizerBase
from gal3d.field.spherical_field.ray.lu_mono_cy import MyPchipInterpolator as PchipInterpolator
from gal3d.optimization.result import ModelResult

__all__ = ["Disk"]

class Disk(CharacterizerBase):
    """
    GalaxyDisk
    ----------
    Characterizer that prepares a monotonic interpolator of a disk-fraction-like quantity
    (e.g. epsilon) as a function of radius (a).
    This class accepts input data (a dict of 1D numpy arrays or a ModelResult with dict-like access),
    validates and sorts the arrays by ascending 'a', and builds a PCHIP interpolator for the chosen
    eps-like quantity. The interpolator is created with extrapolation enabled.

    Attributes
    ----------
    data : dict[str, numpy.ndarray]
        Input arrays sorted by ascending `a`.
    a : numpy.ndarray
        Sorted 1D array of semi-major axis / radius values used as the interpolator abscissa.
    eps : numpy.ndarray
        Sorted 1D array of the eps-like quantity corresponding to `a`.
    """


    def __init__(self, data: dict[str, np.ndarray] | ModelResult, use_key: str = "eps_ac"):
        super().__init__(data)

        dex = np.argsort(data["a"])
        self.data={i: data[i][dex] for i in data}

        self.a = self.data["a"]
        if data.get(use_key) is not None:
            self.data[use_key] = data[use_key][dex]
            self.eps = self.data[use_key]
        else:
            raise KeyError(f"Key '{use_key}' is missing from the data dictionary.")

        assert len(self.a) == len(self.eps), "Inconsistent array lengths."
        assert len(self.a) > 0, "Empty array detected."
        assert len(self.a) > 5, "Insufficient data points."
        assert np.all(np.isfinite(self.a)), "Non-finite values detected in 'a'."
        assert np.all(np.isfinite(self.eps)), "Non-finite values detected in 'eps'."

        self._f_eps_R = PchipInterpolator(self.a, self.eps, extrapolate=True)


    def measure(self,
        eps_cond: float = 0.5,
        range_min: float = 0.2,
        min_n: int = 3,
        detail: bool = False,
        other_keys: list[str] | str | None = None
        ) -> dict:

        """
        Find candidate disk regions where eps >= eps_cond and summarize each region.

        Parameters
        ----------
        eps_cond : float
            Ellipticity threshold for disk detection.
        range_min : float
            Minimum radial extent of a valid region.
        min_n : int
            Minimum number of data points in a valid region.
        detail : bool
            If True, include extra fields per region (n_points, eps_in, eps_ou).
        other_keys : str | list[str] | None
            Additional data keys to interpolate at R_in, R_ou, and R_max for each region.

        Returns
        -------
        dict
            {
              "flag": 1 if any region found else 0,
              "n_regions": number of regions,
              "eps_cond": threshold used,
              "regions": [
                 {
                   "R_in": ...,
                   "R_ou": ...,
                   "length": ...,
                   "eps_max": ...,
                   "R_max": ...,
                   [if detail] "n_points": ...,
                   [if detail] "eps_in": ...,
                   [if detail] "eps_ou": ...,
                   [if other_keys] "R_in_<key>": ..., "R_ou_<key>": ..., "R_max_<key>": ...
                 },
                 ...
              ]
            }
        """

        # 1) find contiguous regions with eps >= threshold
        R_start, R_end = self.select_region(eps_cond=eps_cond)
        # 2) filter by length and min number of samples
        R_start, R_end = self.filter_region_length(R_start, R_end, range_min=range_min, min_n=min_n)

        if R_start.size == 0:
            return {"flag": 0, "n_regions": 0, "eps_cond": float(eps_cond), "regions": []}

        # prepare interpolators for requested other_keys
        key_list: list[str] = []
        key_interps: dict[str, PchipInterpolator] = {}
        if other_keys is not None:
            if isinstance(other_keys, str):
                key_list = [other_keys]
            elif isinstance(other_keys, list):
                key_list = other_keys
            else:
                raise TypeError("Parameter 'other_keys' must be a string, list, or None.")
            for k in key_list:
                key_interps[k] = PchipInterpolator(self.a, self.data[k], extrapolate=False)

        regions: list[dict[str, Any]] = []
        f_eps_R = self._f_eps_R

        for s, e in zip(R_start, R_end, strict=False):
            sel = (self.a >= s) & (self.a <= e)
            a_seg = self.a[sel]
            eps_seg = self.eps[sel]
            if a_seg.size == 0:
                continue

            # per-region maxima within the segment
            imax = int(np.argmax(eps_seg))
            eps_max = float(eps_seg[imax])
            R_max = float(a_seg[imax])

            reg: dict[str, Any] = {
                "R_in": float(s),
                "R_ou": float(e),
                "length": float(e - s),
                "eps_max": eps_max,
                "R_max": R_max,
            }

            if detail:
                reg["n_points"] = int(a_seg.size)
                reg["eps_in"] = float(f_eps_R(s))
                reg["eps_ou"] = float(f_eps_R(e))

            # interpolate additional keys at characteristic radii
            for k, f in key_interps.items():
                reg[f"R_in_{k}"] = float(f(s))
                reg[f"R_ou_{k}"] = float(f(e))
                reg[f"R_max_{k}"] = float(f(R_max))

            regions.append(reg)

        return {
            "flag": 1 if len(regions) > 0 else 0,
            "n_regions": len(regions),
            "eps_cond": float(eps_cond),
            "regions": regions,
        }


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
        m = self.eps >= eps_cond
        if not np.any(m):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Find the start points (False->True) and end points (True->False)
        dm = np.diff(m.astype(np.int8))
        starts_idx = np.where(dm == 1)[0] + 1
        ends_idx = np.where(dm == -1)[0]

        # if the first element is True, we have a start at index 0
        if m[0]:
            starts_idx = np.r_[0, starts_idx]
        # if the last element is True, we have an end at the last index
        if m[-1]:
            ends_idx = np.r_[ends_idx, len(m) - 1]

        R_start_arr = self.a[starts_idx].astype(np.float64)
        if starts_idx.size:
            mask0 = (starts_idx == 0)
            if np.any(mask0):
                R_start_arr[mask0] = 0.0
        R_end_arr = self.a[ends_idx].astype(np.float64)
        return R_start_arr, R_end_arr


    def filter_region_length(
        self, R_start: np.ndarray, R_end: np.ndarray, range_min: float = 0.3, min_n : int = 3,
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
        min_n : int, optional
            Minimum number of data points in the region (default: 3)

        Returns
        -------
        R_start : ndarray
            Filtered starting radii
        R_end : ndarray
            Filtered ending radii
        """
        if R_start.size == 0 or R_end.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # ensure proper shape and ordering
        R_start = np.asarray(R_start, dtype=float).ravel()
        R_end = np.asarray(R_end, dtype=float).ravel()
        valid = R_end > R_start
        R_start = R_start[valid]
        R_end = R_end[valid]

        if R_start.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        sel_range = R_end - R_start > range_min
        # filter by minimum number of samples within each region
        keep = np.zeros(R_start.shape, dtype=bool)
        for i, (s, e) in enumerate(zip(R_start, R_end, strict=False)):
            n = int(np.count_nonzero((self.a >= s) & (self.a <= e)))
            keep[i] = n >= int(min_n)


        R_start = R_start[sel_range&keep]
        R_end = R_end[sel_range&keep]

        return R_start, R_end


