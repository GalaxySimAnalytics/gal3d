"""
Segment profile characterizer: multivariate piecewise-constant segmentation
===========================================================================

Principle
---------
Partition the radii-sorted multi-channel profile :math:`X \\in \\mathbb{R}^{n \times p}` into
contiguous segments so each channel is well-approximated by a constant within each segment
in a weighted least-squares sense, subject to a minimum segment size.

Notation
--------
- Radii: :math:`r_1,\\dots,r_n` (sorted).
- Data: :math:`X_{t,p}` (sample t, channel p), errors :math:`\\mathrm{err}_{t,p}`.
- Weights combine point/channel weights and uncertainty:

  .. math::
     W_{t,p} \\;=\\; \frac{w_{\\mathrm{point}}(t)\\, w_{\\mathrm{chan}}(p)}{V_{t,p}},
     \\quad V_{t,p} \\;=\\; \\mathrm{err}_{t,p}^2 + \\mathrm{scale}_p^2.

Key equations
-------------
- Weighted mean in [i:j) for channel p:

  .. math::
     \\hat{\\mu}_p(i:j) \\;=\\;
     \frac{\\sum_{t=i}^{j-1} W_{t,p} X_{t,p}}{\\sum_{t=i}^{j-1} W_{t,p}}.

- Weighted SSE of segment [i:j):

  .. math::
     \\mathrm{SSE}(i,j) \\;=\\; \\sum_{p=1}^{p}
     \\left[
       \\sum_{t=i}^{j-1} W_{t,p} X_{t,p}^2 \\;-\\;
       \frac{\\left(\\sum_{t=i}^{j-1} W_{t,p} X_{t,p}\right)^2}{\\sum_{t=i}^{j-1} W_{t,p}}
     \right].

- Optimal m-segment partition with min_size via dynamic programming:

  .. math::
     \\mathrm{dp}[m,j] \\;=\\;
     \\min_{k \\in [ (m-1)\\,\\mathrm{min\\_size},\\, j-\\mathrm{min\\_size} ]}
     \big\\{ \\mathrm{dp}[m-1,k] + \\mathrm{SSE}(k,j) \big\\}.

- Segment count (BIC-like selection):

  .. math::
     \\mathrm{Crit}(m) \\;=\\; \\mathrm{dp}[m,n] + \\lambda \\,(m\\,p)\\,\\log n.

Explanation
-----------
Prefix sums of :math:`W,\\, W X,\\, W X^2` give :math:`\\mathrm{SSE}(i,j)` in :math:`\\mathcal{O}(1)`.
Dynamic programming yields the globally optimal partition under the min_size constraint.
The BIC-like criterion balances data fit (SSE) against model complexity (segments × channels).
"""

import logging
from typing import Any, Literal

import numpy as np

from gal3d.characterization.characterizer import CharacterizerBase
from gal3d.optimization.result import ModelResult

__all__ = ["Segment"]

logger = logging.getLogger("gal3d.characterization.segment")

class Segment(CharacterizerBase):
    """
    Multivariate piecewise-constant segmentation for shape profiles.

    This characterizer partitions a radial profile into contiguous segments
    and summarizes each segment across one or more channels by weighted
    statistics (means/medians, etc.).

    Required data keys
    ------------------
    - radial_key (default "a"): the radius array.
    - Additional channel keys to segment, e.g., "eps_ab", "eps_bc", "eps_ac",
      "x_axis_angle", "z_axis_angle", etc.

    Notes
    -----
    Inputs are sorted by `radial_key` at construction time to ensure
    consistent ordering across all channels.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray] | ModelResult,
        radial_key: str = "a",
        channel_keys: tuple[str, ...] =("eps_ab", "eps_bc", "x_axis_angle", "z_axis_angle"),
        error_keys: tuple[str, ...] = ("eps_ab", "eps_bc", "x_axis_angle", "z_axis_angle"),
        ):
        # Store and validate
        super().__init__(data)
        r = data[radial_key]
        dex = np.argsort(r)
        self.data: dict[str, np.ndarray] = {k: np.asarray(data[k])[dex]
                                            for k in channel_keys}

        # Also keep sorted radii
        self.r = r[dex]

        self.err_data: dict[str, np.ndarray] = {}
        for k in error_keys:
            err_key = f"{k}_err"
            self.err_data[k] = np.asarray(data[err_key])[dex]



    def measure(
        self,
        keys: list[str] | None = None,
        selector: Literal["bic", "fixed"] = "bic",
        max_segments: int = 6,
        min_size: int = 5,
        fixed_segments: int | None = None,
        lam_scale: float = 1.0,
        *,
        scale_mode: Literal["std","range","none"] | list[str] = "std",
        with_stats: bool = True,
        weighting: Literal["uniform","linear","log"] = "uniform",
        point_weights: np.ndarray | None = None,
        channel_weights: np.ndarray | None = None,
        normalize_weights: bool = True,
        allowed_ranges: dict[str, tuple[float, float]] | None = None,
        ) -> dict[str, Any]:
        """
        Run the segmentation algorithm on the specified keys.

        Parameters
        ----------
        keys : Iterable[str] | None
            The keys to segment. If None, all channels are used.
        selector : {"bic", "fixed"}, optional
            Segment count selection strategy:
            - "bic": choose m in [1, max_segments] minimizing
              SSE + lam_scale * (m * p) * log(n),
              where p is number of channels;
            - "fixed": use `fixed_segments`.
            Default "bic".
        max_segments : int
            The maximum number of segments to create. Default 6.
        min_size : int
            Minimum number of data points per segment. Default 5.
        fixed_segments : int | None, optional
            Number of segments to use when `selector="fixed"`. If None,
            uses min(3, max_segments). Ignored for "bic".
        lam_scale : float
            The regularization strength. Only used if `selector="bic"`. Default 1.0.
        with_stats : bool
            Whether to compute statistics for each segment. Default True.
        weighting : Literal["uniform","linear","log"]
            The weighting scheme to use. Default "uniform".
        point_weights : np.ndarray or None, optional
            Custom per-point weights overriding `weighting`. Default None.
        channel_weights : np.ndarray or None, optional
            Custom per-channel weights. Default None.
        normalize_weights : bool, optional
            If True and weights are finite and sum > 0, rescale them so that
            sum(w) = n (number of samples) to stabilize the penalty scale.
            Default True.
        scale_mode : Literal["std","range","none"] | list[str]
            The scaling mode to use for the data.
            - "std": standardize each channel to zero mean, unit variance;
            - "range": scale each channel to [0, 1] range;
            - "none": no scaling.
            - If list[str], each channel is scaled according to its name.
            Default "std".
        allowed_ranges : dict[str, tuple[float, float]] | None
            The allowed ranges for each key, first key is the channel name,
            second is a tuple (min, max).
            Will be used to scale the data if scale_mode is "range".
            If not provided, uses the min/max of each channel.

        Returns
        -------
        result : dict
            A dictionary with the following entries:
            - "bkps": list[int]
                Segment end indices (exclusive) in the sorted/filtered arrays.
            - "regions": list[dict]
                One dict per segment with fields:
                - "idx": (start, end) indices in [0, n)
                - "R_in": inner radius (float)
                - "R_ou": outer radius (float)
                - "mean": dict[key -> float], weighted mean per channel
                - If `with_stats`:
                  - "median": dict[key -> float], weighted median
                  - "std": dict[key -> float], weighted population std
                  - "16th": dict[key -> float], weighted 16th percentile
                  - "84th": dict[key -> float], weighted 84th percentile
            - "r": np.ndarray
                Sorted radii after filtering (length n).
            - "keys": list[str]
                Channel names in column order of the feature matrix.

        Notes
        -----
        - The segmentation algorithm is sensitive to the choice of parameters.
        - BIC-like model selection uses p = number of channels.

        Examples
        --------
        >>> seg = Segment(data)
        >>> out = seg.measure(keys=["eps_ab", "eps_bc"], selector="bic", max_segments=5)
        >>> out["bkps"]
        [23, 61, 120]
        """
        # 1) Channel selection and ranges
        r, series_dict, series_ranges = self._prepare_series(
            keys=keys,
            allowed_ranges=allowed_ranges,
        )

        # 2) Preprocess
        names, n_points, n_features, X, X_variance, mu, sd = self._preprocess_series(
            r,
            series_dict,
            scale_mode=scale_mode,
            allowed_ranges=series_ranges,
        )

        # Edge case: not enough data
        edge_case = self._handle_edge_case(
            n_points, min_size, r, series_dict, names, with_stats,)
        if edge_case is not None:
            return edge_case

        # 3) data point weights
        n_points_w = self._compute_point_weights(n_points, r, weighting, point_weights, normalize_weights)
        n_features_w = self._compute_point_weights(n_features, np.arange(n_features), "uniform", channel_weights, normalize_weights)

        # 4) all weights

        w_all = np.array([n_points_w] * n_features).T
        w_all *= n_features_w
        w_all /= X_variance

        # 5) Costs and segmentation
        C = self._segment_cost_matrix(X, w_all, min_size)  # shape (n+1, n+1) with exclusive end indices
        # Dynamic programming to compute optimal partitions
        dp, ptr = self._dp_optimal_costs(C, n_points, max_segments, min_size)
        p = n_features

        if selector == "bic":
            # BIC-like selection on m in [1, max_segments]
            best_score = np.inf
            bkps = [n_points]
            for m in range(1, max_segments + 1):
                sse = float(dp[m, n_points])
                if not np.isfinite(sse):
                    continue
                crit = sse + lam_scale * (m * p) * np.log(max(n_points, 2))
                if crit < best_score:
                    best_score = crit
                    bkps = self._trace_back(ptr, m, n_points)
        elif selector == "fixed":
            m = fixed_segments if fixed_segments is not None else min(3, max_segments)
            m = int(np.clip(m, 1, max_segments))
            bkps = self._trace_back(ptr, m, n_points)
        else:
            raise ValueError(f"Unknown selector: {selector}")

        # 6) Prepare output regions
        return self._prepare_output_regions(
            bkps, r, series_dict, names, n_points_w, with_stats)

    def _prepare_output_regions(
        self,
        bkps: list[int],
        r: np.ndarray,
        series_dict: dict[str, np.ndarray],
        names: list[str],
        weights: np.ndarray,
        with_stats: bool,
    ) -> dict[str, Any]:
        """ Prepare output regions with statistics. """
        regions: list[dict[str, Any]] = []
        start_idx = 0
        for end_idx in bkps:
            reg: dict[str, Any] = {
                "idx": (start_idx, end_idx),
                "R_in": float(r[start_idx]),
                "R_ou": float(r[end_idx - 1]),
                "mean": {},
            }
            sel = slice(start_idx, end_idx)
            w_seg = weights[sel]
            if with_stats:
                reg["median"] = {}
                reg["std"] = {}
                reg["16th"] = {}
                reg["84th"] = {}
            for k in names:
                vals = np.asarray(series_dict[k])[sel]
                w_sum = np.sum(w_seg)
                if w_sum > 0:
                    mean_val = float(np.sum(vals * w_seg) / w_sum)
                else:
                    mean_val = float(np.mean(vals))
                reg["mean"][k] = mean_val

                if with_stats:
                    sorted_indices: np.ndarray = np.argsort(vals)
                    sorted_vals: np.ndarray = vals[sorted_indices]
                    sorted_weights: np.ndarray = w_seg[sorted_indices]
                    cum_weights: np.ndarray = np.cumsum(sorted_weights)
                    total_weight: float = cum_weights[-1]

                    def weighted_percentile(
                        p: float,
                        *,
                        _total_weight: float = total_weight,
                        _cum_weights: np.ndarray = cum_weights,
                        _sorted_vals: np.ndarray = sorted_vals,
                    ) -> float:
                        if _sorted_vals.size == 0:
                            return float("nan")
                        target = p * _total_weight
                        idx = int(np.searchsorted(_cum_weights, target))
                        if idx >= _sorted_vals.shape[0]:
                            idx = _sorted_vals.shape[0] - 1
                        return float(_sorted_vals[idx])

                    median_val = weighted_percentile(0.5)
                    p16_val = weighted_percentile(0.16)
                    p84_val = weighted_percentile(0.84)
                    variance = np.sum(sorted_weights * (sorted_vals - mean_val) ** 2) / w_sum if w_sum > 0 else 0.0
                    std_val = float(np.sqrt(variance))

                    reg["median"][k] = median_val
                    reg["std"][k] = std_val
                    reg["16th"][k] = p16_val
                    reg["84th"][k] = p84_val

            regions.append(reg)
            start_idx = end_idx

        out = {
            "bkps": bkps,
            "regions": regions,
            "r": r,
            "keys": names,
        }
        return out

    def _segment_cost_matrix(
        self,
        X: np.ndarray,
        W: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """
        Build segment cost matrix C where C[i, j] is the weighted SSE of segment [i:j)
        (i inclusive, j exclusive). i in [0..n-1], j in [i+min_size .. n].
        """
        n, p = X.shape
        tiny = 1e-12

        # Pad cumulative sums with a leading zero row so sums over [i:j) = S[j] - S[i]
        Wp    = np.vstack([np.zeros((1, p)), W])
        WXp   = np.vstack([np.zeros((1, p)), W * X])
        WXXp  = np.vstack([np.zeros((1, p)), W * X * X])

        cum_W   = np.cumsum(Wp, axis=0)    # (n+1, p)
        cum_WX  = np.cumsum(WXp, axis=0)
        cum_WXX = np.cumsum(WXXp, axis=0)

        # C is (n+1, n+1) to allow j up to n and i from 0..n-1
        C = np.full((n + 1, n + 1), np.inf, dtype=float)

        for i in range(n):
            j_min = i + min_size
            if j_min > n:
                continue
            # Vectorized j computation is possible; keep clear loop for readability
            for j in range(j_min, n + 1):
                sw   = cum_W[j]   - cum_W[i]    # (p,)
                swx  = cum_WX[j]  - cum_WX[i]
                swx2 = cum_WXX[j] - cum_WXX[i]
                denom = np.maximum(sw, tiny)
                sse_j = swx2 - (swx**2) / denom
                sse_j[sw <= tiny] = 0.0
                sse = float(np.sum(sse_j))
                C[i, j] = max(0.0, sse)
        return C



    def _dp_optimal_costs(
        self,
        C: np.ndarray,
        n: int,
        max_segments: int,
        min_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dynamic programming over segment counts.
        dp[m, j] = min cost to segment [0:j) into m segments.
        ptr[m, j] = argmin k that achieves dp[m, j] via dp[m-1, k] + C[k, j].
        """
        inf = np.inf
        dp = np.full((max_segments + 1, n + 1), inf, dtype=float)
        ptr = np.full((max_segments + 1, n + 1), -1, dtype=int)

        # Base: 0 segments -> cost 0 at j=0 only
        dp[0, 0] = 0.0

        # One segment: from 0 to j if length >= min_size
        for j in range(min_size, n + 1):
            dp[1, j] = C[0, j]
            ptr[1, j] = 0

        # m >= 2
        for m in range(2, max_segments + 1):
            # Minimal j that can host m segments respecting min_size
            j_start = m * min_size
            if j_start > n:
                break
            for j in range(j_start, n + 1):
                k_min = (m - 1) * min_size
                k_max = j - min_size
                best = inf
                best_k = -1
                # Find k minimizing dp[m-1, k] + C[k, j]
                for k in range(k_min, k_max + 1):
                    prev = dp[m - 1, k]
                    if not np.isfinite(prev):
                        continue
                    val = prev + C[k, j]
                    if val < best:
                        best = val
                        best_k = k
                dp[m, j] = best
                ptr[m, j] = best_k
        return dp, ptr

    def _trace_back(self, ptr: np.ndarray, m: int, n: int) -> list[int]:
        """
        Recover breakpoints for exactly m segments ending at n.
        Returns a sorted list of exclusive end indices, ending with n.
        """
        bkps: list[int] = []
        j = n
        seg = m
        while seg > 0 and j >= 0:
            k = ptr[seg, j]
            if k < 0:
                # Fallback: if unreachable, collapse to a single segment
                return [n]
            bkps.append(j)
            j = k
            seg -= 1
        bkps.reverse()
        return bkps


    def _compute_point_weights(
        self,
        n: int,
        r: np.ndarray,
        weighting: Literal["uniform","linear","log"],
        weights: np.ndarray | None,
        normalize_weights: bool,
    ) -> np.ndarray:
        n = r.size
        if weights is not None:
            w = np.asarray(weights, dtype=float)
        elif weighting == "uniform":
            w = np.ones(n)
        elif weighting == "linear":
            w = np.gradient(r)
        elif weighting == "log":
            w = np.gradient(np.log10(r + 1e-3))
        else:
            raise ValueError(f"Unknown weighting scheme: {weighting}")

        if w.size != n:
            raise ValueError(f"weights size {w.size} does not match data point size {n}.")
        # Normalize weights if requested
        if normalize_weights:
            finite_mask = np.isfinite(w)
            sum_w = np.sum(w[finite_mask])
            if sum_w > 0:
                w = w * (n / sum_w)
        return w

    def _handle_edge_case(
        self,
        n: int,
        min_size: int,
        r: np.ndarray,
        series_dict: dict[str, np.ndarray],
        names: list[str],
        with_stats: bool,
    ) -> dict | None:
        if n < 2 * min_size:
            logger.warning("Not enough data points for segmentation.")
            bkps = [n]
            regs: list[dict[str, Any]] = [{
                "idx": (0, n),
                "R_in": float(r[0]),
                "R_ou": float(r[-1]),
                "mean": {k: float(np.mean(series_dict[k])) for k in names},
            }]
            if with_stats:
                regs[0]["median"] = {}
                regs[0]["std"] = {}
                regs[0]["16th"] = {}
                regs[0]["84th"] = {}

                for k in names:
                    vals = np.asarray(series_dict[k])

                    regs[0]["median"][k] = float(np.median(vals))
                    regs[0]["std"][k] = float(np.std(vals, ddof=0))
                    regs[0]["16th"][k] = float(np.percentile(vals, 16))
                    regs[0]["84th"][k] = float(np.percentile(vals, 84))
            out = {"bkps": bkps, "regions": regs, "r": r, "keys": names}
            return out

        # General case
        return None

    def _preprocess_series(
        self,
        r: np.ndarray,
        series_dict: dict[str, np.ndarray],
        scale_mode: Literal["std","range","none"] | list[str],
        allowed_ranges: dict[str, tuple[float, float]],
    ) -> tuple[list[str], int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Implementation of preprocessing logic
        names: list[str] = list(series_dict.keys())
        n_points: int = r.size
        n_features: int = len(names)

        # Stack data into feature matrix
        X: np.ndarray = np.zeros((n_points, n_features))
        X_variance: np.ndarray = np.zeros((n_points, n_features))
        for j, name in enumerate(names):
            X[:, j] = series_dict[name]

        # Handle scaling
        mu: np.ndarray = np.zeros(n_features)
        sd: np.ndarray = np.ones(n_features)

        # scaling based on scale_mode
        if isinstance(scale_mode, str):
            scale_list: list[str] = [scale_mode] * n_features
        else:
            scale_list = list(scale_mode)

        for i, name in enumerate(names):
            x_err: np.ndarray = np.zeros(n_points, dtype=float)
            mode = scale_list[i]
            if mode == "range":
                min_val, max_val = allowed_ranges[name]
                range_val = max_val - min_val
                if range_val > 0:
                    X[:, i] = (X[:, i] - min_val) / range_val

                    if name in self.err_data:
                        x_err = self.err_data[name] / range_val
            mu[i] = np.mean(X[:, i])
            sd[i] = np.std(X[:, i])
            if mode == "std":
                if sd[i] > 0:
                    X[:, i] = (X[:, i] - mu[i]) / sd[i]
                    if name in self.err_data:
                        x_err = self.err_data[name] / sd[i]
                mu[i] = 0.0
                sd[i] = 1.0
            X_variance[:, i] = x_err ** 2 + sd[i] ** 2

        return names, n_points, n_features, X, X_variance, mu, sd

    def _prepare_series(
        self,
        keys: list[str] | None,
        allowed_ranges: dict[str, tuple[float, float]] | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, tuple[float, float]]]:
        # Implementation of data preparation logic
        if keys is None:
            keys = list(self.data.keys())
        series_dict = {k: self.data[k] for k in keys}

        series_ranges = {}
        for k in keys:
            min_val = np.min(series_dict[k])
            max_val = np.max(series_dict[k])
            series_ranges[k] = (min_val, max_val)
        if allowed_ranges is not None:
            series_ranges.update(allowed_ranges)

        return self.r, series_dict, series_ranges
