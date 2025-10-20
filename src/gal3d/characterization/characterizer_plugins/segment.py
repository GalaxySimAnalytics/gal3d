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
The BIC-like criterion balances data fit (SSE) against model complexity (segments x channels).
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
        """
        Parameters
        ----------
        data : dict or ModelResult
            The input data containing radial and channel arrays.
        radial_key : str
            The key for the radius array in `data`. Default "a".
        channel_keys : tuple of str
            The keys for the channels to segment. Default ("eps_ab", "eps_bc", "x_axis_angle", "z_axis_angle").
        error_keys : tuple of str
            The keys for the error channels. Default ("eps_ab", "eps_bc", "x_axis_angle", "z_axis_angle").
        """
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
        max_segments: int = 10,
        min_size: int = 15,
        fixed_segments: int | None = None,
        lam_scale: float = 0.02,
        *,
        scale_mode: Literal["std","range","none"] | list[str] = "std",
        with_stats: bool = True,
        radial_log: bool = False,
        point_weights: np.ndarray | Literal["uniform","linear","log"] = "uniform",
        channel_weights: np.ndarray | None = None,
        normalize_weights: bool = True,
        allowed_ranges: dict[str, tuple[float, float]] | None = None,
        fit: Literal["constant", "linear","mix"] = "constant",
        ) -> dict[str, Any]:
        """
        Run the segmentation algorithm on the specified keys.

        Parameters
        ----------
        keys : Iterable[str] | None
            The keys to segment. If None, all channels are used.
        selector : {"bic", "fixed"}, default "bic"
            Segment count selection strategy:
            - "bic": choose m in [1, max_segments] minimizing
              SSE + lam_scale * (m * p) * log(n),
              where p is number of channels;
            - "fixed": use `fixed_segments`.
        max_segments : int, default 10
            The maximum number of segments to create.
        min_size : int, default 15
            Minimum number of data points per segment.
        fixed_segments : int | None, optional
            Number of segments to use when `selector="fixed"`. If None,
            uses min(3, max_segments). Ignored for "bic".
        lam_scale : float, default 0.02
            The regularization strength. Only used if `selector="bic"`.
        with_stats : bool, default True
            Whether to compute statistics for each segment.
        radial_log : bool, default False
            If True, perform segmentation and optional linear fitting in log10(radius) space.
            Segment indices are unchanged; R_in/R_ou in the output remain in original radius units.
        point_weights : np.ndarray | {"uniform","linear","log"}, default "uniform"
            Per-point weighting. If an array, it must have length n (number of samples).
            If a string:
              - "uniform": all points weight 1;
              - "linear": weights proportional to spacing via gradient of the working radius
                coordinate (r or log10(r) depending on radial_log);
              - "log": weights proportional to gradient of log10(working radius + 1e-3).
        channel_weights : np.ndarray or None, default None.
            Custom per-channel weights.
        normalize_weights : bool, optional, default True.
            If True and weights are finite and sum > 0, rescale them so that
            sum(w) = n (number of samples) to stabilize the penalty scale.
        scale_mode : Literal["std","range","none"] | list[str], default "std".
            The scaling mode to use for the data.
            - "std": standardize each channel to zero mean, unit variance;
            - "range": scale each channel to [0, 1] range;
            - "none": no scaling.
            - If list[str], each channel is scaled according to its name.
        allowed_ranges : dict[str, tuple[float, float]] | None
            The allowed ranges for each key, first key is the channel name, second is a tuple (min, max).
            Will be used to scale the data if scale_mode is "range".
            If not provided, uses the min/max of each channel.
        fit : Literal["constant", "linear", "mix"] = "constant"
            The fitting method to use for the segmentation.
            - "constant": use a constant fit for the segments;
            - "linear": use a linear fit for the segments.
            - "mix": allow each segment to be either constant or linear,
            Default "constant".

        Returns
        -------
        result : dict
            A dictionary with the following entries:
            - "bkps": list[int]
                Segment end indices (exclusive) in the sorted/filtered arrays.
            - "regions": list[dict]
                One dict per segment with fields:
                - "idx": (start, end) indices in [0, n)
                - "R_in": inner radius (float, original units)
                - "R_ou": outer radius (float, original units)
                - "mean": dict[key -> float], weighted mean per channel
                - If `with_stats`:
                  - "median": dict[key -> float], weighted median
                  - "std": dict[key -> float], weighted population std
                  - "16th": dict[key -> float], weighted 16th percentile
                  - "84th": dict[key -> float], weighted 84th percentile
                - If `fit == "linear"`:
                  - "slope" and "intercept" with respect to r or log10(r) if radial_log
            - "r": np.ndarray
                Sorted radii after filtering (length n, original units).
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

        # 2) Working radius coordinate
        r_fit = self._get_working_radius(r, radial_log)

        # 3) Preprocess
        names, n_points, n_features, X, X_variance, mu, sd = self._preprocess_series(
            r,
            series_dict,
            scale_mode=scale_mode,
            allowed_ranges=series_ranges,
        )

        # 4) Edge case
        edge_case = self._handle_edge_case(
            n_points, min_size, r, series_dict, names, with_stats,)
        if edge_case is not None:
            return edge_case

        # 5) Weights
        w_all, n_points_w = self._build_weights(
            r_fit=r_fit,
            X_variance=X_variance,
            point_weights=point_weights,
            channel_weights=(np.ones(n_features) if channel_weights is None else channel_weights),
            normalize_weights=normalize_weights,
            series_dict=series_dict,
        )

        # 6) Costs
        C_const, C_lin = self._segment_cost_matrices(X, w_all, r_fit, min_size)

        # 7) Optimization and selection
        bkps, seg_models = self._optimize_segments(
            C_const=C_const,
            C_lin=C_lin,
            n_points=n_points,
            n_features=n_features,
            selector=selector,
            max_segments=max_segments,
            min_size=min_size,
            lam_scale=lam_scale,
            fit=fit,
            fixed_segments=fixed_segments,
        )

        # 8) Prepare output regions
        return self._prepare_output_regions(
            bkps, r, r_fit, series_dict, names, n_points_w, with_stats, fit, seg_models=seg_models)

    def _specific_weights(self, series_dict: dict[str, np.ndarray]) -> np.ndarray:
        """ Compute specific weights from error data. """
        n_points = next(iter(series_dict.values())).size
        feature_names = list(series_dict.keys())
        n_features = len(series_dict)
        w_spec = np.ones((n_points, n_features), dtype=float)
        if "eps_ab" in series_dict and "x_axis_angle" in series_dict:
            w = series_dict["eps_ab"] ** 2
            w_spec[:, feature_names.index("x_axis_angle")] = w/np.sum(w)*n_points
        if "eps_bc" in series_dict and "z_axis_angle" in series_dict:
            w = series_dict["eps_bc"] ** 2
            w_spec[:, feature_names.index("z_axis_angle")] = w/np.sum(w)*n_points
        return w_spec

    def _optimize_segments(
        self,
        C_const: np.ndarray,
        C_lin: np.ndarray,
        n_points: int,
        n_features: int,
        selector: Literal["bic","fixed"],
        max_segments: int,
        min_size: int,
        lam_scale: float,
        fit: Literal["constant","linear","mix"],
        fixed_segments: int | None,
    ) -> tuple[list[int], list[Literal["constant","linear"]] | None]:
        """
        Run DP according to fit mode and select breakpoints with either BIC-like score or fixed count.
        Returns (bkps, seg_models). seg_models is None for non-mix modes.
        """
        if fit in ("constant", "linear"):
            C = C_const if fit == "constant" else C_lin
            if fit == "constant":
                dp, ptr = self._dp_optimal_costs_dc(C, n_points, max_segments, min_size)
            else:
                dp, ptr = self._dp_optimal_costs(C, n_points, max_segments, min_size)

            p = n_features
            if selector == "bic":
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
                return bkps, None
            else:  # fixed
                m = fixed_segments if fixed_segments is not None else min(3, max_segments)
                m = int(np.clip(m, 1, max_segments))
                bkps = self._trace_back(ptr, m, n_points)
                return bkps, None

        # mix mode: DP over complexity where const=1, lin=2
        dp, ptr_k, ptr_t = self._dp_optimal_costs_mix(C_const, C_lin, n_points, max_segments, min_size)
        p = n_features
        if selector == "bic":
            best_score = np.inf
            bkps = [n_points]
            seg_models: list[Literal["constant", "linear"]] = ["constant"]
            for c in range(1, max_segments + 1):
                sse = float(dp[c, n_points])
                if not np.isfinite(sse):
                    continue
                crit = sse + lam_scale * (c * p) * np.log(max(n_points, 2))
                if crit < best_score:
                    best_score = crit
                    bkps, seg_models = self._trace_back_mix(ptr_k, ptr_t, c, n_points)
            return bkps, seg_models
        else:
            c = fixed_segments if fixed_segments is not None else min(3, max_segments)
            c = int(np.clip(c, 1, max_segments))
            bkps, seg_models = self._trace_back_mix(ptr_k, ptr_t, c, n_points)
            return bkps, seg_models

    def _get_working_radius(
        self,
        r: np.ndarray,
        radial_log: bool,
    ) -> np.ndarray:
        """Choose working radius coordinate for segmentation/fitting."""
        return np.log10(r + 1e-3) if radial_log else r

    def _build_weights(
        self,
        r_fit: np.ndarray,
        X_variance: np.ndarray,
        point_weights: np.ndarray | Literal["uniform","linear","log"],
        channel_weights: np.ndarray,
        normalize_weights: bool,
        series_dict: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build full per-point, per-channel weights and return also the 1D point weights
        used for reporting segment statistics.
        """
        _, n_features = X_variance.shape

        # Per-point
        n_points_w = self._compute_point_weights(r_fit, point_weights)

        # Per-channel
        n_features_w = self._compute_channel_weights(channel_weights, n_features)

        # Specific cross-feature weights
        specific_weights = self._specific_weights(series_dict)

        # Combine
        w_all = np.array([n_points_w] * n_features).T  # shape (n, p)
        w_all *= n_features_w
        # Avoid division by zero by relying on X_variance construction
        w_all /= X_variance
        w_all *= specific_weights

        if normalize_weights:
            finite = np.isfinite(w_all)
            denom = np.mean(w_all[finite]) if np.any(finite) else 1.0
            if denom != 0:
                w_all = w_all / denom

        return w_all, n_points_w

    def _compute_channel_weights(
        self,
        channel_weights: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        w = np.asarray(channel_weights, dtype=float)
        if w.size != n_features:
            raise ValueError(f"channel_weights size {w.size} does not match number of features {n_features}.")
        return w

    def _prepare_output_regions(
        self,
        bkps: list[int],
        r: np.ndarray,
        r_fit: np.ndarray,
        series_dict: dict[str, np.ndarray],
        names: list[str],
        weights: np.ndarray,
        with_stats: bool,
        fit: Literal["constant", "linear","mix"] = "constant",
        *,
        seg_models: list[Literal["constant","linear"]] | None = None,
    ) -> dict[str, Any]:
        """ Prepare output regions with statistics. """
        regions: list[dict[str, Any]] = []
        start_idx = 0
        seg_idx = 0
        for end_idx in bkps:
            seg_model = seg_models[seg_idx] if seg_models is not None else fit

            reg: dict[str, Any] = {
                "idx": (start_idx, end_idx),
                "R_in": float(r[start_idx]),
                "R_ou": float(r[end_idx - 1]),
                "mean": {},
            }
            if seg_model == "linear":
                reg["slope"] = {}
                reg["intercept"] = {}

            sel = slice(start_idx, end_idx)
            w_seg = weights[sel]

            if with_stats:
                reg["median"] = {}
                reg["std"] = {}
                reg["16th"] = {}
                reg["84th"] = {}

            for k in names:
                vals = np.asarray(series_dict[k])[sel]
                w_sum = float(np.sum(w_seg))
                mean_val = float(np.sum(vals * w_seg) / w_sum) if w_sum > 0 else float(np.mean(vals))
                reg["mean"][k] = mean_val

                if seg_model == "linear":
                    b, a = self._weighted_linear_fit(r_fit[sel], vals, w_seg)
                    reg["slope"][k] = b
                    reg["intercept"][k] = a

                if with_stats:
                    p16, med, p84 = self._weighted_percentiles(vals, w_seg, ps=(0.16, 0.5, 0.84))
                    # weighted population std around weighted mean
                    variance = float(np.sum(w_seg * (vals - mean_val) ** 2) / w_sum) if w_sum > 0 else 0.0
                    reg["median"][k] = med
                    reg["std"][k] = float(np.sqrt(variance))
                    reg["16th"][k] = p16
                    reg["84th"][k] = p84

            regions.append(reg)
            start_idx = end_idx
            seg_idx += 1

        return {"bkps": bkps, "regions": regions, "r": r, "keys": names}

    def _weighted_linear_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        tiny: float = 1e-12,
    ) -> tuple[float, float]:
        """Weighted linear regression y ~ a + b x. Returns (b, a). Falls back to mean if ill-conditioned."""
        sw = np.sum(w)
        if sw <= tiny:
            return 0.0, float(np.mean(y)) if y.size else 0.0
        sx = np.sum(w * x)
        sy = np.sum(w * y)
        sxx = np.sum(w * x * x)
        sxy = np.sum(w * x * y)
        det = sw * sxx - sx * sx
        if abs(det) <= tiny:
            return 0.0, sy / sw
        b = (sw * sxy - sx * sy) / det
        a = (sxx * sy - sx * sxy) / det
        return float(b), float(a)

    def _weighted_percentiles(
        self,
        vals: np.ndarray,
        w: np.ndarray,
        ps: tuple[float, ...] = (0.16, 0.5, 0.84),
    ) -> list[float]:
        """Weighted percentiles using the 'cumulative weight crossing' rule."""
        if vals.size == 0:
            return [float("nan")] * len(ps)
        order = np.argsort(vals)
        v = vals[order]
        ww = w[order]
        cw = np.cumsum(ww)
        tot = float(cw[-1])
        if tot <= 0:
            # fall back to unweighted percentiles if all weights 0/non-finite
            return [float(np.percentile(v, p*100)) for p in ps]
        out: list[float] = []
        for p in ps:
            target = p * tot
            idx = int(np.searchsorted(cw, target, side="left"))
            if idx >= v.size:
                idx = v.size - 1
            out.append(float(v[idx]))
        return out

    def _segment_cost_matrix(
        self,
        X: np.ndarray,
        W: np.ndarray,
        r: np.ndarray,
        min_size: int,
        model: Literal["constant", "linear"] = "constant",
    ) -> np.ndarray:
        """
        Build segment cost matrix C where C[i, j] is the weighted SSE of segment [i:j)
        (i inclusive, j exclusive). i in [0..n-1], j in [i+min_size .. n].
        """
        # Delegate to unified builder to avoid duplication
        C_const, C_lin = self._segment_cost_matrices(X, W, r, min_size)
        return C_const if model == "constant" else C_lin

    def _segment_cost_matrices(
        self,
        X: np.ndarray,
        W: np.ndarray,
        r: np.ndarray,
        min_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cost matrices for both constant and linear models:
        returns (C_const, C_lin), each shape (n+1, n+1).
        """
        n, p = X.shape
        cum = self._build_prefix_sums(X, W, r)
        return self._fill_cost_matrices(cum, n, p, min_size)

    def _build_prefix_sums(
        self,
        X: np.ndarray,
        W: np.ndarray,
        r: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Prefix sums for constant and linear models."""
        n, p = X.shape
        zeros = np.zeros((1, p))
        R = r.reshape(-1, 1)

        Wp   = np.vstack([zeros, W])
        WXp  = np.vstack([zeros, W * X])
        WXXp = np.vstack([zeros, W * X * X])

        WRp  = np.vstack([zeros, W * R])
        WRRp = np.vstack([zeros, W * R * R])
        WRXp = np.vstack([zeros, W * R * X])

        return {
            "cum_W":   np.cumsum(Wp, axis=0),
            "cum_WX":  np.cumsum(WXp, axis=0),
            "cum_WXX": np.cumsum(WXXp, axis=0),
            "cum_WR":  np.cumsum(WRp, axis=0),
            "cum_WRR": np.cumsum(WRRp, axis=0),
            "cum_WRX": np.cumsum(WRXp, axis=0),
        }

    def _fill_cost_matrices(
        self,
        cum: dict[str, np.ndarray],
        n: int,
        p: int,
        min_size: int,
        tiny: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute both constant and linear cost matrices in one pass."""
        cum_W   = cum["cum_W"]
        cum_WX  = cum["cum_WX"]
        cum_WXX = cum["cum_WXX"]
        cum_WR  = cum["cum_WR"]
        cum_WRR = cum["cum_WRR"]
        cum_WRX = cum["cum_WRX"]

        C_const = np.full((n + 1, n + 1), np.inf, dtype=float)
        C_lin   = np.full((n + 1, n + 1), np.inf, dtype=float)

        for i in range(n):
            j_min = i + min_size
            if j_min > n:
                continue
            for j in range(j_min, n + 1):
                sw  = cum_W[j]   - cum_W[i]     # (p,)
                sy  = cum_WX[j]  - cum_WX[i]
                syy = cum_WXX[j] - cum_WXX[i]

                denom = np.maximum(sw, tiny)
                sse_const_j = syy - (sy**2) / denom
                sse_const_j[sw <= tiny] = 0.0
                C_const[i, j] = max(0.0, float(np.sum(np.maximum(0.0, sse_const_j))))

                # linear terms
                sx  = cum_WR[j]  - cum_WR[i]
                sxx = cum_WRR[j] - cum_WRR[i]
                sxy = cum_WRX[j] - cum_WRX[i]
                det = sw * sxx - sx * sx
                quad_num = (sxx * (sy**2)) - (2.0 * sx * sy * sxy) + (sw * (sxy**2))
                safe_det = np.where(np.abs(det) <= tiny, np.nan, det)
                sse_lin_j = syy - (quad_num / safe_det)
                # fallback to constant if ill-conditioned or invalid
                use_const = (sw <= tiny) | (np.abs(det) <= tiny) | ~np.isfinite(sse_lin_j)
                sse_lin_j = np.where(use_const, sse_const_j, sse_lin_j)
                C_lin[i, j] = max(0.0, float(np.sum(np.maximum(0.0, sse_lin_j))))
        return C_const, C_lin


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

    def _dp_optimal_costs_dc(
        self,
        C: np.ndarray,
        n: int,
        max_segments: int,
        min_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Divide-and-conquer DP optimization for segmentation with a single model.
        Assumes the cost matrix C has the quadrangle inequality/Monge property,
        which holds for SSE-based constant segments and commonly in practice.
        Complexity: ~O(max_segments * n log n) vs O(max_segments * n^2).
        """
        inf = np.inf
        dp = np.full((max_segments + 1, n + 1), inf, dtype=float)
        ptr = np.full((max_segments + 1, n + 1), -1, dtype=int)

        # Base
        dp[0, 0] = 0.0

        # m = 1: only [0:j)
        for j in range(min_size, n + 1):
            dp[1, j] = C[0, j]
            ptr[1, j] = 0

        def compute_row(m: int) -> None:
            j_start = m * min_size
            if j_start > n:
                return

            def solve(j_lo: int, j_hi: int, k_lo: int, k_hi: int) -> None:
                if j_lo > j_hi:
                    return
                j_mid = (j_lo + j_hi) // 2
                # admissible k range must also satisfy segment length >= min_size
                k_max_adm = min(k_hi, j_mid - min_size)
                if k_lo > k_max_adm:
                    # No feasible split
                    dp[m, j_mid] = inf
                    ptr[m, j_mid] = -1
                else:
                    best_val = inf
                    best_k = -1
                    # linear scan within admissible [k_lo, k_max_adm]
                    for k in range(k_lo, k_max_adm + 1):
                        prev = dp[m - 1, k]
                        if not np.isfinite(prev):
                            continue
                        val = prev + C[k, j_mid]
                        if val < best_val:
                            best_val = val
                            best_k = k
                    dp[m, j_mid] = best_val
                    ptr[m, j_mid] = best_k
                    if best_k == -1:
                        # nothing feasible; still recurse to keep bounds tight
                        best_k = k_lo
                # Recurse with monotone decision boundaries
                solve(j_lo, j_mid - 1, k_lo, max(k_lo, ptr[m, j_mid]))
                solve(j_mid + 1, j_hi, max(k_lo, ptr[m, j_mid]), k_hi)

            solve(j_start, n, (m - 1) * min_size, n - min_size)

        for m in range(2, max_segments + 1):
            compute_row(m)

        return dp, ptr

    def _dp_optimal_costs_mix(
        self,
        C_const: np.ndarray,
        C_lin: np.ndarray,
        n: int,
        max_complexity: int,
        min_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DP over complexity units c where:
        - constant segment consumes 1 unit with cost C_const[k,j]
        - linear   segment consumes 2 units with cost C_lin[k,j]
        Returns:
        - dp[c, j]: minimal SSE to cover [0:j) with complexity c
        - ptr_k[c, j]: previous index k
        - ptr_t[c, j]: segment type used to reach j (1=const, 2=linear)
        """
        inf = np.inf
        dp = np.full((max_complexity + 1, n + 1), inf, dtype=float)
        ptr_k = np.full((max_complexity + 1, n + 1), -1, dtype=int)
        ptr_t = np.full((max_complexity + 1, n + 1), 0, dtype=int)

        dp[0, 0] = 0.0

        for c in range(1, max_complexity + 1):
            # at least one segment must have length >= min_size
            for j in range(min_size, n + 1):
                best = inf
                best_k = -1
                best_t = 0
                # Try constant (cost 1 unit)
                for k in range(j - min_size + 1):
                    prev = dp[c - 1, k]
                    if not np.isfinite(prev):
                        continue
                    val = prev + C_const[k, j]
                    if val < best:
                        best = val
                        best_k = k
                        best_t = 1
                # Try linear (cost 2 units)
                if c >= 2:
                    for k in range(j - min_size + 1):
                        prev = dp[c - 2, k]
                        if not np.isfinite(prev):
                            continue
                        val = prev + C_lin[k, j]
                        if val < best:
                            best = val
                            best_k = k
                            best_t = 2
                dp[c, j] = best
                ptr_k[c, j] = best_k
                ptr_t[c, j] = best_t

        return dp, ptr_k, ptr_t

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

    def _trace_back_mix(
        self,
        ptr_k: np.ndarray,
        ptr_t: np.ndarray,
        c: int,
        n: int,
    ) -> tuple[list[int], list[Literal["constant","linear"]]]:
        """
        Recover breakpoints and per-segment model types for complexity c ending at n.
        Returns (bkps, seg_models), where seg_models[i] in {"constant","linear"}.
        """
        bkps: list[int] = []
        models: list[Literal["constant","linear"]] = []
        j = n
        cc = c
        while cc > 0 and j >= 0:
            k = ptr_k[cc, j]
            t = ptr_t[cc, j]
            if k < 0 or t == 0:
                return [n], ["constant"]
            bkps.append(j)
            models.append("linear" if t == 2 else "constant")
            j = k
            cc -= t  # consume 1 or 2 units
        bkps.reverse()
        models.reverse()
        return bkps, models


    def _compute_point_weights(
        self,
        r: np.ndarray,
        scheme_or_array: np.ndarray | Literal["uniform","linear","log"],
    ) -> np.ndarray:
        """
        Compute per-point weights.

        Parameters
        ----------
        r : np.ndarray
            Working radius coordinate (r or log10(r) depending on caller).
        scheme_or_array : array | {"uniform","linear","log"}
            If array, used directly. If None, uniform. If string:
              - "uniform": all ones
              - "linear": gradient(r)
              - "log": gradient(log10(r + 1e-3))
        normalize_weights : bool
            If True, rescale so sum(w) == len(r) when finite and sum > 0.
        """
        n = r.size
        if isinstance(scheme_or_array, np.ndarray):
            w = scheme_or_array.astype(float, copy=False)
        elif scheme_or_array == "uniform":
            w = np.ones(n, dtype=float)
        elif scheme_or_array == "linear":
            w = np.gradient(r.astype(float))
        elif scheme_or_array == "log":
            w = np.gradient(np.log10(r.astype(float) + 1e-3))
        else:
            raise ValueError(f"Unknown point weight scheme: {scheme_or_array}")

        if w.size != n:
            raise ValueError(f"point_weights size {w.size} does not match data point size {n}.")
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
