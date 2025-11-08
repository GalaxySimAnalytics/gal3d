import logging

logger = logging.getLogger("gal3d.optimization.util")


_model_result_style = """
        <style>
        .mr-wrapper {
            overflow:auto;
            max-height:460px;
            border:1px solid #c8ccd0;
            border-radius:4px;
            background:#fff;
            box-shadow:0 2px 4px rgba(0,0,0,0.08);
            padding:0;
            position:relative;
        }
        .mr-summary {
            position:sticky;
            top:0;
            z-index:5;
            background:#1f2730;
            color:#f2f5f7;
            font-family:monospace;
            font-size:13px;
            line-height:1.25;
            padding:4px 10px 5px 10px;
            border-bottom:1px solid #444;
            letter-spacing:0.3px;
            white-space:nowrap;
        }
        .mr-table {
            border-collapse:collapse;
            font-family:monospace;
            font-size:12.5px;
            min-width:max-content;
        }
        .mr-table th, .mr-table td {
            border:1px solid #d0d4da;
            padding:4px 8px;
            text-align:right;
            white-space:nowrap;
        }
        .mr-table th {
            background:#2d3642;
            color:#f9f9f9;
            font-weight:600;
            position:sticky;
            top:28px; /* summary height (approx) */
            z-index:4;
        }
        .mr-table tbody tr:nth-child(even){
            background:#fafafa;
        }
        .mr-table tbody tr:hover{
            background:#ffe8aa;
        }
        .mr-table td{
            color:#222;
        }
        .mr-table th:first-child,
        .mr-table td:first-child{
            background:#314150;
            color:#fff;
            font-weight:600;
        }
        .mr-table th:nth-child(2),
        .mr-table td:nth-child(2){
            background:#3a4a5a;
            color:#f1f6ff;
        }
        .mr-note{
            font-size:11px;
            color:#555;
            margin-top:4px;
            font-family:monospace;
        }
        </style>
        """

def truncate(num: float, n: int) -> float:
    """
    Truncate a float to n decimal places (without rounding).

    Parameters
    ----------
    num : float
        The number to truncate.
    n : int
        Number of decimal places to keep.

    Returns
    -------
    float
        The truncated number. If num is inf, -inf, or nan, returns num unchanged.
    """
    import numpy as np
    if not np.isfinite(num):
        return num
    factor = 10.0 ** n
    return float(int(num * factor) / factor)
