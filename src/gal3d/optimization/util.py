import logging

logger = logging.getLogger("gal3d.optimization.util")


_model_result_style = """<style>
.mr-wrapper {
    font-family: monospace;
    border: 1px solid #c8ccd0;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.10);
    display: inline-block;
    max-width: 100%;
}
.mr-summary {
    background: #1f2730;
    color: #e8edf2;
    font-size: 13px;
    line-height: 1.5;
    padding: 5px 12px;
    border-bottom: 2px solid #3a4a5a;
    letter-spacing: 0.3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.mr-body {
    overflow: auto;
    max-height: 420px;
}
.mr-table {
    border-collapse: separate;
    border-spacing: 0;
    font-size: 12.5px;
    min-width: max-content;
    width: 100%;
}
.mr-table thead th {
    background: #2d3f50;
    color: #f0f5ff;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #4a5968;
    border-top: 0;
    text-align: right;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 3;
    background-clip: padding-box;
}
.mr-table thead th:first-child {
    text-align: center;
    background: #1e3040;
}
.mr-table td {
    border: 1px solid #d8dce2;
    border-top: 0;
    padding: 3px 10px;
    text-align: right;
    white-space: nowrap;
    color: #1a1a2e !important;
    background: #ffffff !important;
    font-variant-numeric: tabular-nums;
}
.mr-table td:first-child {
    text-align: center;
    background: #eef2f8 !important;
    color: #2a3a4a !important;
    font-weight: 600;
    border-right: 2px solid #b0bac8;
}
.mr-table tbody tr:nth-child(even) td {
    background: #f0f4fa !important;
    color: #1a1a2e !important;
}
.mr-table tbody tr:nth-child(even) td:first-child {
    background: #e4ecf5 !important;
    color: #2a3a4a !important;
}
.mr-table tbody tr:hover td {
    background: #fff3cd !important;
    color: #1a1a2e !important;
}
.mr-table tbody tr:hover td:first-child {
    background: #ffe8a0 !important;
    color: #2a3a4a !important;
}
.mr-table td.mr-col-cost {
    background: #eef3f8 !important;
    color: #0f172a !important;
    font-weight: 600;
}
.mr-table tbody tr:nth-child(even) td.mr-col-cost {
    background: #e6edf5 !important;
    color: #0f172a !important;
}
.mr-table tbody tr:hover td.mr-col-cost {
    background: #ffe8a0 !important;
    color: #0f172a !important;
}
.mr-ellipsis td {
    text-align: center !important;
    color: #888 !important;
    font-style: italic;
    background: #f5f5f5 !important;
    padding: 2px 10px;
    font-size: 14px;
    letter-spacing: 4px;
}
.mr-note {
    font-size: 11px;
    color: #475569;
    padding: 4px 10px 5px;
    border-top: 1px solid #e0e4ea;
    background: #fafbfc;
}
</style>"""


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
    factor = 10.0**n
    return float(int(num * factor) / factor)
