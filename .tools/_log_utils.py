"""Shared CLI output utilities for gal3d developer tool scripts.

Import the shared ``logger`` instance for a zero-config experience::

    from _log_utils import logger

    logger.header("My tool")
    logger.step("Doing something…")
    logger.item("file.py")
    logger.success("Done — 3 files processed")

Color and Unicode detection is automatic; both can be overridden via the
``ToolLogger`` constructor when a custom instance is needed.
"""

from __future__ import annotations

import sys
from datetime import datetime


class _Ansi:
    """ANSI escape-code constants."""

    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"


def _tty_supports_color(stream: object = sys.stdout) -> bool:
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and isatty()


def _stdout_supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower().replace("-", "")
    return enc.startswith("utf")


class ToolLogger:
    """Unified CLI logger for gal3d developer tool scripts.

    Provides semantically-named output methods with a consistent visual style.
    Color and Unicode symbols are auto-detected from the terminal; both can be
    overridden via the constructor.

    Methods
    -------
    header(msg)   Bold cyan section title with a thin rule below.
    step(msg)     Cyan in-progress step (indented one level).
    item(msg)     Blue bullet for individual items (indented two levels).
    success(msg)  Bold green result with ✓  — for key counts / outcomes.
    done(msg)     Plain green result with ✓  — for secondary information.
    warning(msg)  Yellow advisory with ⚠.
    error(msg)    Bold red message to stderr with ✗.
    summary(msg)  Bold green final summary with a thin rule above.
    """

    def __init__(
        self,
        *,
        use_color: bool | None = None,
        use_unicode: bool | None = None,
        timestamps: bool = False,
    ) -> None:
        self._color   = _tty_supports_color()    if use_color   is None else use_color
        self._unicode = _stdout_supports_unicode() if use_unicode is None else use_unicode
        self._ts      = timestamps

    # ── internals ────────────────────────────────────────────────────────────

    def _c(self, codes: str, text: str) -> str:
        """Apply ANSI *codes* around *text* when color is enabled."""
        return f"{codes}{text}{_Ansi.RESET}" if self._color else text

    def _sym(self, unicode_sym: str, ascii_sym: str) -> str:
        return unicode_sym if self._unicode else ascii_sym

    def _prefix(self) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S')}] " if self._ts else ""

    def _emit(self, text: str, *, err: bool = False) -> None:
        print(self._prefix() + text, file=sys.stderr if err else sys.stdout)

    # ── public API ────────────────────────────────────────────────────────────

    def header(self, msg: str) -> None:
        """Bold cyan section header with a rule underneath."""
        rule = self._sym("─", "-") * min(len(msg), 72)
        self._emit(self._c(_Ansi.BOLD + _Ansi.CYAN, msg))
        self._emit(self._c(_Ansi.DIM, rule))

    def step(self, msg: str) -> None:
        """Cyan in-progress step (indented one level)."""
        arrow = self._sym("❯", ">")
        self._emit(self._c(_Ansi.CYAN, f"  {arrow} {msg}"))

    def item(self, msg: str) -> None:
        """Blue bullet for an individual item (indented two levels)."""
        bullet = self._sym("•", "-")
        self._emit(self._c(_Ansi.BLUE, f"    {bullet} {msg}"))

    def success(self, msg: str) -> None:
        """Bold green success with check mark — for key results / counts."""
        check = self._sym("✓", "+")
        self._emit(self._c(_Ansi.BOLD + _Ansi.GREEN, f"  {check} {msg}"))

    def done(self, msg: str) -> None:
        """Plain green check mark — for secondary / informational successes."""
        check = self._sym("✓", "+")
        self._emit(self._c(_Ansi.GREEN, f"  {check} {msg}"))

    def warning(self, msg: str) -> None:
        """Yellow advisory message."""
        warn = self._sym("⚠", "!")
        self._emit(self._c(_Ansi.YELLOW, f"  {warn} {msg}"))

    def error(self, msg: str) -> None:
        """Bold red error message written to stderr."""
        cross = self._sym("✗", "x")
        self._emit(self._c(_Ansi.BOLD + _Ansi.RED, f"  {cross} {msg}"), err=True)

    def summary(self, msg: str) -> None:
        """Bold green final summary line preceded by a thin rule."""
        rule = self._sym("─", "-") * min(len(msg) + 4, 72)
        check = self._sym("✓", "+")
        self._emit(self._c(_Ansi.DIM, rule))
        self._emit(self._c(_Ansi.BOLD + _Ansi.GREEN, f"  {check} {msg}"))


#: Shared module-level instance — import this directly for convenience.
logger: ToolLogger = ToolLogger()
