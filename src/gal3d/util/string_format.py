"""
String formatting utilities for terminal output.
Modified from https://github.com/wx-ys/ansi-string-formatter
"""
from functools import singledispatch

__all__ = ["string_formatter"]

def string_formatter(
    string: str,
    bg_color: str | int | tuple | None = None,
    fg_color: str | int | tuple | None = None,
    bold: bool = False,
    thin: bool = False,
    italics: bool = False,
    underline: bool = False,
    strikethrough: bool = False,
) -> str:
    """
    Format a string with background and foreground colors and font styles.

    Parameters
    ----------
    string : str
        The input string to format.
    bg_color : str | int | tuple, optional
        The background color as a string, integer, or tuple of RGB values.
    fg_color : str | int | tuple, optional
        The foreground color as a string, integer, or tuple of RGB values.
    bold : bool, optional
        Whether to apply bold formatting. Default is False.
    thin : bool, optional
        Whether to apply thin formatting. Default is False.
    italics : bool, optional
        Whether to apply italics formatting. Default is False.
    underline : bool, optional
        Whether to apply underline formatting. Default is False.
    strikethrough : bool, optional
        Whether to apply strikethrough formatting. Default is False.

    Returns
    -------
    str
        The formatted string with ANSI escape codes.
    """
    bg_color = bg_color and background_color(bg_color) or ""
    fg_color = fg_color and foreground_color(fg_color) or ""
    font = fontformat(bold, thin, italics, underline, strikethrough)

    return "".join([bg_color, fg_color, font, string, "\033[0m"])


@singledispatch
def escape_codes_color_fg(color: int | str | tuple) -> str:
    """
    Generate an ANSI escape code for the foreground color.

    Parameters
    ----------
    color : int
        The color code for the foreground color.

    Returns
    -------
    str
        The ANSI escape code for the foreground color.
    """
    return f"\033[{color}m"


@escape_codes_color_fg.register
def _(color: str) -> str:
    """
    Generate an ANSI escape code for the foreground color from a string.

    Parameters
    ----------
    color : str
        The color name (e.g., "red", "blue", "bright_black").

    Returns
    -------
    str
        The ANSI escape code for the foreground color.
    """
    color_fg = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "purple": 35,
        "cyan": 36,
        "white": 37,
        "bright_black": 90,
        "bright_red": 91,
        "bright_green": 92,
        "bright_yellow": 93,
        "bright_blue": 94,
        "bright_purple": 95,
        "bright_cyan": 96,
        "bright_white": 97,
    }
    return f"\033[{color_fg[color]}m"


@escape_codes_color_fg.register
def _(color: tuple) -> str:
    """
    Generate an ANSI escape code for the foreground color from a tuple of RGB values.

    Parameters
    ----------
    color : tuple of int
        A tuple containing the RGB values (e.g., (255, 0, 0) for red).

    Returns
    -------
    str
        The ANSI escape code for the foreground color.
    """
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m"


@escape_codes_color_fg.register
def _(color: int) -> str:
    """
    Generate an ANSI escape code for the foreground color from a 256-color code.

    Parameters
    ----------
    color : int
        The 256-color code.

    Returns
    -------
    str
        The ANSI escape code for the foreground color.
    """
    return f"\033[38;5;{color}m"


@singledispatch
def escape_codes_color_bg(color: int | str | tuple) -> str:
    """
    Generate an ANSI escape code for the background color.

    Parameters
    ----------
    color : int
        The color code for the background color.

    Returns
    -------
    str
        The ANSI escape code for the background color.
    """
    return f"\033[{color}m"


@escape_codes_color_bg.register
def _(color: str) -> str:
    """
    Generate an ANSI escape code for the background color from a string.

    Parameters
    ----------
    color : str
        The color name (e.g., "red", "blue", "bright_black").

    Returns
    -------
    str
        The ANSI escape code for the background color.
    """
    color_bg = {
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "purple": 45,
        "cyan": 46,
        "white": 47,
        "bright_black": 100,
        "bright_red": 101,
        "bright_green": 102,
        "bright_yellow": 103,
        "bright_blue": 104,
        "bright_purple": 105,
        "bright_cyan": 106,
        "bright_white": 107,
    }
    return f"\033[{color_bg[color]}m"


@escape_codes_color_bg.register
def _(color: tuple) -> str:
    """
    Generate an ANSI escape code for the background color from a tuple of RGB values.

    Parameters
    ----------
    color : tuple of int
        A tuple containing the RGB values (e.g., (255, 0, 0) for red).

    Returns
    -------
    str
        The ANSI escape code for the background color.
    """
    return f"\033[48;2;{color[0]};{color[1]};{color[2]}m"


@escape_codes_color_bg.register
def _(color: int) -> str:
    """
    Generate an ANSI escape code for the background color from a 256-color code.

    Parameters
    ----------
    color : int
        The 256-color code.

    Returns
    -------
    str
        The ANSI escape code for the background color.
    """
    return f"\033[48;5;{color}m"


def foreground_color(*args: (str | int | tuple)) -> str:
    """
    Generate the ANSI escape code for the foreground color.

    Parameters
    ----------
    *args : (str | int | tuple)
        The foreground color as a string, integer, or tuple of RGB values.

    Returns
    -------
    str
        The ANSI escape code for the foreground color.

    Raises
    ------
    TypeError
        If the input format is invalid.
    """
    if len(args) == 1:
        return escape_codes_color_fg(args[0])
    elif len(args) == 3 and all(isinstance(a, int) for a in args):
        return escape_codes_color_fg(args)
    else:
        raise TypeError("Invalid input format for foreground_color")


def background_color(*args: (str | int | tuple)) -> str:
    """
    Generate the ANSI escape code for the background color.

    Parameters
    ----------
    *args : (str | int | tuple)
        The background color as a string, integer, or tuple of RGB values.

    Returns
    -------
    str
        The ANSI escape code for the background color.

    Raises
    ------
    TypeError
        If the input format is invalid.
    """
    if len(args) == 1:
        return escape_codes_color_bg(args[0])
    elif len(args) == 3 and all(isinstance(a, int) for a in args):
        return escape_codes_color_bg(args)
    else:
        raise TypeError("Invalid input format for background_color")


def color(
    bg_color: str | int | tuple | None = None, fg_color: str | int | tuple | None = None
) -> str:
    """
    Combine background and foreground color escape codes.

    Parameters
    ----------
    bg_color : str | int | tuple, optional
        The background color as a string, integer, or tuple of RGB values.
    fg_color : str | int | tuple, optional
        The foreground color as a string, integer, or tuple of RGB values.

    Returns
    -------
    str
        The combined ANSI escape codes for the background and foreground colors.
    """
    bg_color = bg_color and background_color(bg_color) or ""
    fg_color = fg_color and foreground_color(fg_color) or ""

    return "".join([bg_color, fg_color])


def fontformat(
    bold: bool = False,
    thin: bool = False,
    italics: bool = False,
    underline: bool = False,
    strikethrough: bool = False,
) -> str:
    """
    Generate ANSI escape codes for font formatting.

    Parameters
    ----------
    bold : bool, optional
        Whether to apply bold formatting. Default is False.
    thin : bool, optional
        Whether to apply thin formatting. Default is False.
    italics : bool, optional
        Whether to apply italics formatting. Default is False.
    underline : bool, optional
        Whether to apply underline formatting. Default is False.
    strikethrough : bool, optional
        Whether to apply strikethrough formatting. Default is False.

    Returns
    -------
    str
        The ANSI escape codes for the font formatting.
    """
    bold_code: str = bold and "\033[1m" or ""
    thin_code: str = thin and "\033[2m" or ""
    italics_code: str = italics and "\033[3m" or ""
    underline_code: str = underline and "\033[4m" or ""
    strikethrough_code: str = strikethrough and "\033[9m" or ""

    return "".join([bold_code, thin_code, italics_code, underline_code, strikethrough_code])
