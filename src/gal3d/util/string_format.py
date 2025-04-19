from functools import singledispatch


__all__ = ['string_formator']


@singledispatch
def escape_codes_color_fg(color):
    return f"\033[{color}m"


@escape_codes_color_fg.register
def _(color: str):
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
def _(color: tuple):
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m"


@escape_codes_color_fg.register
def _(color: int):
    return f"\033[38;5;{color}m"


@singledispatch
def escape_codes_color_bg(color):
    return f"\033[{color}m"


@escape_codes_color_bg.register
def _(color: str):
    color_bg = {
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "purple": 45,
        "cyan": 46,
        "white": 47,
        "black": 100,
        "red": 101,
        "green": 102,
        "yellow": 103,
        "blue": 104,
        "purple": 105,
        "cyan": 106,
        "white": 107,
    }
    return f"\033[{color_bg[color]}m"


@escape_codes_color_bg.register
def _(color: tuple):
    return f"\033[48;2;{color[0]};{color[1]};{color[2]}m"


@escape_codes_color_bg.register
def _(color: int):
    return f"\033[48;5;{color}m"


def foreground_color(*args):
    if len(args) == 1:
        return escape_codes_color_fg(args[0])
    elif len(args) == 3 and all(isinstance(a, int) for a in args):
        return escape_codes_color_fg(args)
    else:
        raise TypeError("Invalid input format for foreground_color")


def background_color(*args):
    if len(args) == 1:
        return escape_codes_color_bg(args[0])
    elif len(args) == 3 and all(isinstance(a, int) for a in args):
        return escape_codes_color_bg(args)
    else:
        raise TypeError("Invalid input format for foreground_color")


def color(
    bg_color: str | int | tuple | None = None, fg_color: str | int | tuple | None = None
) -> str:
    bg_color = bg_color and background_color(bg_color) or ""
    fg_color = bg_color and background_color(fg_color) or ""

    return "".join([bg_color, fg_color])


def fontformat(
    bold: bool = False,
    thin: bool = False,
    italics: bool = False,
    underline: bool = False,
    strikethrough: bool = False,
) -> str:
    bold = bold and "\033[1m" or ""
    thin = thin and "\033[2m" or ""
    italics = italics and "\033[3m" or ""
    underline = underline and "\033[4m" or ""
    strikethrough = strikethrough and "\033[9m" or ""

    return "".join([bold, thin, italics, underline, strikethrough])


def string_formator(
    string: str,
    bg_color: str | int | tuple | None = None,
    fg_color: str | int | tuple | None = None,
    bold: bool = False,
    thin: bool = False,
    italics: bool = False,
    underline: bool = False,
    strikethrough: bool = False,
) -> str:

    bg_color = bg_color and background_color(bg_color) or ""
    fg_color = fg_color and foreground_color(fg_color) or ""
    font = fontformat(bold, thin, italics, underline, strikethrough)

    return "".join([bg_color, fg_color, font, string, "\033[0m"])
