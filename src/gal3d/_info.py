"""
gal3d package information.

version, logo, and information display function.

"""

import logging

from gal3d.util.string_format import string_formatter

__version__ = "1.0.0"

_logo = [
    r"     ╭──────────────────────────────────────────────────────╮",
    r"     │·         .            .     .               .      Ç │",
    r"     │    _______  _______  _      . ______+  ______ .  ø˜  │",
    r"     │   (  ____ \(. ___  )( \      / ___ .\ (  __ .\       │",
    r"     │   |.(    \/| (   ) || (     .\/   \  \| (+ \  )      │",
    r"     │   | |      |.(___) || | .       ___) /| |   ) |  .   │",
    r"     │   |.| ____ |  ___  || |  .    ³(___ ( | |   |.|      │",
    r"     │   | | \_  )| ( + ) || |  .      .  ) \| |   ) | .    │",
    r"     │   | (___) || )  .( || (____/\/\___/  /| (__/  )      │",
    r"     │   (_______)|/     \|(_______/\______/ (______/  .    │",
    r"     │      .     .      .        .      .    .    .        │",
    r"     │ --✦--               .      .             .           │",
    r"     ╰──────────────────────────────────────────────────────╯",
]
logo = "\n".join(_logo)
logo_color = logo

def print_gal3d_info(show_plugins=True):
    """
    Print gal3d package information, including logo, version and plugins.

    This function provides a comprehensive overview of the gal3d installation,
    displaying the logo, version information, and optionally listing all
    available plugin managers and their plugins.

    Parameters
    ----------
    show_plugins : bool, optional
        Whether to display plugin information. Default is True.
    """
    import shutil

    from gal3d.log import DirectOutputHandler

    info_logger = logging.getLogger("gal3d.info")
    info_logger.setLevel(logging.INFO)
    info_logger.propagate = False  # prevent propagation to parent logger

    # Add direct output handler
    handler = DirectOutputHandler()
    info_logger.addHandler(handler)

    try:
        # Define consistent width for all components
        width = shutil.get_terminal_size().columns
        # Display welcome message and version information
        welcome_msg = string_formatter(
            "\n" + "Welcome to Gal3D !!! \n        3D Morphological Models of Galaxies in Simulations",
            fg_color="bright_cyan",
            bold=True
        )
        # Display colored logo
        info_logger.info(welcome_msg)
        info_logger.info(logo_color)

        # Display version information
        version_info = string_formatter(
            f"\n{f'gal3d version: {__version__}'.rjust(70, ' ')}\n",
            fg_color="bright_cyan",
            bold=True
        )
        info_logger.info(version_info)

        # Display plugin information
        if show_plugins:
            # Add a separator line
            separator = string_formatter(
                "=" * width,
                fg_color="bright_blue"
            )
            info_logger.info(separator)

            # Import and use PluginManagerRegistry
            from gal3d.plugin import PluginManagerRegistry
            PluginManagerRegistry.print_plugins()

            # Add another separator line at the end
            info_logger.info(separator)
    finally:
        # Clean up handlers
        info_logger.removeHandler(handler)

# If the script is run directly, display information automatically
if __name__ == "__main__":
    print_gal3d_info()
