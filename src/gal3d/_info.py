import logging

from .util.string_format import string_formatter

__version__ = "1.0.0"



_logo = [
    r"     ╭──────────────────────────────────────────────────────╮",
    r"    .│          .            .        .               .     │",
    r"     │   _______  _______  _      . ______+  ______ .       │",
    r"     │  (  ____ \(. ___  )( \      / ___ .\ (  __ .\        │",
    r"     │  |.(    \/| (   ) || (     .\/   \  \| (+ \  )       │",
    r"     │  | |      |.(___) || | .       ___) /| |   ) |  .    │",
    r"     │  |.| ____ |  ___  || |  .     (___ ( | |   |.|       │",
    r"     │  | | \_  )| ( + ) || |  .      .  ) \| |   ) | .     │",
    r"     │  | (___) || )  .( || (____/\/\___/  /| (__/  )       │",
    r"     │  (_______)|/     \|(_______/\______/ (______/  .     │",
    r"     │      .     .      .        .      .    .    .        │",
    r"     │.     Galaxy 3D Modeling.& Analysis.Framework     .   │",
    r"     │               .      .             .        .        │",
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
    from .log import DirectOutputHandler

    info_logger = logging.getLogger("gal3d.info")
    info_logger.setLevel(logging.INFO)
    info_logger.propagate = False  # prevent propagation to parent logger

    # Add direct output handler
    handler = DirectOutputHandler()
    info_logger.addHandler(handler)

    try:
        # Display colored logo
        info_logger.info(logo_color)

        # Display version information
        version_info = string_formatter(
            f"\n{f'gal3d version: {__version__}'.rjust(64, ' ')}\n",
            fg_color="bright_cyan",
            bold=True
        )
        info_logger.info(version_info)

        # Display plugin information
        if show_plugins:
            # Add a separator line
            separator = string_formatter(
                "=" * 80,
                fg_color="bright_blue"
            )
            info_logger.info(separator)

            # Import and use PluginManagerRegistry
            from .plugin import PluginManagerRegistry
            PluginManagerRegistry.print_plugins()

            # Add another separator line at the end
            info_logger.info(separator)
    finally:
        # Clean up handlers
        info_logger.removeHandler(handler)

# If the script is run directly, display information automatically
if __name__ == "__main__":
    print_gal3d_info()
