
import os
import sys
import json
import logging
from .util.string_format import string_formatter

logger = logging.getLogger('gal3d.executor')
def print_info():
    from ._plugins import load_plugins_info_json
    plugins = load_plugins_info_json()
    for i in plugins:
        logger.log(2025, f"\n - Available {string_formatter(i, fg_color='cyan')} plug-ins: ")
        for name in plugins[i]:
            logger.log(2025, f"           '{name['name']}'")

def main():

    if len(sys.argv) == 1:
        print_info()
        return

    #from gal3d.analyzer import Gal3DAnalyzer

if __name__ == "__main__":
    main()