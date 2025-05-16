import logging
import os
import configparser

from .util.string_format import string_formatter



def _set_config_parser()-> configparser.RawConfigParser:
    config_parser = configparser.RawConfigParser(
        comment_prefixes=('#', ';', '---'),
        inline_comment_prefixes=('#'),
        interpolation=configparser.ExtendedInterpolation(),
    )
    config_parser.optionxform = str  # it prevents option key from being lowercase,
    return config_parser

def _get_config_parser_with_defaults() -> configparser.RawConfigParser:
    # Create config dictionaries which will be required by subpackages

    config_parser = _set_config_parser()
    default_config_path = os.path.join(os.path.dirname(__file__), "default_config.ini")
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default configuration file not found: {default_config_path}")
    config_parser.read(default_config_path)
    return config_parser

def _get_basic_config_from_parser(config_parser: configparser.RawConfigParser):
    config = {}
    config['logger'] = {}
    for i in ['level','stream_level','file_level']:
        config['logger'][i] = config_parser['logger'].getint(i, fallback=20)
    config['logger']["save_file"] = config_parser['logger'].getboolean("save_file", fallback=True)
    config['logger']["file_name"] = config_parser['logger'].get("file_name", fallback="gal3d.log")
    
    config['general'] = {}
    config['general']["update_stub"] = config_parser['general'].getboolean("update_stub", fallback=False)
    config['general']["batchsize"] = config_parser['general'].getint("batchsize", fallback=200000)

    default_thread_count = max(os.cpu_count() // 3, 1)
    config['general']['number_of_threads'] = config_parser.getint('general', 'number_of_threads', fallback=default_thread_count)
    config['general']['use_cython'] = config_parser.getboolean('general', 'use_cython', fallback=True)

    if config['general']['number_of_threads']<0:
        config['general']['number_of_threads']=default_thread_count
    return config
class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    FORMATS = {
        logging.DEBUG: "".join(
            [
                string_formatter(
                    "[%(asctime)s.%(msecs)03d] ",
                    italics=True,
                ),"< ",
                string_formatter(
                    "%(name)s", fg_color='bright_blue', underline=True
                )," >",
                string_formatter(" line: %(lineno)d ", fg_color='purple', italics=True),
                "\n",
                "  >>>  ",
                string_formatter("| %(levelname)s | ", fg_color='cyan', bold=True),
                "%(message)s",
            ]
        ),
        logging.INFO: "".join(
            [
                string_formatter(
                    "[%(asctime)s.%(msecs)03d] ", italics=True, underline=False
                ),"< ",
                string_formatter(
                    "%(name)s", fg_color='bright_blue', underline=True
                )," >",
                "\n",
                "  >>>  ",
                string_formatter("| %(levelname)s | ", fg_color='green', bold=True),
                "%(message)s",
            ]
        ),
        logging.WARNING: "".join(
            [
                string_formatter(
                    "[%(asctime)s.%(msecs)03d] ",
                    fg_color='yellow',
                    italics=True,
                    underline=False,
                ),"< ",
                string_formatter(
                    "%(filename)s", fg_color='bright_blue', underline=True
                )," >",
                string_formatter(" line: %(lineno)d ", fg_color='purple', italics=True),
                "\n",
                "  >>>  ",
                string_formatter("| %(levelname)s | ", fg_color='yellow', bold=True),
                "%(message)s",
            ]
        ),
        logging.ERROR: "".join(
            [
                string_formatter(
                    "[%(asctime)s.%(msecs)03d] ",
                    fg_color='red',
                    italics=True,
                    underline=False,
                ),"< ",
                string_formatter(
                    "%(filename)s", fg_color='bright_blue', underline=True
                )," >",
                string_formatter(" line: %(lineno)d ", fg_color='purple', italics=True),
                "\n",
                "  >>>  ",
                string_formatter(
                    "| %(levelname)s | %(message)s", fg_color='red', bold=True
                ),
            ]
        ),
        logging.CRITICAL: "".join(
            [
                string_formatter(
                    "[%(asctime)s.%(msecs)03d] ",
                    fg_color='red',
                    italics=True,
                    underline=False,
                ),
                string_formatter(
                    "< %(filename)s >", fg_color='bright_blue', underline=True
                ),
                string_formatter(" line: %(lineno)d ", fg_color='purple', italics=True),
                "\n",
                "  >>>  ",
                string_formatter(
                    "| %(levelname)s | %(message)s",
                    fg_color='red',
                    bg_color='white',
                    bold=True,
                ),
            ]
        ),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class NoColorFormatter(logging.Formatter):
    """
    A logging formatter that outputs log messages without color formatting.

    This formatter is useful for environments where ANSI escape codes for colors
    are not supported or desired. It removes any ANSI escape sequences from the
    log messages and provides a plain-text format, unlike `ColorFormatter` which
    adds color and styling to the log output.
    """
    import re

    FORMATS = {
        logging.DEBUG: "[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.INFO: "[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.WARNING: "[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.ERROR: "[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.CRITICAL: "[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
    }
    ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        if record.levelno not in logging._levelToName:
            return self.ANSI_ESCAPE.sub('', formatter.format(record))
        return formatter.format(record)


def _setup_logging(cfg: dict) -> logging.Logger:
    logger = logging.getLogger("gal3d")
    if cfg['level'] not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        raise ValueError(f"Invalid logging level: {cfg['level']}")
    logger.setLevel(cfg['level'])
    
    ch = logging.StreamHandler()
    ch.setLevel(cfg['stream_level'])

    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)
    
    file_handle = cfg['save_file']
    if file_handle:
        fh = logging.FileHandler(cfg['file_name'], mode='w', encoding="utf-8")
        fh.setLevel(cfg['file_level'])
        fh.setFormatter(NoColorFormatter())
        logger.addHandler(fh)

    return logger


def set_logging_level(level=logging.INFO):
    """
    Set the logging level for the 'gal3d' logger.

    Valid levels:
    - logging.DEBUG: Detailed information, typically of interest only when diagnosing problems.
    - logging.INFO: Confirmation that things are working as expected.
    - logging.WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.
    - logging.ERROR: Due to a more serious problem, the software has not been able to perform some function.
    - logging.CRITICAL: A very serious error, indicating that the program itself may be unable to continue running.
    """
    logger = logging.getLogger('gal3d')
    logger.setLevel(level)



config_parser = _get_config_parser_with_defaults()
config = _get_basic_config_from_parser(config_parser)
logger = _setup_logging(config['logger'])

