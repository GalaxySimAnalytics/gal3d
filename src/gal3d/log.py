import logging

from .config import config, LoggerConfig
from .util.string_format import string_formatter

class ColorFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors.

    Formats log messages with ANSI color codes for different log levels.
    """

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
                string_formatter(" | %(levelname)s | ", fg_color='cyan', bold=True),
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
                string_formatter(" | %(levelname)s | ", fg_color='green', bold=True),
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
                ), "from < ",
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
                    fg_color=(205, 0, 0),
                    italics=True,
                    underline=False,
                ),"from < ",
                string_formatter(
                    "%(filename)s", fg_color='bright_blue', underline=True
                )," >",
                string_formatter(" line: %(lineno)d ", fg_color='purple', italics=True),
                "\n",
                "  >>>  ",
                string_formatter(
                    "| %(levelname)s | %(message)s", fg_color=(205, 0, 0), bold=True
                ),
            ]
        ),
        logging.CRITICAL: "".join(
            [
                string_formatter(
                    "[%(asctime)s.%(msecs)03d] ",
                    fg_color=(255, 20, 147),
                    italics=True,
                    underline=False,
                ),"from < ",
                string_formatter(
                    "%(filename)s", fg_color='bright_blue', underline=True
                )," >",
                string_formatter(" line: %(lineno)d ", fg_color='purple', italics=True),
                "\n",
                "  >>>  ",
                string_formatter(
                    "| %(levelname)s | %(message)s",
                    fg_color=(255, 20, 147),
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
    Format the specified record as text.

    Parameters
    ----------
    record : logging.LogRecord
        The log record to format.

    Returns
    -------
    str
        The formatted log message.
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


def _setup_logging(cfg: LoggerConfig) -> logging.Logger:
    """ 
    Set up the gal3d logger according to the provided configuration.

    Parameters
    ----------
    cfg : LoggerConfig
        Logger configuration dataclass.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("gal3d")
    if cfg.level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        raise ValueError(f"Invalid logging level: {cfg.level}")
    logger.setLevel(cfg.level)

    ch = logging.StreamHandler()
    ch.setLevel(cfg.stream_level)

    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)
    
    file_handle = cfg.save_file
    if file_handle:
        fh = logging.FileHandler(cfg.file_name, mode='w', encoding="utf-8")
        fh.setLevel(cfg.file_level)
        fh.setFormatter(NoColorFormatter())
        logger.addHandler(fh)

    return logger


def set_logging_level(level=logging.INFO) -> None:
    """
    Set the logging level for the 'gal3d' logger.

    Parameters
    ----------
    level : int, optional
        Logging level to set for the 'gal3d' logger. Valid values are:
            - logging.DEBUG
            - logging.INFO
            - logging.WARNING
            - logging.ERROR
            - logging.CRITICAL
        Default is logging.INFO.
    """
    logger = logging.getLogger('gal3d')
    logger.setLevel(level)
    


logger = _setup_logging(config.logger)

