"""
Logging configuration and utilities for the gal3d package.

This module sets up the logging framework for the gal3d package, including
custom formatters and handlers to provide colorized console output and
optional file logging without color codes.

"""

import logging
import sys

from .configuration import LoggerConfig, config
from .util.string_format import string_formatter


class DirectOutputHandler(logging.Handler):
    """
    A logging handler that directly outputs the message without any formatting.

    This handler is useful when you want to output pre-formatted messages,
    such as colorized text or custom formatted outputs, directly to stdout
    without the logger's standard formatting being applied.

    The handler will just output the raw message content and a newline.
    Any record formatting needs to be done before passing the message to the logger.
    """

    def __init__(self, stream=None):
        """
        Initialize the handler with an optional output stream.

        Parameters
        ----------
        stream : file-like object, optional
            The stream to which messages are written. If not specified,
            sys.stdout is used.
        """
        super().__init__()
        self.stream = stream if stream is not None else sys.stdout

    def emit(self, record):
        """
        Output the raw message content directly to the stream.

        Parameters
        ----------
        record : logging.LogRecord
            The record to be output.
        """
        try:
            msg = self.format(record)
            self.stream.write(msg + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)

    def format(self, record):
        """
        Format the record. For this handler, we just return the raw message.

        Parameters
        ----------
        record : logging.LogRecord
            The record to format.

        Returns
        -------
        str
            The raw message string.
        """
        # Just return the message part without any formatting
        return record.msg


class ColorFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors.

    Formats log messages with ANSI color codes for different log levels.
    """

    FORMATS = {
        logging.DEBUG: "".join(
            [
                string_formatter("[%(asctime)s.%(msecs)03d] ", italics=True),
                "< ",
                string_formatter("%(name)s", fg_color="bright_blue", underline=True),
                " >",
                string_formatter(" line: %(lineno)d ", fg_color="purple", italics=True),
                string_formatter(" | %(levelname)s | ", fg_color="cyan", bold=True),
                "%(message)s",
            ]
        ),
        logging.INFO: "".join(
            [
                string_formatter("[%(asctime)s.%(msecs)03d] ", italics=True, underline=False),
                "< ",
                string_formatter("%(name)s", fg_color="bright_blue", underline=True),
                " >",
                string_formatter(" | %(levelname)s | ", fg_color="green", bold=True),
                "%(message)s",
            ]
        ),
        logging.WARNING: "".join(
            [
                string_formatter("[%(asctime)s.%(msecs)03d] ", fg_color="yellow", italics=True, underline=False),
                "from < ",
                string_formatter("%(filename)s", fg_color="bright_blue", underline=True),
                " >",
                string_formatter(" line: %(lineno)d ", fg_color="purple", italics=True),
                "\n",
                "  >>>  ",
                string_formatter("| %(levelname)s | ", fg_color="yellow", bold=True),
                "%(message)s",
            ]
        ),
        logging.ERROR: "".join(
            [
                string_formatter("[%(asctime)s.%(msecs)03d] ", fg_color=(205, 0, 0), italics=True, underline=False),
                "from < ",
                string_formatter("%(filename)s", fg_color="bright_blue", underline=True),
                " >",
                string_formatter(" line: %(lineno)d ", fg_color="purple", italics=True),
                "\n",
                "  >>>  ",
                string_formatter("| %(levelname)s | %(message)s", fg_color=(205, 0, 0), bold=True),
            ]
        ),
        logging.CRITICAL: "".join(
            [
                string_formatter("[%(asctime)s.%(msecs)03d] ", fg_color=(255, 20, 147), italics=True, underline=False),
                "from < ",
                string_formatter("%(filename)s", fg_color="bright_blue", underline=True),
                " >",
                string_formatter(" line: %(lineno)d ", fg_color="purple", italics=True),
                "\n",
                "  >>>  ",
                string_formatter("| %(levelname)s | %(message)s", fg_color=(255, 20, 147), bold=True),
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
    ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        if record.levelno not in logging._levelToName:
            return self.ANSI_ESCAPE.sub("", formatter.format(record))
        return formatter.format(record)


def _setup_logging(cfg: LoggerConfig, logger_name: str = "gal3d") -> logging.Logger:
    """
    Set up the logger according to the provided configuration.

    Parameters
    ----------
    cfg : LoggerConfig
        Logger configuration dataclass.
    logger_name : str
        Name of the logger to configure (default: "gal3d").

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    if cfg.level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        raise ValueError(f"Invalid logging level: {cfg.level}")
    logger.setLevel(cfg.level)

    # Remove all handlers before adding new ones
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    ch = logging.StreamHandler()
    ch.setLevel(cfg.stream_level)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    if cfg.save_file:
        fh = logging.FileHandler(cfg.file_name, encoding="utf-8")
        fh.setLevel(cfg.file_level)
        fh.setFormatter(NoColorFormatter())
        logger.addHandler(fh)

    return logger


def set_logging_level(level: int = logging.INFO) -> None:
    """
    Set the logging level for the 'gal3d' logger.

    Parameters
    ----------
    level : int, optional
        Logging level for the ``gal3d`` logger. Common values include
        ``logging.DEBUG`` (10), ``logging.INFO`` (20), ``logging.WARNING`` (30),
        ``logging.ERROR`` (40), and ``logging.CRITICAL`` (50). The default is
        ``logging.INFO`` (20).
    """
    logger = logging.getLogger("gal3d")
    logger.setLevel(level)


logger = _setup_logging(config.logger)
