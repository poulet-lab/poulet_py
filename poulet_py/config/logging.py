from io import StringIO
from logging import FileHandler, Formatter, Logger, LogRecord, getLogger
from re import sub

from prompt_toolkit import ANSI, print_formatted_text
from rich.console import Console
from rich.logging import RichHandler

from poulet_py import SETTINGS


class Handler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.string_buffer = StringIO()
        self.console = Console(file=self.string_buffer, force_terminal=True)

    def emit(self, record: LogRecord) -> None:
        """Emit the record using prompt_toolkit to preserve colors."""
        try:
            super().emit(record)

            colored_output = sub(r";id=\d+|;file://[^;]*", "", self.string_buffer.getvalue())
            self.string_buffer.truncate(0)
            self.string_buffer.seek(0)

            print_formatted_text(ANSI(colored_output))
        except Exception:
            super().emit(record)


def setup_logging(
    logger: Logger,
    *,
    show_time: bool = False,
    show_path: bool = True,
    markup: bool = True,
    rich_tracebacks: bool = True,
    tracebacks_extra_lines: int = 4,
    tracebacks_word_wrap: bool = True,
    tracebacks_show_locals: bool = True,
    level: int | str = "warning",
    file: str | None = None,
) -> None:
    """
    Configure logging for the provided logger with optional rich formatting
    and file logging.

    Parameters
    ----------
    logger : Logger
        The logger instance to configure.
    show_time : bool, optional
        Whether to show the time in the log output. Default is False.
    show_path : bool, optional
        Whether to show the path in the log output. Default is True.
    markup : bool, optional
        Whether to enable markup in the log output. Default is True.
    rich_tracebacks : bool, optional
        Whether to enable rich tracebacks. Default is True.
    tracebacks_extra_lines : int, optional
        Number of extra lines to show in tracebacks. Default is 4.
    tracebacks_word_wrap : bool, optional
        Whether to enable word wrap in tracebacks. Default is True.
    tracebacks_show_locals : bool, optional
        Whether to show local variables in tracebacks. Default is True.
    level : int or str, optional
        The logging level to set for the logger. Default is warning.
    file : str, optional
        The file to log to. If None, logs are output to the console.
    Returns
    -------
    None
    """
    logger.handlers.clear()

    if file is not None:
        file_handler = FileHandler(file)
        file_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    rich_handler = Handler(
        show_time=show_time,
        show_level=True,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=tracebacks_show_locals,
        tracebacks_word_wrap=tracebacks_word_wrap,
        tracebacks_extra_lines=tracebacks_extra_lines,
        markup=markup,
        show_path=show_path,
    )
    rich_handler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    logger.setLevel(level.upper() if isinstance(level, str) else level)
    logger.propagate = False


# Global instance of the `logger` object
LOGGER = getLogger()
setup_logging(LOGGER, level=SETTINGS.log.level, file=SETTINGS.log.file)
"""
An instance of the `logger` object.

This instance holds can be imported and used throughout the application
for logging.

Example
-------
>>> from poulet_py.config.logging import LOGGER
>>> LOGGER.warning("This is a warning message.")
"""
