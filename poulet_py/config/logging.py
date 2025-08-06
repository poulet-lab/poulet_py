from logging import FileHandler, Formatter, Logger, getLogger

from rich.console import Console
from rich.logging import RichHandler

from poulet_py.config.settings import SETTINGS


def setup_logging(
    logger: Logger,
    *,
    terminal_width: int | None = None,
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
    terminal_width : int, optional
        The width of the terminal for rich console output.
        If None, the default width is used.
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
    if file is not None:
        file_handler = FileHandler(file)
        file_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    else:
        console = Console(width=terminal_width) if terminal_width else Console()
        rich_handler = RichHandler(
            show_time=show_time,
            show_level=True,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=tracebacks_show_locals,
            tracebacks_word_wrap=tracebacks_word_wrap,
            tracebacks_extra_lines=tracebacks_extra_lines,
            markup=markup,
            show_path=show_path,
            console=console,
        )
        rich_handler.setFormatter(Formatter("%(message)s"))
        logger.addHandler(rich_handler)

    logger.setLevel(level.upper())
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
