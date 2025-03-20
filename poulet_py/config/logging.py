from logging import FileHandler, Formatter, Logger, getLogger

from rich.console import Console
from rich.logging import RichHandler

from poulet_py.config.settings import settings


def setup_logging(
    logger: Logger,
    terminal_width: int | None = None,
    show_time: bool = False,
    show_path: bool = True,
    markup: bool = True,
    rich_tracebacks: bool = True,
    tracebacks_extra_lines: int = 4,
    tracebacks_word_wrap: bool = True,
    tracebacks_show_locals: bool = True,
    level: int | str = settings.log.level.upper(),
) -> None:
    if settings.log.file_path:
        file_handler = FileHandler(settings.log.file_path + "/poulet_py.log")
        file_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    else:
        console = Console(width=terminal_width) if terminal_width else None
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

    logger.setLevel(level)
    logger.propagate = False


LOGGER = getLogger()
setup_logging(logger=LOGGER)
