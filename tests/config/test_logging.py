from logging import DEBUG, FileHandler, _nameToLevel, getLogger

from rich.logging import RichHandler

from poulet_py import LOGGER, SETTINGS, setup_logging


def test_setup_logging_with_file_handler(tmpdir):
    logger = getLogger("test_logger_file")
    setup_logging(logger=logger, level="debug", file=str(tmpdir.join("test.log")))

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], FileHandler)
    assert logger.level == DEBUG
    del logger


def test_setup_logging_with_rich_handler():
    logger = getLogger("test_logger_rich")
    setup_logging(logger=logger, level="debug")

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RichHandler)
    assert logger.level == DEBUG
    del logger


def test_logger_instance():
    assert LOGGER.level == _nameToLevel[SETTINGS.log.level.upper()]
