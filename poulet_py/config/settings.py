"""
Configuration settings.

This module defines the application's configuration settings using Pydantic's
`BaseSettings` and `BaseModel`. It supports loading configuration from environment
variables and a `.env` file.

Usage Examples
--------------
1. Accessing Default Settings:
    >>> from poulet_py.config.settings import settings
    >>> print(settings.log.level)  # Output: "info"
    >>> print(settings.log.file)  # Output: None

2. Overriding Settings with Environment Variables:
    Set environment variables before running the application:
    ```bash
    export LOG__LEVEL="debug"
    export LOG__FILE="/var/log/myapp.log"
    ```
    Then access the settings in your code:
    >>> from poulet_py.config.settings import settings
    >>> print(settings.log.level)  # Output: "debug"
    >>> print(settings.log.file)  # Output: "/var/log/myapp.log"

3. Loading Settings from a `.env` File:
    Create a `.env` file in the project root:
    ```env
    LOG__LEVEL=warning
    LOG__FILE=/tmp/app.log
    ```
    Then access the settings in your code:
    >>> from poulet_py.config.settings import settings
    >>> print(settings.log.level)  # Output: "warning"
    >>> print(settings.log.file)  # Output: "/tmp/app.log"

4. Programmatically Updating Settings:
    >>> from poulet_py.config.settings import Settings
    >>> custom_settings = Settings(log={"level": "error", "file": "/custom/path.log"})
    >>> print(custom_settings.log.level)  # Output: "error"
    >>> print(custom_settings.log.file)  # Output: "/custom/path.log"
"""

from dotenv import find_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Log(BaseModel):
    level: str = Field("info", description="The logging level")
    file: str | None = Field(
        None,
        description="""The file path for logging. If `None`,
        logging is done to the console.
        """,
    )


class Settings(BaseSettings):
    """
    This class loads configuration from environment variables and a `.env` file.
    It supports nested configuration using the `env_nested_delimiter`
    (e.g., `LOG__LEVEL`).
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    log: Log = Field(Log(), description="Logging configuration settings")


# Global instance of the `Settings` class
SETTINGS = Settings()
"""
An instance of the `Settings` class.

This instance holds the application's configuration, loaded from environment
variables and a `.env` file.
It can be imported and used throughout the application to access
configuration settings.

Example
-------
To access the logging level:
>>> from poulet_py.config.settings import SETTINGS
>>> print(SETTINGS.log.level)
"""
