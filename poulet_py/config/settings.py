from dotenv import find_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Log(BaseModel):
    level: str = "info"
    file_path: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    log: Log = Log()


settings = Settings()
