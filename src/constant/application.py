"""Web application settings."""

import os

from src.constant.env_variable import CORS_ORIGINS_ENV_KEY

APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "5000"))


def get_cors_origins() -> list:
    """Return the list of origins the API accepts requests from.

    ``CORS_ORIGINS`` is a comma separated list, for example
    ``https://my-app.azurewebsites.net``. It falls back to ``["*"]`` which is
    convenient locally but should not be used in production.
    """
    raw_origins = os.getenv(CORS_ORIGINS_ENV_KEY, "*")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
