"""Project wide logging configuration.

Importing this module is enough to start logging: the handlers are attached to
the root logger on first import, and every module in ``src`` does
``from src.logger import logging`` instead of importing ``logging`` directly.
"""

import logging
import os

from from_root import from_root

from src.constant.training_pipeline import ARTIFACT_DIR, LOG_DIR, LOG_FILE, PIPELINE_NAME

LOG_FILE_PATH: str = os.path.join(
    from_root(), PIPELINE_NAME, ARTIFACT_DIR, LOG_DIR, LOG_FILE
)

os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
