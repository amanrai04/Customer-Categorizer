"""Object storage locations.

The name can be overridden with an environment variable so the project can be
pointed at your own bucket without editing code. The default is the value used
by the accompanying infrastructure scripts.
"""

import os

#: Bucket holding the serialised model produced by the training pipeline and
#: served by the prediction pipeline.
TRAINING_BUCKET_NAME: str = os.getenv(
    "TRAINING_BUCKET_NAME", "customer-segmentation-bucket"
)
