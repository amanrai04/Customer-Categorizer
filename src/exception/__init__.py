"""Project specific exception.

Every component wraps low level errors in :class:`CustomerException` so the
log always records which file and line actually failed, instead of a bare
traceback pointing at a deeply nested ``except`` block.
"""

import os
import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Build a ``file, line, message`` description of a raised exception.

    Args:
        error: The exception instance being re-raised.
        error_detail: The ``sys`` module captured at the call site, used to walk
            the traceback.

    Returns:
        A single line describing where the failure happened.
    """
    _, _, exc_traceback = error_detail.exc_info()
    file_name = os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1]
    return (
        f"Error occurred in [{file_name}] at line [{exc_traceback.tb_lineno}]: "
        f"{str(error)}"
    )


class CustomerException(Exception):
    """Custom exception enriched with the failing file and line number."""

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
