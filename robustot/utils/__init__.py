import io
import sys
from typing import TextIO


class suppress_print:
    enabled: bool
    _original_stdout: TextIO | None

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._original_stdout = None

    def __enter__(self):
        if self.enabled:
            self._original_stdout = sys.stdout
            sys.stdout = io.StringIO()
        else:
            self._original_stdout = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
