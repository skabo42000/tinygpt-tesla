"""One generation owner at a time, including disconnect cleanup."""
import threading


class OneAtATime:
    """A running generation never expires into a second concurrent writer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def try_acquire(self):
        with self._lock:
            if self._owner is not None:
                return None
            self._owner = object()
            return self._owner

    def release(self, owner):
        with self._lock:
            if owner is not None and self._owner is owner:
                self._owner = None
                return True
            return False
