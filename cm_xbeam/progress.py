"""Live console progress without reading tensors or synchronizing CUDA in the timer."""
import sys
import threading
import time
from tqdm.auto import tqdm


class TrainingProgress:
    def __init__(self, total, description, initial=0, enabled=True, position=0,
                 leave=True, unit='batch', file=None, refresh_seconds=1.0):
        self.enabled = enabled
        self.refresh_seconds = refresh_seconds
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._phase = 'loading'
        self._phase_started = time.monotonic()
        self._fields = {}
        self._bar = tqdm(total=total, initial=initial, desc=description, unit=unit,
                         position=position, leave=leave, dynamic_ncols=True,
                         mininterval=0.5, disable=not enabled,
                         file=file if file is not None else sys.stderr)

    def __enter__(self):
        if self.enabled:
            self._render()
            self._thread = threading.Thread(target=self._heartbeat, name='training-progress', daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _render(self):
        if not self.enabled:
            return
        with self._lock:
            elapsed = time.monotonic() - self._phase_started
            fields = {'phase': f'{self._phase} {elapsed:.0f}s', **self._fields}
            self._bar.set_postfix(fields, refresh=True)

    def _heartbeat(self):
        while not self._stop.wait(self.refresh_seconds):
            try:
                self._render()
            except (OSError, ValueError):
                # A closed console must not interrupt the actual training job.
                self._stop.set()

    def set_phase(self, phase, **fields):
        with self._lock:
            self._phase = phase
            self._phase_started = time.monotonic()
            self._fields.update(fields)
        self._render()

    def set_description(self, description):
        with self._lock:
            self._bar.set_description_str(description, refresh=False)

    def advance(self, **fields):
        with self._lock:
            self._fields.update(fields)
            if self.enabled:
                self._bar.update(1)
            else:
                self._bar.n += 1
        self._render()

    @property
    def completed(self):
        return self._bar.n

    @staticmethod
    def write(message):
        # Keep machine-readable stdout separate from progress bars on stderr.
        tqdm.write(message, file=sys.stdout)
        sys.stdout.flush()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            self._bar.close()
