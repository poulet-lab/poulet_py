from threading import Event, Thread

from numpy import ndarray, zeros
from pydantic import Field, PrivateAttr

from poulet_py import DCAM, LOGGER, BaseSource, precise_sleep


class DCAMSource(BaseSource, DCAM):
    drain_batch_frames: int = Field(default=2, ge=1)
    drain_wait_timeout_s: float = Field(default=0.05, gt=0)

    _temp_buffer: ndarray = PrivateAttr()
    _acquisition_thread: Thread = PrivateAttr()
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self) -> None:
        self._source_buffer_dtype = self._dcam_buffer.dtype

    def open(self) -> None:
        if self._is_open:
            return

        super().open()

        self._stop_acquisition_event.clear()
        self._acquisition_thread = Thread(
            target=self._acquire,
            daemon=True,
        )
        self._acquisition_thread.start()

    def close(self) -> None:
        if not self._is_open:
            return

        self._stop_acquisition_event.set()
        self._acquisition_thread.join(timeout=self.drain_wait_timeout_s + 1.0)

        super().close()

    def _open(self) -> None:
        DCAM.open(self)
        self._temp_buffer = zeros(
            self.drain_batch_frames,
            dtype=self._dcam_buffer.dtype,
        )

    def _close(self) -> None:
        DCAM.close(self)

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000)
        return True

    def _acquire(self) -> None:
        try:
            while not self._stop_acquisition_event.is_set():
                samples = self.read_many_sample(
                    data=self._temp_buffer,
                    n=self.drain_batch_frames,
                    timeout=self.drain_wait_timeout_s,
                )
                if samples:
                    self._write_samples(self._temp_buffer[:samples])

        except Exception:
            LOGGER.exception("DCAM acquisition thread failed")
