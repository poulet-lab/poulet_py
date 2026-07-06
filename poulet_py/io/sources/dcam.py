"""Drop-in DCAMSource for poulet_py widefield acquisition.

Design preserved from your current implementation:
    class DCAMSource(BaseSource, DCAM)

This file does NOT introduce dcam_kwargs and does NOT wrap a separate DCAM
object. All camera settings must be passed directly to DCAMSource(...), because
DCAMSource inherits the DCAM pydantic fields.

Place this file where your current DCAMSource implementation lives, replacing
that module/file, or copy the class body into the existing source file.
"""

try:
    from time import monotonic_ns

    from numpy import ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import DCAM, LOGGER, AcquisitionType, BaseSource, precise_sleep

except ImportError as e:
    msg = """
Missing 'sources' module. Install options:

    Dedicated: pip install poulet_py[sources]
    Module: pip install poulet_py[io]
    Full: pip install poulet_py[all]

"""
    raise ImportError(msg) from e


class DCAMSource(BaseSource, DCAM):
    """poulet_py source that is also a DCAM instance.

    Important for the widefield trigger setup:
    - Because this class inherits DCAM directly, pass camera settings directly
      to DCAMSource(...), e.g. timing_mode="masterpulse", frame_rate=10.0.
    - Do not pass dcam_kwargs={...}; this class does not consume that wrapper.
    """

    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = self._dcam_buffer.dtype

    def _open(self):
        # Optional but very useful while verifying that StimulatorRuntime uses
        # the patched DCAM fields, not an older exported class.
        if getattr(self, "debug_output", False):
            print("\n--- DCAMSource debug before DCAM.open ---", flush=True)
            print(f"DCAMSource class: {type(self)}", flush=True)
            print(f"DCAM base class: {DCAM}", flush=True)
            print(f"DCAM base module: {getattr(DCAM, '__module__', '<unknown>')}", flush=True)

            for name in (
                "device_index",
                "acquisition_type",
                "exposure_time",
                "frame_rate",
                "timing_mode",
                "trigger_source",
                "trigger_mode",
                "trigger_active",
                "trigger_polarity",
                "trigger_global_exposure",
                "masterpulse_mode",
                "masterpulse_triggersource",
                "output_trigger_connector",
                "output_trigger_kind",
                "output_trigger_polarity",
                "output_trigger_basesensor",
            ):
                print(f"{name}: {getattr(self, name, '<MISSING>')}", flush=True)

            print("--- end DCAMSource debug ---\n", flush=True)

        # Since this class inherits DCAM, open the DCAM part of self.
        DCAM.open(self)
        self._temp_buffer = zeros(self.buffer_size, dtype=self._dcam_buffer.dtype)

    def _close(self):
        DCAM.close(self)

    def _fire(self) -> bool:
        if self.acquisition_type == AcquisitionType.FINITE:
            deadline = monotonic_ns() + self._max_stimulus_duration_ms * 1_000_000

            while monotonic_ns() < deadline:
                sample = self.read_sample()

                if sample is None:
                    LOGGER.error("DCAMSource error in reading sample, drop frame")
                    continue

                self._write_samples(sample)

        else:
            precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            samples = self.read_many_sample(data=self._temp_buffer, n=-1, timeout=-1)
            if samples > 0:
                self._write_samples(self._temp_buffer[:samples])

        return True