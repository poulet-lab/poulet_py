try:
    from collections.abc import Callable
    from dataclasses import dataclass
    from functools import partial
    from time import monotonic_ns

    from adafruit_bus_device.i2c_device import I2CDevice
    from board import SCL, SDA
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr
    from pyftdi.i2c import I2cIOError, I2cNackError

    from poulet_py import (
        LOGGER,
        BaseSource,
        DRV2605Stimulus,
        precise_sleep,
    )

except ImportError as e:
    raise ImportError(
        """
Missing DRV2605L source dependencies.

Install options:
- Dedicated: pip install poulet_py[sources] adafruit-blinka
- Module:    pip install poulet_py[io] adafruit-blinka
- Full:      pip install poulet_py[all] adafruit-blinka
"""
    ) from e


DRV2605_ADDR = 0x5A

REG_MODE = 0x01
REG_RTP_INPUT = 0x02
REG_LIBRARY = 0x03
REG_WAVESEQ1 = 0x04
REG_GO = 0x0C
REG_OD_CLAMP = 0x17
REG_CONTROL2 = 0x1C
REG_CONTROL3 = 0x1D

MODE_INTERNAL_TRIGGER = 0x00
MODE_RTP = 0x05
MODE_STANDBY = 0x40

LIBRARY_TS2200A = 0x01

BIDIR_INPUT_MASK = 0x80
ERM_OPEN_LOOP_MASK = 0x20
DATA_FORMAT_RTP_UNSIGNED = 0x08
LRA_OPEN_LOOP_MASK = 0x01

MODE_CODE_RTP = 0
MODE_CODE_PLAY_WAVEFORM = 1

RTP_ZERO_DRIVE = 0x00
REGISTER_FULL_SCALE_V = 5.6


@dataclass(frozen=True, slots=True)
class _PreparedDRVStimulus:
    mode: str
    duration_ms: int
    drive_voltage: float
    waveform: int
    repeat_count: int
    drive_value: int
    od_clamp: int


class DRV2605Source(BaseSource):
    """
    DRV2605L source using the original shared direct busio.I2C ownership.

    RTP is configured as closed-loop, unidirectional and unsigned:
    - RTP_INPUT=0xFF requests full-scale input.
    - RTP_INPUT=0x00 requests 0% input and automatic braking.
    - OD_CLAMP limits drive, overdrive and braking voltage.

    After the first RTP configuration, every repeated RTP stimulus uses only:
    - one RTP_INPUT write to start;
    - one RTP_INPUT=0 write to brake.

    Mode, control-register and waveform configuration are cached. Configuration
    work is performed inside pre_delay so it does not extend stimulus onset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=DRV2605_ADDR)
    bus_frequency: int = Field(default=400_000, gt=1)

    ftdi_latency_ms: int | None = Field(
        default=None,
        ge=1,
        le=255,
        description="FTDI latency timer in ms. Use None to leave unchanged.",
    )

    i2c: I2C | None = Field(
        default=None,
        description=(
            "Original speed_up_dev ownership: pass one experiment-level "
            "shared busio.I2C object to the DRV and both INA228 sources. "
            "If None, this source creates and owns an internal bus."
        ),
    )

    maximum_voltage: float = Field(
        default=3.3,
        gt=0.0,
        le=REGISTER_FULL_SCALE_V,
        description=("Maximum voltage converted and written to OD_CLAMP. Default: 3.3 V."),
    )

    i2c_retry_attempts: int = Field(default=5, ge=0)
    i2c_retry_backoff_s: float = Field(default=0.005, ge=0.0)
    continue_on_i2c_error: bool = Field(default=True)

    _device: I2CDevice | None = PrivateAttr(default=None)
    _internal_i2c: bool = PrivateAttr(default=False)

    _last_mode: str | None = PrivateAttr(default=None)
    _last_waveform: tuple[int, int] | None = PrivateAttr(default=None)
    _library_configured: bool = PrivateAttr(default=False)

    _control2_original: int = PrivateAttr(default=0)
    _control3_original: int = PrivateAttr(default=0)
    _control2_current: int = PrivateAttr(default=0)
    _control3_current: int = PrivateAttr(default=0)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"),
            ("time_roundtrip", "uint64"),
            ("mode", "uint8"),
            ("waveform", "uint8"),
            ("repeat_count", "uint8"),
            ("drive_voltage", "float32"),
            ("drive_value", "uint8"),
            ("maximum_voltage", "float32"),
            ("od_clamp", "uint8"),
        ]

    def _open(self):
        if self.i2c is None:
            LOGGER.info("Opening internal I2C bus for DRV2605L %s", self.name)
            self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
            self._internal_i2c = True

        self._set_ftdi_latency_timer()
        self._device = I2CDevice(self.i2c, self.address)

        try:
            self._initialize_drv2605()
        except Exception:
            self._device = None
            if self._internal_i2c and self.i2c is not None:
                self.i2c.deinit()
                self.i2c = None
                self._internal_i2c = False
            raise

    def _close(self):
        self.stop()
        self._device = None

        if self._internal_i2c and self.i2c is not None:
            LOGGER.info("Closing internal I2C bus for DRV2605L %s", self.name)
            self.i2c.deinit()
            self.i2c = None
            self._internal_i2c = False

    def _initialize_drv2605(self):
        """Initialize safely without entering an active stimulus state."""
        self._write_register(REG_MODE, MODE_STANDBY)

        self._control2_original = self._read_register(REG_CONTROL2)
        self._control3_original = self._read_register(REG_CONTROL3)
        self._control2_current = self._control2_original
        self._control3_current = self._control3_original

        self._write_register(
            REG_OD_CLAMP,
            self._voltage_to_register(self.maximum_voltage),
        )
        self._write_register(REG_RTP_INPUT, RTP_ZERO_DRIVE)

        self._last_mode = None
        self._last_waveform = None
        self._library_configured = False

    def _fire(self) -> bool:
        for stimulus in self._stimuli:
            if not isinstance(stimulus, DRV2605Stimulus):
                continue

            prepared = self._prepare_stimulus(stimulus)
            if prepared is None:
                if not self.continue_on_i2c_error:
                    return False
                continue

            if not self._run_prepared_stimulus(stimulus, prepared):
                return False

        return True

    def _prepare_stimulus(
        self,
        stimulus: DRV2605Stimulus,
    ) -> _PreparedDRVStimulus | None:
        config = stimulus.build()
        mode = str(config["mode"])
        duration_ms = int(config["duration"])
        drive_voltage = float(config["drive_voltage"])
        waveform = int(config["waveform"])
        repeat_count = int(config["repeat_count"])
        drive_value = self._voltage_to_register(drive_voltage)
        od_clamp = self._voltage_to_register(self.maximum_voltage)

        pre_delay_start = monotonic_ns()
        configured = self._run_with_retries(
            description=f"{mode} configuration",
            operation=partial(
                self._configure_for_stimulus,
                mode=mode,
                specify_mode=bool(config["specify_mode"]),
                waveform=waveform,
                repeat_count=repeat_count,
                od_clamp=od_clamp,
            ),
        )
        self._sleep_remaining_ms(
            start_ns=pre_delay_start,
            duration_ms=stimulus.pre_delay,
        )

        if not configured:
            self._emergency_stop(mode)
            precise_sleep(duration_ms / 1000.0)
            precise_sleep(stimulus.post_delay / 1000.0)
            return None

        return _PreparedDRVStimulus(
            mode=mode,
            duration_ms=duration_ms,
            drive_voltage=drive_voltage,
            waveform=waveform,
            repeat_count=repeat_count,
            drive_value=drive_value,
            od_clamp=od_clamp,
        )

    def _run_prepared_stimulus(
        self,
        stimulus: DRV2605Stimulus,
        prepared: _PreparedDRVStimulus,
    ) -> bool:
        start_request_ns = monotonic_ns()
        started = self._run_with_retries(
            description=f"{prepared.mode} start",
            operation=partial(
                self._start_stimulus,
                mode=prepared.mode,
                drive_value=prepared.drive_value,
            ),
        )
        start_answer_ns = monotonic_ns()

        if not started:
            self._emergency_stop(prepared.mode)
            precise_sleep(prepared.duration_ms / 1000.0)
            precise_sleep(stimulus.post_delay / 1000.0)
            return self.continue_on_i2c_error

        self._record_start_edge(
            prepared=prepared,
            request_ns=start_request_ns,
            answer_ns=start_answer_ns,
        )

        self._sleep_until_ns(start_answer_ns + prepared.duration_ms * 1_000_000)

        stop_request_ns = monotonic_ns()
        stopped = self._run_with_retries(
            description=f"{prepared.mode} stop",
            operation=partial(self._stop_stimulus, prepared.mode),
        )
        stop_answer_ns = monotonic_ns()

        if stopped:
            self._record_stop_edge(
                prepared=prepared,
                request_ns=stop_request_ns,
                answer_ns=stop_answer_ns,
            )
        else:
            self._last_mode = None
            self._emergency_stop(prepared.mode)
            if not self.continue_on_i2c_error:
                return False

        precise_sleep(stimulus.post_delay / 1000.0)
        return True

    def _record_start_edge(
        self,
        *,
        prepared: _PreparedDRVStimulus,
        request_ns: int,
        answer_ns: int,
    ) -> None:
        mode_code, waveform, repeat_count = self._stored_mode_fields(prepared)
        self._write_command_sample(
            timestamp_ns=request_ns,
            time_roundtrip_ns=0,
            mode_code=mode_code,
            waveform=waveform,
            repeat_count=repeat_count,
            drive_voltage=0.0,
            drive_value=RTP_ZERO_DRIVE,
            od_clamp=prepared.od_clamp,
        )
        self._write_command_sample(
            timestamp_ns=answer_ns,
            time_roundtrip_ns=answer_ns - request_ns,
            mode_code=mode_code,
            waveform=waveform,
            repeat_count=repeat_count,
            drive_voltage=prepared.drive_voltage,
            drive_value=prepared.drive_value,
            od_clamp=prepared.od_clamp,
        )

    def _record_stop_edge(
        self,
        *,
        prepared: _PreparedDRVStimulus,
        request_ns: int,
        answer_ns: int,
    ) -> None:
        mode_code, waveform, repeat_count = self._stored_mode_fields(prepared)
        self._write_command_sample(
            timestamp_ns=request_ns,
            time_roundtrip_ns=0,
            mode_code=mode_code,
            waveform=waveform,
            repeat_count=repeat_count,
            drive_voltage=prepared.drive_voltage,
            drive_value=prepared.drive_value,
            od_clamp=prepared.od_clamp,
        )
        self._write_command_sample(
            timestamp_ns=answer_ns,
            time_roundtrip_ns=answer_ns - request_ns,
            mode_code=mode_code,
            waveform=waveform,
            repeat_count=repeat_count,
            drive_voltage=0.0,
            drive_value=RTP_ZERO_DRIVE,
            od_clamp=prepared.od_clamp,
        )

    @staticmethod
    def _stored_mode_fields(
        prepared: _PreparedDRVStimulus,
    ) -> tuple[int, int, int]:
        if prepared.mode == "rtp":
            return MODE_CODE_RTP, 0, 0
        return (
            MODE_CODE_PLAY_WAVEFORM,
            prepared.waveform,
            prepared.repeat_count,
        )

    def _configure_for_stimulus(
        self,
        *,
        mode: str,
        specify_mode: bool,
        waveform: int,
        repeat_count: int,
        od_clamp: int,
    ):
        if specify_mode:
            self._write_register(REG_OD_CLAMP, od_clamp)

        if mode == "rtp":
            if specify_mode or self._last_mode != "rtp":
                # Closed-loop, unidirectional, unsigned RTP.
                control2 = self._control2_original & ~BIDIR_INPUT_MASK
                control3 = (
                    self._control3_original & ~ERM_OPEN_LOOP_MASK & ~LRA_OPEN_LOOP_MASK
                ) | DATA_FORMAT_RTP_UNSIGNED

                self._set_control_registers(
                    control2=control2,
                    control3=control3,
                    force=specify_mode,
                )

                # Zero the retained RTP command before leaving standby or
                # switching back from waveform mode.
                self._write_register(REG_RTP_INPUT, RTP_ZERO_DRIVE)
                self._write_register(REG_MODE, MODE_RTP)
                self._last_mode = "rtp"
            return

        if specify_mode or self._last_mode != "play_waveform":
            self._set_control_registers(
                control2=self._control2_original,
                control3=self._control3_original,
                force=specify_mode,
            )
            self._write_register(REG_MODE, MODE_INTERNAL_TRIGGER)
            self._last_mode = "play_waveform"

        if specify_mode or not self._library_configured:
            self._write_register(REG_LIBRARY, LIBRARY_TS2200A)
            self._library_configured = True

        waveform_config = (waveform, repeat_count)
        if specify_mode or waveform_config != self._last_waveform:
            slots = [waveform] * repeat_count + [0] * (8 - repeat_count)
            self._write_block(REG_WAVESEQ1, slots)
            self._last_waveform = waveform_config

    def _set_control_registers(
        self,
        *,
        control2: int,
        control3: int,
        force: bool,
    ):
        write_control2 = force or control2 != self._control2_current
        write_control3 = force or control3 != self._control3_current

        if write_control2 and write_control3:
            # CONTROL2 and CONTROL3 are consecutive. One block transaction
            # avoids a second I2C lock acquisition and address phase.
            self._write_block(REG_CONTROL2, [control2, control3])
            self._control2_current = control2
            self._control3_current = control3
            return

        if write_control2:
            self._write_register(REG_CONTROL2, control2)
            self._control2_current = control2

        if write_control3:
            self._write_register(REG_CONTROL3, control3)
            self._control3_current = control3

    def _start_stimulus(
        self,
        *,
        mode: str,
        drive_value: int,
    ):
        if mode == "rtp":
            # Exactly one direct shared-bus register transaction.
            self._write_register(REG_RTP_INPUT, drive_value)
        else:
            self._write_register(REG_GO, 1)

    def _stop_stimulus(self, mode: str):
        if mode == "rtp":
            # Exactly one owner request and one register transaction. In
            # closed-loop unidirectional mode, 0x00 is 0% input and invokes
            # automatic braking while back-EMF indicates continued motion.
            self._write_register(REG_RTP_INPUT, RTP_ZERO_DRIVE)
        else:
            self._write_register(REG_GO, 0)

    def _emergency_stop(self, mode: str):
        """Best-effort stop used only after a failed start/configuration/stop."""
        try:
            if mode == "play_waveform":
                self._write_register(REG_GO, 0)
            else:
                self._write_register(REG_RTP_INPUT, RTP_ZERO_DRIVE)
        except (I2cIOError, I2cNackError, OSError, TimeoutError) as error:
            LOGGER.error(
                "DRV2605L %s emergency zero command failed: %s",
                self.name,
                error,
            )
            try:
                self._write_register(REG_MODE, MODE_STANDBY)
            except (I2cIOError, I2cNackError, OSError, TimeoutError) as standby_error:
                LOGGER.error(
                    "DRV2605L %s emergency standby failed: %s",
                    self.name,
                    standby_error,
                )

    def stop(self):
        if self._device is None:
            return

        try:
            if self._last_mode == "play_waveform":
                self._write_register(REG_GO, 0)
            else:
                # Keep the device in RTP mode at a 0% command so closed-loop
                # braking can complete without another routine mode write.
                self._write_register(REG_RTP_INPUT, RTP_ZERO_DRIVE)
        except (I2cIOError, I2cNackError, OSError, TimeoutError) as error:
            LOGGER.warning(
                "DRV2605L %s failed to stop cleanly: %s",
                self.name,
                error,
            )

    def _write_command_sample(
        self,
        *,
        timestamp_ns: int,
        time_roundtrip_ns: int,
        mode_code: int,
        waveform: int,
        repeat_count: int,
        drive_voltage: float,
        drive_value: int,
        od_clamp: int,
    ):
        self._write_sample(
            (
                timestamp_ns,
                time_roundtrip_ns,
                mode_code,
                waveform,
                repeat_count,
                drive_voltage,
                drive_value,
                self.maximum_voltage,
                od_clamp,
            )
        )

    def _write_register(self, register: int, value: int):
        if self._device is None:
            raise RuntimeError("DRV2605L I2C device is not open.")

        payload = bytes([register & 0xFF, value & 0xFF])
        with self._device as device:
            device.write(payload)

    def _read_register(self, register: int) -> int:
        if self._device is None:
            raise RuntimeError("DRV2605L I2C device is not open.")

        result = bytearray(1)
        with self._device as device:
            device.write_then_readinto(
                bytes([register & 0xFF]),
                result,
            )
        return result[0]

    def _write_block(
        self,
        start_register: int,
        values: list[int],
    ):
        if self._device is None:
            raise RuntimeError("DRV2605L I2C device is not open.")

        payload = bytes([start_register & 0xFF] + [value & 0xFF for value in values])
        with self._device as device:
            device.write(payload)

    def _run_with_retries(
        self,
        *,
        description: str,
        operation: Callable[[], None],
    ) -> bool:
        last_error: Exception | None = None

        for attempt in range(self.i2c_retry_attempts + 1):
            try:
                operation()
                return True
            except (
                I2cIOError,
                I2cNackError,
                OSError,
                TimeoutError,
            ) as error:
                last_error = error
                LOGGER.warning(
                    "DRV2605L %s transient I2C error during %s attempt %s/%s: %s",
                    self.name,
                    description,
                    attempt + 1,
                    self.i2c_retry_attempts + 1,
                    error,
                )
                self._recover_i2c_backend()
                precise_sleep(self.i2c_retry_backoff_s)

        LOGGER.error(
            "DRV2605L %s %s failed after %s attempts. Last error: %s",
            self.name,
            description,
            self.i2c_retry_attempts + 1,
            last_error,
        )
        return False

    def _recover_i2c_backend(self):
        if self.i2c is None:
            return

        try:
            self.i2c._i2c._i2c.flush()
        except Exception as error:
            LOGGER.debug(
                "DRV2605L %s backend flush failed during recovery: %s",
                self.name,
                error,
            )

    def _set_ftdi_latency_timer(self):
        if self.ftdi_latency_ms is None or self.i2c is None:
            return

        try:
            pyftdi_i2c_controller = self.i2c._i2c._i2c
            pyftdi_i2c_controller.ftdi.set_latency_timer(self.ftdi_latency_ms)
            LOGGER.info(
                "DRV2605L %s: set FTDI latency timer to %s ms",
                self.name,
                self.ftdi_latency_ms,
            )
        except AttributeError as error:
            raise RuntimeError("Could not access pyftdi controller through Blinka.") from error

    @staticmethod
    def _voltage_to_register(voltage: float) -> int:
        """Convert the existing 0-5.6 V API scale to an 8-bit register value."""
        return max(
            0,
            min(
                255,
                round(float(voltage) / REGISTER_FULL_SCALE_V * 255),
            ),
        )

    @staticmethod
    def _sleep_remaining_ms(
        *,
        start_ns: int,
        duration_ms: float,
    ):
        elapsed_ms = (monotonic_ns() - start_ns) / 1_000_000
        remaining_ms = max(0.0, float(duration_ms) - elapsed_ms)
        precise_sleep(remaining_ms / 1000.0)

    @staticmethod
    def _sleep_until_ns(deadline_ns: int):
        remaining_ns = deadline_ns - monotonic_ns()
        if remaining_ns > 0:
            precise_sleep(remaining_ns / 1_000_000_000)
