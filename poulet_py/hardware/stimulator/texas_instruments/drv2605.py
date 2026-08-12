try:
    from collections.abc import Callable
    from dataclasses import dataclass
    from enum import IntEnum
    from math import sqrt
    from threading import Condition, Event, Thread
    from time import monotonic, monotonic_ns
    from typing import Literal

    from adafruit_bus_device.i2c_device import I2CDevice
    from board import SCL, SDA
    from busio import I2C
    from numpy import ndarray, zeros
    from pydantic import ConfigDict, Field, PrivateAttr, model_validator

    from poulet_py import LOGGER, AcquisitionType, BaseSource, DRV2605Stimulus, precise_sleep
    from poulet_py.hardware.stimulator.texas_instruments._drv260x_regs import (
        AutoCalibrationBackEMFResult,
        AutoCalibrationCompensationResult,
        Libraries,
        Modes,
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


class DRV2605(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=0x5A)
    bus_frequency: int = Field(default=400_000, gt=1)
    ftdi_latency_ms: int | None = Field(default=None, ge=1, le=255)
    i2c: I2C | None = Field(default=None)
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="Type of data acquisition, continuous or finite"
    )
    mode: Modes = Field(
        default=Modes.INTERNAL_TRIGGER
    )  # TODO maybe hide and automate based on stimulus
    motor_type: Literal["erm", "lra"] = Field(default="erm")
    library: Libraries = Field(default=Libraries.ERM_A)  # TODO if motor_type switch to lra lib

    loop_mode: Literal["closed_loop", "open_loop"] = Field(default="closed_loop")
    rated_voltage: int = Field(default=3.0, gt=0.0)
    maximum_voltage: float = Field(default=5.0, gt=0.0)
    calibrate: bool = Field(default=False)
    calibration_timeout: float = Field(default=2.0, gt=0.0, le=10.0)
    calibration_compensation: int = Field(
        default=AutoCalibrationCompensationResult.DEFAULT, ge=0, le=0xFF
    )
    calibration_back_emf: int = Field(default=AutoCalibrationBackEMFResult.DEFAULT, ge=0, le=0xFF)
    bemf_gain: int | None = Field(
        default=0x01,
        ge=0,
        le=0x03,
        repr=False,
    )
    lra_frequency_hz: float = Field(default=175.0, ge=80.0, le=500.0)

    i2c_retry_attempts: int = Field(default=5, ge=0)
    i2c_retry_backoff_s: float = Field(default=0.001, ge=0.0)
    continue_on_i2c_error: bool = Field(default=True)

    _is_open: bool = PrivateAttr(default=False)
    _device: I2CDevice = PrivateAttr()
    _internal_i2c: bool = PrivateAttr(default=False)

    _drv_buffer_idx: int = PrivateAttr(0)
    _drv_buffer_needle: int = PrivateAttr(0)
    _drv_buffer: ndarray = PrivateAttr()

    _stimulus: DRV2605Stimulus = PrivateAttr()
    _done_trigger: Event = PrivateAttr(default_factory=Event)
    _start_trigger: Event = PrivateAttr(default_factory=Event)
    _stop_trigger_thread: Event = PrivateAttr(default_factory=Event)
    _trigger_thread: Thread = PrivateAttr()

    _stop_acquisition_thread: Event = PrivateAttr(default_factory=Event)
    _acquisition_thread: Thread = PrivateAttr()
    _acquisition_cond: Condition = PrivateAttr(default_factory=Condition)

    _waveform: tuple[int, int] | None = PrivateAttr(default=None)
    _library: Library | None = PrivateAttr(default=None)
    _controls: tuple[int, int] = PrivateAttr(default=(0, 0))
    _waveform_controls: tuple[int, int] = PrivateAttr(default=(0, 0))
    _rtp_controls: tuple[int, int] = PrivateAttr(default=(0, 0))
    _rtp_drive: int | None = PrivateAttr(default=None)
    _od_clamp: int = PrivateAttr(default=0)
    _rtp_zero: int = PrivateAttr(default=0)
    _motor_code: int = PrivateAttr(default=0)
    _loop_code: int = PrivateAttr(default=0)
    _resonance_frequency: float | None = PrivateAttr(default=None)
    _calibration: DRV2605CalibrationResult | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_configuration(self):
        if self.loop_mode == "closed_loop" and self.maximum_voltage < self.rated_voltage:
            raise ValueError("maximum_voltage must be >= rated_voltage in closed loop.")

        if self.calibrate and self.loop_mode != "closed_loop":
            raise ValueError("Auto-calibration requires loop_mode='closed_loop'.")

        if self.loop_mode == "closed_loop" and not self.calibrate:
            values = (
                self.auto_cal_comp,
                self.auto_cal_bemf,
                self.bemf_gain,
            )
            if any(value is None for value in values):
                raise ValueError(
                    "Closed-loop operation requires saved calibration values. "
                    "Run calibrate_current_erm.py and install its constants block."
                )

        return self

    @property
    def calibration_result(self) -> DRV2605CalibrationResult | None:
        return self._calibration

    @property
    def resonance_frequency(self) -> float | None:
        return self._resonance_frequency

    def open(self):
        if self._is_open:
            return
        self._set_i2c()
        self._set_i2c_device()
        try:
            self._set_params()
            self._initialize_device()
            self._drv_set_buffer()

            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                self._start()
                self._start_acquisition_thread()
        except Exception:
            self._release_i2c_device()
            self._release_i2c()
            raise

        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._stop_acquisition_thread()
            self._stop()

        self._drv_release_buffer()
        self._release_i2c_device()
        self._release_i2c()

        self._is_open = False

    def info(self):
        self._ensure_open()

        LOGGER.info(
            "DRV2605L %s: %s, %s, RTP %s, rated %.3f V, max %.3f V%s",
            self.name,
            self.motor_type.upper(),
            self.loop_mode,
            "unidirectional" if self.loop_mode == "closed_loop" else "midpoint",
            self.rated_voltage,
            self.maximum_voltage,
            f", {self.lra_frequency_hz:.2f} Hz" if self.motor_type == "lra" else "",
        )
        if self.loop_mode == "closed_loop":
            LOGGER.info(
                "DRV2605L %s: %s calibration values %s",
                self.name,
                self.motor_type.upper(),
                "measured" if self.calibrate else "loaded",
            )

    def stop(self):
        self._ensure_open()
        self._write_register(self.Register.MODE, self.Mode.STANDBY)

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("DCAM is not open")

    def _initialize_device(self):
        self._write_register(self.Register.MODE, self.Mode.STANDBY)

        (_, _, current_cal_comp, current_cal_bemf, feedback, control1, control2, control3) = (
            self._read_registers(self.Register.RATED_VOLTAGE, 8)
        )

        motor_feedback = feedback | N_ERM_LRA if self.motor_type == "lra" else feedback & ~N_ERM_LRA
        if self.loop_mode == "closed_loop" and not self.calibrate:
            motor_feedback = (motor_feedback & ~BEMF_GAIN) | int(self.bemf_gain)

        loop_control = control3 & ~(ERM_OPEN_LOOP | LRA_OPEN_LOOP)
        if self.loop_mode == "open_loop":
            loop_control |= ERM_OPEN_LOOP if self.motor_type == "erm" else LRA_OPEN_LOOP

        # TS2200 ROM data is bidirectional. Closed-loop RTP uses TI's recommended
        # full-resolution unidirectional, unsigned input format.
        self._waveform_controls = (control2 | BIDIR_INPUT, loop_control)
        self._rtp_controls = (
            control2 & ~BIDIR_INPUT if self.loop_mode == "closed_loop" else control2,
            loop_control | RTP_UNSIGNED,
        )
        self._controls = (control2, loop_control)
        self._rtp_zero = 0x00 if self.loop_mode == "closed_loop" else 0x7F
        self._od_clamp = self._overdrive_register()

        if self.loop_mode == "closed_loop":
            cal_comp = current_cal_comp if self.calibrate else int(self.auto_cal_comp)
            cal_bemf = current_cal_bemf if self.calibrate else int(self.auto_cal_bemf)
            # One contiguous write installs the voltage and motor-specific
            # closed-loop calibration while preserving unrelated feedback bits.
            self._write_block(
                self.Register.RATED_VOLTAGE,
                (
                    self._rated_voltage_register(control2),
                    self._od_clamp,
                    cal_comp,
                    cal_bemf,
                    motor_feedback,
                ),
            )
        else:
            self._write_register(self.Register.OD_CLAMP, self._od_clamp)
            if motor_feedback != feedback:
                self._write_register(self.Register.FEEDBACK_CONTROL, motor_feedback)

        if self.calibrate:
            calibrated_control1 = control1 & ~DRIVE_TIME | self._drive_time()
            if calibrated_control1 != control1:
                self._write_register(self.Register.CONTROL1, calibrated_control1)

        if loop_control != control3:
            self._write_register(self.Register.CONTROL3, loop_control)

        if self.motor_type == "lra" and self.loop_mode == "open_loop":
            self._write_register(
                self.Register.OL_LRA_PERIOD,
                self._frequency_to_period(self.lra_frequency_hz),
            )

        if self.calibrate:
            self._calibration = self._run_auto_calibration()
            if not self._calibration.success:
                raise RuntimeError(
                    f"DRV2605L auto calibration failed: status=0x{self._calibration.status:02X}"
                )
            self.calibration_rated_voltage = self.rated_voltage
            self.calibration_maximum_voltage = self.maximum_voltage
            self.auto_cal_comp = self._calibration.auto_cal_comp
            self.auto_cal_bemf = self._calibration.auto_cal_bemf
            self.bemf_gain = self._calibration.bemf_gain

        self._write_register(self.Register.MODE, self.Mode.INTERNAL_TRIGGER)

        self._waveform = self._library = self._rtp_drive = None
        self._motor_code = int(self.motor_type == "lra")
        self._loop_code = int(self.loop_mode == "open_loop")

    def _set_i2c(self):
        if self.i2c is None:
            self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
            self._internal_i2c = True

        self._set_ftdi_latency_timer()

    def _release_i2c(self):
        del self._device

        if self._internal_i2c and self.i2c is not None:
            self.i2c.deinit()
            self.i2c = None
            self._internal_i2c = False

    def _set_ftdi_latency_timer(self):
        if self.ftdi_latency_ms is None:
            return

        try:
            self.i2c._i2c._i2c.ftdi.set_latency_timer(self.ftdi_latency_ms)
        except AttributeError as error:
            raise RuntimeError("Could not access PyFtdi through Blinka.") from error

    def _set_i2c_device(self):
        self._device = I2CDevice(self.i2c, self.address)

    def _release_i2c_device(self):
        del self._device

    def _drv_set_buffer(self):
        try:
            self._drv_buffer = zeros(
                self.buffer_size,
                dtype=[
                    ("timestamp", "uint64"),
                    ("time_roundtrip", "uint64"),
                    ("mode", "uint8"),
                    ("waveform", "uint8"),
                    ("repeat_count", "uint8"),
                    ("drive_voltage", "float32"),
                    ("drive_value", "uint8"),
                    ("maximum_voltage", "float32"),
                    ("od_clamp", "uint8"),
                    ("motor_type", "uint8"),
                    ("loop_mode", "uint8"),
                    ("resonance_frequency", "float32"),
                ],
            )
            self._drv_buffer_idx = 0
            self._drv_buffer_needle = 0
        except Exception as e:
            raise RuntimeError("Buffer initialization failed") from e

    def _drv_release_buffer(self):
        del self._drv_buffer

    def _write_register(self, register: int, value: int):
        self._write_registers(register, (value,))

    def _write_registers(self, register: int, values: tuple[int, ...]):
        with self._device as device:
            device.write(bytes((register, *values)))

    def _read_register(self, register: int) -> int:
        return self._read_registers(register, 1)[0]

    def _read_registers(self, register: int, size: int) -> bytearray:
        data = bytearray(size)
        with self._device as device:
            device.write_then_readinto(bytes((register,)), data)
        return data

    def _run_auto_calibration(self) -> DRV2605CalibrationResult:
        self._write_register(self.Register.MODE, self.Mode.AUTO_CALIBRATION)

        start = monotonic()
        self._write_register(self.Register.GO, 1)

        while self._read_register(self.Register.GO) & 1:
            if monotonic() - start >= self.calibration_timeout:
                self._write_register(self.Register.MODE, self.Mode.STANDBY)
                raise TimeoutError("DRV2605L auto calibration timed out.")
            precise_sleep(0.01)

        status = self._read_register(self.Register.STATUS)
        result_regs = self._read_registers(self.Register.AUTO_CAL_COMP, 3)
        result = DRV2605CalibrationResult(
            success=not bool(status & DIAG_RESULT),
            status=status,
            auto_cal_comp=result_regs[0],
            auto_cal_bemf=result_regs[1],
            bemf_gain=result_regs[2] & BEMF_GAIN,
            elapsed_s=monotonic() - start,
        )
        self._write_register(self.Register.MODE, self.Mode.STANDBY)
        return result

    def _read_resonance(self):
        period = self._read_register(self.Register.LRA_PERIOD)
        if period:
            self._resonance_hz = self._period_to_frequency(period)

    def _fire(self) -> bool:
        for stimulus in self._stimuli:
            if isinstance(stimulus, DRV2605Stimulus):
                prepared = self._prepare_stimulus(stimulus)
                if prepared is None:
                    if not self.continue_on_i2c_error:
                        return False
                elif not self._play_stimulus(stimulus, prepared):
                    return False
        return True

    def _prepare_stimulus(self, stimulus: DRV2605Stimulus) -> _PreparedStimulus | None:
        config = stimulus.build()
        mode = config["mode"]
        voltage = float(config["drive_voltage"])

        if mode == "play_waveform" and self.motor_type == "erm" and self.loop_mode != "open_loop":
            raise ValueError("ERM ROM-library playback requires loop_mode='open_loop'.")
        if mode == "rtp":
            limit = self.rated_voltage if self.loop_mode == "closed_loop" else self.maximum_voltage
            if voltage > limit:
                raise ValueError(f"RTP drive_voltage must be <= {limit:.3f} V for this source.")

        prepared = _PreparedStimulus(
            mode=mode,
            duration_ms=int(config["duration"]),
            voltage=voltage,
            waveform=int(config["waveform"]),
            repeats=int(config["repeat_count"]),
            drive=self._rtp_register(voltage) if mode == "rtp" else self._rtp_zero,
        )

        start = monotonic_ns()
        configured = self._retry("configuration", self._configure_stimulus, prepared)
        self._sleep_remaining_ms(start, stimulus.pre_delay)

        if configured:
            return prepared

        self._emergency_stop()
        precise_sleep((prepared.duration_ms + stimulus.post_delay) / 1000.0)
        return None

    def _configure_stimulus(self, prepared: _PreparedStimulus):
        """Configure and preload while ready-idle with GO=0."""
        self._set_idle()

        if prepared.mode == "rtp":
            self._set_controls(self._rtp_controls)
            if prepared.drive != self._rtp_drive:
                self._write_register(self.Register.RTP_INPUT, prepared.drive)
                self._rtp_drive = prepared.drive
            return

        self._set_controls(self._waveform_controls)
        library = self.Library.LRA if self.motor_type == "lra" else self.Library.ERM_A
        if library != self._library:
            self._write_register(self.Register.LIBRARY, library)
            self._library = library

        waveform = prepared.waveform, prepared.repeats
        if waveform != self._waveform:
            self._write_block(
                self.Register.WAVESEQ1,
                [prepared.waveform] * prepared.repeats + [0] * (8 - prepared.repeats),
            )
            self._waveform = waveform

    def _play_stimulus(self, stimulus: DRV2605Stimulus, prepared: _PreparedStimulus) -> bool:
        request = monotonic_ns()
        started = self._retry("start", self._start_stimulus, prepared.mode)
        answer = monotonic_ns()

        if not started:
            self._emergency_stop()
            precise_sleep((prepared.duration_ms + stimulus.post_delay) / 1000.0)
            return self.continue_on_i2c_error

        self._record_edge(prepared, request, answer, start=True)
        self._sleep_until_ns(answer + prepared.duration_ms * 1_000_000)

        if self.calibrate and self.motor_type == "lra" and prepared.mode == "rtp":
            try:
                self._read_resonance()
            except I2C_ERRORS as error:
                LOGGER.warning("DRV2605L %s resonance read failed: %s", self.name, error)

        request = monotonic_ns()
        stopped = self._retry("stop", self._stop_stimulus, prepared.mode)
        answer = monotonic_ns()

        if stopped:
            self._record_edge(prepared, request, answer, start=False)
        else:
            self._emergency_stop()
            if not self.continue_on_i2c_error:
                return False

        precise_sleep(stimulus.post_delay / 1000.0)
        return True

    def _start_stimulus(self, mode: str):
        if mode == "rtp":
            self._write_register(self.Register.MODE, self.Mode.RTP)
        else:
            self._write_register(self.Register.GO, 1)

    def _stop_stimulus(self, mode: str):
        if mode == "rtp":
            self._write_register(self.Register.MODE, self.Mode.INTERNAL_TRIGGER)
        else:
            self._write_register(self.Register.GO, 0)

    def _set_idle(self):
        self._write_register(self.Register.MODE, self.Mode.INTERNAL_TRIGGER)

    def _set_controls(self, controls: tuple[int, int]):
        if controls != self._controls:
            self._write_block(self.Register.CONTROL2, controls)
            self._controls = controls

    def _record_edge(self, prepared: _PreparedStimulus, request: int, answer: int, *, start: bool):
        mode, waveform, repeats = (
            (MODE_CODE_RTP, 0, 0)
            if prepared.mode == "rtp"
            else (MODE_CODE_WAVEFORM, prepared.waveform, prepared.repeats)
        )
        before_voltage, before_drive, after_voltage, after_drive = (
            (0.0, self._rtp_zero, prepared.voltage, prepared.drive)
            if start
            else (prepared.voltage, prepared.drive, 0.0, self._rtp_zero)
        )
        self._write_command_sample(
            request, 0, mode, waveform, repeats, before_voltage, before_drive
        )
        self._write_command_sample(
            answer, answer - request, mode, waveform, repeats, after_voltage, after_drive
        )

    def _write_command_sample(
        self,
        timestamp: int,
        roundtrip: int,
        mode: int,
        waveform: int,
        repeats: int,
        voltage: float,
        drive: int,
    ):
        self._write_sample(
            (
                timestamp,
                roundtrip,
                mode,
                waveform,
                repeats,
                voltage,
                drive,
                self.maximum_voltage,
                self._od_clamp,
                self._motor_code,
                self._loop_code,
                float("nan") if self._resonance_hz is None else self._resonance_hz,
            )
        )

    def _retry(self, action: str, operation: Callable[..., None], *args) -> bool:
        attempts = self.i2c_retry_attempts + 1
        for attempt in range(attempts):
            try:
                operation(*args)
                return True
            except I2C_ERRORS as error:
                if attempt == attempts - 1:
                    LOGGER.error(
                        "DRV2605L %s %s failed after %d attempt%s: %s",
                        self.name,
                        action,
                        attempts,
                        "" if attempts == 1 else "s",
                        error,
                    )
                    return False
                if attempt == 0:
                    LOGGER.warning(
                        "DRV2605L %s %s failed; retrying: %s",
                        self.name,
                        action,
                        error,
                    )
                self._recover_i2c()
                if self.i2c_retry_backoff_s:
                    precise_sleep(self.i2c_retry_backoff_s)
        return False

    def _recover_i2c(self):
        try:
            self.i2c._i2c._i2c.flush()
        except Exception as error:
            LOGGER.debug("DRV2605L %s I2C flush failed: %s", self.name, error)

    def _emergency_stop(self):
        try:
            self._write_register(self.Register.MODE, self.Mode.STANDBY)
        except I2C_ERRORS as error:
            LOGGER.error("DRV2605L %s emergency standby failed: %s", self.name, error)

    def _rtp_register(self, voltage: float) -> int:
        if self.loop_mode == "closed_loop":
            return min(255, round(voltage / self.rated_voltage * 255))
        return min(0xFF, 0x7F + round(voltage / self.maximum_voltage * 128))

    def _drive_time(self) -> int:
        value = (
            round((500.0 / self.lra_frequency_hz - 0.5) / 0.1) if self.motor_type == "lra" else 0x13
        )
        return max(0, min(31, value))

    def _rated_voltage_register(self, control2: int) -> int:
        if self.motor_type == "erm":
            value = self.rated_voltage / 0.02118
        else:
            sample_time_s = LRA_SAMPLE_TIME_US[(control2 & SAMPLE_TIME) >> 4] * 1e-6
            factor = 1.0 - 4.0 * (sample_time_s + 300e-6) * self.lra_frequency_hz
            if factor <= 0:
                raise ValueError(
                    "Invalid LRA frequency/sample-time combination for rated-voltage calculation."
                )
            value = self.rated_voltage * sqrt(factor) / 0.02058
        return max(0, min(255, round(value)))

    def _overdrive_register(self) -> int:
        if self.motor_type == "erm":
            return max(0, min(255, round(self.maximum_voltage / REGISTER_FULL_SCALE_V * 255)))
        factor = sqrt(max(1e-12, 1.0 - self.lra_frequency_hz * 800e-6))
        value = (
            self.maximum_voltage / (0.02132 * factor)
            if self.loop_mode == "open_loop"
            else self.maximum_voltage / 0.02122
        )
        return max(0, min(255, round(value)))
