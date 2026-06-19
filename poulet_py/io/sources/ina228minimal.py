try:
    from threading import Event, Thread
    from time import time_ns
    from typing import Literal
    from adafruit_blinka.microcontroller.generic_linux.rpi_gpio_pin import Pin
    from adafruit_ina228 import (
        INA228,
    )
    from board import SDA, SCL
    from busio import I2C
    from digitalio import DigitalInOut, Direction
    from pydantic import ConfigDict, Field, PrivateAttr

    from poulet_py import LOGGER, BaseSource, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


#INCLUDE SET calibration settings
#especially for up to 16V and 400 mA


class INA228Source_minimal(BaseSource):
    #incase multiple ina228 sources are used, we can specify the i2c address for each one
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    #I2C clock frequency in Hz:
    #Fast mode: 400 kHz
    #I2C 2940 kHz
    #Check how to set the clock frequency for the i2c bus in adafruit_blinka
    #baud_rate: int = Field(default=400_000, description="I2C clock speed in Hz")
    
    sda: Pin | None = Field(default=SDA, description="I2C SDA pin")
    scl: Pin | None = Field(default=SCL, description="I2C SCL pin")
    
    bus_voltage_conv_time: Literal[50, 84, 150, 280, 540, 1052, 2074, 4120] = Field(
        default=50, description="ADC conversion time for bus voltage measurement in microseconds"
    )
    shunt_voltage_conv_time: Literal[50, 84, 150, 280, 540, 1052, 2074, 4120] = Field(
        default=50, description="ADC conversion time for shunt voltage measurement in microseconds, higher conversion times can improve the accurcay of a signal but also increase the time it takes to acquire a signal"
    )
    averaging_count: Literal[1, 4, 16, 64, 128, 256, 512, 1024] = Field(
        default=1, description="Number of samples to average for each reading, higher values can improve the accurcay of a signal but also increase the time it takes to acquire a signal"
    )
    
    #Set calibration mode (closest to expected current and voltage range, check adafruit library for details)
    #need to test whether useful as no mesaurements are currently done through over the shunt res
    #default 32V 2A
    #32V 1A
    #16V 400mA

    _sda: SDA | None = PrivateAttr(None)
    _scl: SCL | None = PrivateAttr(None)
    _ina28: INA228 = PrivateAttr()

    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._buffer_dtype = [
            ("timestamp", "uint64"),
            ("current", "float32"), #default unit: µA
            ("bus_voltage", "float32"), #default unit: V
            ("shunt_voltage", "float32"), #default unit: µV
            ("power", "float32"), #default unit: µW
            ("energy", "float32"), #default unit: J
            ("temperature", "float32"), #default unit: °C
            #("faults", "uint8"),
        ]

    def _open(self):
        try:
            self._i2c = I2C(self._sda, self._scl)


            self._ina228 = INA228(
                self._i2c,
                #rate of 400 kHz schould be deufault and not need to be set
                #baudrate=self.baud_rate,
                )
            #max voltage as temperature maximum
            self._ina228 = self.bus_voltage
            #temperature threshold for faults schould be ~50-60°C, (check during continous measurement) and not changable
            #temp reading in lib
            #self._ina228.reference_temperature_thresholds = self.reference_temperature_thresholds

        except Exception as e:
            msg = f"Failed to initialize INA228: {e}"
            raise RuntimeError(msg) from e

        self._start_acquisition_thread()

    def _start_acquisition_thread(self):
        """Start the background acquisition thread."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            return

        self._stop_acquisition_event.clear()

        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func,
            daemon=True,
            name=f"INA228-Acquisition-{self.name}",
        )
        self._acquisition_thread.start()

    def _close(self):
        """Close the I2C device and stop acquisition thread."""
        self._stop_acquisition_event.set()

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)
            if self._acquisition_thread.is_alive():
                LOGGER.warning("INA228: Acquisition thread is still alive after closure")

        if self._i2c:
            self._i2c.deinit()


        self._acquisition_thread = None

    def _acquisition_thread_func(self):
        """Background thread for continuous SPI data acquisition."""
        while not self._stop_acquisition_event.is_set():
            try:
                #faults = self._ina228._read_register(_INA228_SR_REG, 1)[0]
                #self._ina228._perform_one_shot_measurement()
                timestamp = time_ns()
                temperature = self._ina228.read_temperature()
                current = self._ina228.read_current()
                bus_voltage = self._ina228.read_bus_voltage()            
                shunt_voltage = self._ina228.read_shunt_voltage()
                power = self._ina228.read_power()
                energy = self._ina228.read_energy()

                #reference = self._ina228.unpack_reference_temperature()

                #self._log_faults(faults)
                self._write_sample(timestamp, temperature, current, bus_voltage, shunt_voltage, power, energy)

                precise_sleep(0.1)

            except Exception as e:
                LOGGER.error(f"INA228 acquisition error: {e}")
                break

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        return True
