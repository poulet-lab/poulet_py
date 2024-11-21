import serial
import time


class JulaboChiller:
    """Class to interact with a Julabo water chiller via serial port."""

    def __init__(self, port=None, baudrate=9600, timeout=1):
        """
        Initialize the JulaboChiller with the given serial port configuration.

        Args:
            port (str): The serial port (e.g., 'COM12').
            baudrate (int, optional): The baud rate for the serial communication. Default is 9600.
            timeout (int or float, optional): The read timeout value. Default is 1 second.
        """
        if port is None:
            # check in env cariable
            from dotenv import load_dotenv, find_dotenv
            import os
            dotenv_path = find_dotenv(usecwd=True)
            load_dotenv(dotenv_path)
            self.port = os.getenv('CHILLER_PORT')
            if self.port is None:
                raise ValueError("No serial port specified in .env file or argument.")
        else:
            self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = self._configure_serial_port()

    def _configure_serial_port(self):
        """Configure the serial port with the given settings."""
        return serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

    def read(self):
        """Read data from the chiller.

        Returns:
            str: The response from the chiller, or None if no data is available.
        """
        try:
            time.sleep(0.1)  # Give the device some time to respond
            if self.ser.in_waiting > 0:
                data = self.ser.readline().decode('ascii').strip()
                return data
            else:
                return None
        except Exception as e:
            print(f"Error reading from serial port: {e}")
            return None

    def write(self, command):
        """Write a command to the chiller.

        Args:
            command (str): The command to send to the chiller.
        """
        try:
            self.ser.write(command.encode('ascii') + b'\r\n')
            time.sleep(0.1)  # Give the device some time to process the command
        except Exception as e:
            print(f"Error writing to serial port: {e}")

    def close_port(self):
        """Close the serial port connection."""
        self.ser.close()

    def set_temperature(self, temperature):
        """Set the temperature of the chiller.

        Args:
            temperature (float): The temperature to set (in Celsius).
        """
        command = f'OUT_SP_00 {temperature:.1f}'
        self.write(command)

    def get_temperature(self):
        """Get the current temperature from the chiller.

        Returns:
            str: The current temperature reported by the chiller.
        """
        self.write('IN_PV_00')
        return self.read()

    def start(self):
        """Turn on the chiller."""
        self.write('OUT_MODE_05 1')

    def stop(self):
        """Turn off the chiller."""
        self.write('OUT_MODE_05 0')

    def check_version(self):
        """Check the version of the chiller.

        Returns:
            str: The version information.
        """
        self.write('VERSION')
        return self.read()

    def check_status(self):
        """Check the status of the chiller.

        Returns:
            str: The status information.
        """
        self.write('STATUS')
        return self.read()

    def check_started(self):
        """Check whether the chiller has started.

        Returns:
            str: The started status.
        """
        self.write('IN_MODE_05')
        return self.read()

    def get_target_temperature(self):
        """Get the target temperature set point.

        Returns:
            str: The target temperature.
        """
        self.write('IN_SP_00')
        return self.read()