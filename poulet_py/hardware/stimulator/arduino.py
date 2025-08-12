try:
    import csv
    import os
    import time

    import serial

    from poulet_py.config.logging import LOGGER, setup_logging
except ImportError as e:
    msg = """
Missing 'arduino' module. Install options:
- Dedicated:    pip install poulet_py[arduino]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class Arduino:
    def __init__(self, ports=None):
        """
        Initialize the Arduino class with the given ports.

        Args:
            ports (list): List of serial port addresses.
        """
        if ports is None:
            ports = []
        self.arduinos = {}

        for index, port in enumerate(ports):
            temp_arduino = serial.Serial(port, 9600, timeout=1)
            temp_arduino.flushInput()
            self.arduinos[index] = {"port": port, "arduino": temp_arduino}

    def read_data(self, data_parser=float):
        """
        Read data from the Arduino and parse it.

        Args:
            data_parser (callable): Function to parse the data read from the Arduino.
                                    Defaults to float.

        Sets:
            self.parsed_data: Parsed data from Arduino.
        """
        for key in self.arduinos.keys():
            try:
                data = self.arduinos[key]["arduino"].readline()
                self.timestamp = time.time() - self.start_time
                self.arduinos[key]["last_value"] = data_parser(data)

            except Exception as e:
                self.arduinos[key]["last_value"] = float("nan")
                LOGGER.exception("Exception from arduino read_data method")
                self.log_error(f"Exception from arduino {key} read_data method: {e}")

    def set_error_log_path(self, folder_path, error_file_name):
        """
        Set the error log path for the Arduino.

        Args:
            folder_path (str): Path to the folder where the error log will be stored.
            error_file_name (str): Name of the error log file.
        """
        self.error_log_file = os.path.join(folder_path, error_file_name)
        self._file_logger = LOGGER.getChild(f"hardware.stimulator.arduino.{id(self)}")
        setup_logging(self._file_logger, level="error", file=self.error_log_file)

    def set_output_file(self, path, extra_name, data_columns=["data"], base_file_name="arduino"):
        """
        Set the output file for the Arduino.

        Args:
            path (str): Path to the folder where the output file will be saved.
            extra_name (str): An additional name to be added to the base file name.
            base_file_name (str): Base name of the output file. Defaults to 'arduino'.
        """
        for key in self.arduinos.keys():
            extra_name = extra_name + str(key)
            self.output_file_name = f"{base_file_name}_{extra_name}.csv"
            self.arduinos[key]["output_file"] = os.path.join(path, self.output_file_name)

            if not os.path.isfile(self.arduinos[key]["output_file"]):
                with open(self.arduinos[key]["output_file"], mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["timestamp", *data_columns])

    def save_data(self, data):
        """
        Save the data to a CSV file.

        Args:
            timestamp: The timestamp to be saved.
        """
        for key in self.arduinos.keys():
            try:
                with open(self.arduinos[key]["output_file"], mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([self.timestamp, *data])
            except Exception as e:
                self.log_error(f"Error from arduino {key} save_data method: {e}")

    def set_timer(self, start_time):
        """
        Sets the timer for the camera.

        Args:
            start_time (float): The time at which the camera recording started.
        """
        self.start_time = start_time

    def close_port(self):
        """
        Close all serial connections.
        """
        for key in self.arduinos.keys():
            try:
                self.arduinos[key]["arduino"].close()
                LOGGER.info(f"Closed connection on port {self.arduinos[key]['port']}")
            except Exception as e:
                self.log_error(
                    f"Error closing connection on port {self.arduinos[key]['port']}: {e}"
                )

    def log_error(self, error_message):
        LOGGER.error(error_message)
        file_logger = getattr(self, "_file_logger", None)
        if file_logger is not None:
            file_logger.error(error_message)
