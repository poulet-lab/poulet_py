import serial
import os
import csv
import logging
import time

def print_me(message):
    """Print a message with newlines before and after."""
    print(f"\n{message}\n")

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

        # Establish serial connections for each port
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
                self.arduinos[key]["last_value"] = float('nan')
                print(f"Exception from arduino read_data method: {e}")
                self.log_error(f"Exception from arduino {key} read_data method: {e}")

    def set_error_log_path(self, folder_path, error_file_name):
        """
        Set the error log path for the Arduino.

        Args:
            folder_path (str): Path to the folder where the error log will be stored.
            error_file_name (str): Name of the error log file.
        """
        self.error_log_path = os.path.join(folder_path, error_file_name)

    def set_output_file(self, path, extra_name, data_columns = ['data'], base_file_name="arduino"):
        """
        Set the output file for the Arduino.

        Args:
            path (str): Path to the folder where the output file will be saved.
            extra_name (str): An additional name to be added to the base file name.
            base_file_name (str): Base name of the output file. Defaults to 'arduino'.
        """
        for key in self.arduinos.keys():
            # Create the output file name
            extra_name = extra_name + str(key)
            self.output_file_name = f"{base_file_name}_{extra_name}.csv"
            self.arduinos[key]["output_file"] = os.path.join(path, self.output_file_name)

            # Create the CSV file and write the header if it doesn't exist
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
                print(f"Closed connection on port {self.arduinos[key]['port']}")
            except Exception as e:
                self.log_error(f"Error closing connection on port {self.arduinos[key]['port']}: {e}")

    @staticmethod
    def log_error(self, error_message):
        """
        Logs an error message to the error log file.
        """
        if self.error_log_file is not None:
            logging.error(error_message)
        else:
            print(f"An error occurred: {error_message}")
            print("Set the error log file path to log the error with set_error_log_path().")
        