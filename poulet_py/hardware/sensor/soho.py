"""
Soho (Ponemah) data collection interface module.

This module provides a Python interface for collecting physiological data
from DSI Ponemah software via TCP socket connection. It supports real-time
data streaming, keyboard-controlled recording sessions, and CSV export.

Examples
--------
>>> soho = Soho(host="192.168.1.100", port=9000)
>>> soho.set_output_file("/data", "experiment", "subject01.csv")
>>> soho.start()
>>> soho.wait_for_completion()
>>> soho.save()
"""

try:
    from collections.abc import Callable
    from os.path import basename, exists, join
    from re import sub
    from socket import AF_INET, SOCK_STREAM, create_connection
    from socket import socket as Socket
    from struct import unpack
    from subprocess import Popen
    from threading import Thread
    from time import sleep, time
    from typing import Any

    from pandas import DataFrame, MultiIndex, concat
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
    from pynput.keyboard import Key, Listener
    from rich.console import Console
    from rich.prompt import Confirm

    from poulet_py import LOGGER

except ImportError as e:
    msg = """
Missing 'soho' module. Install options:
- Dedicated:    pip install poulet_py[soho]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e

console = Console()

HOST = "localhost"
PORT = 6732
PONEMAH_EXECUTABLE = r"C:\\Ponemah\\PPP3.exe"


class Soho(BaseModel):
    """
    Interface for collecting physiological data from DSI Ponemah software.

    This class establishes a TCP socket connection to the Ponemah server and
    collects streaming data in real-time. It provides keyboard controls for
    managing recording sessions and supports CSV export of collected data.

    Parameters
    ----------
    host : str
        IP address or hostname of the Ponemah server.
    port : int
        TCP port number for the Ponemah connection (1-65535).
    output_path : str, optional
        Default output directory path for data files.
    error_log_path : str, optional
        Default directory path for error log files.

    Attributes
    ----------
    host : str
        IP address or hostname of the Ponemah server.
    port : int
        TCP port number for the Ponemah connection.
    output_path : str or None
        Default output directory path for data files.
    error_log_path : str or None
        Default directory path for error log files.
    error_log_file : str or None
        Full path to the error log file.
    output_file : str or None
        Full path to the output CSV file.
    data : DataFrame or None
        Collected data as a pandas DataFrame with MultiIndex columns.
    experiment_start_time : float or None
        Unix timestamp marking the start of the experiment.
    on_data_callback : Callable or None
        Optional callback function invoked when data is received.

    Methods
    -------
    set_error_log_path(path, file_name)
        Configure the error log file path.
    set_output_file(path, extra_name, base_file_name)
        Configure the output CSV file path.
    set_experiment_start_time(start_time)
        Set the experiment start timestamp.
    set_on_data_callback(callback)
        Register a callback for data reception events.
    collect()
        Collect data from the Ponemah server.
    save()
        Save collected data to CSV file.
    start()
        Begin data collection with keyboard controls.
    stop()
        Stop data collection.
    wait_for_completion()
        Block until collection threads complete.
    pause_and_test_connection()
        Pause recording and test the server connection.
    log_error(message, log_file)
        Log an error message.

    Examples
    --------
    >>> soho = Soho(host="192.168.1.100", port=9000)
    >>> soho.set_output_file("/data", "experiment", "subject01.csv")
    >>> soho.set_error_log_path("/logs", "errors.log")
    >>> soho.start()
    >>> soho.wait_for_completion()
    >>> soho.save()
    >>> print(soho.data.head())
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    host: str = Field(HOST, min_length=1)
    port: int = Field(PORT, ge=1, le=65535)
    output_path: str | None = None
    error_log_path: str | None = None
    error_log_file: str | None = None
    output_file: str | None = None
    data: DataFrame | None = None
    experiment_start_time: float | None = None
    on_data_callback: Callable[[], None] | None = None

    _stop: bool = PrivateAttr(False)
    _active: bool = PrivateAttr(True)
    _collection_thread: Thread | None = PrivateAttr(None)
    _keyboard_listener_instance: Listener | None = PrivateAttr(None)

    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        output_path: str | None = None,
        error_log_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Soho data collector instance.

        Parameters
        ----------
        host : str
            IP address or hostname of the Ponemah server.
        port : int
            TCP port number for the Ponemah connection (1-65535).
        output_path : str, optional
            Default output directory path for data files.
        error_log_path : str, optional
            Default directory path for error log files.
        **kwargs
            Additional keyword arguments passed to BaseModel.
        """
        super().__init__(
            host=host, port=port, output_path=output_path, error_log_path=error_log_path, **kwargs
        )

    def set_error_log_path(self, path: str, file_name: str) -> None:
        """
        Configure the file path for error logging.

        Parameters
        ----------
        path : str
            Directory path where the error log file will be stored.
        file_name : str
            Name of the error log file.
        """
        self.error_log_file = join(path, file_name)

    def set_output_file(self, path: str, extra_name: str, base_file_name: str) -> None:
        """
        Configure the output CSV file path.

        The final filename is constructed as "{extra_name}_{base_file_name}".

        Parameters
        ----------
        path : str
            Directory path where the output file will be stored.
        extra_name : str
            Prefix to prepend to the base filename.
        base_file_name : str
            Base name of the output file (typically includes .csv extension).
        """
        name = f"{extra_name}_{base_file_name}"
        self.output_file = join(path, name)

    def set_experiment_start_time(self, start_time: float) -> None:
        """
        Set the experiment start timestamp for relative timing.

        When set, collected data will include an experiment_timestamp column
        calculated as the offset from this start time.

        Parameters
        ----------
        start_time : float
            Unix timestamp (seconds since epoch) marking experiment start.
        """
        self.experiment_start_time = start_time

    def set_on_data_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function for data reception events.

        The callback is invoked each time a data packet is received from
        the Ponemah server. Exceptions in the callback are logged but do
        not interrupt data collection.

        Parameters
        ----------
        callback : Callable[[], None]
            A function with no arguments to be called on data reception.
        """
        self.on_data_callback = callback

    def collect(self) -> DataFrame:
        """
        Collect data from the Ponemah server via TCP socket.

        Establishes a connection to the configured host and port, then
        continuously reads and parses data packets until stopped. Data is
        accumulated and merged with any previously collected data.

        Returns
        -------
        DataFrame
            Collected data with MultiIndex columns organized by channel
            and parameter name. Includes metadata columns for timing and
            subject identification.

        Raises
        ------
        ConnectionError
            If the socket connection fails or is unexpectedly closed.
        OSError
            If a network-related error occurs during data collection.
        """
        metadata: dict[str, dict[str, Any]] = {}
        rows: dict[tuple[str, str], dict[tuple[str, str], str]] = {}
        previous_data = self.data

        try:
            with Socket(AF_INET, SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                sock.settimeout(1.0)
                while not self._stop:
                    try:
                        raw_len = self._read_exact(sock, 2)
                        length = unpack(">H", raw_len)[0]
                        payload = self._read_exact(sock, length)
                        text = payload.decode("ascii", errors="ignore").strip()
                        fields = text.split(";")
                        msg_type = fields[0]

                        if msg_type == "1":
                            inst_id = fields[1]
                            metadata[inst_id] = {
                                "subject_id": fields[3],
                                "channel_number": fields[4],
                                "channel_label": fields[5],
                                "channel_analysis": fields[6],
                                "param_names": fields[7].split(","),
                            }
                        elif msg_type == "2":
                            ref_id = fields[2]
                            values = fields[3].split(",")
                            meta = metadata.get(ref_id)
                            if meta:
                                param_names = meta["param_names"]
                                data = dict(zip(param_names, values, strict=False))

                                elapsed = data.pop("ElapsedTime", "")
                                real = data.pop("RealTime", "")
                                event = data.pop("Event", "")
                                key = (elapsed, real)

                                row = rows.setdefault(
                                    key,
                                    {
                                        ("meta", "ElapsedTime"): elapsed,
                                        ("meta", "RealTime"): real,
                                        ("meta", "Event"): event,
                                    },
                                )
                                if self.experiment_start_time is not None:
                                    timestamp = time() - self.experiment_start_time
                                    row[("meta", "experiment_timestamp")] = f"{timestamp:.6f}"
                                row[("meta", "subject_id")] = meta["subject_id"]
                                channel = meta["channel_label"]
                                for name, value in data.items():
                                    row[(channel, name)] = value

                                if self.on_data_callback is not None:
                                    try:
                                        self.on_data_callback()
                                    except Exception as e:
                                        Soho.log_error(
                                            f"Error in data callback: {e}", self.error_log_file
                                        )
                    except TimeoutError:
                        continue
                    except (ConnectionError, OSError) as err:
                        Soho.log_error(str(err), self.error_log_file)
                        break
        except (ConnectionError, OSError) as err:
            Soho.log_error(str(err), self.error_log_file)

        ordered_rows = list(rows.values())
        df = DataFrame(ordered_rows)
        if not df.empty and len(df.columns) > 0:
            df.columns = MultiIndex.from_tuples(df.columns)

        if previous_data is not None and not previous_data.empty:
            if not df.empty:
                self.data = concat([previous_data, df], ignore_index=True)
            else:
                self.data = previous_data
        else:
            self.data = df

        return self.data

    def save(self) -> None:
        """
        Save collected data to the configured CSV file.

        Writes the accumulated DataFrame to the output file path configured
        via set_output_file(). If no output file or data is available,
        an error is logged and the method returns without action.

        Raises
        ------
        OSError
            If the file cannot be written due to permissions or disk errors.
        """
        if self.output_file is None or self.data is None:
            Soho.log_error("Output file or data not set.", self.error_log_file)
            return
        try:
            self.data.to_csv(self.output_file, index=False)
        except OSError as err:
            Soho.log_error(str(err), self.error_log_file)

    def start(self) -> None:
        """
        Start data collection with interactive keyboard controls.

        Launches background threads for data collection and keyboard
        monitoring. Prompts the user to confirm readiness before starting.
        During recording, press 'e' to end or 't' to test the connection.
        """
        self._active = True
        self._stop = False

        console.input(
            "[bold yellow]\nWARNING: Ensure that the continuous sampling is "
            "activated. Press Enter when you're ready to start recording."
        )

        console.rule("[bold cyan]🟢 RECORDING WITH PONEMAH[/bold cyan]", style="bold cyan")
        console.print("Press 'e' to end recording, 't' to test connection.")

        self._collection_thread = Thread(target=self.collect)
        self._collection_thread.start()

        self._start_keyboard_listener()

        console.print("[bold green]Soho recording started.[/bold green]")

    def stop(self) -> None:
        """
        Signal the data collector to stop recording.

        Sets internal flags to terminate the collection loop and stops
        the keyboard listener. The collection thread will complete its
        current operation before fully stopping.
        """
        self._stop = True
        self._active = False
        self._stop_keyboard_listener()

    def wait_for_completion(self) -> None:
        """
        Block until all collection threads have completed.

        Stops the keyboard listener and waits for the data collection
        thread to finish. Call this method after start() to ensure all
        data has been collected before saving or processing.
        """
        self._stop_keyboard_listener()
        if self._collection_thread:
            self._collection_thread.join()

    def pause_and_test_connection(self) -> None:
        """
        Pause recording to test the Ponemah server connection.

        Temporarily stops data collection, guides the user through testing
        the remote connection in Ponemah, and offers the option to resume
        recording afterward. Previously collected data is preserved.
        """
        console.print("[bold yellow]Pausing recording...[/bold yellow]")
        self._stop = True

        if self._collection_thread:
            self._collection_thread.join()

        console.print("[bold cyan]🔌 CONNECTION TEST[/bold cyan]")
        console.input(
            "Go to Ponemah 'Experiment Setup' and click 'Test' remote "
            "connection. Press Enter when ready to test."
        )

        test_ponemah_connection(self.host, self.port)

        console.input(
            "Close the remote connection test window by pressing 'OK'. Press Enter when done."
        )

        confirm_resume = Confirm.ask("[yellow]Resume recording? (y/n)[/yellow]", default=True)

        if confirm_resume:
            console.input(
                "[bold yellow]\nWARNING: Ensure that the continuous sampling is "
                "activated. Press Enter when you're ready to start recording."
            )
            self._stop = False
            self._collection_thread = Thread(target=self.collect)
            self._collection_thread.start()
            console.print("[bold green]Recording resumed.[/bold green]")
        else:
            console.print("[bold red]Recording not resumed.[/bold red]")
            self._active = False

    def _start_keyboard_listener(self) -> None:
        """Start the pynput keyboard listener."""
        self._keyboard_listener_instance = Listener(on_press=self._on_key_press)
        self._keyboard_listener_instance.start()

    def _stop_keyboard_listener(self) -> None:
        """Stop the pynput keyboard listener."""
        if self._keyboard_listener_instance is not None:
            self._keyboard_listener_instance.stop()
            self._keyboard_listener_instance = None

    def _on_key_press(self, key: Key) -> bool | None:
        """Handle key press events from pynput."""
        if not self._active:
            return False

        try:
            key_char = key.char if hasattr(key, "char") else None
        except AttributeError:
            return None

        if key_char is None:
            return None

        if key_char.lower() == "e":
            confirm = Confirm.ask("Are you sure you want to stop recording? (y/n): ", default=True)

            if confirm:
                console.print("[bold green]Stopping recording...[/bold green]")
                self.stop()
                return False

        elif key_char.lower() == "t":
            confirm = Confirm.ask("Pause recording to test connection? (y/n): ", default=True)

            if confirm:
                self.pause_and_test_connection()

        return None

    @staticmethod
    def _read_exact(sock: Socket, n: int) -> bytes:
        """
        Read exactly n bytes from a socket.

        Parameters
        ----------
        sock : Socket
            Connected socket to read from.
        n : int
            Exact number of bytes to read.

        Returns
        -------
        bytes
            The requested bytes read from the socket.

        Raises
        ------
        ConnectionError
            If the socket is closed before all bytes are read.
        """
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed while reading")
            buf.extend(chunk)
        return bytes(buf)

    @staticmethod
    def log_error(message: str, log_file: str | None = None) -> None:
        """
        Log an error message to the application logger and optionally to a file.

        Parameters
        ----------
        message : str
            The error message to log.
        log_file : str, optional
            Path to a file where the message should also be appended.
            If None, the message is only logged to LOGGER.
        """
        LOGGER.error(message)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")


def test_ponemah_connection(host: str, port: int) -> bool:
    """
    Test the TCP connection to a Ponemah server.

    Attempts to connect to the specified host and port, waiting for a valid
    response from the Ponemah remote connection test. Retries indefinitely
    until a successful connection is established.

    Parameters
    ----------
    host : str
        IP address or hostname of the Ponemah server.
    port : int
        TCP port number for the Ponemah connection.

    Returns
    -------
    bool
        True when a successful connection response is received.
    """
    console.print("Testing remote connection with PONEMAH")
    while True:
        try:
            with create_connection((host, port), timeout=1) as sock:
                sock.settimeout(1)
                raw = sock.recv(1024).decode("ascii", errors="ignore")
                clean = sub(r"[^ -~]", "", raw).strip()
                if ";;0" in clean or "0;;" in clean:
                    console.print("[bold green]Response successful[/bold green]")
                    return True
                console.print(f"[red]Unexpected response:[/red] {clean!r}")
        except OSError:
            console.print(
                "[red]Connection refused, start test in setup remote connection test[/red]"
            )
        sleep(1)


def open_ponemah(
    executable: str = PONEMAH_EXECUTABLE,
    wait_seconds: float = 4.0,
    run_as_admin: bool = True,
) -> bool:
    """
    Launch Ponemah executable if available.

    Ponemah is the DSI telemetry software that Soho connects to for data
    collection. This helper launches it, typically with elevated privileges.

    Parameters
    ----------
    executable : str, optional
        Full path to the Ponemah executable.
    wait_seconds : float, optional
        Time to wait after launching, allowing the UI to open.
    run_as_admin : bool, optional
        If True, launch with RunAs (admin). Default True.

    Returns
    -------
    bool
        True if Ponemah is already running or launch succeeds, otherwise False.
    """
    try:
        import psutil
    except ImportError:
        psutil = None

    name = basename(executable)
    if psutil is not None:
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == name:
                console.print("[bold green]Ponemah already running.[/bold green]")
                return True

    if not exists(executable):
        msg = f"Ponemah executable not found: {executable}"
        LOGGER.error(msg)
        console.print(f"[red]{msg}[/red]")
        return False

    try:
        if run_as_admin:
            import subprocess

            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f'Start-Process "{executable}" -Verb RunAs',
                ],
                check=True,
            )
        else:
            Popen(executable)
        LOGGER.info("Ponemah launched successfully")
        console.print("[bold green]Ponemah launched.[/bold green]")
        sleep(wait_seconds)
        return True
    except OSError as err:
        msg = f"Failed to open Ponemah: {err}"
        LOGGER.error(msg)
        console.print(f"[red]{msg}[/red]")
        return False
