"""
Soho (Ponemah) data collection interface module.

This module provides a Python interface for collecting physiological data
from DSI Ponemah software via TCP socket connection. It supports real-time
data streaming, keyboard-controlled recording sessions, and CSV export.

Interactive prompts use prompt_toolkit and assume a real terminal (TTY)

Examples
--------
>>> soho = Soho(host="192.168.1.100", port=9000)
>>> soho.set_output_file("/data", "experiment", "subject01.csv")
>>> soho.start()
>>> soho.wait_for_completion()
>>> soho.save()
"""

try:
    import asyncio
    from collections.abc import Callable
    from os.path import basename, exists, join
    from re import sub
    from socket import AF_INET, SOCK_STREAM, create_connection
    from socket import socket as Socket
    from struct import unpack
    from subprocess import Popen
    from threading import Lock, Thread
    from time import sleep, time
    from typing import Any

    from pandas import DataFrame, MultiIndex, concat
    from prompt_toolkit.application import Application, get_app_or_none
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.shortcuts import yes_no_dialog
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
    from rich.console import Console

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


def _build_press_space_application() -> Application:
    """Minimal prompt_toolkit app that exits on Space (cross-platform)."""
    kb = KeyBindings()

    @kb.add("space")
    def _(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    return Application(
        layout=Layout(Window(FormattedTextControl(""))),
        key_bindings=kb,
        full_screen=False,
    )


def _wait_for_space(prompt: str) -> None:
    """
    Wait for user to press Space before continuing.

    Uses Space instead of Enter to avoid conflicts with Ponemah's connection
    test. Only captures keystrokes when the terminal has focus.
    """
    console.print(prompt)
    _build_press_space_application().run(handle_sigint=False)


async def _wait_for_space_async(prompt: str) -> None:
    """Async variant for callers already inside a prompt_toolkit event loop."""
    console.print(prompt)
    await _build_press_space_application().run_async(handle_sigint=False)


async def _ptk_yes_no_async(text: str, *, default: bool = True) -> bool:
    """Async yes/no dialog for use from key-binding background tasks."""
    result = await yes_no_dialog(title="", text=text).run_async(handle_sigint=False)
    return default if result is None else result


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
    _skip_keyboard_join: bool = PrivateAttr(False)
    _collection_thread: Thread | None = PrivateAttr(None)
    _keyboard_thread: Thread | None = PrivateAttr(None)
    _data_lock: Lock = PrivateAttr(default_factory=Lock)
    _packet_count: int = PrivateAttr(0)

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

    def _update_data_from_rows(
        self,
        rows: dict[tuple[str, str], dict[tuple[str, str], str]],
        metadata: dict[str, dict[str, Any]],
        previous_data: DataFrame | None,
    ) -> None:
        """Update self.data from current rows for incremental save support."""
        ordered_rows = list(rows.values())
        df = DataFrame(ordered_rows)
        if not df.empty and len(df.columns) > 0:
            df.columns = MultiIndex.from_tuples(df.columns)
        if previous_data is not None and not previous_data.empty:
            if not df.empty:
                combined = concat([previous_data, df], ignore_index=True)
            else:
                combined = previous_data
        else:
            combined = df
        with self._data_lock:
            self.data = combined

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
        self._packet_count = 0

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
                                self._packet_count += 1
                                if self._packet_count % 100 == 0:
                                    self._update_data_from_rows(rows, metadata, previous_data)
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
                result = concat([previous_data, df], ignore_index=True)
            else:
                result = previous_data
        else:
            result = df

        with self._data_lock:
            self.data = result
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
        if self.output_file is None:
            Soho.log_error("Output file not set.", self.error_log_file)
            return
        with self._data_lock:
            data = self.data
        if data is None:
            Soho.log_error("No data to save.", self.error_log_file)
            return
        try:
            data.to_csv(self.output_file, index=False)
        except OSError as err:
            Soho.log_error(str(err), self.error_log_file)

    def start(
        self,
        *,
        keyboard_handler: str = "internal",
    ) -> None:
        """
        Start data collection with interactive keyboard controls.

        Launches background threads for data collection and optionally
        keyboard monitoring. Prompts the user to confirm readiness before
        starting. During recording, press 'e' to end or 't' to test the
        connection.

        Parameters
        ----------
        keyboard_handler : str, optional
            "internal" (default): Soho runs its own keyboard thread for E/T.
            "external": Caller handles E/T; no keyboard thread started.
        """
        self._active = True
        self._stop = False

        _wait_for_space(
            "[bold yellow]\nWARNING: Ensure that the continuous sampling is "
            "activated. Press Space when you're ready to start recording."
        )

        console.rule("[bold cyan]🟢 RECORDING WITH PONEMAH[/bold cyan]", style="bold cyan")
        console.print("Press 'e' to end recording, 't' to test connection.")

        self._collection_thread = Thread(target=self.collect)
        self._collection_thread.start()

        if keyboard_handler == "internal":
            self._keyboard_thread = Thread(target=self._keyboard_listener_loop)
            self._keyboard_thread.start()

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
        if self._keyboard_thread is not None and not self._skip_keyboard_join:
            self._keyboard_thread.join(timeout=2.0)
            self._keyboard_thread = None

    def wait_for_completion(self) -> None:
        """
        Block until all collection threads have completed.

        Stops the keyboard listener and waits for the data collection
        thread to finish. Call this method after start() to ensure all
        data has been collected before saving or processing.
        """
        self._stop_keyboard_listener()
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=2.0)
            self._keyboard_thread = None
        if self._collection_thread:
            self._collection_thread.join()

    def pause_and_test_connection(self) -> None:
        """
        Pause recording to test the Ponemah server connection.

        Temporarily stops data collection, guides the user through testing
        the remote connection in Ponemah, and offers the option to resume
        recording afterward. Previously collected data is preserved.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._pause_and_test_connection_async())
        else:
            msg = "pause_and_test_connection() cannot be called from a running event loop"
            raise RuntimeError(msg)

    async def _pause_and_test_connection_async(self) -> None:
        """Async implementation (keyboard handler and internal use)."""
        console.print("[bold yellow]Pausing recording...[/bold yellow]")
        self._stop = True

        if self._collection_thread:
            await asyncio.to_thread(self._collection_thread.join)

        console.print("[bold cyan]🔌 CONNECTION TEST[/bold cyan]")
        await _wait_for_space_async(
            "Go to Ponemah 'Experiment Setup' and click 'Test' remote "
            "connection. Press Space when ready to test."
        )

        await asyncio.to_thread(test_ponemah_connection, self.host, self.port)

        await _wait_for_space_async(
            "Close the remote connection test window by pressing 'OK'. Press Space when done."
        )

        confirm_resume = await _ptk_yes_no_async("Resume recording? (y/n)", default=True)

        if confirm_resume:
            await _wait_for_space_async(
                "[bold yellow]\nWARNING: Ensure that the continuous sampling is "
                "activated. Press Space when you're ready to start recording."
            )
            self._stop = False
            self._collection_thread = Thread(target=self.collect)
            self._collection_thread.start()
            console.print("[bold green]Recording resumed.[/bold green]")
        else:
            console.print("[bold red]Recording not resumed.[/bold red]")
            self._active = False

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener thread (no-op if external handler)."""
        self._stop = True
        self._active = False

    def _keyboard_listener_loop(self) -> None:
        """prompt_toolkit keyboard loop for E/T while recording."""
        kb = KeyBindings()

        @kb.add("e")
        def _(event: KeyPressEvent) -> None:
            event.app.create_background_task(self._handle_key_async("e"))

        @kb.add("t")
        def _(event: KeyPressEvent) -> None:
            event.app.create_background_task(self._handle_key_async("t"))

        exit_requested: list[bool] = [False]  # list to allow assignment in closure

        def stop_if_inactive(app: Application) -> None:
            if not self._active and not exit_requested[0]:
                exit_requested[0] = True
                app.exit(result=None)

        app = Application(
            layout=Layout(Window(FormattedTextControl(""))),
            key_bindings=kb,
            full_screen=False,
            refresh_interval=0.1,
            before_render=stop_if_inactive,
        )
        app.run(in_thread=True, handle_sigint=False)

    async def _handle_key_async(self, c: str) -> None:
        """Handle E or T via prompt_toolkit dialogs (same thread as listener app)."""
        if c == "e":
            if await _ptk_yes_no_async(
                "Are you sure you want to stop recording? (y/n)", default=True
            ):
                console.print("[bold green]Stopping recording...[/bold green]")
                self._skip_keyboard_join = True
                try:
                    self.stop()
                finally:
                    self._skip_keyboard_join = False

        elif c == "t":
            if await _ptk_yes_no_async("Pause recording to test connection? (y/n)", default=True):
                await self._pause_and_test_connection_async()

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
