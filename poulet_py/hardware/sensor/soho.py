try:
    from typing import Any, Dict, Optional, Tuple, Callable

    import os
    import socket
    import struct
    import time
    import threading
    import pandas as pd
    import re
    import msvcrt

    from rich.console import Console
    from rich.prompt import Confirm

    from pydantic import BaseModel, Field, PrivateAttr
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

class Soho(BaseModel):
    """Interface for collecting data from Ponemah."""

    host: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    output_path: Optional[str] = None
    error_log_path: Optional[str] = None

    _stop: bool = PrivateAttr(False)
    _active: bool = PrivateAttr(True)
    _collection_thread: Optional[threading.Thread] = PrivateAttr(None)
    _listener_thread: Optional[threading.Thread] = PrivateAttr(None)

    def __init__(self, host: str, port: int, output_path: Optional[str] = None, error_log_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(host=host, port=port, output_path=output_path, error_log_path=error_log_path, **kwargs)
        self.error_log_file: Optional[str] = None
        self.output_file: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None
        self.experiment_start_time: Optional[float] = None
        self.on_data_callback: Optional[Callable[[], None]] = None

    def set_error_log_path(self, path: str, file_name: str) -> None:
        """Set the file used for error logging."""
        self.error_log_file = os.path.join(path, file_name)

    def set_output_file(self, path: str, extra_name: str, base_file_name: str) -> None:
        """Configure the output CSV file."""
        name = f"{extra_name}_{base_file_name}"
        self.output_file = os.path.join(path, name)

    def set_experiment_start_time(self, start_time: float) -> None:
        """Set the shared experiment start time."""
        self.experiment_start_time = start_time

    def set_on_data_callback(self, callback: Callable[[], None]) -> None:
        """Set a callback to be called when data is received."""
        self.on_data_callback = callback

    def collect(self) -> pd.DataFrame:
        """Collect data from the Ponemah server."""
        metadata: Dict[str, Dict[str, Any]] = {}
        rows: Dict[Tuple[str, str], Dict[Tuple[str, str], str]] = {}
        previous_data = self.data

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                sock.settimeout(1.0)
                while not self._stop:
                    try:
                        raw_len = self._read_exact(sock, 2)
                        length = struct.unpack(">H", raw_len)[0]
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
                                data = dict(zip(param_names, values))

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
                                    timestamp = time.time() - self.experiment_start_time
                                    row[("meta", "experiment_timestamp")] = f"{timestamp:.6f}"
                                row[("meta", "subject_id")] = meta["subject_id"]
                                channel = meta["channel_label"]
                                for name, value in data.items():
                                    row[(channel, name)] = value
                                
                                if self.on_data_callback is not None:
                                    try:
                                        self.on_data_callback()
                                    except Exception as e:
                                        Soho.log_error(f"Error in data callback: {e}", self.error_log_file)
                    except socket.timeout:
                        continue
                    except (ConnectionError, OSError) as err:
                        Soho.log_error(str(err), self.error_log_file)
                        break
        except (ConnectionError, OSError) as err:
            Soho.log_error(str(err), self.error_log_file)

        ordered_rows = list(rows.values())
        df = pd.DataFrame(ordered_rows)
        if not df.empty and len(df.columns) > 0:
            df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        if previous_data is not None and not previous_data.empty:
            if not df.empty:
                self.data = pd.concat(
                    [previous_data, df], ignore_index=True
                )
            else:
                self.data = previous_data
        else:
            self.data = df
        
        return self.data

    def save(self) -> None:
        """Save collected data to the configured CSV file."""
        if self.output_file is None or self.data is None:
            Soho.log_error("Output file or data not set.", self.error_log_file)
            return
        try:
            self.data.to_csv(self.output_file, index=False)
        except OSError as err:
            Soho.log_error(str(err), self.error_log_file)

    def start(self) -> None:
        """Start data collection and keyboard listener threads."""
        self._active = True
        self._stop = False

        console.input(
            "[bold yellow]\nWARNING: Ensure that the continuous sampling is "
            "activated. Press Enter when you're ready to start recording."
        )

        console.rule("[bold cyan]🟢 RECORDING WITH PONEMAH[/bold cyan]", style="bold cyan")
        console.print("Press 'e' to end recording, 't' to test connection.")

        self._collection_thread = threading.Thread(target=self.collect)
        self._collection_thread.start()
        
        self._listener_thread = threading.Thread(
            target=self._keyboard_listener
        )
        self._listener_thread.start()
        
        console.print("[bold green]Soho recording started.[/bold green]")

    def stop(self) -> None:
        """Signal the collector to stop."""
        self._stop = True
        self._active = False

    def wait_for_completion(self) -> None:
        """Wait for both threads to complete."""
        if self._listener_thread:
            self._listener_thread.join()
        if self._collection_thread:
            self._collection_thread.join()

    def pause_and_test_connection(self) -> None:
        """Pause recording, test connection, and optionally resume."""
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
            "Close the remote connection test window by pressing 'OK'. "
            "Press Enter when done."
        )
        
        confirm_resume = Confirm.ask("[yellow]Resume recording? (y/n)[/yellow]", default=True)
        
        if confirm_resume:
            console.input(
                "[bold yellow]\nWARNING: Ensure that the continuous sampling is "
                "activated. Press Enter when you're ready to start recording."
            )
            self._stop = False
            self._collection_thread = threading.Thread(target=self.collect)
            self._collection_thread.start()
            console.print("[bold green]Recording resumed.[/bold green]")
        else:
            console.print("[bold red]Recording not resumed.[/bold red]")
            self._active = False

    def _keyboard_listener(self) -> None:
        """Monitor keyboard input for stopping or testing connection."""
        if msvcrt is None:
            input()
            console.print(
                "Recording finished, stopping Soho, "
                "waiting for the last reading..."
            )
            self.stop()
            return
        
        while self._active:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                
                if key.lower() == "e":
                    confirm = Confirm.ask(
                        "Are you sure you want to stop recording? (y/n): "
                    , default=True)
                    
                    if confirm:
                        console.print("[bold green]Stopping recording...[/bold green]")
                        self.stop()
                        break
                
                elif key.lower() == "t":
                    confirm = Confirm.ask(
                        "Pause recording to test connection? (y/n): "
                    , default=True)
                    
                    if confirm:
                        self.pause_and_test_connection()
            
            time.sleep(0.1)


    @staticmethod
    def _read_exact(sock: socket.socket, n: int) -> bytes:
        """Read exactly n bytes from a socket."""
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed while reading")
            buf.extend(chunk)
        return bytes(buf)


def test_ponemah_connection(HOST: str, PORT: int) -> None:
    """Test connection to Ponemah server."""
    console.print("Testing remote connection with PONEMAH")
    while True:
        try:
            with socket.create_connection((HOST, PORT), timeout=1) as sock:
                sock.settimeout(1)
                raw = sock.recv(1024).decode("ascii", errors="ignore")
                clean = re.sub(r"[^ -~]", "", raw).strip()
                if ";;0" in clean or "0;;" in clean:
                    console.print("[bold green]Response successful[/bold green]")
                    return True
                console.print(f"[red]Unexpected response:[/red] {clean!r}")
        except OSError:
            console.print(
                "[red]Connection refused, start test in setup remote "
                "connection test[/red]"
            )
        time.sleep(1)
