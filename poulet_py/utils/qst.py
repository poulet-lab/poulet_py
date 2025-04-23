from threading import Lock


try:
    from secrets import choice
    from time import sleep, time
    from typing import Literal

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from pandas import DataFrame
    from tqdm import tqdm

    from poulet_py.hardware.sensors.qst import TCS, TCSStimulus
    from poulet_py.tools.generators import generate_stimulus_sequence

except ImportError as e:
    msg = "Missing 'qst' module. To install it use: pip install poulet_py[qst]"
    raise ImportError(msg) from e


class TCSInterface(TCS):
    """
    This class is used to interface with the QST TCS device.
    It handles the serial communication and provides methods to send and receive data.
    """

    def __init__(
        self,
        port: str,
        *,
        maximum_temperature: float = 40,
        beep: bool = False,
        trigger_out_channel: int = 255,
        read_timeout: float = 2,
        response_timeout: float = 2,
        n_trials: int = 1,
        stimuli: list[TCSStimulus] | None = None,
        mode: Literal["random", "fixed"] = "random",
        interstimulus_period: int | list[int] = 0,
        max_samples: int = 1000,
        plot_backend: str = "Qt5Agg",
    ):
        super().__init__(
            port=port,
            maximum_temperature=maximum_temperature,
            beep=beep,
            trigger_out_channel=trigger_out_channel,
            read_timeout=read_timeout,
            response_timeout=response_timeout,
        )

        self.interstimulus_period = interstimulus_period
        self.n_trials = n_trials
        self.mode = mode

        self._stimuli = []
        if self.stimuli is not None:
            self.stimuli = stimuli

        self._readings: list[dict[str, float | int]] = []

        self._plotting_lock = Lock()
        self.max_samples = max_samples
        self._lines = {}
        self._animation = None
        if plot_backend:
            mpl.use(plot_backend)

    @property
    def stimuli(self) -> list[TCSStimulus]:
        return self._stimuli

    @stimuli.setter
    def stimuli(self, value: list[TCSStimulus]) -> None:
        msg = ""
        if not isinstance(value, list):
            msg = "Stimuli must be a list"

        for stimulus in value:
            if not isinstance(stimulus, TCSStimulus):
                msg = "Stimulus must be of type TCSStimulus"

            if self.maximum_temperature < stimulus.target:
                msg = (
                    f"Target temperature {stimulus.target} exceeds "
                    f"maximum temperature {self.maximum_temperature}"
                )

            if stimulus.baseline > self.maximum_temperature:
                msg = (
                    f"Baseline temperature {stimulus.baseline} exceeds "
                    f"maximum temperature {self.maximum_temperature}"
                )
            if msg:
                break
        if msg:
            raise ValueError(msg)

        self._stimuli = generate_stimulus_sequence(
            n=self.n_trials, stimuli_options=value, mode=self.mode
        )

    @stimuli.deleter
    def stimuli(self) -> None:
        self._stimulus = []

    def run(self, plot: bool = False) -> list[dict]:
        if not self.stimuli:
            msg = "Stimuli must be set before running the experiment"
            raise ValueError(msg)

        self.init()
        self._readings = []

        if plot:
            self._init_plot()
            self._start_plotting_thread()

        for idx, stimulus in tqdm(enumerate(self.stimuli), total=self.n_trials):
            self.stimulus = stimulus
            self.trigger()

            interstimulus_period = (
                choice(self.interstimulus_period)
                if isinstance(self.interstimulus_period, list)
                else self.interstimulus_period
            )

            t = (
                int(time() * 1000)
                + self.stimulus.duration
                + interstimulus_period
            )
            t_1 = int(time() * 1000)

            while t > t_1:
                reading = self.get_readings()
                if reading:
                    reading["trial"] = idx
                    with self._plotting_lock:
                        self._readings.append(reading)
                t_1 = int(time() * 1000)
                sleep(0.0001)
        self._stop_plotting_thread()
        self.close()
        return self._readings

    def to_df(self) -> DataFrame:
        data = DataFrame(self._readings)
        data["timestamp"] = data["timestamp"].astype("datetime64[ns]")
        data.set_index("timestamp", inplace=True)
        return data

    def _init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Temperature (°C)")
        self.ax.set_title("Real-time Temperature Reading")

    def _start_plotting_thread(self):
        self._animation = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=None,
            interval=50,  # Update every 50ms
            cache_frame_data=False,
            blit=True,
            save_count=0,
        )
        plt.show(block=False)
        plt.pause(0.1)

    def _stop_plotting_thread(self):
        """Stop the plotting thread"""
        self._animation.event_source.stop()
        self._animation = None
        plt.close(self.fig)

    def _update_plot(self, frame):
        """Main loop for the plotting thread"""
        with self._plotting_lock:
            if len(self._readings) == 0:
                return []

            readings = (
                self._readings[-self.max_samples :]
                if len(self._readings) > self.max_samples
                else self._readings
            )
            data = DataFrame(readings)
            data["timestamp"] = data["timestamp"].astype("datetime64[ns]")
            data.set_index("timestamp", inplace=True)

            artists = []
            for col in data.columns:
                if col == "trial":
                    continue

                if col not in self._lines:
                    (self._lines[col],) = self.ax.plot(
                        data.index, data[col], label=col, animated=True
                    )
                    self.ax.legend()
                else:
                    self._lines[col].set_data(data.index, data[col])

                artists.append(self._lines[col])

            self.ax.relim()
            self.ax.autoscale_view(scalex=True, scaley=True)
            self.fig.canvas.draw_idle()

            return artists
