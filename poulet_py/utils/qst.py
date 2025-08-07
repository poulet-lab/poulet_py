try:
    from secrets import choice
    from time import sleep, time
    from typing import Literal

    from pandas import DataFrame
    from tqdm import tqdm

    from poulet_py.hardware.stimulator.qst import TCS, TCSStimulus
    from poulet_py.tools.generators import generate_stimulus_sequence
    from poulet_py.utils.oscilloscope import Oscilloscope

except ImportError as e:
    msg = """
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[osc, qst]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class TCSInterface(TCS):
    """
    Interface class for the Thermal Cutaneous Stimulator (TCS) device.

    Handles serial communication with the device and provides methods to configure
    and execute thermal stimulus experiments.

    Parameters
    ----------
    port : str
        Serial port to which the device is connected.
    maximum_temperature : float, optional
        Maximum allowed temperature in °C (default: 40).
    beep : bool, optional
        Whether to enable audible beeps (default: False).
    trigger_out_channel : int, optional
        Output channel for trigger signals (default: 255).
    read_timeout : float, optional
        Timeout for read operations in seconds (default: 2).
    response_timeout : float, optional
        Timeout for device responses in seconds (default: 2).
    n_trials : int, optional
        Number of trials to run (default: 1).
    stimuli : list[TCSStimulus], optional
        List of stimulus configurations (default: None).
    mode : {'random', 'fixed'}, optional
        Stimulus presentation mode (default: 'random').
    interstimulus_period : int or list[int], optional
        Time between stimuli in milliseconds (default: 0).

    Attributes
    ----------
    stimuli : list[TCSStimulus]
        List of configured stimulus sequences.
    n_trials : int
        Number of trials to be executed.
    mode : str
        Stimulus presentation mode ('random' or 'fixed').
    interstimulus_period : int or list[int]
        Time between stimuli in milliseconds.
        assign a list of integers for random selection
        or a single integer for fixed time.

    Methods
    -------
    run(plot=False, max_plot_samples=1000)
        Execute the configured experiment.
    to_df()
        Convert collected data to a pandas DataFrame.
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
    ):
        super().__init__(
            port=port,
            maximum_temperature=maximum_temperature,
            beep=beep,
            trigger_out_channel=trigger_out_channel,
            read_timeout=read_timeout,
            response_timeout=response_timeout,
        )

        self.interstimulus_period: int | list[int] = interstimulus_period
        self.n_trials: int = n_trials
        self.mode: Literal["random", "fixed"] = mode

        if stimuli is not None:
            self.stimuli = stimuli

        self._stimuli: list[TCSStimulus] = []
        self._readings: list[dict[str, float | int]] = []
        self._oscilloscope: Oscilloscope | None = None

    @property
    def stimuli(self) -> list[TCSStimulus]:
        """
        Get or set the stimulus sequence.

        Returns
        -------
        list[TCSStimulus]
            Currently configured stimulus sequence.

        Notes
        -----
        When setting the stimuli, they are automatically validated and a sequence
        is generated according to the configured mode (random/fixed) and number of trials.
        """
        return self._stimuli

    @stimuli.setter
    def stimuli(self, value: list[TCSStimulus]) -> None:
        """Set the stimulus sequence with validation."""
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
        """Clear the stimulus sequence."""
        self._stimulus = []

    def run(self, *, plot: bool = False, max_plot_samples: int = 1000) -> list[dict]:
        """
        Execute the configured experiment.

        Parameters
        ----------
        plot : bool, optional
            Whether to show real-time plotting (default: False).
        max_plot_samples : int, optional
            Maximum number of samples to show in the plot (default: 1000).

        Returns
        -------
        list[dict]
            List of readings collected during the experiment.

        Raises
        ------
        ValueError
            If no stimuli are configured before running.

        Notes
        -----
        Each reading contains:
        - timestamp: Time of reading
        - temperature: Current temperature
        - trial: Trial number
        - Other relevant parameters
        """
        self._setup_run(plot=plot, max_plot_samples=max_plot_samples)

        try:
            for idx, stimulus in tqdm(enumerate(self.stimuli), total=self.n_trials):
                self._execute_trial(idx, stimulus)
        finally:
            self._cleanup_run()

        return self._readings

    def to_df(self) -> DataFrame:
        """
        Convert collected data to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Formatted DataFrame with timestamp as index and trial data.

        Examples
        --------
        >>> df = device.to_df()
        >>> df.plot()
        """
        data = DataFrame(self._readings)
        data["timestamp"] = data["timestamp"].astype("datetime64[ns]")
        data.set_index("timestamp", inplace=True)
        return data

    def _setup_run(self, *, plot: bool = False, max_plot_samples: int = 1000) -> None:
        """Initialize experiment run."""
        if not self.stimuli:
            msg = "Stimuli must be set before running the experiment"
            raise ValueError(msg)

        self.init()
        self._readings = []

        if plot:
            self._oscilloscope = Oscilloscope(
                max_plot_samples,
                title="TCS Readings",
                xlabel="Time",
                ylabel="Temperature (°C)",
            )
            self._plot_active = True
            self._oscilloscope.start()
            self._last_plot_update = time()
            self._plot_update_interval = 0.05

    def _execute_trial(self, idx: int, stimulus: TCSStimulus) -> None:
        """Execute a single trial with the given stimulus."""
        self.stimulus = stimulus
        self.trigger()

        interstimulus_period = self._get_interstimulus_period()
        start_time = int(time() * 1000)

        while self._should_continue_trial(start_time, stimulus.duration + interstimulus_period):
            reading = self.get_readings()

            if reading:
                reading["trial"] = idx
                self._readings.append(reading.copy())
                if self._oscilloscope is not None:
                    self._handle_plotting(reading)

            sleep(0.0001)

    def _cleanup_run(self) -> None:
        """Clean up after experiment completion."""
        if self._oscilloscope is not None:
            self._oscilloscope.stop()
            self._oscilloscope = None
        self.close()

    def _handle_plotting(self, reading: dict) -> None:
        """Update the real-time plot if enabled."""
        if self._oscilloscope is not None:
            timestamp = reading.pop("timestamp")
            self._oscilloscope.add_data(reading, x=timestamp)
            if time() - self._last_plot_update > self._plot_update_interval:
                self._oscilloscope.force_redraw()
                self._last_plot_update = time()

    def _get_interstimulus_period(self) -> int:
        """Get the inter-stimulus period (random if list provided)."""
        if isinstance(self.interstimulus_period, list):
            return choice(self.interstimulus_period)
        return self.interstimulus_period

    def _should_continue_trial(self, start_time: int, duration: int) -> bool:
        """Check if trial should continue based on elapsed time."""
        return (time() * 1000 - start_time) < duration
