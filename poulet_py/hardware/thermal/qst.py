try:
    from time import sleep
    from typing import Literal

    from deprecated import deprecated
    from numpy.random import randint
    from pandas import DataFrame, concat
    from pydantic import BaseModel, Field
    from pytcsii import tcsii_serial
    from tqdm import tqdm
    from typing_extensions import Self

    from poulet_py.tools import generate_trials
except ImportError as e:
    msg = "Missing 'qst' module. To install it use: pip install poulet_py[qst]"
    raise ImportError(msg) from e


class TCSStimulus(BaseModel):
    """
    A stimulus configuration for the TCS thermal stimulator.
    """

    target: int = Field(description="Target temperature in Celsius.")
    rise_rate: int = Field(
        description="Temperature rise rate in Celsius/second."
    )
    return_rate: int = Field(
        description="Temperature return rate in Celsius/second."
    )
    dur_ms: int | None = Field(
        None,
        description="Duration in milliseconds. Interpretation depends on `dur_mode`.",
    )
    dur_mode: Literal["fix_stim", "fixed_plateau", "fixed_total"] = Field(
        "fixed_stim",
        description="""
            Duration interpretation mode:
            - 'fix_stim': rise + plateau are total time, return is 0
            - 'fixed_plateau': duration is for plateau, rise/return rates are additional
            - 'fixed_total': duration is total time including rise/return rates"
        """,
    )
    trigger_code: int = Field(255, description="Trigger code.")
    trigger_dur_ms: int = Field(
        10, description="Trigger duration in milliseconds "
    )
    surfaces: int = Field(0, description="Surfaces parameter.")


@deprecated(
    version="0.0.2",
    reason="""This class is deprecated and it is going to be removed in the next release.
        You should use the generalized TCSStimulus instead.
        """,
)
class TCSIIStimulus(TCSStimulus):
    """
    A stimulus configuration for the TCS-II thermal stimulator.
    """


@deprecated(
    version="0.0.2",
    reason="""This class is deprecated and it is going to be removed in the next release.
        You should use the generalized TCSController instead.
        """,
)
class TCSIIController(tcsii_serial):
    def __init__(
        self,
        port,
        *,
        baseline=30,
        surfaces=0,
        max_temp=50,
        beep=False,
        trigger_in=True,
        temp_profile=False,
    ):
        super().__init__(
            port, baseline, surfaces, max_temp, beep, trigger_in, temp_profile
        )

        self.data = DataFrame()
        self.trials = []

    def trials(
        self,
        n: int,
        stimuli: list[TCSIIStimulus],
        mode: Literal["random", "fixed"],
    ) -> Self:
        self.trials = generate_trials(n=n, stimuli_options=stimuli, mode=mode)
        return self

    def run(
        self,
        duration_ms: int | None = None,
        frequency: int = 1000,
        offset_s: int = 1,
        delay_bounds: tuple[int, int] = (0, 2),
        keep: Literal["first", "last", "all"] = "all",
    ) -> Self:
        if not self.trials:
            msg = "Trials were not set. Please use trials() first"
            raise RuntimeError(msg)

        for idx, trial in enumerate(tqdm(self.trials)):
            self.set_stim(**trial.model_dump())
            self.trigger_and_save_temp(duration_ms, frequency, offset_s)

            random_delay = randint(*delay_bounds)

            temp = self.read_outs
            temp["iti"] = random_delay
            temp["target"] = trial.target
            temp["trial"] = idx

            if keep == "all":
                self.data = concat([self.data, temp], ignore_index=True)
            else:
                self.data = temp

            if keep == "first":
                break

            sleep(random_delay)

        return self
