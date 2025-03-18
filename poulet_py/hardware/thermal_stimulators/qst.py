from typing import Literal
from typing_extensions import Self
from pydantic import BaseModel, Field
from pytcsii import tcsii_serial
from time import sleep
from random import randint
from pandas import DataFrame, concat
from tqdm import tqdm

from poulet_py.tools import generate_trials


class TCSIIStimulus(BaseModel):
    target: int = Field(description="target temperature in C ")
    rise_rate: int = Field(description="rise rate in C/s")
    return_rate: int = Field(description="return rate in C/s")
    dur_ms: int | None = Field(
        None, description="duration in ms. Phases duration depends on dur_mode."
    )
    dur_mode: Literal["fix_stim", "fixed_plateau", "fixed_total"] = Field(
        "fixed_stim",
        description="""
            'fix_stim' (rise + plateau are total time and return is 0) 
            'fixed_plateau' (duration is for plateau and rise/return rates are additional time) 
            'fixed_total' (duration is total time and rise/return rates are included)""",
    )
    trigger_code: int = Field(255, description="trigger code. Defaults to 255.")
    trigger_dur_ms: int = Field(10, description="trigger duration. Defaults to 10.")
    surfaces: int = Field(0)


class TCSIIController(tcsii_serial):
    def __init__(
        self,
        port,
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

    def trials(
        self, n: int, stimuli: list[TCSIIStimulus], mode: Literal["random", "fixed"]
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
        if not hasattr(self, "trials"):
            raise RuntimeError("Trials were not set. Please run trials first")

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
