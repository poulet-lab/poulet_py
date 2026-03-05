try:
    from collections.abc import Sequence
    from typing import ClassVar

    from pydantic import Field

    from poulet_py import BaseStimulus, TCSCommand

except ImportError as e:
    msg = """
Missing 'qst' module. Install options:
- Module:       pip install poulet_py[qst]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class TCSStimulus(BaseStimulus):
    """
    Configuration for thermal stimulation parameters.

    Attributes
    ----------
    surface : int
        Target surface (0-5, where 0 means all surfaces).
    baseline : float
        Baseline temperature in °C (20-45).
    target : float
        Target temperature in °C (0-60).
    rise_rate : float
        Temperature rise rate in °C/s (0.1-999.9).
    return_speed : float
        Temperature return speed in °C/s (0.1-999.9).
    duration : int
        Stimulation duration in ms (10-99999).

    Methods
    -------
    commands() -> list
        Generate the sequence of commands needed to configure this stimulus.

    Examples
    --------
    >>> stimulus = TCSStimulus(surface=1)
    >>> stimulus.commands()
    [b'S10000', b'N300', b'C1000', b'V10010', b'D100100', b'R10010']
    """

    SURFACE_MAP: ClassVar = {0: 11111, 1: 10000, 2: 1000, 3: 100, 4: 10, 5: 1}

    surface: int = Field(
        default=0,
        description="Target surface (0-5, where 0 means all surfaces)",
        ge=0,
        le=5,
    )
    baseline: float = Field(
        default=30,
        description="Baseline temperature in °C (20-45)",
        ge=20,
        le=45,
    )
    target: float = Field(
        default=10,
        description="Target temperature in °C (0-60)",
        ge=0,
        le=60,
    )
    rise_rate: float = Field(
        default=1,
        description="Temperature rise rate in °C/s (0.1-999.9)",
        ge=0.1,
        le=999.9,
    )
    return_speed: float = Field(
        default=1,
        description="Temperature return speed in °C/s (0.1-999.9)",
        ge=0.1,
        le=999.9,
    )
    duration: int = Field(
        default=100,
        description="Stimulation duration in ms (10-99999)",
        ge=10,
        le=99999,
    )

    def build(self, *args, **kwargs) -> Sequence[bytes]:
        """
        Generate the sequence of commands needed to configure this stimulus.

        Returns
        -------
        list
            Sequence of formatted command strings

        Examples
        --------
        >>> stimulus = TCSStimulus(surface=1)
        >>> stimulus.commands()
        [b'S10000', b'N300', b'C1000', b'V10010', b'D100100', b'R10010']
        """
        return [
            TCSCommand.SURFACE_SELECTION.format(self.SURFACE_MAP[self.surface]),
            TCSCommand.BASELINE_TEMPERATURE.format(int(self.baseline * 10)),
            TCSCommand.TARGET_TEMPERATURE.format(self.surface, int(self.target * 10)),
            TCSCommand.STIMULATION_RATE.format(self.surface, int(self.rise_rate * 10)),
            TCSCommand.STIMULATION_DURATION.format(self.surface, self.duration),
            TCSCommand.RETURN_SPEED.format(self.surface, int(self.return_speed * 10)),
        ]
