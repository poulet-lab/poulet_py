try:
    from abc import ABC, abstractmethod
    from collections.abc import Sequence

    from numpy import (
        arange,
        arcsin,
        asarray,
        ceil,
        cumsum,
        deg2rad,
        float64,
        ones,
        ones_like,
        pi,
        sin,
        stack,
        tile,
        uint32,
        zeros,
        zeros_like,
    )
    from numpy.random import default_rng
    from numpy.typing import NDArray
    from pydantic import Field

    from poulet_py import BaseStimulus

except ImportError as e:
    raise ImportError("""
Missing 'stim' module. Install options:
- Dedicated:    pip install poulet_py[nidaq]
- Module:       pip install poulet_py[stim]
- Full:         pip install poulet_py[all]
""") from e


class NIAnalogBaseStimulus(BaseStimulus):
    offset: float = Field(default=0.0)

    def build(self, rate: float) -> NDArray[float64]:
        if rate <= 0:
            raise ValueError("Sampling rate must be positive")

        active_samples = int(self.duration * rate / 1000)
        pre_samples = int(self.pre_delay * rate / 1000)
        post_samples = int(self.post_delay * rate / 1000)

        total_samples = pre_samples + active_samples + post_samples
        signal = zeros(total_samples)

        t = arange(active_samples) / rate
        waveform = self._generate(t, rate)

        signal[pre_samples : pre_samples + active_samples] = waveform
        signal += self.offset

        return signal.reshape((1, -1))

    @abstractmethod
    def _generate(self, t: NDArray[float64], rate: float) -> NDArray[float64]: ...


class NIConstantAnalogStimulus(NIAnalogBaseStimulus):
    amplitude: float = Field(1.0, description="", ge=0)

    def _generate(self, t, rate) -> NDArray[float64]:
        return ones_like(t) * self.amplitude


class NISineAnalogStimulus(NIAnalogBaseStimulus):
    frequency: float = Field(..., gt=0)
    amplitude: float = Field(default=1.0, description="", ge=0)
    phase: float = Field(default=0.0)

    def _generate(self, t, rate) -> NDArray[float64]:
        omega = 2 * pi * self.frequency
        return self.amplitude * sin(omega * t + deg2rad(self.phase))


class NISquareAnalogStimulus(NIAnalogBaseStimulus):
    frequency: float = Field(..., gt=0)
    amplitude: float = Field(default=1.0)
    duty_cycle: float = Field(default=0.5, ge=0, le=1)

    def _generate(self, t, rate) -> NDArray[float64]:
        period = 1 / self.frequency
        return (self.amplitude * ((t % period) < (self.duty_cycle * period))).astype(float64)


class NITriangleAnalogStimulus(NIAnalogBaseStimulus):
    frequency: float = Field(..., gt=0)
    amplitude: float = Field(default=1.0)

    def _generate(self, t, rate) -> NDArray[float64]:
        return self.amplitude * (2 / pi) * arcsin(sin(2 * pi * self.frequency * t))


class NISawAnalogStimulus(NIAnalogBaseStimulus):
    frequency: float = Field(..., gt=0)
    amplitude: float = Field(default=1.0)

    def _generate(self, t, rate) -> NDArray[float64]:
        return self.amplitude * (2 * (t * self.frequency % 1) - 1)


class NIPulseAnalogStimulus(NIAnalogBaseStimulus):
    pulse_width: int = Field(..., gt=0)
    amplitude: float = Field(default=1.0)

    def _generate(self, t, rate) -> NDArray[float64]:
        samples = int(self.pulse_width * rate / 1000)
        y = zeros_like(t)
        y[:samples] = self.amplitude
        return y


class NIPulseTrainAnalogStimulus(NIAnalogBaseStimulus):
    pulse_width: int = Field(..., gt=0)
    pulse_interval: int = Field(..., gt=0)
    amplitude: float = Field(default=1.0)

    def _generate(self, t, rate) -> NDArray[float64]:
        width = int(self.pulse_width * rate / 1000)
        interval = int(self.pulse_interval * rate / 1000)
        period = width + interval

        idx = arange(len(t))
        mask = (idx % period) < width
        return self.amplitude * mask.astype(float)


class NIChirpAnalogStimulus(NIAnalogBaseStimulus):
    f_start: float = Field(..., gt=0)
    f_end: float = Field(..., gt=0)
    amplitude: float = Field(default=1.0)

    def _generate(self, t, rate) -> NDArray[float64]:
        dt = 1 / rate
        sweep = self.f_start + (self.f_end - self.f_start) * (t / t[-1])
        phase = 2 * pi * cumsum(sweep) * dt
        return self.amplitude * sin(phase)


class NIWhiteNoiseAnalogStimulus(NIAnalogBaseStimulus):
    amplitude: float = Field(default=1.0)
    std: float = Field(default=0.1)
    seed: int | None = Field(default=None)

    def _generate(self, t, rate) -> NDArray[float64]:
        rng = default_rng(self.seed)
        return self.amplitude * rng.normal(0, self.std, len(t))


class NIArbitraryAnalogStimulus(NIAnalogBaseStimulus):
    waveform: Sequence[float] = Field(...)

    def _generate(self, t, rate) -> NDArray[float64]:
        data = asarray(self.waveform, dtype=float)
        tiled = tile(data, int(ceil(len(t) / len(data))))
        return tiled[: len(t)]


class NISteppedAnalogStimulus(NIAnalogBaseStimulus):
    step_values: Sequence[float] = Field(...)
    step_durations: Sequence[int] = Field(...)

    def _generate(self, t, rate) -> NDArray[float64]:
        y = zeros_like(t)
        cursor = 0

        for value, duration in zip(self.step_values, self.step_durations, strict=False):
            samples = int(duration * rate / 1000)
            y[cursor : cursor + samples] = value
            cursor += samples
            if cursor >= len(t):
                break

        return y


class NIAnalogCompositeStimulus(BaseStimulus):
    stimuli: Sequence[NIAnalogBaseStimulus] = Field(...)

    def build(self, rate: float) -> NDArray[float64]:
        if rate <= 0:
            raise ValueError("Sampling rate must be positive")
        build = []
        for stim in self.stimuli:
            build.append(stim.build(rate))
        return stack(build).squeeze()


class NIDigitalBaseStimulus(BaseStimulus, ABC):
    duration: int = Field(..., ge=1)
    pre_delay: int = Field(default=0, ge=0)
    post_delay: int = Field(default=0, ge=0)

    def build(self, rate: float) -> NDArray[uint32]:
        if rate <= 0:
            raise ValueError("Sampling rate must be positive")

        active_samples = int(self.duration * rate / 1000)
        pre_samples = int(self.pre_delay * rate / 1000)
        post_samples = int(self.post_delay * rate / 1000)

        total_samples = pre_samples + active_samples + post_samples
        signal = zeros(total_samples, dtype=uint32)

        active = self._generate(active_samples, rate)
        signal[pre_samples : pre_samples + active_samples] = active
        return signal.reshape((1, -1))

    @abstractmethod
    def _generate(self, samples: int, rate: float) -> NDArray[uint32]: ...


class NIConstantDigitalStimulus(NIDigitalBaseStimulus):
    def _generate(self, samples: int, rate: float) -> NDArray[uint32]:
        return ones(samples, dtype=uint32)


class NIPulseDigitalStimulus(NIDigitalBaseStimulus):
    pulse_width: int = Field(..., description="in ms")

    def _generate(self, samples: int, rate: float) -> NDArray[uint32]:
        width_samples = int(self.pulse_width * rate / 1000)
        y = zeros(samples, dtype=uint32)
        y[:width_samples] = True
        return y


class NIPulseTrainDigitalStimulus(NIDigitalBaseStimulus):
    pulse_width: int = Field(..., description="in ms")
    pulse_interval: int = Field(..., description="in ms")

    def _generate(self, samples: int, rate: float) -> NDArray[uint32]:
        width_samples = int(self.pulse_width * rate / 1000)
        interval_samples = int(self.pulse_interval * rate / 1000)
        period = width_samples + interval_samples
        idx = arange(samples)
        return ((idx % period) < width_samples).astype(uint32)


class NIDigitalCompositeStimulus(BaseStimulus):
    stimuli: Sequence[NIDigitalBaseStimulus] = Field(...)

    def build(self, rate: float) -> NDArray[float64]:
        if rate <= 0:
            raise ValueError("Sampling rate must be positive")

        build = []
        for stim in self.stimuli:
            build.append(stim.build(rate))
        return stack(build).squeeze()
