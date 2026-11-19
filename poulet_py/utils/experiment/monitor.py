try:
    from enum import IntEnum

    from pydantic import PrivateAttr

    from poulet_py import BaseEvent, BaseSink
    from poulet_py.utils.stimulator import Trial

except ImportError as e:
    raise ImportError("""
Missing 'stimulator' module. Install options:
- Dedicated:    pip install poulet_py[stm]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
""") from e


class MonitorCommands(IntEnum):
    NONE = 0x0
    START = 0x1
    STOP = 0x2
    PAUSE = 0x4
    RESUME = 0x8


class BaseMonitor(BaseSink):
    _command: MonitorCommands = PrivateAttr(default=MonitorCommands.NONE)

    @property
    def command(self) -> MonitorCommands:
        return self._command

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def _on_event(self, event: BaseEvent) -> None:
        pass

    def check_trial(self, trial: Trial) -> Trial | None:
        return trial
