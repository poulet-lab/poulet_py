try:
    from numpy import ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import (
        AcquisitionType,
        BaseSource,
        NIAnalogBaseStimulus,
        NIAnalogCompositeStimulus,
        NIAnalogInputTask,
        NIDaQ,
        NIDigitalBaseStimulus,
        NIDigitalCompositeStimulus,
        NIDigitalInputTask,
    )
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class NIDaQSource(BaseSource, NIDaQ):
    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self):
        dtype = [("timestamp", "uint64")]

        for task in self._read_tasks:
            if isinstance(task, NIAnalogInputTask):
                for ch in task.channels:
                    dtype.append((f"{task.name}_{ch.name}", "float64"))

            elif isinstance(task, NIDigitalInputTask):
                for ch in task.channels:
                    dtype.append((f"{task.name}_{ch.name}", "uint32"))

        self._source_buffer_dtype = dtype

    def _open(self):
        self._set_buffer_dtype()
        NIDaQ.open(self)
        self._temp_buffer = zeros(self._clock_task.samps_per_chan, dtype=self._source_buffer_dtype)

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.start()

    def _close(self):
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.stop()

        NIDaQ.close(self)

    def _fire(self) -> bool:
        for st in self._stimuli:
            if isinstance(
                st,
                (
                    NIAnalogCompositeStimulus,
                    NIDigitalCompositeStimulus,
                    NIAnalogBaseStimulus,
                    NIDigitalBaseStimulus,
                ),
            ):
                self.write(st)

        if self.acquisition_type == AcquisitionType.FINITE:
            self.start()
            self._rw_data()
            self.stop()

        return True

    def _acquire(self) -> None:
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._rw_data()

    def _rw_data(self):
        data = self.read(-1, -1)

        if data:
            lengths = [len(v) for v in data.values() if v is not None and len(v) > 0]

            if lengths:
                # TODO fix to avoid data loss
                n = min(lengths)

                t_prev = 0
                task_name = ""

                for task in self._read_tasks:
                    if task.name in data:
                        # TODO concat instead of just first timestamp
                        if not task_name or t_prev > data[task.name]["timestamp"][0]:
                            task_name = task.name
                            t_prev = data[task.name]["timestamp"][0]

                        for ch in task.channels:
                            self._temp_buffer[f"{task.name}_{ch.name}"][:n] = data[task.name][
                                ch.name
                            ][:n]

                self._temp_buffer["timestamp"][:n] = data[task_name]["timestamp"][:n]

                self._write_samples(self._temp_buffer[:n])
