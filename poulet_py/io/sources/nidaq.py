try:
    from numpy import arange, uint64

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
        precise_sleep,
    )
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class NIDaQSource(BaseSource, NIDaQ):
    def _set_buffer_dtype(self):
        self._buffer_dtype = [("timestamp", "uint64")]

        for task in self._read_tasks:
            if isinstance(task, NIAnalogInputTask):
                dt = [(ch.name, "float64") for ch in task.channels]
            elif isinstance(task, NIDigitalInputTask):
                dt = [(ch.name, "uint32") for ch in task.channels]
            else:
                continue

            self._buffer_dtype.append((task.name, dt))

    def _open(self):
        NIDaQ.open(self)
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

        data = self.read(-1, -1)
        if data:
            lengths = [len(v) for v in data.values() if len(v) > 0]
            if not lengths:
                return True

            n = min(lengths)

            rate = self._clock_task._clock_handle.rate
            dt = uint64(1e9 / rate)

            t0 = min(data[t.name]["timestamp"][0] for t in self._read_tasks if t.name in data)
            t0 = t0 - (n - 1) * dt

            timestamps = t0 + dt * arange(n, dtype="uint64")

            with self._lock:
                idx = self._buffer_idx % self.buffer_size
                end = idx + n
                self._buffer_idx += n

                if end <= self.buffer_size:
                    self._buffer[idx:end]["timestamp"] = timestamps

                    for task in self._read_tasks:
                        if task.name in data:
                            task_data = data[task.name]
                            for ch in task.channels:
                                self._buffer[idx:end][task.name][ch.name] = task_data[ch.name][:n]
                else:
                    split = self.buffer_size - idx

                    self._buffer[idx:]["timestamp"] = timestamps[:split]
                    self._buffer[: end % self.buffer_size]["timestamp"] = timestamps[split:]

                    for task in self._read_tasks:
                        if task.name in data:
                            task_data = data[task.name]
                            for ch in task.channels:
                                self._buffer[idx:][task.name][ch.name] = task_data[ch.name][:split]
                                self._buffer[: end % self.buffer_size][task.name][ch.name] = (
                                    task_data[ch.name][split:n]
                                )

        if self.acquisition_type == AcquisitionType.FINITE:
            self.stop()

        return True
