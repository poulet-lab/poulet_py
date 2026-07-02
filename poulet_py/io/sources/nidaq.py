try:
    from numpy import empty

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
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class NIDaQSource(BaseSource, NIDaQ):
    def _set_buffer_dtype(self):
        dtype = [("timestamp", "uint64")]

        for task in self._read_tasks:
            if isinstance(task, NIAnalogInputTask):
                for ch in task.channels:
                    dtype.append((f"{task.name}_{ch.name}", "float64"))

            elif isinstance(task, NIDigitalInputTask):
                for ch in task.channels:
                    dtype.append((f"{task.name}_{ch.name}", "uint32"))

        self._buffer_dtype = dtype

    def _open(self):
        NIDaQ.open(self)
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.start()

    def _close(self):
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.stop()

        NIDaQ.close(self)

    def _trigger(self) -> bool:
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

            if lengths:
                n = min(lengths)
                samples = empty(n, dtype=self._buffer_dtype)

                t_prev = 0
                task_name = ""

                for task in self._read_tasks:
                    if task.name in data:
                        if not task_name or t_prev > data[task.name]["timestamp"][0]:
                            task_name = task.name
                            t_prev = data[task.name]["timestamp"][0]

                        for ch in task.channels:
                            samples[f"{task.name}_{ch.name}"] = data[task.name][ch.name][:n]

                samples["timestamp"] = data[task_name]["timestamp"][:n]
                self._write_samples(samples)

        if self.acquisition_type == AcquisitionType.FINITE:
            self.stop()

        return True
