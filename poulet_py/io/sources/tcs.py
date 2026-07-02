try:
    from poulet_py import (
        TCS,
        BaseSource,
        TCSStimulus,
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


class TCSSource(TCS, BaseSource):
    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [("timestamp", "uint64"), *((f"s{i}", "float64") for i in range(5))]

    def _open(self):
        TCS.open(self)

    def _close(self):
        TCS.close(self)

    def _trigger(self) -> bool:
        for st in self._stimuli:
            if isinstance(st, TCSStimulus):
                precise_sleep(st.pre_delay / 1000.0)

                self.trigger(st)

                while self.stimulus_running:
                    pass

                precise_sleep(st.post_delay / 1000.0)

        return True
