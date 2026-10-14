from poulet_py import BaseEvent, BaseSink, Oscilloscope, SinkEvent


#TODO macos need osc to run on main thread and the rest on subthreads
class OscSink(BaseSink, Oscilloscope):
    def _open(self) -> None:
        Oscilloscope.open(self)

    def _close(self) -> None:
        Oscilloscope.close(self)

    def _on_event(self, event: BaseEvent) -> None:
        if isinstance(event, SinkEvent):
            self.add_data({event.name: event.payload})
