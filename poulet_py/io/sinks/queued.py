try:
    import queue
    import threading

    from poulet_py import LOGGER, DataPacket, DataSink
except ImportError as e:
    msg = """
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class QueueDataSink(DataSink):
    def __init__(self, writer, maxsize=100):
        self.queue = queue.Queue(maxsize=maxsize)
        self.writer = writer
        self._running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.thread.start()

    def push(self, packet: DataPacket):
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            LOGGER.warning("DataSink queue full — dropping packet")

    def _run(self):
        while self._running or not self.queue.empty():
            try:
                packet = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self.writer.write(packet)

    def close(self):
        self._running = False
        self.thread.join()
