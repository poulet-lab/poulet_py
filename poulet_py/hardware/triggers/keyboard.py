try:
    from threading import Event, Thread

    from pydantic import Field

    from poulet_py import LOGGER, BaseTrigger
except ImportError as e:
    msg = """
Missing 'triggers' module. Install options:
- Dedicated:    pip install poulet_py[triggers]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class KeyboardTrigger(BaseTrigger):
    """Keyboard-based trigger (press Enter key)."""

    key: str = Field(..., description="Key to trigger")

    def wait(self) -> bool:
        """Wait for keyboard input with optional timeout."""
        result = False
        input_received = Event()

        def input_thread():
            nonlocal result
            try:
                LOGGER.info(f"Press the '{self.key}' key to trigger...")
                while not input_received.is_set():
                    user_input = input()
                    if user_input == self.key:
                        result = True
                        input_received.set()
                    else:
                        LOGGER.warning(f"Wrong key '{user_input}'")
            except EOFError:
                LOGGER.error("EOF reached while waiting for input")
            except Exception as e:
                LOGGER.error(f"Error in input thread: {e}")
            finally:
                input_received.set()

        thread = Thread(target=input_thread)
        thread.daemon = True
        thread.start()

        # Wait with timeout if specified
        if self.timeout is not None:
            input_received.wait(timeout=self.timeout)
        else:
            input_received.wait()

        # If thread is still alive after timeout, interrupt it
        if thread.is_alive():
            LOGGER.warning(f"Timeout waiting for key '{self.key}'")

        return result

    def cleanup(self) -> None:
        """No cleanup needed for keyboard."""
        pass
