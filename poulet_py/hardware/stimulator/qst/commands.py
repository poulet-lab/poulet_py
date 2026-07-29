try:
    from enum import Enum

    from poulet_py import LOGGER
except ImportError as e:
    raise ImportError("""
Missing 'qst' module. Install options:
- Dedicated:    pip install poulet_py[qst]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class TCSCommand(bytes, Enum):
    """
    Enumeration of all available TCS commands with their byte representations.

    Each command includes formatting capability for parameterized commands.

    Examples
    --------
    >>> TCSCommand.READ_TEMPERATURES
    <TCSCommand.READ_TEMPERATURES: b'E'>
    >>> TCSCommand.BASELINE_TEMPERATURE.format(300)
    b'N300'
    """

    READ_INFO = b"H"
    # Neutral temperature then each surface
    READ_TEMPERATURES = b"E"
    # Display the current values of the stimulator parameters
    READ_STIMULATION_VALUES = b"P"
    # Return the status of buttons 1 and 2.
    # 10 button 1 pressed; 01 button 2 pressed; 11 both pressed
    READ_BUTTON_STATUS = b"K"
    # Display voltage and % battery charge
    READ_BATTERY = b"B"
    # Return error codes for probe diagnosis
    # Returns “xxxxxx” for each zone and the basic temperature;
    # x = 0 : OK / x > 1 : ERROR
    READ_ERRORS = b"Q"

    # Allow regular display of current temperatures, 1Hz
    DISPLAY_TEMPERATURES_BETWEEN_STIMULATION = b"Oa"
    # Allow the display of temperatures during stimulation, 100 Hz
    DISPLAY_TEMPERATURES_DURING_STIMULATION = b"Ob"
    # Reset the TCS (same action as switching OFF and ON again)
    RESET = b"Oc"

    # Define a maximum stimulation temperature, xxx' 1/10 °C
    SET_MAX_TEMPERATURE = b"Om%03d"

    # Automatic calibration of the reference temperature,
    # Displays Nxxx with neutral t° in case of success
    AUTOMATIC_CALIBRATION = b"G"
    # Deactivate the display of current temperatures
    DEACTIVATE_DISPLAY = b"F"
    # Trigger stimulation with the current settings
    TRIGGER_STIMULATION = b"L"
    # Force a halt to the current stimulation A
    HALT_STIMULATION = b"A"

    # xxx=200-450, unit=0.1°C, default: 300
    BASELINE_TEMPERATURE = b"N%03d"
    # xxxxx=0 or 1 per surface , default: 00000
    SURFACE_SELECTION = b"S%05d"

    # s=0-5 (surface number), xxx=000-600, unit=0.1°C, default: 100
    TARGET_TEMPERATURE = b"C%d%03d"
    # s=0-5 (surface number), xxxx=0001-9999, unit=0.1°C/s,
    # default: Depends on the type of sensor
    STIMULATION_RATE = b"V%d%04d"
    # s=0-5 (surface number). xxxx=0001-9999, unit=0.1°C/s,
    # default: Depends on the type of sensor
    RETURN_SPEED = b"R%d%04d"
    # s=0-5 (surface number). xxxxx=00010-99999, unit=ms, default: 00100
    STIMULATION_DURATION = b"D%d%05d"
    # xxx=001-255 (trigger_channel), yyy=010-999 (duration), unit=ms, default: 255300
    TRIGGER_CHANNEL_DURATION = b"T%03d%03d"
    # Buzzer ddd: duration in 10X ms, fff: frequency in 10× Hz
    BUZZER = b"Z%03d%03d"

    def format(self, *args: int | float) -> bytes:
        """
        Format the command with the given arguments.

        Parameters
        ----------
        *args : int, float
            Arguments to format into the command string

        Returns
        -------
        bytes
            Formatted command string

        Raises
        ------
        ValueError
            If arguments don't match the command's format requirements

        Examples
        --------
        >>> TCSCommand.TARGET_TEMPERATURE.format(1, 350)
        b'C1350'
        """
        try:
            LOGGER.debug(f"Formatting command {self.name} with args {args}")
            return self.value % args
        except TypeError as e:
            raise ValueError(f"Wrong number/type of arguments for {self.name}: {e}") from e
