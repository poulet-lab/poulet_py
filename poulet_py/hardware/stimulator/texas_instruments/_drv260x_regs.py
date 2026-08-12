from enum import IntEnum, IntFlag


class Register(IntEnum):
    STATUS = 0x00
    MODE = 0x01
    RTP_INPUT = 0x02  # Real-Time Playback Input
    LIBRARY = 0x03
    WAV_FRM_SEQ_1 = 0x04  # Waveform Sequencer
    WAV_FRM_SEQ_2 = 0x05
    WAV_FRM_SEQ_3 = 0x06
    WAV_FRM_SEQ_4 = 0x07
    WAV_FRM_SEQ_5 = 0x08
    WAV_FRM_SEQ_6 = 0x09
    WAV_FRM_SEQ_7 = 0x0A
    WAV_FRM_SEQ_8 = 0x0B
    GO = 0x0C
    ODT = 0x0D  # Overdrive Time Offset
    SPT = 0x0E  # Sustain Positive Time Offset
    SNT = 0x0F  # Sustain Negative Time Offset
    BRT = 0x10  # Brake Time Offset
    ATH_CONTROL = 0x11  # AUDIO_TO_VIBE
    ATH_MIN_INPUT = 0x12
    ATH_MAX_INPUT = 0x13
    ATH_MIN_DRIVE = 0x14
    ATH_MAX_DRIVE = 0x15
    RATED_VOLTAGE = 0x16
    OD_CLAMP = 0x17  # Overdrive Clamp Voltage
    A_CAL_COMP = 0x18  # Auto-calibration Compensation Result
    A_CAL_BEMF = 0x19  # Auto-calibration Back EMF Result
    FEEDBACK_CONTROL = 0x1A
    CONTROL1 = 0x1B
    CONTROL2 = 0x1C
    CONTROL3 = 0x1D
    CONTROL4 = 0x1E
    CONTROL5 = 0x1F
    OL_LRA_PERIOD = 0x20  # Open-Loop LRA Resonance-Period
    VBAT = 0x21  # Voltage-Monitor
    LRA_PERIOD = 0x22  # LRA Resonance-Period


class Devices(IntEnum):
    DRV2605 = 0x60
    DRV2604 = 0x80
    DRV2604L = 0xC0
    DRV2605L = 0xE0


class Status(IntFlag):
    OC_DETECT = 0x01  # 0: No overcurrent event is detected, 1: Overcurrent event is detected
    OVER_TEMP = 0x02  # 0: functioning normally, 1:  exceeded the temperature threshold
    FB_STS = 0x04  # 0: Feedback controller has not timed out, 1: timed out
    DIAG_RESULT = 0x08  # Auto-calibration result
    DEVICE_ID_MASK = 0xE0


class Modes(IntEnum):
    INTERNAL_TRIGGER = 0x00  # Default
    EXTERNAL_EDGE_TRIGGER = 0x01
    EXTERNAL_LEVEL_TRIGGER = 0x02
    PWM_ANALOG = 0x03
    AUDIO_TO_VIBE = 0x04
    RTP = 0x05  # Real-time playback mode
    _DIAGNOSTIC = 0x06
    _AUTO_CALIBRATION = 0x07


class Mode(IntFlag):
    MASK = 0x07
    STANDBY = 0x40  # 0: Device ready, 1: Device in software standby (default)
    DEV_RESET = 0x80


class RTP(IntEnum):
    MASK = 0xFF
    DEFAULT = 0x00


class Libraries(IntEnum):
    EMPTY = 0x00
    ERM_A = 0x01  # Default
    ERM_B = 0x02
    ERM_C = 0x03
    ERM_D = 0x04
    ERM_E = 0x05
    LRA = 0x06
    ERM_F = 0x07


class Library(IntFlag):
    MASK = 0x07
    HI_Z = 0x10


class WaveformSequencer(IntFlag):
    WAIT = 0x80
    MASK = 0x7F
    DEFAULT = 0x00


class Go(IntFlag):
    GO = 0x01


class AudioToVibeControl(IntEnum):
    LP_FILTER_100 = 0x00  # Hz
    LP_FILTER_125 = 0x01  # Default
    LP_FILTER_150 = 0x02
    LP_FILTER_200 = 0x03
    LP_FILTER_MASK = 0x03
    PEAK_TIME_10 = 0x00  # ms
    PEAK_TIME_20 = 0x04  # Default
    PEAK_TIME_30 = 0x08
    PEAK_TIME_40 = 0x0C
    PEAK_TIME_MASK = 0x0C


class AudioToVibeMinInput(IntEnum):
    # MIN_INPUT[7:0] × 1.8 V / 255
    MASK = 0xFF
    DEFAULT = 0x19


class AudioToVibeMaxInput(IntEnum):
    # MAX_INPUT[7:0] × 1.8 V / 255
    MASK = 0xFF
    DEFAULT = 0xFF


class AudioToVibeMinDrive(IntEnum):
    # MIN_DRIVE[7:0] / 255 × 100%
    MASK = 0xFF
    DEFAULT = 0x19


class AudioToVibeMaxDrive(IntEnum):
    # MAX_DRIVE[7:0] / 255 × 100%
    MASK = 0xFF
    DEFAULT = 0xFF


class RatedVoltage(IntEnum):
    MASK = 0xFF
    DEFAULT = 0x3F


class OverdriveClampVoltage(IntEnum):
    MASK = 0xFF
    DEFAULT = 0x8C


class AutoCalibrationCompensationResult(IntEnum):
    # Auto-calibration compensation coefficient = 1 + A_CAL_COMP[7:0] / 255
    MASK = 0xFF
    DEFAULT = 0x0C


class AutoCalibrationBackEMFResult(IntEnum):
    # Auto-calibration back-EMF (V) = (A_CAL_BEMF[7:0] / 255) × 1.22 V / BEMF_GAIN[1:0]
    MASK = 0xFF
    DEFAULT = 0x6F


class FeedbackControl(IntEnum):
    BEMF_GAIN_MASK = 0x03
    ERM_0_255X = 0x00
    ERM_0_7875X = 0x01
    ERM_1_365X = 0x02  # Default
    ERM_3_0X = 0x03
    LRA_3_75X = 0x00
    LRA_7_5X = 0x01
    LRA_15X = 0x02  # Default
    LRA_22_5X = 0x03

    LOOP_GAIN_MASK = 0x0C
    LOOP_GAIN_LOW = 0x00
    LOOP_GAIN_MEDIUM = 0x04  # Default
    LOOP_GAIN_HIGH = 0x08
    LOOP_GAIN_VERY_HIGH = 0x0C

    FB_BRAKE_FACTOR_MASK = 0x70
    FB_BRAKE_FACTOR_1X = 0x00
    FB_BRAKE_FACTOR_2X = 0x10
    FB_BRAKE_FACTOR_3X = 0x20
    FB_BRAKE_FACTOR_4X = 0x30  # Default
    FB_BRAKE_FACTOR_6X = 0x40
    FB_BRAKE_FACTOR_8X = 0x50
    FB_BRAKE_FACTOR_16X = 0x60
    FB_BRAKE_FACTOR_DISABLED = 0x70

    N_ERM_LRA = 0x80  # 0: ERM Mode (default), 1: LRA Mode


class Control1(IntEnum):
    # LRA Mode: DRIVE_TIME[4:0] × 0.1 ms + 0.5 ms ERM Mode: DRIVE_TIME[4:0] × 0.2 ms + 1 ms
    DRIVE_TIME_MASK = 0x1F
    DRIVE_TIME_DEFAULT = 0x13

    AC_COUPLE = 0x20  # 0: DC-coupled (default), 1: AC-coupled
    STARTUP_BOOST = 0x80


class Control2(IntEnum):
    IDISS_TIME_MASK = 0x03
    IDISS_TIME_ERM_45 = 0x00  # us
    IDISS_TIME_ERM_75 = 0x01  # Default
    IDISS_TIME_ERM_150 = 0x02
    IDISS_TIME_ERM_300 = 0x03
    IDISS_TIME_LRA_15 = 0x00  # us. second part if on register LRA_AUTO_OPEN_LOOP (0x1F)
    IDISS_TIME_LRA_25 = 0x01  # Default
    IDISS_TIME_LRA_50 = 0x02
    IDISS_TIME_LRA_75 = 0x03
    IDISS_TIME_LRA_90 = 0x04
    IDISS_TIME_LRA_105 = 0x05
    IDISS_TIME_LRA_120 = 0x06
    IDISS_TIME_LRA_135 = 0x07
    IDISS_TIME_LRA_150 = 0x08
    IDISS_TIME_LRA_165 = 0x09
    IDISS_TIME_LRA_180 = 0x0A
    IDISS_TIME_LRA_195 = 0x0B
    IDISS_TIME_LRA_210 = 0x0C
    IDISS_TIME_LRA_235 = 0x0D
    IDISS_TIME_LRA_260 = 0x0E
    IDISS_TIME_LRA_285 = 0x0F

    BLANKING_TIME_MASK = 0x0C
    BLANKING_TIME_ERM_45 = 0x00  # us
    BLANKING_TIME_ERM_75 = 0x40  # Default
    BLANKING_TIME_ERM_150 = 0x80
    BLANKING_TIME_ERM_300 = 0xC0
    BLANKING_TIME_LRA_15 = 0x00  # us. second part if on register LRA_AUTO_OPEN_LOOP (0x1F)
    BLANKING_TIME_LRA_25 = 0x01  # Default
    BLANKING_TIME_LRA_50 = 0x02
    BLANKING_TIME_LRA_75 = 0x03
    BLANKING_TIME_LRA_90 = 0x04
    BLANKING_TIME_LRA_105 = 0x05
    BLANKING_TIME_LRA_120 = 0x06
    BLANKING_TIME_LRA_135 = 0x07
    BLANKING_TIME_LRA_150 = 0x08
    BLANKING_TIME_LRA_165 = 0x09
    BLANKING_TIME_LRA_180 = 0x0A
    BLANKING_TIME_LRA_195 = 0x0B
    BLANKING_TIME_LRA_210 = 0x0C
    BLANKING_TIME_LRA_235 = 0x0D
    BLANKING_TIME_LRA_260 = 0x0E
    BLANKING_TIME_LRA_285 = 0x0F

    SAMPLE_TIME_MASK = 0x30
    SAMPLE_TIME_150 = 0x00  # us
    SAMPLE_TIME_200 = 0x10
    SAMPLE_TIME_250 = 0x20
    SAMPLE_TIME_300 = 0x30  # Default

    BRAKE_STABILIZER = 0x04  # 0: Disabled, 1: Enabled (default)
    BIDIR_INPUT = 0x80  # 0: Unidirectional input, 1: Bidirectional input (default)


class Control3(IntEnum):
    LRA_OPEN_LOOP = 0x01  # 0: Auto-resonance mode, 1: LRA Open-loop mode
    N_PWM_ANALOG = 0x02  # 0: PWM input mode, 1: Analog input mode
    LRA_DRIVE_MODE = 0x04  # 0: Once per cycle, 1: Twice per cycle
    DATA_FORMAT_RTP = 0x08  # 0: Signed, 1: Unsigned
    SUPPLY_COMP_DISABLE = 0x10  # 0: Supply compensation enabled, 1: disabled
    ERM_OPEN_LOOP = 0x20  # 0: Closed-loop, 1: Open-loop

    # noise-gate threshold
    NG_THRESHOLD_MASK = 0xC0
    NG_THRESHOLD_DISABLED = 0x00
    NG_THRESHOLD_2 = 0x40
    NG_THRESHOLD_4 = 0x80  # Default
    NG_THRESHOLD_8 = 0xC0


class Control4(IntEnum):
    OTP_PROGRAM = 0x01  # 0: Normal operation, 1: Program OTP memory
    OTP_STATUS = 0x04  # 0: OTP memory is not programmed, 1: OTP memory is programmed

    AUTO_CAL_TIME_MASK = 0x30
    AUTO_CAL_TIME_150_350 = 0x00  # ms
    AUTO_CAL_TIME_250_450 = 0x10
    AUTO_CAL_TIME_500_700 = 0x20  # Default
    AUTO_CAL_TIME_1000_1200 = 0x30

    ZC_DET_TIME_MASK = 0xC0
    ZC_DET_TIME_100 = 0x00  # us, Default
    ZC_DET_TIME_200 = 0x40
    ZC_DET_TIME_300 = 0x80
    ZC_DET_TIME_390 = 0xC0


class Control5(IntEnum):
    IDISS_TIME_MASK = 0x03  # Second part of IDISS_TIME_LRA_xx in register CONTROL2 (0x1C)
    BLANKING_TIME_MASK = 0x0C  # Second part of BLANKING_TIME_LRA_xx in register CONTROL2 (0x1C)
    PLAYBACK_INTERVAL = 0x10  # 0: 5ms (default), 1: 1ms
    LRA_AUTO_OPEN_LOOP = (
        0x20  # 0: Never transitions to open loop, 1: Automatically transitions to open loop
    )
    # Number of cycles required to attempt synchronization before transitioning to open loop
    AUTO_OL_CNT_MASK = 0xC0
    AUTO_OL_CNT_3 = 0x00  # attempts
    AUTO_OL_CNT_4 = 0x40
    AUTO_OL_CNT_5 = 0x80
    AUTO_OL_CNT_6 = 0xC0


class LRAOpenLoopPeriod(IntEnum):
    MASK = 0x7F
    DEFAULT = 0x00


class VBAT(IntEnum):
    MASK = 0xFF
    DEFAULT = 0x00


class LRAResonancePeriod(IntEnum):
    MASK = 0xFF
    DEFAULT = 0x00
