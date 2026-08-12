"""
One-time calibration for the current ERM.

Set ERM_RATED_VOLTAGE to the actuator's actual rated average voltage before
running this script. Auto-calibration does not discover a safe rated voltage.
The script prints the constants block to install in
DRC2605SSource.py.
"""

from poulet_py import DRV2605Source

ERM_RATED_VOLTAGE = 3.0  # Replace when the current ERM rating is established.
ERM_MAXIMUM_VOLTAGE = 5


def main() -> None:
    drv = DRV2605Source(
        name="drv2605_erm_calibration",
        motor_type="erm",
        loop_mode="closed_loop",
        rated_voltage=ERM_RATED_VOLTAGE,
        maximum_voltage=ERM_MAXIMUM_VOLTAGE,
        calibrate=True,
    )

    try:
        drv.open()
        result = drv.calibration_result
        if result is None or not result.success:
            raise RuntimeError("DRV2605L ERM calibration did not succeed.")

        print("\nReplace the current-ERM constants block with:\n")
        print(f"CURRENT_ERM_RATED_VOLTAGE = {ERM_RATED_VOLTAGE:.6f}")
        print(f"CURRENT_ERM_MAXIMUM_VOLTAGE = {ERM_MAXIMUM_VOLTAGE:.6f}")
        print(f"CURRENT_ERM_CALIBRATED_RATED_VOLTAGE: float | None = {ERM_RATED_VOLTAGE:.6f}")
        print(f"CURRENT_ERM_CALIBRATED_MAXIMUM_VOLTAGE: float | None = {ERM_MAXIMUM_VOLTAGE:.6f}")
        print(f"CURRENT_ERM_AUTO_CAL_COMP: int | None = 0x{result.auto_cal_comp:02X}")
        print(f"CURRENT_ERM_AUTO_CAL_BEMF: int | None = 0x{result.auto_cal_bemf:02X}")
        print(f"CURRENT_ERM_BEMF_GAIN: int | None = 0x{result.bemf_gain:02X}")
        print(f"\nCalibration elapsed: {result.elapsed_s:.3f} s")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
