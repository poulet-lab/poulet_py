import time

import adafruit_ina228
from busio import I2C
import board
i2c = I2C(board.SCL, board.SDA, frequency=400_000)
ina228_1 = adafruit_ina228.INA228(i2c, address=0x41)
ina228_2 = adafruit_ina228.INA228(i2c, address=0x40)

while True:
    print(f"Bus Voltage: {ina228_1.bus_voltage:.2f} V")
    print(f"Bus Voltage: {ina228_2.bus_voltage:.2f} V")

    time.sleep(1)