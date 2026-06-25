from time import sleep

from poulet_py import DCAM

NAME = "dcam_example"
running = True
devices = DCAM.get_available_devices()

if devices:
    print(devices)
else:
    raise RuntimeError("No Devices")

dcam = DCAM(device_index=0)
with dcam:
    while running:
        sample = dcam.read_last_sample()
        sleep(0.001)
