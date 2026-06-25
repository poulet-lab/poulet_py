# import cv2
# import numpy as np

from poulet_py import DCAM

NAME = "dcam_example"


# def show_framedata(data):
#     maxval = np.amax(data)
#     if data.dtype == np.uint16 and maxval > 0:
#         imul = int(65535 / maxval)
#         data = data * imul

#     cv2.imshow(NAME, data)


devices = DCAM.get_available_devices()

if devices:
    print(devices)
else:
    raise RuntimeError("No Devices")

dcam = DCAM(device_index=0)

try:
    with dcam:
        # cv2.namedWindow(
        #     NAME,
        #     cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
        # )

        while True:
            sample = dcam.read_last_sample()
            # show_framedata(sample["dcam"])

            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     break

    # cv2.destroyAllWindows()
except KeyboardInterrupt:
    print("\nShutdown initiated. Cleaning up resources...")
    print("Shutdown complete.")
