from random import random
from time import sleep, time

from poulet_py.utils.oscilloscope import Oscilloscope

osc = Oscilloscope()
osc.start()
last_plot_update = time()
plot_update_interval = 0.05
for i in range(2000):
    osc.add_data({"temperature": random(), "humidity": i**2}, i)
    if time() - last_plot_update > plot_update_interval:
        osc.force_redraw()
        last_plot_update = time()
    sleep(0.01)
osc.stop()
