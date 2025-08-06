from poulet_py import TCSInterface, TCSStimulus

stimuli = [
    TCSStimulus(
        surface=0,
        baseline=32,
        target=20,
        rise_rate=10,
        return_speed=10,
        duration=3000,
    ),
    TCSStimulus(
        surface=0,
        baseline=32,
        target=25,
        rise_rate=10,
        return_speed=10,
        duration=3000,
    ),
]

tcs = TCSInterface(
    port="/dev/tty.usbmodem1403",
    maximum_temperature=50,
    stimuli=stimuli,
    n_trials=4,  # number of total trials
    mode="random",  # random choice between the two stimuli
    interstimulus_period=[1000, 5000],  # random choice between 1 and 5 seconds
)

tcs.run(plot=True)
data = tcs.to_df()
print(data)
input("Press Enter to continue...")
