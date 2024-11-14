from random import shuffle
from typing import Any, Literal


def generate_trials(
    n: int, *, stimuli_options: list[Any], mode: Literal["random", "fixed"] = "random"
) -> list[Any]:
    """
    Generates a list of trials, randomly or fixed, based on a list of options.

    Parameters:
    - n (int):
        Number of trials to generate.
    - options (list):
        List of stimuli options. For example the temperature options of an experiment.
          If it contains one temperature, all trials will use it.
    - mode (str):
      "random" for a random mix of temperatures, "fixed" to use only the provided temperature(s).

    Returns:
    - list : List of temperatures for each trial.
    """
    n_temp = len(stimuli_options)

    if n % n_temp != 0:
        raise ValueError(
            f"Number of trials ({n}) must be divisible by the number of temperatures ({n_temp}) to have equal representation."
        )

    trials_per_temp = n // n_temp

    if mode == "random":
        # Ensure equal representation by repeating each temperature an equal number of times
        stim = stimuli_options * trials_per_temp
        shuffle(stim)
        return stim
    elif mode == "fixed":
        if len(stimuli_options) == 1:
            # All trials at the single specified temperature
            return stimuli_options * n
        else:
            # All trials use the list of temperatures in sequence until reaching num_trials
            return (stimuli_options * (n // n_temp + 1))[:n]

    raise ValueError("Invalid mode. Choose 'random' or 'fixed'.")
