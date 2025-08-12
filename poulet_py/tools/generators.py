try:
    from random import shuffle
    from typing import Any, Literal
except ImportError as e:
    msg = """
Missing 'tools' module. Install options:
- Module:       pip install poulet_py[tools]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


def generate_stimulus_sequence(
    n: int,
    *,
    stimuli_options: list[Any],
    mode: Literal["random", "fixed"] = "random",
) -> list[Any]:
    """
    Generate a list of trials with specified stimuli distribution.

    Parameters
    ----------
    n : int
        Number of trials to generate. Must be divisible by the number of
        stimuli options when mode is 'random' or when multiple stimuli
        are provided in 'fixed' mode.
    stimuli_options : List[Any]
        List of possible stimulus values. For a single stimulus, all trials
        will use it. For multiple stimuli, distribution depends on mode.
    mode : {'random', 'fixed'}, optional
        Distribution mode:
        - 'random': Shuffled trials with equal representation of each stimulus
        - 'fixed': Trials use stimuli in sequence (or single stimulus repeated)
        (default: 'random')

    Returns
    -------
    List[Any]
        Generated list of stimuli for each trial

    Raises
    ------
    ValueError
        If n is not divisible by number of stimuli options (for relevant modes),
        or if mode is invalid.

    Notes
    -----
    - For 'random' mode with multiple stimuli, each appears exactly
        n//len(stimuli_options) times.
    - For 'fixed' mode with multiple stimuli, stimuli are repeated in sequence
        until n is reached.
    - For 'fixed' mode with single stimulus, that stimulus is repeated n times.

    Examples
    --------
    >>> generate_trials(4, stimuli_options=[1, 2], mode="random")
    [2, 1, 2, 1]  # Random order with equal representation

    >>> generate_trials(3, stimuli_options=[5], mode="fixed")
    [5, 5, 5]
    """
    n_stim = len(stimuli_options)

    if n_stim == 0:
        msg = "stimuli_options cannot be empty"
        raise ValueError(msg)

    # Validate input for modes that require equal representation
    if mode == "random" or (mode == "fixed" and n_stim > 1):
        if n % n_stim != 0:
            msg = f"Number of trials ({n}) must be divisible by the number"
            "of stimuli ({n_stim}) for equal representation in mode '{mode}'."
            raise ValueError(msg)

    if mode == "random":
        # Create balanced representation then shuffle
        trials = stimuli_options * (n // n_stim)
        shuffle(trials)
        return trials
    elif mode == "fixed":
        if n_stim == 1:
            return stimuli_options * n
        # For multiple stimuli in fixed mode, repeat the sequence
        return (stimuli_options * ((n // n_stim) + 1))[:n]

    msg = f"Invalid mode '{mode}'. Choose 'random' or 'fixed'."
    raise ValueError(msg)
