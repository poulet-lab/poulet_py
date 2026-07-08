try:
    from collections.abc import Sequence
    from random import shuffle
    from typing import Any, Literal
except ImportError as e:
    raise ImportError("""
Missing 'tools' module. Install options:
- Module:       pip install poulet_py[tools]
- Full:         pip install poulet_py[all]
""") from e


def repeat(
    l: Sequence[Any], n: int, *, mode: Literal["random", "sequential"] = "random"
) -> Sequence[Any]:
    """
    Generate a list of trials with specified stimuli distribution.

    Parameters
    ----------
    n : int
        Number of trials to generate. Must be divisible by the number of
        stimuli options when mode is 'random' or when multiple stimuli
        are provided in 'fixed' mode.
    stimuli_options : Sequence[Any]
        List of possible stimulus values. For a single stimulus, all trials
        will use it. For multiple stimuli, distribution depends on mode.
        note use sequential instead of fixed
        as it is deprecated and will be removed in future releases.
    mode : {'random', 'fixed', 'sequential'}, optional
        Distribution mode:
        - 'random': Shuffled trials with equal representation of each stimulus
        - 'fixed': Trials use stimuli in sequence (or single stimulus repeated)
        (default: 'random')

    Returns
    -------
    Sequence[Any]
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
    if len(l) == 0:
        return l

    _l = list(l) * n
    if mode == "random":
        shuffle(_l)
        return _l
    elif mode == "sequential":
        return _l

    raise ValueError(f"Invalid mode '{mode}'. Choose 'random' or 'sequential'.")
