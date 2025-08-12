from random import seed

from pytest import raises

from poulet_py import generate_stimulus_sequence

# Set random seed for reproducible tests
seed(42)


def test_single_stimulus_random():
    """Test with single stimulus in random mode (should just repeat)"""
    result = generate_stimulus_sequence(3, stimuli_options=[5], mode="random")
    assert result == [5, 5, 5]


def test_single_stimulus_fixed():
    """Test with single stimulus in fixed mode"""
    result = generate_stimulus_sequence(4, stimuli_options=["A"], mode="fixed")
    assert result == ["A", "A", "A", "A"]


def test_multiple_stimuli_random():
    """Test with multiple stimuli in random mode"""
    result = generate_stimulus_sequence(4, stimuli_options=[1, 2], mode="random")
    # Since we seeded random, we know the output will be [1, 2, 2, 1]
    assert sorted(result) == [1, 1, 2, 2]  # Check correct counts
    assert set(result) == {1, 2}  # Check all stimuli are present


def test_multiple_stimuli_fixed():
    """Test with multiple stimuli in fixed mode"""
    result = generate_stimulus_sequence(5, stimuli_options=["X", "Y", "Z", "I", "J"], mode="fixed")
    assert result == ["X", "Y", "Z", "I", "J"]


def test_divisibility_error_random():
    """Test error when n not divisible by number of stimuli in random mode"""
    with raises(ValueError):
        generate_stimulus_sequence(5, stimuli_options=[1, 2], mode="random")


def test_divisibility_error_fixed_multiple():
    """Test error when n not divisible by number of stimuli in fixed mode with multiple stimuli"""
    with raises(ValueError):
        generate_stimulus_sequence(5, stimuli_options=[1, 2, 3], mode="fixed")


def test_no_error_fixed_single():
    """Test no error when n not divisible by number of stimuli in fixed mode with single stimulus"""
    result = generate_stimulus_sequence(5, stimuli_options=[1], mode="fixed")
    assert result == [1, 1, 1, 1, 1]


def test_invalid_mode():
    """Test error when invalid mode is provided"""
    with raises(ValueError):
        generate_stimulus_sequence(4, stimuli_options=[1, 2], mode="invalid")


def test_empty_stimuli_options():
    """Test error when stimuli_options is empty"""
    with raises(ValueError):
        generate_stimulus_sequence(4, stimuli_options=[], mode="random")


def test_zero_trials():
    """Test with zero trials requested"""
    result = generate_stimulus_sequence(0, stimuli_options=[1, 2], mode="random")
    assert result == []


def test_large_input():
    """Test with large input to verify performance and correctness"""
    n = 1000
    options = list(range(10))
    result = generate_stimulus_sequence(n, stimuli_options=options, mode="random")
    assert len(result) == n
    assert all(x in options for x in result)
    # Check equal representation (since n is divisible by len(options))
    for option in options:
        assert result.count(option) == n // len(options)


def test_fixed_mode_sequence():
    """Verify fixed mode maintains exact sequence"""
    options = ["A", "B", "C"]
    result = generate_stimulus_sequence(6, stimuli_options=options, mode="fixed")
    assert result == ["A", "B", "C", "A", "B", "C"]


def test_random_mode_distribution():
    """Verify random mode produces different orderings on subsequent calls"""
    options = [1, 2, 3, 4]
    result1 = generate_stimulus_sequence(4, stimuli_options=options, mode="random")
    result2 = generate_stimulus_sequence(4, stimuli_options=options, mode="random")
    # Very small chance this could fail, but with seed set it shouldn't
    assert result1 != result2
