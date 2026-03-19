try:
    from time import perf_counter_ns, sleep
except ImportError as e:
    msg = """
Missing 'tools' module. Install options:
- Module:       pip install poulet_py[tools]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


def precise_sleep(t: float, precision: float = 0.0001):
    duration_ns = int(t * 1e9)
    end = perf_counter_ns() + duration_ns
    precision_ns = int(precision * 1e9)

    while True:
        now = perf_counter_ns()
        remaining = end - now

        if remaining <= 0:
            break

        if remaining > precision_ns:
            sleep(precision)
        else:
            while perf_counter_ns() < end:
                pass
            break
