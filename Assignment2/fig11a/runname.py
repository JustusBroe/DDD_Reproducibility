"""Run names and the default Figure 11a grid. No third-party imports."""

FULL_STEPS = 500000


def default_widths():
    """1..40 step 1, then 42..80 step 2. 40 is included once."""
    return list(range(1, 41)) + list(range(42, 81, 2))


def default_ns():
    return [12500, 25000, 50000]


def run_name(width, n, seed, max_steps=FULL_STEPS):
    name = f"c{int(width)}_n{int(n)}_s{int(seed)}"
    if int(max_steps) != FULL_STEPS:
        name += f"_t{int(max_steps)}"
    return name
