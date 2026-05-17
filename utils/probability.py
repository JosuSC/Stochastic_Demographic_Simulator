import random


def get_uniform(min_val: float, max_val: float) -> float:
    """Draw a value from a uniform distribution."""
    return random.uniform(min_val, max_val)


def get_exponential(mean: float) -> float:
    """Draw a value from an exponential distribution with the given mean.

    Uses random.expovariate(lambda=1/mean), which gives the requested mean.
    """
    if mean <= 0:
        return 0.0
    return random.expovariate(1.0 / mean)


def evaluate_probability(prob: float) -> bool:
    """Run a Bernoulli trial with probability prob."""
    return random.random() < prob
